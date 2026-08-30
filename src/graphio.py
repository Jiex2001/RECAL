"""Graph + feature loading, per-relation edge caches, calibration split (§5.3).

Reads what `preprocess/` produced (`graph_{tag}.npz`, `feat_{tag}.npy`,
`meta.json`, `relation_vocab.json`, `feature_info.json`) and hands the training
code plain torch tensors, so the training path is backend-agnostic.

Never touches `malicious.pkl` (§13.3 label discipline); test labels are loaded
by `src/metrics.py` only.
"""

import json
import os
from collections import OrderedDict

import numpy as np
import torch

from .utils import proc_dir


def _read_json(dataset, name):
    with open(os.path.join(proc_dir(dataset), name), 'r', encoding='utf-8') as f:
        return json.load(f)


def load_meta(dataset):
    return _read_json(dataset, 'meta.json')


def load_feature_info(dataset):
    return _read_json(dataset, 'feature_info.json')


def load_relation_vocab(dataset):
    return _read_json(dataset, 'relation_vocab.json')


def relation_names(dataset):
    """id -> relation name, ordered by id."""
    r2i = load_relation_vocab(dataset)['relation_to_id']
    names = [None] * len(r2i)
    for k, v in r2i.items():
        names[v] = k
    return names


def split_tags(meta, calib_index=-1):
    """(train tags, calib tag, test tags).  §5.3: calib = last training graph."""
    n_tr = meta['n_train']
    c = calib_index % n_tr
    train = [f'train{i}' for i in range(n_tr) if i != c]
    test = [f'test{i}' for i in range(meta['n_test'])]
    return train, f'train{c}', test


def relation_freq(dataset, train_tags):
    """f_r = event count summed over the graphs actually trained on.

    §6.2 says "training-set event counts"; the calibration graph is excluded from
    training by §5.3, so it is excluded here too.
    """
    rv = load_relation_vocab(dataset)
    n_rel = len(rv['relation_to_id'])
    f = np.zeros(n_rel, dtype=np.float64)
    for tag in train_tags:
        for k, v in rv['event_counts'][tag].items():
            f[int(k)] += v
    return f


def feature_columns(dataset, use_semantic=True, use_profile=True):
    """Column index of the enabled Eq.1 blocks (feature ablations 8 / semantic)."""
    sl = load_feature_info(dataset)['block_slices']
    cols = [np.arange(*sl['type'])]
    if use_semantic:
        cols.append(np.arange(*sl['semantic']))
    if use_profile:
        cols.append(np.arange(*sl['profile']))
    return np.concatenate(cols)


# ------------------------------------------------------------------ graph ----

class Graph:
    """One provenance graph, tensors all on the same device."""

    def __init__(self, tag, num_nodes, x, edge_index, etype, edge_attr, ntype,
                 rel_src, rel_dst_local, rel_part):
        self.tag = tag
        self.num_nodes = num_nodes
        self.x = x                       # (N, F) float32
        self.edge_index = edge_index     # (2, E) int64, row 0 = src, row 1 = dst
        self.etype = etype               # (E,) int64
        self.edge_attr = edge_attr       # (E, R) float32 one-hot (F7)
        self.ntype = ntype               # (N,) int64
        self.rel_src = rel_src           # per relation: (E_r*,) src node ids
        self.rel_dst_local = rel_dst_local   # per relation: (E_r*,) index into part
        self.rel_part = rel_part         # per relation: (P_r,) node ids, sorted

    @property
    def n_feat(self):
        return self.x.shape[1]

    @property
    def num_edges(self):
        return self.edge_index.shape[1]

    @property
    def device(self):
        return self.x.device

    def nbytes(self):
        n = self.x.numel() * 4 + self.edge_attr.numel() * 4
        n += (self.edge_index.numel() + self.etype.numel() + self.ntype.numel()) * 8
        n += sum(t.numel() for t in self.rel_src + self.rel_dst_local + self.rel_part) * 8
        return n

    def to(self, device):
        if self.x.device == device:
            return self
        mv = lambda t: t.to(device, non_blocking=True)  # noqa: E731
        return Graph(self.tag, self.num_nodes, mv(self.x), mv(self.edge_index),
                     mv(self.etype), mv(self.edge_attr), mv(self.ntype),
                     [mv(t) for t in self.rel_src], [mv(t) for t in self.rel_dst_local],
                     [mv(t) for t in self.rel_part])


def _rel_caches(edge_index, etype, n_rel, num_nodes):
    """§6.3: per relation the symmetrized, deduplicated edge set + participants.

    m_v^r averages over neighbours reachable by an r edge in either direction, so
    the r subgraph is symmetrized and (dst, src) duplicates removed.  Participants
    are exactly the nodes appearing as a dst of the symmetrized set.
    """
    src, dst = edge_index[0], edge_index[1]
    rel_src, rel_dst_local, rel_part = [], [], []
    for r in range(n_rel):
        idx = np.flatnonzero(etype == r)
        if idx.size == 0:
            rel_src.append(np.empty(0, dtype=np.int64))
            rel_dst_local.append(np.empty(0, dtype=np.int64))
            rel_part.append(np.empty(0, dtype=np.int64))
            continue
        u, v = src[idx].astype(np.int64), dst[idx].astype(np.int64)
        s = np.concatenate([u, v])
        d = np.concatenate([v, u])
        _, keep = np.unique(d * num_nodes + s, return_index=True)
        s, d = s[keep], d[keep]                      # d is sorted ascending
        part = np.unique(d)
        rel_src.append(s)
        rel_dst_local.append(np.searchsorted(part, d))
        rel_part.append(part)
    return rel_src, rel_dst_local, rel_part


class GraphStore:
    """Loads graphs once, keeps a byte-bounded LRU of device-resident copies.

    MAGIC re-reads every graph from disk on every epoch; here the CPU copy is
    read once and the device copy reused, which is why 50 epochs cost seconds.
    """

    def __init__(self, dataset, cfg, device):
        self.dataset = dataset
        self.device = device
        self.meta = load_meta(dataset)
        self.n_rel = self.meta['n_relation']
        self.cols = feature_columns(dataset, cfg['feature']['use_semantic'],
                                    cfg['feature']['use_profile'])
        finfo = load_feature_info(dataset)
        self.full_feat_dim = finfo['feat_dim']
        self.n_feat = len(self.cols)
        self._cpu = {}
        self._gpu = OrderedDict()
        self._budget = int(cfg['train'].get('gpu_cache_gb', 16) * (1 << 30))

    def cpu(self, tag):
        g = self._cpu.get(tag)
        if g is not None:
            return g
        d = proc_dir(self.dataset)
        z = np.load(os.path.join(d, f'graph_{tag}.npz'))
        edge_index = z['edge_index'].astype(np.int64)
        etype = z['etype'].astype(np.int64)
        ntype = z['ntype'].astype(np.int64)
        num_nodes = int(z['num_nodes'])

        feat = np.load(os.path.join(d, f'feat_{tag}.npy'), mmap_mode='r')
        assert feat.shape[0] == num_nodes, f'{tag}: feature/node count mismatch'
        assert feat.shape[1] == self.full_feat_dim
        if len(self.cols) == feat.shape[1]:
            x = np.ascontiguousarray(feat)
        else:
            x = np.ascontiguousarray(feat[:, self.cols])
        del feat

        edge_attr = np.zeros((etype.shape[0], self.n_rel), dtype=np.float32)
        edge_attr[np.arange(etype.shape[0]), etype] = 1.0

        rs, rd, rp = _rel_caches(edge_index, etype, self.n_rel, num_nodes)
        t = torch.from_numpy
        g = Graph(tag, num_nodes, t(x), t(edge_index), t(etype), t(edge_attr),
                  t(ntype), [t(a) for a in rs], [t(a) for a in rd], [t(a) for a in rp])
        self._cpu[tag] = g
        return g

    def get(self, tag):
        """Graph on `self.device`, cached."""
        g = self._gpu.get(tag)
        if g is not None:
            self._gpu.move_to_end(tag)
            return g
        g = self.cpu(tag)
        if self.device.type == 'cpu':
            return g
        g = g.to(self.device)
        self._gpu[tag] = g
        used = sum(v.nbytes() for v in self._gpu.values())
        while used > self._budget and len(self._gpu) > 1:
            _, old = self._gpu.popitem(last=False)
            used -= old.nbytes()
            del old
        return g

    def drop_gpu(self):
        self._gpu.clear()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


def test_offsets(store, test_tags):
    """Node-id offset of each test graph in the concatenated test node space."""
    off, cur = [], 0
    for tag in test_tags:
        off.append(cur)
        cur += int(store.meta['graphs'][tag]['num_nodes'])
    return off, cur


def concat_edge_index(dataset, tags):
    """(2, E) int64 edges of `tags` shifted into their concatenated node space.

    Only the `edge_index` array of each npz is touched, so this is cheap enough to
    call from a script that has no GraphStore (the metrics module needs it for the
    2-hop counting protocol).
    """
    d = proc_dir(dataset)
    parts, cur = [], 0
    for tag in tags:
        z = np.load(os.path.join(d, f'graph_{tag}.npz'))
        parts.append(z['edge_index'].astype(np.int64) + cur)
        cur += int(z['num_nodes'])
    return (np.concatenate(parts, axis=1) if parts
            else np.zeros((2, 0), dtype=np.int64))
