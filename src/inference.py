"""Inference-side error computation and embeddings.

Nodes are split into `n_rounds` mutually exclusive batches; each batch is masked
once with the rest of the graph visible, so every node gets exactly one error per
relation it participates in.

Errors are stored as three parallel arrays instead of the nested
nested `{node: {relation: error}}`, which is not representable at 3.4M test
nodes; `ErrorTable.as_dict()` reproduces it on demand.
"""

import numpy as np
import torch


class ErrorTable:
    """Flat (node, relation, error) triples for one graph."""

    def __init__(self, node, rel, err, num_nodes):
        self.node = np.asarray(node, dtype=np.int64)
        self.rel = np.asarray(rel, dtype=np.int32)
        self.err = np.asarray(err, dtype=np.float32)
        self.num_nodes = int(num_nodes)

    def __len__(self):
        return self.node.shape[0]

    def as_dict(self):
        out = {}
        for v, r, e in zip(self.node, self.rel, self.err):
            out.setdefault(int(v), {})[int(r)] = float(e)
        return out

    def by_relation(self, n_rel):
        return [self.err[self.rel == r] for r in range(n_rel)]

    def shifted(self, offset):
        return ErrorTable(self.node + offset, self.rel, self.err, self.num_nodes)

    @staticmethod
    def concat(tables, offsets):
        node = np.concatenate([t.node + o for t, o in zip(tables, offsets)])
        rel = np.concatenate([t.rel for t in tables])
        err = np.concatenate([t.err for t in tables])
        return ErrorTable(node, rel, err, sum(t.num_nodes for t in tables))


def round_batches(num_nodes, n_rounds, seed=0):
    """Random mutually exclusive node batches, deterministic given `seed`."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_nodes)
    return np.array_split(perm, n_rounds)


@torch.no_grad()
def compute_errors(model, g, n_rounds=10, seed=0):
    """§6.5 -> ErrorTable for one graph."""
    model.eval()
    nodes, rels, errs = [], [], []
    for batch in round_batches(g.num_nodes, n_rounds, seed):
        if batch.size == 0:
            continue
        mask = torch.zeros(g.num_nodes, dtype=torch.bool, device=g.x.device)
        mask[torch.from_numpy(batch).to(g.x.device)] = True
        for r, item in enumerate(model.node_errors(g, mask)):
            if item is None:
                continue
            n, e = item
            nodes.append(n.cpu().numpy())
            rels.append(np.full(n.shape[0], r, dtype=np.int32))
            errs.append(e.float().cpu().numpy())
    if not nodes:
        return ErrorTable([], [], [], g.num_nodes)
    return ErrorTable(np.concatenate(nodes), np.concatenate(rels),
                      np.concatenate(errs), g.num_nodes)


@torch.no_grad()
def compute_embeddings(model, g, sample=None, seed=0):
    """Node embeddings (§6.1), optionally subsampled to `sample` rows."""
    model.eval()
    rep = model.embed(g)
    if sample is not None and rep.shape[0] > sample:
        idx = np.random.RandomState(seed).permutation(rep.shape[0])[:sample]
        rep = rep[torch.from_numpy(np.sort(idx)).to(rep.device)]
    return rep.float()
