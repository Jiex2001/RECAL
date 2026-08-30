"""CDM18 -> provenance graph parser.

Adapted from MAGIC/utils/trace_parser.py, with four changes:

  1. the full (pre-dedup) event stream is written to events_{split}{i}.parquet
  2. multi-edge dedup is by (u, v, r) instead of (u, v)
  3. no malicious-label filtering on training graphs (switch kept)
  4. relation_vocab.json / node_type_vocab.json are written out

Entity names: MAGIC's three name regexes are
position-sensitive -- they require the wanted key to be the FIRST key of
`properties.map` -- which only ever holds for trace.  On theia they match 18 of
2623 processes and 0 of 9702 files; on cadets 0 of both, because cadets keeps
entity names only on Event records.  We therefore parse non-Event records with
json.loads and dig the real paths, and additionally harvest names off Event
records (predicateObjectPath / predicateObject2Path / properties.map.exec).

Usage:  python -m preprocess.parse_cdm --dataset cadets
"""

import argparse
import array
import json
import os
import pickle as pkl
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import HERE, proc_dir  # noqa: E402

# Dataset file split, copied verbatim from MAGIC/utils/trace_parser.py (F12).
metadata = {
    'trace': {
        'train': ['ta1-trace-e3-official-1.json', 'ta1-trace-e3-official-1.json.1',
                  'ta1-trace-e3-official-1.json.2', 'ta1-trace-e3-official-1.json.3'],
        'test': ['ta1-trace-e3-official-1.json', 'ta1-trace-e3-official-1.json.1',
                 'ta1-trace-e3-official-1.json.2', 'ta1-trace-e3-official-1.json.3',
                 'ta1-trace-e3-official-1.json.4']
    },
    'theia': {
        'train': ['ta1-theia-e3-official-6r.json', 'ta1-theia-e3-official-6r.json.1',
                  'ta1-theia-e3-official-6r.json.2', 'ta1-theia-e3-official-6r.json.3'],
        'test': ['ta1-theia-e3-official-6r.json.8']
    },
    'cadets': {
        'train': ['ta1-cadets-e3-official.json', 'ta1-cadets-e3-official.json.1',
                  'ta1-cadets-e3-official.json.2', 'ta1-cadets-e3-official-2.json.1'],
        'test': ['ta1-cadets-e3-official-2.json']
    }
}

CDM = 'com.bbn.tc.schema.avro.cdm18.'
EVENT_TAG = CDM + 'Event'

# Regexes copied from MAGIC/utils/trace_parser.py (used on Event records only).
pattern_src = re.compile(r'subject\":{\"' + CDM + r'UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"' + CDM + r'UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"' + CDM + r'UUID\":\"(.*?)\"}')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')
# D6: names carried by Event records.
pattern_obj_path = re.compile(r'predicateObjectPath\":\{\"string\":\"(.*?)\"')
pattern_obj2_path = re.compile(r'predicateObject2Path\":\{\"string\":\"(.*?)\"')
pattern_event_exec = re.compile(r'\"exec\":\"(.*?)\"')

SKIP_TYPES = {'Event', 'Host', 'TimeMarker', 'StartMarker', 'UnitDependency', 'EndMarker'}
NAME_KEYS = ('path', 'filename', 'name', 'exec', 'cmdLine')
VOLUME_RE = re.compile(r'\.json(\.\d+)?$')
NULL_UUID = '00000000-0000-0000-0000-000000000000'


def volume_files(dataset):
    """Every raw volume of the dataset, deterministic order.

    Stricter than MAGIC's `'json' in file` scan, which would also swallow the
    ta1-*.json.tar.gz archives sitting next to the extracted volumes.
    """
    d = os.path.join(HERE, 'data', 'dataset', dataset)
    return sorted(f for f in os.listdir(d) if VOLUME_RE.search(f))


def record_type(line):
    """CDM record class of one line, read off the head of the JSON."""
    i = line.find(CDM, 0, 256)
    if i < 0:
        return None
    i += len(CDM)
    j = line.find('"', i)
    return line[i:j] if j > 0 else None


# ------------------------------------------------------------- pass 1 -------

def _dig_name(rec_type, body):
    """Pull the display name out of one decoded non-Event CDM record (D6)."""
    if rec_type == 'NetFlowObject':
        addr = body.get('remoteAddress')
        return addr if isinstance(addr, str) and addr else None
    props = (body.get('properties') or {}).get('map') or {}
    base = body.get('baseObject') if isinstance(body.get('baseObject'), dict) else {}
    base_props = (base.get('properties') or {}).get('map') or {}
    for src in (props, base_props):
        for k in NAME_KEYS:
            v = src.get(k)
            if isinstance(v, str) and v:
                return v
    cmd = body.get('cmdLine')
    if isinstance(cmd, dict):
        cmd = cmd.get('string')
    if isinstance(cmd, str) and cmd:
        return cmd
    return None


def pass1_entities(dataset, verbose=True):
    """uuid -> type string and uuid -> name string, over *all* volumes."""
    id_type, id_name = {}, {}
    for file in volume_files(dataset):
        path = os.path.join(HERE, 'data', 'dataset', dataset, file)
        t0, n = time.time(), 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                rec_type = record_type(line)
                if rec_type is None or rec_type in SKIP_TYPES:
                    continue
                n += 1
                try:
                    datum = json.loads(line)['datum']
                except Exception:
                    continue
                rec_key = next(iter(datum))
                body = datum[rec_key]
                uuid = body.get('uuid')
                if not isinstance(uuid, str) or uuid == NULL_UUID:
                    continue
                # MAGIC takes the first `type":"..."` on the line; records with
                # no type field fall back to the record class name.
                subject_type = body.get('type')
                if not isinstance(subject_type, str):
                    base = body.get('baseObject')
                    subject_type = base.get('type') if isinstance(base, dict) else None
                if not isinstance(subject_type, str):
                    subject_type = rec_type
                if subject_type == 'SUBJECT_UNIT':
                    continue
                id_type[uuid] = subject_type
                if uuid not in id_name:
                    name = _dig_name(rec_type, body)
                    if name:
                        id_name[uuid] = name
        if verbose:
            print(f'  pass1 {file}: {n} entity records in {time.time() - t0:.0f}s '
                  f'(types={len(id_type)}, names={len(id_name)})', flush=True)
    return id_type, id_name


# ------------------------------------------------------------- pass 2 -------

class Vocab:
    """Insertion-ordered string -> int, shared across a dataset's graphs."""

    def __init__(self):
        self.d = {}

    def __call__(self, key):
        i = self.d.get(key)
        if i is None:
            i = len(self.d)
            self.d[key] = i
        return i

    def __len__(self):
        return len(self.d)


def parse_volume(dataset, file, id_type, id_name, node_types, relations,
                 malicious=None, verbose=True):
    """One raw volume -> (graph arrays, full event stream).

    Mirrors MAGIC read_single_graph: reverse READ/RECV/LOAD (F3), sort by
    timestamp, number nodes in time order, then dedup -- by (u, v, r) for us.
    Event records are also mined for entity names (D6 pass-2 backfill).
    """
    path = os.path.join(HERE, 'data', 'dataset', dataset, file)
    src_a, dst_a = array.array('i'), array.array('i')
    rel_a, ts_a = array.array('h'), array.array('q')
    uuid_to_local, n_local = {}, 0

    t0, n_lines, n_skip = time.time(), 0, 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if EVENT_TAG not in line:
                continue
            n_lines += 1
            m_src = pattern_src.search(line)
            if m_src is None:
                continue
            srcId = m_src.group(1)
            srcType = id_type.get(srcId)
            if srcType is None:
                continue
            edge_type = pattern_type.search(line).group(1)
            ts = int(pattern_time.search(line).group(1))
            reverse = 'READ' in edge_type or 'RECV' in edge_type or 'LOAD' in edge_type

            m = pattern_event_exec.search(line)          # D6: cadets process names
            if m is not None and srcId not in id_name:
                id_name[srcId] = m.group(1)

            for pat_dst, pat_path in ((pattern_dst1, pattern_obj_path),
                                      (pattern_dst2, pattern_obj2_path)):
                m_dst = pat_dst.search(line)
                if m_dst is None or m_dst.group(1) == 'null':
                    continue
                dstId = m_dst.group(1)
                dstType = id_type.get(dstId)
                if dstType is None:
                    continue
                m = pat_path.search(line)                 # D6: cadets file paths
                if m is not None and dstId not in id_name:
                    id_name[dstId] = m.group(1)

                u, ut, v, vt = srcId, srcType, dstId, dstType
                if malicious is not None:   # F4 / D1, only when magic_label_filter=true
                    if (u in malicious and ut != 'MemoryObject') or \
                       (v in malicious and vt != 'MemoryObject'):
                        n_skip += 1
                        continue
                if reverse:                               # F3 data-flow direction
                    u, ut, v, vt = v, vt, u, ut
                node_types(ut)
                node_types(vt)
                iu = uuid_to_local.get(u)
                if iu is None:
                    iu = uuid_to_local[u] = n_local
                    n_local += 1
                iv = uuid_to_local.get(v)
                if iv is None:
                    iv = uuid_to_local[v] = n_local
                    n_local += 1
                src_a.append(iu)
                dst_a.append(iv)
                rel_a.append(relations(edge_type))
                ts_a.append(ts)

    src = np.frombuffer(src_a, dtype=np.int32).copy()
    dst = np.frombuffer(dst_a, dtype=np.int32).copy()
    rel = np.frombuffer(rel_a, dtype=np.int16).copy()
    ts = np.frombuffer(ts_a, dtype=np.int64).copy()
    del src_a, dst_a, rel_a, ts_a
    n_ev = len(src)
    assert n_local > 0 and n_ev > 0, f'{file}: no events parsed'

    order = np.argsort(ts, kind='stable')            # MAGIC: lines.sort(key=ts)
    src, dst, rel, ts = src[order], dst[order], rel[order], ts[order]
    del order

    # Renumber nodes by first appearance in time order, src before dst, exactly
    # as MAGIC's node_map does while walking the sorted event list.
    inter = np.empty(2 * n_ev, dtype=np.int32)
    inter[0::2], inter[1::2] = src, dst
    _, first_pos = np.unique(inter, return_index=True)   # indexed by local id
    del inter
    assert len(first_pos) == n_local
    remap = np.empty(n_local, dtype=np.int32)
    remap[np.argsort(first_pos, kind='stable')] = np.arange(n_local, dtype=np.int32)
    src, dst = remap[src], remap[dst]

    uuids = np.empty(n_local, dtype='S36')
    ntype = np.empty(n_local, dtype=np.int16)
    for u, i in uuid_to_local.items():
        j = remap[i]
        uuids[j] = u.encode()
        ntype[j] = node_types.d[id_type[u]]

    # (u, v, r) dedup, earliest edge wins: arrays are ts-sorted and
    # np.unique(return_index=True) reports the first occurrence of each key.
    key = (src.astype(np.int64) * n_local + dst) * (len(relations) + 1) + rel
    _, keep = np.unique(key, return_index=True)
    del key
    keep.sort()
    edge_index = np.stack([src[keep], dst[keep]]).astype(np.int64)
    etype = rel[keep].copy()

    if verbose:
        print(f'  {file}: {n_lines} event lines -> {n_ev} events, {n_local} nodes, '
              f'{len(keep)} edges (label-skipped {n_skip}), {time.time() - t0:.0f}s',
              flush=True)
    events = pd.DataFrame({'src_id': src, 'dst_id': dst, 'relation_id': rel,
                           'ts_nanos': ts})
    return dict(edge_index=edge_index, etype=etype, ntype=ntype,
                num_nodes=np.int64(n_local), uuids=uuids), events


def run(dataset, magic_label_filter=False):
    out = proc_dir(dataset)
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(HERE, 'data', 'label', f'{dataset}.txt'), 'r', encoding='utf-8') as f:
        malicious_uuids = {l.strip() for l in f if l.strip()}
    print(f'{dataset}: {len(malicious_uuids)} labelled malicious entities')

    cache = os.path.join(out, 'entities.pkl')
    if os.path.exists(cache):
        print('loading cached pass-1 entity maps')
        with open(cache, 'rb') as f:
            id_type, id_name = pkl.load(f)
    else:
        print('pass 1: entity types and names')
        id_type, id_name = pass1_entities(dataset)
        with open(cache, 'wb') as f:
            pkl.dump((id_type, id_name), f)
    print(f'  {len(id_type)} typed entities, {len(id_name)} named entities')

    node_types, relations = Vocab(), Vocab()
    mfilter = malicious_uuids if magic_label_filter else None

    print('pass 2: events and graphs')
    graphs, counts, built = {}, {}, {}
    for split in ('train', 'test'):
        for i, file in enumerate(metadata[dataset][split]):
            tag = f'{split}{i}'
            filt = mfilter if split == 'train' else None
            ck = (file, filt is not None)
            if ck in built:   # trace lists the same 4 volumes under both splits
                other = built[ck]
                print(f'  {tag}: identical to {other} ({file}), copying')
                for pat in (f'graph_{{}}.npz', f'events_{{}}.parquet'):
                    shutil.copyfile(os.path.join(out, pat.format(other)),
                                    os.path.join(out, pat.format(tag)))
                counts[tag] = dict(counts[other])
                graphs[tag] = graphs[other]
                continue
            g, ev = parse_volume(dataset, file, id_type, id_name, node_types, relations, filt)
            np.savez(os.path.join(out, f'graph_{tag}.npz'), **g)
            ev.to_parquet(os.path.join(out, f'events_{tag}.parquet'), index=False)
            vals, cnts = np.unique(ev['relation_id'].to_numpy(), return_counts=True)
            counts[tag] = {int(k): int(v) for k, v in zip(vals, cnts)}
            graphs[tag] = g
            built[ck] = tag
            del ev

    # malicious node ids in the concatenated test graphs (MAGIC read_graphs)
    test_node_map, offset = {}, 0
    for i in range(len(metadata[dataset]['test'])):
        g = graphs[f'test{i}']
        for local, u in enumerate(g['uuids']):
            u = u.decode()
            if u not in test_node_map:
                test_node_map[u] = local + offset
        offset += int(g['num_nodes'])
    final_malicious, malicious_names = [], []
    for e in sorted(malicious_uuids):
        if e in test_node_map and id_type.get(e) not in (None, 'MemoryObject', 'UnnamedPipeObject'):
            final_malicious.append(test_node_map[e])
            malicious_names.append(id_name.get(e, e))
    with open(os.path.join(out, 'malicious.pkl'), 'wb') as f:
        pkl.dump((final_malicious, malicious_names), f)

    with open(os.path.join(out, 'names.json'), 'w', encoding='utf-8') as f:
        json.dump(id_name, f)
    with open(os.path.join(out, 'types.json'), 'w', encoding='utf-8') as f:
        json.dump(id_type, f)
    with open(os.path.join(out, 'relation_vocab.json'), 'w', encoding='utf-8') as f:
        json.dump({'relation_to_id': relations.d, 'event_counts': counts}, f, indent=1)
    with open(os.path.join(out, 'node_type_vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(node_types.d, f, indent=1)
    meta = {
        'dataset': dataset,
        'n_train': len(metadata[dataset]['train']),
        'n_test': len(metadata[dataset]['test']),
        'n_node_type': len(node_types),
        'n_relation': len(relations),
        'n_malicious': len(final_malicious),
        'n_named_entities': len(id_name),
        'magic_label_filter': magic_label_filter,
        'graphs': {t: {'num_nodes': int(g['num_nodes']), 'num_edges': int(g['etype'].shape[0])}
                   for t, g in graphs.items()},
    }
    with open(os.path.join(out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='RECAL CDM parser')
    ap.add_argument('--dataset', required=True, choices=['trace', 'theia', 'cadets'])
    ap.add_argument('--magic_label_filter', action='store_true',
                    help='restore MAGIC F4 behaviour (drop malicious nodes from train graphs)')
    a = ap.parse_args()
    run(a.dataset, a.magic_label_filter)
