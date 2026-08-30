"""Node features (Eq.1/Eq.2).

x_v = t_v (entity-type one-hot, |T|) + s_v (word2vec, d=32) + p_v (temporal
relation-transition profile, |R|^2), written to feat_{split}{i}.npy.

Usage:  python -m preprocess.build_features --dataset cadets
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import proc_dir, set_random_seed  # noqa: E402

TOKEN_SPLIT = re.compile(r'[/\\._\-: ]+')
LAMBDA_DEFAULT = 1.925e-4     # ln(2)/3600, one-hour half life
BUFFER_L_DEFAULT = 20


def tokenize(name: str):
    """§5.2: split on / \\ . _ - : space, lowercase, drop pure-digit tokens."""
    return [t for t in TOKEN_SPLIT.split(name.lower()) if t and not t.isdigit()]


def transition_profile(src, dst, rel, ts_nanos, num_nodes, n_rel,
                       buffer_L=BUFFER_L_DEFAULT, lam=LAMBDA_DEFAULT,
                       ts_is_seconds=False, normalize=True):
    """Eq.2, vectorized.

    Per node: keep the last `buffer_L` events (FIFO buffer B_v), accumulate the
    weight of every *consecutive* pair inside the buffer (D3, continuous bigram)

        P_v[r(e_{i-1}), r(e_i)] += exp(-lam * (t_now - t(e_i)))

    with t_now the node's own last event time (D5) and time in seconds.
    Post-processing: flatten to |R|^2, log1p, L2-normalize (zero stays zero).

    A node participates in an event as either endpoint, so each event
    contributes one buffer entry to its src and one to its dst.
    """
    node = np.concatenate([np.asarray(src), np.asarray(dst)])
    r = np.concatenate([np.asarray(rel), np.asarray(rel)]).astype(np.int64)
    t = np.concatenate([np.asarray(ts_nanos), np.asarray(ts_nanos)])
    t = t.astype(np.float64) if ts_is_seconds else t.astype(np.float64) / 1e9
    ev_idx = np.tile(np.arange(len(src), dtype=np.int64), 2)

    order = np.lexsort((ev_idx, t, node))   # by node, then time, then event order
    node, r, t = node[order], r[order], t[order]
    del order, ev_idx

    n_rows = len(node)
    P = np.zeros((num_nodes, n_rel * n_rel), dtype=np.float32)
    if n_rows == 0:
        return P

    # contiguous group layout of the node-sorted rows
    ids, starts, counts = np.unique(node, return_index=True, return_counts=True)
    grp = np.repeat(np.arange(len(ids)), counts)
    pos = np.arange(n_rows) - starts[grp]                    # index inside group
    keep = pos >= (counts[grp] - buffer_L)                   # last L events only
    node, r, t, grp = node[keep], r[keep], t[keep], grp[keep]

    # after trimming, groups are still contiguous; pair each row with the next
    same = grp[:-1] == grp[1:]
    if not same.any():
        return P
    i = np.flatnonzero(same)                                 # first of the pair
    t_now = np.empty(len(ids), dtype=np.float64)
    last_of_group = np.flatnonzero(np.r_[grp[:-1] != grp[1:], True])
    t_now[grp[last_of_group]] = t[last_of_group]

    w = np.exp(-lam * (t_now[grp[i]] - t[i + 1]))
    flat = r[i] * n_rel + r[i + 1]
    np.add.at(P, (node[i], flat), w.astype(np.float32))

    np.log1p(P, out=P)
    if normalize:
        nrm = np.linalg.norm(P, axis=1, keepdims=True)
        np.divide(P, nrm, out=P, where=nrm > 0)
    return P


def node_names(uuids, names):
    """uuid bytes array -> list of name strings ('' when unknown)."""
    return [names.get(u.decode(), '') for u in uuids]


def build(dataset, w2v_dim=32, buffer_L=BUFFER_L_DEFAULT, lam=LAMBDA_DEFAULT, seed=0):
    from gensim.models import Word2Vec

    set_random_seed(seed)
    out = proc_dir(dataset)
    with open(os.path.join(out, 'meta.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)
    with open(os.path.join(out, 'relation_vocab.json'), 'r', encoding='utf-8') as f:
        rvocab = json.load(f)
    with open(os.path.join(out, 'names.json'), 'r', encoding='utf-8') as f:
        names = json.load(f)
    n_type, n_rel = meta['n_node_type'], meta['n_relation']
    tags = ([f'train{i}' for i in range(meta['n_train'])] +
            [f'test{i}' for i in range(meta['n_test'])])

    # ---- word2vec corpus: training-graph node names only (no test leakage)
    corpus, n_named = [], 0
    for tag in tags:
        if not tag.startswith('train'):
            continue
        g = np.load(os.path.join(out, f'graph_{tag}.npz'))
        for nm in node_names(g['uuids'], names):
            if nm:
                toks = tokenize(nm)
                if toks:
                    corpus.append(toks)
                    n_named += 1
    print(f'word2vec corpus: {n_named} named training nodes, '
          f'{len(set(t for s in corpus for t in s))} distinct tokens')
    w2v = Word2Vec(sentences=corpus, vector_size=w2v_dim, window=5, min_count=1,
                   epochs=10, seed=seed, workers=1)
    kv = w2v.wv
    del corpus

    stats = {}
    for tag in tags:
        t0 = time.time()
        g = np.load(os.path.join(out, f'graph_{tag}.npz'))
        N = int(g['num_nodes'])

        onehot = np.zeros((N, n_type), dtype=np.float32)
        onehot[np.arange(N), g['ntype'].astype(np.int64)] = 1.0

        sem = np.zeros((N, w2v_dim), dtype=np.float32)
        n_hit = 0
        for i, nm in enumerate(node_names(g['uuids'], names)):
            if not nm:
                continue
            vecs = [kv[t] for t in tokenize(nm) if t in kv]
            if vecs:
                sem[i] = np.mean(vecs, axis=0)
                n_hit += 1

        ev = pd.read_parquet(os.path.join(out, f'events_{tag}.parquet'))
        prof = transition_profile(ev['src_id'].to_numpy(), ev['dst_id'].to_numpy(),
                                 ev['relation_id'].to_numpy(), ev['ts_nanos'].to_numpy(),
                                 N, n_rel, buffer_L=buffer_L, lam=lam)
        del ev

        feat = np.concatenate([onehot, sem, prof], axis=1)
        assert np.isfinite(feat).all(), f'{tag}: non-finite features'
        np.save(os.path.join(out, f'feat_{tag}.npy'), feat)
        stats[tag] = {'nodes': N, 'dim': int(feat.shape[1]),
                      'semantic_coverage': round(n_hit / max(N, 1), 4),
                      'profile_nonzero_rows': int((prof != 0).any(axis=1).sum())}
        print(f'  {tag}: {feat.shape} semantic={n_hit}/{N} '
              f'profile_rows={stats[tag]["profile_nonzero_rows"]}/{N} '
              f'{time.time() - t0:.0f}s', flush=True)
        del onehot, sem, prof, feat

    info = {'n_node_type': n_type, 'n_relation': n_rel, 'w2v_dim': w2v_dim,
            'feat_dim': n_type + w2v_dim + n_rel * n_rel,
            'buffer_L': buffer_L, 'lambda_decay': lam,
            'block_slices': {'type': [0, n_type], 'semantic': [n_type, n_type + w2v_dim],
                             'profile': [n_type + w2v_dim, n_type + w2v_dim + n_rel * n_rel]},
            'per_graph': stats,
            'relation_to_id': rvocab['relation_to_id']}
    with open(os.path.join(out, 'feature_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    print(f"feature dim = {info['feat_dim']} = {n_type} + {w2v_dim} + {n_rel}^2")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='RECAL feature builder')
    ap.add_argument('--dataset', required=True, choices=['trace', 'theia', 'cadets'])
    ap.add_argument('--w2v_dim', type=int, default=32)
    ap.add_argument('--buffer_L', type=int, default=BUFFER_L_DEFAULT)
    ap.add_argument('--lambda_decay', type=float, default=LAMBDA_DEFAULT)
    a = ap.parse_args()
    build(a.dataset, a.w2v_dim, a.buffer_L, a.lambda_decay)
