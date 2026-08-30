"""Offline sweep of M2 scoring rules.

    python -m scripts.sweep_scoring [--datasets cadets theia trace] [--tables a b c d e]

Reuses the trained model and the saved `errors_test.pkl` of `runs/{ds}_full`, so
nothing is retrained: only the detection stage changes.  The stage-1 knn score is
recomputed once per dataset and cached as `runs/{ds}_full/knn.npz`.

    a  relation combination C, with the alpha=0.90 screen kept in place
    b  stage combination, with C fixed at `--combine`
    c  malicious survival and best-F1 against alpha
    d  ranking candidates by lam*rank(knn) + (1-lam)*rank(C)
    e  lexicographic fallback: non-candidates ranked by knn instead of zeroed

Table (a)'s `fisher` row reproduces `runs/{ds}_full/metrics.json`, which is what
pins the sweep to the production path (at alpha=0.90 the production lex fallback
leaves the metrics unchanged, so the zero fallback used here is equivalent).
Tables (b)-(e) hold C at `--combine`.  Every row is scored under the THREATRACE
protocol, same as the reported numbers.  Test labels stay inside `src.metrics`:
this script only receives the aggregate numbers back.
"""

import argparse
import os
import pickle as pkl
import sys

import numpy as np
import torch
from scipy.stats import chi2, rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import detect, graphio                                     # noqa: E402
from src.calibrate import CalibrationTable                          # noqa: E402
from src.inference import compute_embeddings                        # noqa: E402
from src.metrics import score_report, screen_survival              # noqa: E402
from src.model.autoencoder import RECALModel                        # noqa: E402
from src.utils import (HERE, load_config, pick_device,              # noqa: E402
                       set_random_seed)

RULES = ('max', 'mean', 'sidak', 'fisher')
FUSE = ('screen', 'product', 'noisy_or', 'max', 'knn_only', 'q_only')
ALPHAS = (0.50, 0.80, 0.90, 0.95, 0.99)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
HEAD = f'{"ds":7s} {"table":5s} {"variant":26s} {"F1":>7s} {"P":>7s} {"R":>7s} {"AUC":>7s}'


def knn_scores(ds, cfg, out):
    """Stage-1 scores of the calibration and the test graphs, cached on disk."""
    cache = os.path.join(out, 'knn.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        return z['calib'], z['test']
    set_random_seed(cfg['train']['seed'])
    dev = pick_device(cfg)
    store = graphio.GraphStore(ds, cfg, dev)
    train_tags, calib_tag, test_tags = graphio.split_tags(
        store.meta, cfg['data']['calib_index'])
    ck = torch.load(os.path.join(out, 'model.pt'), map_location=dev,
                    weights_only=False)
    model = RECALModel(store.n_feat, store.n_rel, cfg).to(dev)
    model.load_state_dict(ck['model'])
    model.eval()
    x = torch.cat([compute_embeddings(model, store.get(t)) for t in train_tags], 0)
    scorer = detect.KnnScorer(detect.knn_k(ds, cfg), cfg['detect']['knn_ref_max'],
                              cfg['train']['seed']).fit(x)
    del x
    store.drop_gpu()
    kc = np.asarray(scorer.score(compute_embeddings(model, store.get(calib_tag))))
    store.drop_gpu()
    parts = []
    for t in test_tags:
        parts.append(np.asarray(scorer.score(compute_embeddings(model, store.get(t)))))
        store.drop_gpu()
    kt = np.concatenate(parts)
    np.savez(cache, calib=kc, test=kt)
    del model, scorer, store
    torch.cuda.empty_cache()
    return kc, kt


def combine(rule, node, q, n_nodes):
    """Relation combination C over the relations a node actually participates in."""
    k = np.bincount(node, minlength=n_nodes).astype(np.float64)
    seen = k > 0
    s = np.zeros(n_nodes, dtype=np.float64)
    if rule in ('max', 'sidak'):
        mx = np.full(n_nodes, -np.inf)
        np.maximum.at(mx, node, q)
        mx[~np.isfinite(mx)] = 0.0
        s = mx if rule == 'max' else np.power(np.clip(mx, 0.0, 1.0),
                                              np.where(seen, k, 1.0))
    elif rule == 'mean':
        tot = np.bincount(node, weights=q, minlength=n_nodes)
        s[seen] = tot[seen] / k[seen]
    elif rule == 'fisher':
        w = -2.0 * np.log(np.clip(1.0 - q, 1e-12, None))
        x = np.bincount(node, weights=w, minlength=n_nodes)
        s[seen] = chi2.cdf(x[seen], 2.0 * k[seen])
    s[~seen] = 0.0
    return s


def rank01(x):
    """Within-test empirical CDF, ties averaged: strictly monotone in x.

    The calibration-graph ECDF saturates at 1 for every test node above the
    calibration maximum, i.e. exactly where the anomalies are, so it cannot be
    used to compare two test nodes.  This map is label-free either way.
    """
    return rankdata(x, method='average') / float(x.size)


def _row(ds, tbl, variant, counts, s, ei):
    m = score_report(ds, s, counts, ei)
    print(f'{ds:7s} {tbl:5s} {variant:26s} {m["f1"]:7.4f} {m["precision"]:7.4f} '
          f'{m["recall"]:7.4f} {m["auc"]:7.4f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['cadets', 'theia', 'trace'])
    ap.add_argument('--tables', nargs='+', default=list('abcde'))
    ap.add_argument('--combine', default='fisher', choices=RULES,
                    help='relation combination C held fixed in tables b-e '
                         '(default: fisher, the rule used for the results)')
    args = ap.parse_args()
    tabs = set(args.tables)
    print(HEAD)
    for ds in args.datasets:
        cfg = load_config(os.path.join(HERE, 'configs', f'{ds}.yaml'))
        out = os.path.join(HERE, 'runs', f'{ds}_full')
        meta = graphio.load_meta(ds)
        _, _, test_tags = graphio.split_tags(meta, cfg['data']['calib_index'])
        counts = [int(meta['graphs'][t]['num_nodes']) for t in test_tags]
        n_nodes, n_rel = sum(counts), len(graphio.relation_names(ds))

        kc, kt = knn_scores(ds, cfg, out)
        ei = graphio.concat_edge_index(ds, test_tags)
        with open(os.path.join(out, 'errors_test.pkl'), 'rb') as f:
            err = pkl.load(f)
        q = CalibrationTable.load(
            os.path.join(out, 'calib_table.pkl')).transform(err, n_rel)
        node = err.node.astype(np.int64)
        kq = rank01(kt.astype(np.float64))
        cand0 = kt >= detect.candidate_threshold(kc, cfg['detect']['alpha_cand'])
        c_fix = combine(args.combine, node, q, n_nodes)     # held fixed in b-e
        c_rank = rank01(c_fix)
        tag = f'C={args.combine}'

        if 'a' in tabs:
            for rule in RULES:
                c = c_fix if rule == args.combine else combine(rule, node, q, n_nodes)
                _row(ds, 'a', f'C={rule}', counts, np.where(cand0, c, 0.0), ei)
        if 'b' in tabs:
            for op in FUSE:
                if op == 'screen':
                    s = np.where(cand0, c_fix, 0.0)
                elif op == 'product':
                    s = kq * c_rank
                elif op == 'noisy_or':
                    s = 1.0 - (1.0 - kq) * (1.0 - c_rank)
                elif op == 'max':
                    s = np.maximum(kq, c_rank)
                elif op == 'knn_only':
                    s = kq
                else:
                    s = c_rank
                _row(ds, 'b', f'{tag} {op}', counts, s, ei)
        for alpha in (ALPHAS if tabs & set('ce') else ()):
            cand = kt >= detect.candidate_threshold(kc, alpha)
            keep = screen_survival(ds, cand, counts)
            if 'c' in tabs:
                _row(ds, 'c', f'screen a={alpha:.2f} keep={keep:.4f}', counts,
                     np.where(cand, c_fix, 0.0), ei)
            if 'e' in tabs:
                _row(ds, 'e', f'lex    a={alpha:.2f} keep={keep:.4f}', counts,
                     np.where(cand, 1.0 + c_fix, kq), ei)
        if 'd' in tabs:
            for lam in LAMBDAS:
                _row(ds, 'd', f'lam={lam:.2f}', counts,
                     np.where(cand0, 1.0 + lam * kq + (1.0 - lam) * c_rank, kq), ei)
        del err, q
    print('done')


if __name__ == '__main__':
    main()
