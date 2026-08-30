"""Metrics and operating points.

**This is the only module allowed to read test labels.**  `malicious.pkl` is
opened here and nowhere else; scores arrive already computed.

MAGIC's test-set convention is replicated: test graphs other than the last are
also training graphs, so their benign nodes are excluded from evaluation
(`eval.py:56-79`).  Only trace is affected (n_test = 5); theia and cadets have a
single test graph.

Counting follows THREATRACE's node-level protocol (§VI-C-1 of the TIFS'22
paper, p.3982):

    TP  a malicious node that is flagged, *or* whose <=2-hop ancestors or
        descendants contain a flagged node (alert tracing would reach it)
    FP  a flagged benign node whose <=2-hop ancestors and descendants contain no
        malicious node at all
    TN  every other benign node -- including a flagged one that sits within two
        hops of the attack
    FN  every other malicious node

`tolerant_scores` turns that rule into a score transform, so thresholds, PR
curves and operating points stay exactly as before.  Strict per-node counting is
also implemented, as `eval.protocol: strict`.
"""

import pickle as pkl
import os

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

from .utils import proc_dir


def load_malicious(dataset):
    """-> (global test-node ids, names) from malicious.pkl."""
    with open(os.path.join(proc_dir(dataset), 'malicious.pkl'), 'rb') as f:
        nodes, names = pkl.load(f)
    return np.asarray(nodes, dtype=np.int64), names


def label_vector(dataset, n_test_nodes, test_node_counts):
    """(y, eval_idx): labels and the evaluated subset (MAGIC skip_benign)."""
    mal, _ = load_malicious(dataset)
    y = np.zeros(n_test_nodes, dtype=np.float64)
    y[mal] = 1.0
    skip_benign = int(sum(test_node_counts[:-1]))
    idx = np.flatnonzero((np.arange(n_test_nodes) >= skip_benign) | (y == 1.0))
    return y, idx


def _dilate_max(x, src, dst, hops):
    """Max of `x` over each node's <=`hops` ancestors and over its <=`hops`
    descendants (the two directions are propagated separately, so a sibling
    reached by going up and then down is *not* included).  Nodes with no such
    neighbour get -inf.  `x` may be (N,) or (N, k): a second column costs one
    extra scatter column instead of a second pass over the edges."""
    out = np.full(x.shape, -np.inf)
    for step_src, step_dst in ((src, dst), (dst, src)):   # ancestors, descendants
        cur = x
        for _ in range(hops):
            nxt = np.full(x.shape, -np.inf)
            np.maximum.at(nxt, step_dst, cur[step_src])
            out = np.maximum(out, nxt)
            cur = nxt
    return out


def tolerant_scores(scores, y, edge_index, hops=2):
    """THREATRACE's 2-hop tolerant counting, expressed as a score transform.

    A malicious node counts as detected as soon as *anything* in its closed
    <=2-hop ancestor/descendant set is flagged, i.e. iff the maximum score over
    that set clears the threshold -- so give it that maximum.  A benign node that
    has a malicious node within two hops can never be a false positive, so push
    it below every threshold.  Every other benign node keeps its own score.

    The transform is monotone in the threshold, therefore `best_f1_threshold`,
    `confusion` and `roc_auc_score` applied to the result reproduce the paper's
    TP/FP/TN/FN definitions exactly, and the returned tau is still on the
    original score scale.
    """
    if hops <= 0 or edge_index is None or edge_index.shape[1] == 0:
        return scores.astype(np.float64, copy=True)
    src, dst = edge_index[0], edge_index[1]
    mal = y == 1.0
    near = _dilate_max(np.stack([scores.astype(np.float64),
                                 mal.astype(np.float64)], axis=1), src, dst, hops)
    near_score, near_mal = near[:, 0], near[:, 1] > 0.0
    floor = float(np.min(scores)) - 1.0
    out = np.where(mal, np.maximum(scores, near_score),
                   np.where(near_mal, floor, scores))
    return out.astype(np.float64)


def five_metrics(tp, fp, tn, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = (tp + tn) / max(tp + fp + tn + fn, 1)
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {'precision': p, 'recall': r, 'f1': f1, 'accuracy': acc, 'fpr': fpr}


def confusion(y, s, tau):
    alarm = s >= tau
    tp = int(np.sum(alarm & (y == 1.0)))
    fp = int(np.sum(alarm & (y == 0.0)))
    fn = int(np.sum(~alarm & (y == 1.0)))
    tn = int(np.sum(~alarm & (y == 0.0)))
    return tp, fp, tn, fn


def best_f1_threshold(y, s):
    """D4: the community best-F1 operating point on the PR curve."""
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    i = int(np.argmax(f1))
    tau = float(thr[min(i, len(thr) - 1)])
    return tau, {'precision': float(prec[i]), 'recall': float(rec[i]),
                 'f1': float(f1[i])}, (prec, rec, thr)


def fixed_calib_threshold(calib_scores, target_alarm):
    """tau such that the calibration graph's alarm rate equals `target_alarm`."""
    return float(np.quantile(calib_scores, 1.0 - target_alarm))


def per_relation_recall(y, s, tau, participation, rel_names):
    """Recall of malicious nodes grouped by the relations they participate in."""
    alarm = s >= tau
    mal = y == 1.0
    out = {}
    for r, nodes in participation.items():
        m = mal[nodes]
        if not m.any():
            continue
        out[rel_names[r]] = float(alarm[nodes][m].mean())
    return out


def tolerance_hops(cfg):
    """K of the counting protocol: 2 for `threatrace_2hop`, 0 for `strict`."""
    ecfg = cfg.get('eval', {})
    if ecfg.get('protocol', 'threatrace_2hop') == 'strict':
        return 0
    return int(ecfg.get('tolerance_hops', 2))


def score_report(dataset, scores, test_node_counts, edge_index=None, hops=2):
    """-> {'precision', 'recall', 'f1', 'auc'} at the best-F1 point (D4).

    Diagnostic entry point for `scripts/sweep_scoring.py`, which compares scoring
    rules offline: the labels stay inside this module, the caller only ever sees
    the aggregate numbers (§13.3).  Pass `edge_index` to score under the
    THREATRACE protocol; without it the numbers are strict per-node.
    """
    y, idx = label_vector(dataset, scores.shape[0], test_node_counts)
    s = tolerant_scores(scores, y, edge_index, hops)
    y_e, s_e = y[idx], s[idx]
    _, bf1, _ = best_f1_threshold(y_e, s_e)
    bf1['auc'] = (float(roc_auc_score(y_e, s_e))
                  if 0 < y_e.sum() < len(y_e) else None)
    return bf1


def screen_survival(dataset, keep, test_node_counts):
    """Share of the evaluated positives that a boolean mask keeps.

    Same contract as `score_report`: lets `scripts/sweep_scoring.py` report how
    much of the attack a candidate screen retains without handing it any labels.
    """
    y, idx = label_vector(dataset, keep.shape[0], test_node_counts)
    mal = y[idx] == 1.0
    return float(keep[idx][mal].mean()) if mal.any() else 0.0


def evaluate(cfg, dataset, scores, calib_scores, test_node_counts, participation,
             rel_names, edge_index=None, extra=None):
    """-> (metrics.json dict, diagnostics.json dict).

    Labels are loaded here and never leave this function, so no other module ever
    holds them.  Reported numbers use THREATRACE's counting (see the module
    docstring); the strict per-node counts are also written to
    `diagnostics.json` as `points.*_strict`.
    """
    hops = tolerance_hops(cfg)
    y, eval_idx = label_vector(dataset, scores.shape[0], test_node_counts)
    s_tol = tolerant_scores(scores, y, edge_index, hops)
    y_e = y[eval_idx]
    s_e, s_st = s_tol[eval_idx], scores[eval_idx].astype(np.float64)

    tau_bf1, _, (prec, rec, thr) = best_f1_threshold(y_e, s_e)
    tau_fc = fixed_calib_threshold(calib_scores, cfg['detect']['target_calib_alarm'])
    tau_st, _, _ = best_f1_threshold(y_e, s_st)

    points = {}
    for name, tau, s in (('best_f1', tau_bf1, s_e), ('fixed_calib', tau_fc, s_e),
                         ('best_f1_strict', tau_st, s_st),
                         ('fixed_calib_strict', tau_fc, s_st)):
        tp, fp, tn, fn = confusion(y_e, s, tau)
        points[name] = {'tau': tau, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                        **five_metrics(tp, fp, tn, fn)}

    chosen = cfg['eval']['operating_point']
    part_e = {r: np.flatnonzero(np.isin(eval_idx, nodes, assume_unique=False))
              for r, nodes in participation.items()}
    metrics = {
        'exp': cfg['exp_name'], 'dataset': dataset, 'seed': cfg['train']['seed'],
        **{k: points[chosen][k] for k in ('tp', 'fp', 'tn', 'fn', 'precision',
                                          'recall', 'f1', 'accuracy', 'fpr')},
        'tau': points[chosen]['tau'],
        'operating_point': chosen,
        'protocol': cfg['eval'].get('protocol', 'threatrace_2hop'),
        'tolerance_hops': hops,
        'per_relation_recall': per_relation_recall(
            y_e, s_e, points[chosen]['tau'], part_e, rel_names),
    }
    if extra:
        metrics.update(extra)

    auc = lambda s: (float(roc_auc_score(y_e, s))         # noqa: E731
                     if 0 < y_e.sum() < len(y_e) else None)
    diagnostics = {
        'exp': cfg['exp_name'], 'dataset': dataset,
        'n_eval': int(len(eval_idx)), 'n_malicious_eval': int(y_e.sum()),
        'protocol': metrics['protocol'], 'tolerance_hops': hops,
        'auc': auc(s_e), 'auc_strict': auc(s_st),
        'points': points,
        'pr_curve': {'precision': prec.tolist(), 'recall': rec.tolist(),
                     'threshold': thr.tolist()},
    }
    return metrics, diagnostics
