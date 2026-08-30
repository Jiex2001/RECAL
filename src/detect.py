"""Two-stage detection (Eq.6).

Stage 1 reuses MAGIC's KNN scorer: embeddings standardized with the training
mean/std, score = mean distance to the k nearest *training* embeddings divided by
the training set's own mean k-NN distance.  Implemented as chunked `torch.cdist`
+ `topk` on the GPU rather than sklearn, same arithmetic.

Note on the reference set: in MAGIC's `model/eval.py` the neighbour index is
fitted on every training embedding (line 172) and the 50k subsample is only the
*query* set used to average the denominator (line 178).  Using the subsample as
the index instead costs rare-but-benign patterns their true neighbours and
inflates their scores, so `knn_ref_max` here bounds the denominator sample, not
the index.

Stage 2 turns per-relation errors into s_v = C_r(F_hat(r, e_v^r)) and ranks every
non-candidate below every candidate.  No test label is read anywhere in this file.

After Eq.5 each q_v^r is a benign quantile, so p_v^r = 1 - q_v^r is a p-value and
combining the k relations a node participates in is a multiple-comparison
problem.  `max` is the uncorrected minimum-p statistic: its null CDF is q^k,
expectation k/(k+1), so a benign node on 15 relations is expected at 0.94 while a
single-relation malicious leaf sits at 0.5.  The default is therefore Fisher's
combination (1932),

    X_v = -2 sum_r ln p_v^r ~ chi2_{2k}   ->   s_v = F_{chi2, 2k}(X_v),

which is uniform under the null for every k -- the degrees of freedom absorb the
node's degree -- and accumulates evidence across relations instead of listening
to the loudest one only.  Its independence assumption is an approximation: the
relations of one node share an embedding.  `max`, `mean` and Sidak stay available
through `detect.aggregation` as ablation rows.

`fallback='zero'` sets every non-candidate to a single sentinel, which throws
away the stage-1 ordering below the screen; `fallback='lex'` (the default) keeps
it (`1 + s` for candidates, the stage-1 quantile otherwise).  Both give identical
metrics at alpha=0.90 on all three datasets, but with `lex` the detector degrades
to stage-1-only rather than to "alarm on everything" if the screen ever drops the
attack.
"""

import numpy as np
import torch
from scipy.stats import chi2, norm, rankdata

KNN_K_BY_DATASET = {'cadets': 200}     # F11: 200 for cadets, 10 otherwise
KNN_K_DEFAULT = 10


def knn_k(dataset, cfg):
    k = cfg['detect'].get('knn_k')
    if k:
        return int(k)
    return KNN_K_BY_DATASET.get(dataset, KNN_K_DEFAULT)


class KnnScorer:
    """`ref_max` bounds the denominator sample, not the index.

    `chunk` is the number of query rows held against the whole index at once; it
    is capped so that the distance block stays under `BUDGET` elements, because
    the index is now the full training set (~10^6 rows on cadets).
    """

    BUDGET = 256 << 20                                  # 1 GiB of fp32 distances

    def __init__(self, k, ref_max=50000, seed=0, chunk=2048):
        self.k = k
        self.ref_max = ref_max
        self.seed = seed
        self.chunk = chunk

    def fit(self, train_emb):
        """`train_emb`: (N, d) tensor of training-graph embeddings."""
        self.mean = train_emb.mean(dim=0)
        std = train_emb.std(dim=0, unbiased=False)      # numpy .std() semantics
        self.std = torch.where(std > 0, std, torch.ones_like(std))
        self.ref = ((train_emb - self.mean) / self.std).contiguous()
        n = self.ref.shape[0]
        self.k = min(self.k, n)
        self.chunk = max(32, min(self.chunk, self.BUDGET // max(n, 1)))
        if n > self.ref_max:                            # MAGIC: 50k queries, full index
            idx = np.random.RandomState(self.seed).permutation(n)[:self.ref_max]
            q = self.ref[torch.from_numpy(np.sort(idx)).to(self.ref.device)]
        else:
            q = self.ref
        self.mean_distance = float(self._mean_knn_dist(q).mean())
        return self

    def _mean_knn_dist(self, q):
        out = torch.empty(q.shape[0], device=q.device)
        for i in range(0, q.shape[0], self.chunk):
            d = torch.cdist(q[i:i + self.chunk], self.ref)
            out[i:i + self.chunk] = d.topk(self.k, dim=1, largest=False).values.mean(1)
        return out

    def score(self, emb):
        """-> (N,) numpy array of KNN anomaly scores."""
        z = (emb - self.mean) / self.std
        return (self._mean_knn_dist(z) / self.mean_distance).cpu().numpy()


P_FLOOR = 1e-12                 # ln p is unbounded below; clip before -2 ln p


def aggregate_scores(err_table, q, num_nodes, aggregation='fisher', sentinel=0.0,
                     quantile=True):
    """Eq.6 over the calibrated per-relation scores.

    `q` holds one calibrated score per row of `err_table`, i.e. per (node,
    relation) pair.  `quantile` says what it is: a benign CDF value in [0, 1]
    (Eq.5, so p = 1 - q) or a z-score, in which case the p-value is the
    Gaussian tail `sf(z)` -- Fisher needs p-values, `max` and `mean` do not care.
    `fisher` and `sidak` always return a value in [0, 1]; `max` and `mean` inherit
    the range of `q`.  Nodes with no per-relation error (isolated, or never
    decoded) get `sentinel`.
    """
    node = err_table.node
    if aggregation == 'max':
        s = np.full(num_nodes, -np.inf, dtype=np.float64)
        np.maximum.at(s, node, q)
        s[~np.isfinite(s)] = sentinel
        return s
    k = np.bincount(node, minlength=num_nodes).astype(np.float64)
    seen = k > 0
    s = np.full(num_nodes, sentinel, dtype=np.float64)
    if aggregation == 'sidak':
        mx = np.full(num_nodes, -np.inf, dtype=np.float64)
        np.maximum.at(mx, node, q)
        qm = np.clip(mx[seen], 0.0, 1.0) if quantile else norm.cdf(mx[seen])
        s[seen] = np.power(qm, k[seen])       # P(max Q <= q) = q^k, the Sidak null
        return s
    if aggregation == 'fisher':
        p = np.clip(1.0 - q, P_FLOOR, None) if quantile else \
            np.clip(norm.sf(q), P_FLOOR, None)
        x = np.bincount(node, weights=-2.0 * np.log(p), minlength=num_nodes)
        s[seen] = chi2.cdf(x[seen], 2.0 * k[seen])
        return s
    tot = np.bincount(node, weights=q, minlength=num_nodes)
    s[seen] = tot[seen] / k[seen]
    return s


def candidate_threshold(calib_knn, alpha_cand):
    return float(np.quantile(calib_knn, alpha_cand))


def apply_candidates(scores, knn, cand_thr, sentinel=0.0, fallback='zero',
                     bounded=True):
    """Rank non-candidates below every candidate.

    `zero` collapses all non-candidates onto `sentinel`.
    `lex` keeps the same ordering between the two groups -- candidates are offset
    by 1 and therefore still outrank everything else, and stage 2 orders them
    exactly as before -- but orders the non-candidates by their stage-1 score
    instead of flattening them.  `bounded` says whether `scores` is already in
    [0, 1] (the quantile table); a z-score table is mapped through its own ranks
    first so that the offset still separates the two groups.
    """
    cand = knn >= cand_thr
    if fallback != 'lex':
        out = scores.copy()
        out[~cand] = sentinel
        return out
    hi = scores if bounded else _rank01(scores)
    return np.where(cand, 1.0 + hi, _rank01(knn))


def _rank01(x):
    """Empirical CDF within `x`, ties averaged: strictly monotone, in (0, 1]."""
    return rankdata(x, method='average') / float(max(x.size, 1))
