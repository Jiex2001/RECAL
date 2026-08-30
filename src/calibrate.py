"""M2 per-relation calibration (Eq.5, §7.1).

Benign errors from the calibration graph, grouped by relation and sorted.  A
relation with fewer than `min_samples` observations is served by the `fallback`
bucket (all relations' errors mixed).

`quantile` returns the interpolated empirical CDF in [0,1]; `zscore` returns
(e - mu_r) / sigma_r (ablation variant 5).
"""

import pickle as pkl

import numpy as np


def _ecdf(xs, e):
    """Empirical CDF of `xs` at `e`, linearly interpolated between order stats.

    Step values at the knots (so §12's F(3) = 3/5 on [1..5] holds exactly), and
    interpolated in between; 0 below the minimum, 1 at or above the maximum.
    """
    n = xs.shape[0]
    e = np.asarray(e, dtype=np.float64)
    i = np.searchsorted(xs, e, side='right')
    out = i.astype(np.float64)
    mid = (i >= 1) & (i <= n - 1)
    if mid.any():
        j = i[mid]
        lo, hi = xs[j - 1], xs[j]
        d = hi - lo
        out[mid] += np.where(d > 0, (e[mid] - lo) / np.where(d > 0, d, 1.0), 0.0)
    return np.clip(out / n, 0.0, 1.0)


class CalibrationTable:
    def __init__(self, per_rel, fallback, mode='quantile', min_samples=50):
        self.per_rel = per_rel            # {relation_id: sorted np.array}
        self.fallback = fallback          # sorted np.array of every error
        self.mode = mode
        self.min_samples = min_samples
        self.stats = {r: (float(v.mean()), float(v.std())) for r, v in per_rel.items()}
        self.fallback_stats = (float(fallback.mean()), float(fallback.std())) \
            if fallback.size else (0.0, 1.0)

    @classmethod
    def build(cls, err_table, n_rel, mode='quantile', min_samples=50):
        per_rel = {}
        for r in range(n_rel):
            v = err_table.err[err_table.rel == r]
            if v.size >= min_samples:
                per_rel[r] = np.sort(v.astype(np.float64))
        fallback = np.sort(err_table.err.astype(np.float64))
        return cls(per_rel, fallback, mode, min_samples)

    def _bucket(self, r):
        return self.per_rel.get(r)

    def __call__(self, r, e):
        """Calibrated score q for a vector of errors of one relation."""
        xs = self._bucket(r)
        if self.mode == 'quantile':
            return _ecdf(xs if xs is not None else self.fallback, e)
        mu, sd = self.stats[r] if xs is not None else self.fallback_stats
        return (np.asarray(e, dtype=np.float64) - mu) / (sd if sd > 0 else 1.0)

    def transform(self, err_table, n_rel):
        """Calibrate a whole ErrorTable, relation by relation.  -> q array."""
        q = np.empty(len(err_table), dtype=np.float64)
        for r in range(n_rel):
            sel = err_table.rel == r
            if sel.any():
                q[sel] = self(r, err_table.err[sel])
        return q

    def save(self, path):
        with open(path, 'wb') as f:
            pkl.dump({'per_rel': self.per_rel, 'fallback': self.fallback,
                      'mode': self.mode, 'min_samples': self.min_samples}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            d = pkl.load(f)
        return cls(d['per_rel'], d['fallback'], d['mode'], d['min_samples'])
