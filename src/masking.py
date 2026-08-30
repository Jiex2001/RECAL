"""M1 relation-balanced masking (Eq.3, §6.2).

    f_bar = median(f_r)
    p_r   = clip(p0 * (f_bar / f_r) ** gamma, p_min, p_max)
    mask  = union over r of { v in participants[r] : Bernoulli(p_r) }

`participants[r]` (nodes touched by at least one r edge, either direction) is the
cache built by `graphio._rel_caches`.  A node participating in several relations
therefore gets one independent draw per relation, which is what the union means.

`mask.mode: uniform` is the GraphMAE/MAGIC baseline instead: exactly
`int(rate * N)` nodes drawn uniformly, no relation structure at all.
"""

import numpy as np
import torch


def relation_mask_rates(f, p0=0.3, gamma=0.5, p_min=0.05, p_max=0.9):
    """Eq.3.  `f` = per-relation event counts; relations unseen in training
    (f_r = 0) are treated as maximally rare."""
    f = np.asarray(f, dtype=np.float64)
    seen = f[f > 0]
    f_bar = float(np.median(seen)) if seen.size else 1.0
    ratio = np.where(f > 0, f_bar / np.where(f > 0, f, 1.0), np.inf)
    with np.errstate(over='ignore'):
        p = p0 * np.power(ratio, gamma)
    return np.clip(p, p_min, p_max).astype(np.float64), f_bar


class RelationMasker:
    def __init__(self, cfg, f_r):
        mk = cfg['mask']
        self.mode = mk['mode']
        self.uniform_rate = float(mk['uniform_rate'])
        self.p_r, self.f_bar = relation_mask_rates(
            f_r, mk['p0'], mk['gamma'], mk['p_min'], mk['p_max'])

    def sample(self, g, generator=None):
        """-> bool tensor (num_nodes,) of masked nodes."""
        dev = g.x.device
        mask = torch.zeros(g.num_nodes, dtype=torch.bool, device=dev)
        if self.mode == 'uniform':
            n_mask = int(self.uniform_rate * g.num_nodes)
            perm = torch.randperm(g.num_nodes, device=dev, generator=generator)
            mask[perm[:n_mask]] = True
            return mask
        for r, part in enumerate(g.rel_part):
            if part.numel() == 0:
                continue
            u = torch.rand(part.numel(), device=dev, generator=generator)
            mask[part[u < self.p_r[r]]] = True
        return mask
