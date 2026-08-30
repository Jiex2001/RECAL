"""Per-relation aggregation m_v^r and decoder heads D_r (§6.3, Eq.4).

m_v^r = mean{ h_u : u --r-- v, direction ignored }.  The symmetrized, deduplicated
r edge set and its participant list are cached per graph by `graphio._rel_caches`,
so a forward pass is one scatter-mean per relation over that relation's edges
(all relations together touch 2E rows).

Only masked participants are decoded, which is what "inside M_r" means in Eq.4;
edges are filtered by their *destination* so every neighbour of a decoded node is
still averaged in.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class PerRelationHeads(nn.Module):
    def __init__(self, in_dim, n_feat, n_rel):
        super().__init__()
        self.n_rel = n_rel
        self.heads = nn.ModuleList([nn.Linear(in_dim, n_feat) for _ in range(n_rel)])

    def forward(self, h, g, mask_bool):
        """-> list over relations of (node_ids, x_hat) for the masked participants.

        Relations with no masked participant yield None (Eq.4 skips empty M_r).
        """
        out = []
        for r in range(self.n_rel):
            part = g.rel_part[r]
            if part.numel() == 0:
                out.append(None)
                continue
            sel = mask_bool[part]                       # masked, per participant
            n_sel = int(sel.sum())
            if n_sel == 0:
                out.append(None)
                continue
            dst_local = g.rel_dst_local[r]
            new_idx = torch.full((part.numel(),), -1, dtype=torch.long, device=h.device)
            new_idx[sel] = torch.arange(n_sel, device=h.device)
            keep = sel[dst_local]
            dst_new = new_idx[dst_local[keep]]
            m = scatter(h[g.rel_src[r][keep]], dst_new, dim=0, dim_size=n_sel,
                        reduce='mean')
            out.append((part[sel], self.heads[r](m)))
        return out
