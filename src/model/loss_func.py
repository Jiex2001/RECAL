"""Scaled cosine error, copied from MAGIC/model/loss_func.py.

Only change: `reduction` -- §6.5 needs the per-node scalar e_v^r, and Eq.4 needs
means taken inside each M_r rather than over all masked nodes.
"""

import torch.nn.functional as F


def sce_loss(x, y, alpha=3, reduction='mean'):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    if reduction == 'none':
        return loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    raise ValueError(reduction)
