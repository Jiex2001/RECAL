"""Edge-type-conditioned GAT, ported from MAGIC/model/gat.py.

The op chain is DGL's, line for line:

    eh = (fc(h)     * attn_h).sum(-1)                      # per src node
    et = (fc(h)     * attn_t).sum(-1)                      # per dst node
    ee = (edge_fc(a)* attn_e).sum(-1)                       # per edge
    e  = leaky_relu(eh[src] + ee + et[dst])                 # u_add_e + e_add_v
    a  = attn_drop(softmax(e, index=dst))                   # dgl edge_softmax
    rst= scatter_add(fc(h)[src] * a, dst) + bias + res_fc(h)
    rst= (concat heads | mean heads) -> norm -> activation

`torch_geometric.utils.softmax` is the exact analogue of `dgl.ops.edge_softmax`
and `scatter(..., reduce='sum')` of `u_mul_e` + `fn.sum`.  PyG's own GATConv is
deliberately *not* used: it adds self-loops by default and places bias/residual
differently.  Nodes with no in-edges get zeros here, same as DGL's `update_all`.

The weighted sum is written out rather than driven through `MessagePassing.
propagate` because DGL's `u_mul_e`+`sum` is a fused kernel that never
materialises the per-edge message, while PyG's does: with the shared GAT decoder
the message is (E, heads, |x|) = 1.65M x 1 x 822 floats = 5 GB on cadets, and
autograd wants several of those, which OOMs a 32 GB card.  `_weighted_sum`
slices the feature dimension so the peak stays under `MSG_BUDGET` elements, and
`_edge_attention` folds `attn_e` into `edge_fc`'s weight so the equally large
edge projection is never built.  Both keep the arithmetic; only summation order
moves; the result is pinned against MAGIC's DGL implementation in testing.

Dropped from upstream: the bipartite/`is_block` branches (we never build blocks)
and `fc_node_embedding`, which upstream allocates but never reads.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter, softmax

from ..utils import create_activation


class GATConv(nn.Module):
    MSG_BUDGET = 128 << 20              # elements held in one per-edge message

    def __init__(self,
                 in_dim,
                 e_dim,
                 out_dim,
                 n_heads,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 bias=True,
                 norm=None,
                 concat_out=True):
        super().__init__()
        self.n_heads = n_heads
        self.src_feat = self.dst_feat = in_dim
        self.edge_feat = e_dim
        self.out_feat = out_dim
        self.concat_out = concat_out

        self.fc = nn.Linear(self.src_feat, self.out_feat * self.n_heads, bias=False)
        self.edge_fc = nn.Linear(self.edge_feat, self.out_feat * self.n_heads, bias=False)
        self.attn_h = nn.Parameter(torch.empty(1, self.n_heads, self.out_feat))
        self.attn_e = nn.Parameter(torch.empty(1, self.n_heads, self.out_feat))
        self.attn_t = nn.Parameter(torch.empty(1, self.n_heads, self.out_feat))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.empty(1, self.n_heads, self.out_feat))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self.dst_feat != self.n_heads * self.out_feat:
                self.res_fc = nn.Linear(self.dst_feat, self.n_heads * self.out_feat,
                                        bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.norm = norm
        if norm is not None:
            self.norm = norm(self.n_heads * self.out_feat)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.edge_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_h, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        nn.init.xavier_normal_(self.attn_t, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, x, edge_index, edge_attr):
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        h = self.feat_drop(x)
        feat = self.fc(h).view(n, self.n_heads, self.out_feat)
        eh = (feat * self.attn_h).sum(-1, keepdim=True)
        et = (feat * self.attn_t).sum(-1, keepdim=True)
        ee = self._edge_attention(edge_attr)
        e = self.leaky_relu(eh[src] + ee + et[dst])
        a = self.attn_drop(softmax(e, dst, num_nodes=n))
        rst = self._weighted_sum(feat, a, src, dst, n)

        if self.bias is not None:
            rst = rst + self.bias
        if self.res_fc is not None:
            rst = rst + self.res_fc(h).view(n, -1, self.out_feat)
        rst = rst.flatten(1) if self.concat_out else torch.mean(rst, dim=1)
        if self.norm is not None:
            rst = self.norm(rst)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst

    def _edge_attention(self, edge_attr):
        """(edge_fc(a) * attn_e).sum(-1), contracted on the weights instead.

        Upstream builds the (E, heads, out_dim) edge projection and immediately
        reduces it away.  With the shared GAT decoder out_dim is |x| = 822, so on
        cadets that throwaway tensor alone is 5 GB.  Folding attn_e into edge_fc's
        weight first gives the same value from an (E, heads) tensor; only the order
        of the sum over out_dim changes.
        """
        w = self.edge_fc.weight.view(self.n_heads, self.out_feat, self.edge_feat)
        v = torch.einsum('hd,hdk->hk', self.attn_e[0], w)
        return (edge_attr @ v.t()).unsqueeze(-1)

    def _weighted_sum(self, feat, a, src, dst, n):
        """scatter_add(feat[src] * a, dst), sliced along the feature dimension."""
        n_edge = a.shape[0]
        step = self.out_feat
        if n_edge * self.n_heads * self.out_feat > self.MSG_BUDGET:
            step = max(1, self.MSG_BUDGET // (n_edge * self.n_heads))
        if step >= self.out_feat:
            return scatter(feat[src] * a, dst, dim=0, dim_size=n, reduce='sum')
        return torch.cat(
            [scatter(feat[:, :, i:i + step][src] * a, dst, dim=0, dim_size=n,
                     reduce='sum')
             for i in range(0, self.out_feat, step)], dim=-1)


class GAT(nn.Module):
    """Layer stack, structure copied from MAGIC/model/gat.py."""

    def __init__(self, n_dim, e_dim, hidden_dim, out_dim, n_layers, n_heads,
                 n_heads_out, activation, feat_drop, attn_drop, negative_slope,
                 residual, norm, concat_out=False, encoding=False):
        super().__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.concat_out = concat_out
        self.gats = nn.ModuleList()

        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if self.n_layers == 1:
            self.gats.append(GATConv(
                n_dim, e_dim, out_dim, n_heads_out, feat_drop, attn_drop,
                negative_slope, last_residual, norm=last_norm,
                concat_out=self.concat_out))
        else:
            self.gats.append(GATConv(
                n_dim, e_dim, hidden_dim, n_heads, feat_drop, attn_drop,
                negative_slope, residual, create_activation(activation),
                norm=norm, concat_out=self.concat_out))
            for _ in range(1, self.n_layers - 1):
                self.gats.append(GATConv(
                    hidden_dim * self.n_heads, e_dim, hidden_dim, n_heads,
                    feat_drop, attn_drop, negative_slope, residual,
                    create_activation(activation), norm=norm,
                    concat_out=self.concat_out))
            self.gats.append(GATConv(
                hidden_dim * self.n_heads, e_dim, out_dim, n_heads_out,
                feat_drop, attn_drop, negative_slope, last_residual,
                last_activation, norm=last_norm, concat_out=self.concat_out))
        self.head = nn.Identity()

    def forward(self, x, edge_index, edge_attr, return_hidden=False):
        h = x
        hidden_list = []
        for layer in range(self.n_layers):
            h = self.gats[layer](h, edge_index, edge_attr)
            hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list
        return self.head(h)
