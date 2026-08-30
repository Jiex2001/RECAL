"""RECAL model (§6.1/§6.3/§6.4), structure from MAGIC/model/autoencoder.py.

Differences from upstream, all required by the document:
  * decoder defaults to |R| per-relation linear heads (Eq.4) instead of one GAT;
    `model.decoder: shared_gat` restores upstream's single 1-layer GAT decoder.
  * masking comes from `src/masking.py` (Eq.3) instead of a flat rate.
  * `embed()` returns the three encoder layers concatenated and passed through
    `encoder_to_decoder`, as §6.1 prescribes (upstream returns the last layer
    only, which is inconsistent with its own decoder input).
  * link prediction is off by default (F8 / §6.3); when on, the term matches
    upstream's Structural Reconstruction block, with DGL's global uniform
    negative sampling replaced by PyG's `negative_sampling`.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling

from ..utils import create_norm
from .gat import GAT
from .heads import PerRelationHeads
from .loss_func import sce_loss


class RECALModel(nn.Module):
    def __init__(self, n_feat, n_rel, cfg):
        super().__init__()
        m = cfg['model']
        hidden, n_layers, n_heads = m['hidden'], m['n_layers'], m['n_heads']
        assert hidden % n_heads == 0
        self.n_feat, self.n_rel = n_feat, n_rel
        self.n_layers = n_layers
        self.hidden = hidden
        self.decoder_kind = m['decoder']
        self.use_link_pred = bool(m['use_link_pred'])
        self.alpha_l = m['alpha_l']

        enc_hidden = hidden // n_heads
        self.encoder = GAT(
            n_dim=n_feat, e_dim=n_rel, hidden_dim=enc_hidden, out_dim=enc_hidden,
            n_layers=n_layers, n_heads=n_heads, n_heads_out=n_heads, concat_out=True,
            activation=m['activation'], feat_drop=m['feat_drop'],
            attn_drop=m['attn_drop'], negative_slope=m['negative_slope'],
            residual=m['residual'], norm=create_norm(m['norm']), encoding=True)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, n_feat))
        self.encoder_to_decoder = nn.Linear(hidden * n_layers, hidden, bias=False)

        if self.decoder_kind == 'per_relation':
            self.decoder = PerRelationHeads(hidden, n_feat, n_rel)
        else:
            self.decoder = GAT(
                n_dim=hidden, e_dim=n_rel, hidden_dim=hidden, out_dim=n_feat,
                n_layers=1, n_heads=n_heads, n_heads_out=1, concat_out=True,
                activation=m['activation'], feat_drop=m['feat_drop'],
                attn_drop=m['attn_drop'], negative_slope=m['negative_slope'],
                residual=m['residual'], norm=create_norm(m['norm']), encoding=False)

        if self.use_link_pred:
            self.recon_loss = nn.BCELoss(reduction='mean')
            self.edge_recon_fc = nn.Sequential(
                nn.Linear(hidden * n_layers * 2, hidden),
                nn.LeakyReLU(m['negative_slope']),
                nn.Linear(hidden, 1),
                nn.Sigmoid())
            self.edge_recon_fc.apply(self._init_linear)

    @staticmethod
    def _init_linear(mod):
        if isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            nn.init.constant_(mod.bias, 0)

    # ------------------------------------------------------------------ core --

    def _encode(self, g, mask_bool=None):
        """-> (concatenated hidden, encoder_to_decoder projection)."""
        x = g.x
        if mask_bool is not None:
            x = torch.where(mask_bool.unsqueeze(1), self.enc_mask_token.to(x.dtype), x)
        _, all_hidden = self.encoder(x, g.edge_index, g.edge_attr, return_hidden=True)
        enc_rep = torch.cat(all_hidden, dim=1)
        return enc_rep, self.encoder_to_decoder(enc_rep)

    def embed(self, g):
        """Node embedding for KNN and the decoder heads (§6.1)."""
        return self._encode(g, None)[1]

    def node_errors(self, g, mask_bool):
        """Per-relation reconstruction error of the masked nodes (§6.5).

        -> list over relations of (node_ids, per-node error) or None.
        """
        _, rep = self._encode(g, mask_bool)
        out = []
        if self.decoder_kind == 'per_relation':
            for item in self.decoder(rep, g, mask_bool):
                if item is None:
                    out.append(None)
                    continue
                nodes, x_hat = item
                err = sce_loss(x_hat, g.x[nodes], self.alpha_l, reduction='none')
                out.append((nodes, err))
        else:
            recon = self.decoder(rep, g.edge_index, g.edge_attr)
            nodes = torch.nonzero(mask_bool, as_tuple=False).flatten()
            err = sce_loss(recon[nodes], g.x[nodes], self.alpha_l, reduction='none')
            out.append((nodes, err))          # relation-agnostic, index 0
        return out

    def forward(self, g, mask_bool):
        return self.compute_loss(g, mask_bool)

    def compute_loss(self, g, mask_bool):
        """Eq.4 (+ optional link prediction).  -> (total, {relation: loss})."""
        enc_rep, rep = self._encode(g, mask_bool)

        per_rel = {}
        if self.decoder_kind == 'per_relation':
            loss = rep.new_zeros(())
            for r, item in enumerate(self.decoder(rep, g, mask_bool)):
                if item is None:
                    continue
                nodes, x_hat = item
                lr = sce_loss(x_hat, g.x[nodes], self.alpha_l)
                loss = loss + lr
                per_rel[r] = float(lr.detach())
        else:
            recon = self.decoder(rep, g.edge_index, g.edge_attr)
            nodes = torch.nonzero(mask_bool, as_tuple=False).flatten()
            loss = sce_loss(recon[nodes], g.x[nodes], self.alpha_l)
            per_rel[-1] = float(loss.detach())

        if self.use_link_pred:
            loss = loss + self._link_pred_loss(g, enc_rep)
        return loss, per_rel

    def _link_pred_loss(self, g, enc_rep):
        """MAGIC's Structural Reconstruction term (F8)."""
        n_edge = g.edge_index.shape[1]
        threshold = min(10000, g.num_nodes)
        neg = negative_sampling(g.edge_index, num_nodes=g.num_nodes,
                                num_neg_samples=threshold, method='sparse')
        pos_idx = torch.randperm(n_edge, device=enc_rep.device)[:threshold]
        pos = g.edge_index[:, pos_idx]
        src = enc_rep[torch.cat([pos[0], neg[0]])]
        dst = enc_rep[torch.cat([pos[1], neg[1]])]
        y_pred = self.edge_recon_fc(torch.cat([src, dst], dim=-1)).squeeze(-1)
        y = torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])]).to(y_pred.device)
        return self.recon_loss(y_pred, y)
