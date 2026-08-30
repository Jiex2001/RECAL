"""M1 training (§6.4).  python -m scripts.train --config configs/cadets.yaml

Per-file full-graph gradient updates (F10), Adam, 50 epochs, seed 0.  The
calibration graph (§5.3) is excluded from training.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import graphio                                       # noqa: E402
from src.masking import RelationMasker                        # noqa: E402
from src.model.autoencoder import RECALModel                  # noqa: E402
from src.utils import (dump_json, get_logger, load_config, pick_device,  # noqa: E402
                       run_dir, set_random_seed, snapshot_config)


def build(cfg, device):
    dataset = cfg['data']['dataset']
    store = graphio.GraphStore(dataset, cfg, device)
    train_tags, calib_tag, test_tags = graphio.split_tags(
        store.meta, cfg['data']['calib_index'])
    f_r = graphio.relation_freq(dataset, train_tags)
    masker = RelationMasker(cfg, f_r)
    model = RECALModel(store.n_feat, store.n_rel, cfg).to(device)
    return store, (train_tags, calib_tag, test_tags), masker, model


def main():
    ap = argparse.ArgumentParser(description='RECAL training')
    ap.add_argument('--config', required=True)
    ap.add_argument('--override', default=None, help='json dict of config overrides')
    a = ap.parse_args()

    over = json.loads(a.override) if a.override else None
    cfg = load_config(a.config, over)
    set_random_seed(cfg['train']['seed'])
    device = pick_device(cfg)
    out = run_dir(cfg)
    snapshot_config(cfg)
    log = get_logger(cfg, 'train')

    store, (train_tags, calib_tag, test_tags), masker, model = build(cfg, device)
    rel_names = graphio.relation_names(cfg['data']['dataset'])
    log.info(f"{cfg['exp_name']} dataset={cfg['data']['dataset']} device={device}")
    log.info(f'train={train_tags} calib={calib_tag} test={test_tags}')
    log.info(f'n_feat={store.n_feat} n_rel={store.n_rel} '
             f'params={sum(p.numel() for p in model.parameters())}')
    log.info('mask f_bar=%.1f p_r=%s' % (
        masker.f_bar, {rel_names[r]: round(float(p), 4)
                       for r, p in enumerate(masker.p_r)}))

    opt = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'],
                           weight_decay=cfg['train']['weight_decay'])
    curve = []
    t0 = time.time()
    for epoch in range(cfg['train']['epochs']):
        model.train()
        tot, per_rel_sum = 0.0, {}
        n_masked = 0
        for tag in train_tags:
            g = store.get(tag)
            mask = masker.sample(g)
            loss, per_rel = model(g, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n_masked += int(mask.sum())
            for r, v in per_rel.items():
                per_rel_sum[r] = per_rel_sum.get(r, 0.0) + v
        assert np.isfinite(tot), f'epoch {epoch}: loss is {tot}'
        curve.append({'epoch': epoch, 'loss': tot, 'masked': n_masked,
                      'per_relation': {rel_names[r] if r >= 0 else 'shared': v
                                       for r, v in sorted(per_rel_sum.items())}})
        log.info(f'epoch {epoch:3d} loss {tot:.4f} masked {n_masked} '
                 f'{time.time() - t0:.0f}s')
        if epoch == 0 or (epoch + 1) % 10 == 0:
            log.info('  per-relation ' + json.dumps(
                {k: round(v, 4) for k, v in curve[-1]['per_relation'].items()}))

    torch.save({'model': model.state_dict(), 'n_feat': store.n_feat,
                'n_rel': store.n_rel, 'p_r': masker.p_r},
               os.path.join(out, 'model.pt'))
    dump_json(curve, os.path.join(out, 'train_curve.json'))
    log.info(f'saved {out}/model.pt  ({time.time() - t0:.0f}s total)')


if __name__ == '__main__':
    main()
