"""Fig.2 data export (§9).  python -m scripts.export_obs_data

Panel (a): per-relation training-set event counts        -> obs_freq_{ds}.csv
Panel (b): benign per-relation errors of the *backbone*  -> obs_errors_{ds}.csv
           variant on the calibration graph (uniform mask + shared decoder, so
           its single error per node is regrouped onto the relations the node
           participates in, as §9 prescribes), <=10k samples per relation, seed 0.
The mixed-error 95th percentile is written as a csv header comment and drawn as
the global reference line by plots/fig2_observations.py.

Requires the backbone variant to have been evaluated first:
    python -m scripts.run_ablation --dataset {ds} --only v4_backbone
"""

import argparse
import csv
import os
import pickle as pkl
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import graphio                                          # noqa: E402
from src.utils import HERE, load_config, proc_dir, run_dir       # noqa: E402

BACKBONE = os.path.join('configs', 'ablation', 'v4_backbone.yaml')
SAMPLE_PER_REL = 10000


def _cfg(dataset, variant_cfg):
    path = os.path.join(HERE, variant_cfg)
    cfg = load_config(path)
    if cfg['data']['dataset'] != dataset:
        exp = cfg['exp_name'].replace(cfg['data']['dataset'], dataset, 1)
        cfg = load_config(path, {'data': {'dataset': dataset}, 'exp_name': exp})
    return cfg


def participation(dataset, tag, n_rel):
    """Nodes touching an edge of each relation, straight off graph_{tag}.npz."""
    z = np.load(os.path.join(proc_dir(dataset), f'graph_{tag}.npz'))
    src, dst = z['edge_index'][0], z['edge_index'][1]
    etype = z['etype']
    return [np.unique(np.concatenate([src[etype == r], dst[etype == r]]))
            for r in range(n_rel)]


def export_freq(dataset, out_dir):
    meta = graphio.load_meta(dataset)
    train_tags, calib_tag, _ = graphio.split_tags(meta, -1)
    f = graphio.relation_freq(dataset, train_tags)
    names = graphio.relation_names(dataset)
    path = os.path.join(out_dir, f'obs_freq_{dataset}.csv')
    with open(path, 'w', newline='') as fh:
        fh.write(f'# dataset={dataset} train_graphs={",".join(train_tags)} '
                 f'calib={calib_tag}\n')
        w = csv.writer(fh)
        w.writerow(['relation', 'count'])
        for r in np.argsort(-f):
            w.writerow([names[r], int(f[r])])
    print(f'  {path}: {len(f)} relations, {int(f.sum())} events')
    return f, names


def export_errors(dataset, out_dir, cfg, n_rel, names):
    out = run_dir(cfg)
    with open(os.path.join(out, 'errors_calib.pkl'), 'rb') as fh:
        tab = pkl.load(fh)
    meta = graphio.load_meta(dataset)
    _, calib_tag, _ = graphio.split_tags(meta, cfg['data']['calib_index'])

    if cfg['model']['decoder'] == 'per_relation':
        groups = {r: tab.err[tab.rel == r] for r in range(n_rel)}
    else:                       # shared head: regroup the single error (§9)
        err = np.full(tab.num_nodes, np.nan, dtype=np.float32)
        err[tab.node] = tab.err
        groups = {}
        for r, part in enumerate(participation(dataset, calib_tag, n_rel)):
            e = err[part]
            groups[r] = e[~np.isnan(e)]

    pool = np.concatenate([g for g in groups.values() if g.size]) if groups else \
        np.empty(0, dtype=np.float32)
    p95 = float(np.quantile(pool, 0.95)) if pool.size else 0.0

    rng = np.random.RandomState(0)
    path = os.path.join(out_dir, f'obs_errors_{dataset}.csv')
    with open(path, 'w', newline='') as fh:
        fh.write(f'# dataset={dataset} exp={cfg["exp_name"]} calib={calib_tag} '
                 f'decoder={cfg["model"]["decoder"]}\n')
        fh.write(f'# global_p95={p95:.8g} n_errors={pool.size} '
                 f'sample_per_relation={SAMPLE_PER_REL} seed=0\n')
        w = csv.writer(fh)
        w.writerow(['relation', 'error'])
        for r in range(n_rel):
            e = groups.get(r, np.empty(0, dtype=np.float32))
            if e.size > SAMPLE_PER_REL:
                e = e[rng.permutation(e.size)[:SAMPLE_PER_REL]]
            for v in e:
                w.writerow([names[r], f'{float(v):.8g}'])
    print(f'  {path}: {pool.size} errors, global_p95={p95:.6f}')


def main():
    ap = argparse.ArgumentParser(description='Fig.2 data export')
    ap.add_argument('--datasets', nargs='+', default=['cadets', 'theia', 'trace'])
    ap.add_argument('--variant', default=BACKBONE, help='config for panel (b)')
    ap.add_argument('--out', default=os.path.join('runs', 'obs'))
    a = ap.parse_args()

    out_dir = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    os.makedirs(out_dir, exist_ok=True)
    for ds in a.datasets:
        print(ds)
        cfg = _cfg(ds, a.variant)
        f, names = export_freq(ds, out_dir)
        errors = os.path.join(run_dir(cfg), 'errors_calib.pkl')
        if not os.path.exists(errors):
            print(f'  skip panel (b): {errors} missing '
                  f'(run scripts.run_ablation --only v4_backbone first)')
            continue
        export_errors(ds, out_dir, cfg, len(f), names)


if __name__ == '__main__':
    main()
