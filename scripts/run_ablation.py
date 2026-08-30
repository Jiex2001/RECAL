"""Ablation driver (§10).  python -m scripts.run_ablation --dataset cadets

Runs train + eval for every configs/ablation/*.yaml in turn (separate processes,
so GPU memory is released between variants) and collects the operating-point
metrics into runs/ablation_{dataset}.csv.  `--eval_only` reuses the trained
`model.pt` of each variant and re-runs the detection/metrics stage alone, which is
how the grids are refreshed after a change that touches only scoring or counting.
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import HERE, load_config, run_dir  # noqa: E402

# Config knobs that define each variant, flattened into the csv.
KNOBS = [('mask', 'mode'), ('model', 'decoder'), ('model', 'use_link_pred'),
         ('detect', 'mode'), ('detect', 'calibration'), ('detect', 'aggregation'),
         ('detect', 'fallback'), ('feature', 'use_profile')]
FIELDS = ['variant', 'exp', 'precision', 'recall', 'f1', 'accuracy', 'fpr', 'auc',
          'tp', 'fp', 'tn', 'fn', 'tau', 'n_candidates', 'protocol',
          'f1_strict', 'auc_strict', 'train_s', 'eval_s']


def _run(mod, config, override, log_path):
    cmd = [sys.executable, '-u', '-m', mod, '--config', config]
    if override:
        cmd += ['--override', json.dumps(override)]
    t0 = time.time()
    with open(log_path, 'w') as f:
        p = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise SystemExit(f'{mod} failed for {config}, see {log_path}')
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description='RECAL ablation')
    ap.add_argument('--dataset', default='cadets')
    ap.add_argument('--only', nargs='*', help='variant names to run (default: all)')
    ap.add_argument('--skip_existing', action='store_true',
                    help='reuse a variant whose metrics.json already exists')
    ap.add_argument('--eval_only', action='store_true',
                    help='reuse each variant model.pt if present and re-run only '
                         'eval (train just the variants that have none)')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(HERE, 'configs', 'ablation', '*.yaml')))
    if a.only:
        files = [f for f in files
                 if os.path.basename(f)[:-5] in a.only]
    assert files, 'no ablation configs selected'
    os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
    out_csv = os.path.join(HERE, 'runs', f'ablation_{a.dataset}.csv')
    rows = []

    for path in files:
        variant = os.path.basename(path)[:-5]
        cfg = load_config(path)
        over = None
        if cfg['data']['dataset'] != a.dataset:
            exp = cfg['exp_name'].replace(cfg['data']['dataset'], a.dataset, 1)
            over = {'data': {'dataset': a.dataset}, 'exp_name': exp}
            cfg = load_config(path, over)
        out = run_dir(cfg)
        mpath = os.path.join(out, 'metrics.json')
        t_tr = t_ev = None
        if a.skip_existing and os.path.exists(mpath):
            print(f'{variant}: reusing {mpath}', flush=True)
        else:
            if a.eval_only and os.path.exists(os.path.join(out, 'model.pt')):
                print(f'{variant}: reusing model.pt', flush=True)
            else:
                print(f'{variant}: training', flush=True)
                t_tr = _run('scripts.train', path, over,
                            os.path.join(out, 'ablation_train.log'))
            print(f'{variant}: evaluating', flush=True)
            t_ev = _run('scripts.eval', path, over, os.path.join(out, 'ablation_eval.log'))
        with open(mpath) as f:
            m = json.load(f)
        with open(os.path.join(out, 'diagnostics.json')) as f:
            d = json.load(f)
        row = {'variant': variant, 'exp': m['exp'], 'auc': d.get('auc'),
               'auc_strict': d.get('auc_strict'),
               'f1_strict': d['points'].get('best_f1_strict', {}).get('f1'),
               'train_s': None if t_tr is None else round(t_tr, 1),
               'eval_s': None if t_ev is None else round(t_ev, 1)}
        row.update({k: m.get(k) for k in FIELDS if k not in row})
        row.update({f'{s}.{k}': cfg[s][k] for s, k in KNOBS})
        rows.append(row)
        print(f"{variant}: f1={m['f1']:.4f} precision={m['precision']:.4f} "
              f"recall={m['recall']:.4f}", flush=True)

        fields = FIELDS + [f'{s}.{k}' for s, k in KNOBS]
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f'-> {out_csv}')


if __name__ == '__main__':
    main()
