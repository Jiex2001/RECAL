"""Fig.2 (§9).  python plots/fig2_observations.py [--datasets cadets ...]

Reads the csv pair written by `scripts.export_obs_data` and draws, per dataset,
two panels on a shared x axis (relations ordered by descending training-set
frequency):

  (a) log-scale bar chart of the per-relation training event count,
  (b) violin with an inner box plot of the backbone's benign reconstruction
      errors on the calibration graph, plus the global 95th-percentile line that
      a single relation-agnostic threshold would use.

Colour follows the frequency tercile: high #4C72B0, mid #8C8C8C, rare #DD8452.
Output is vector PDF, one file per dataset.
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import HERE                                        # noqa: E402

HIGH, MID, RARE = '#4C72B0', '#8C8C8C', '#DD8452'
BAND = ('high', 'mid', 'rare')


def _read(path):
    """-> (header comments as dict, list of rows)."""
    meta, rows = {}, []
    with open(path, newline='') as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            for tok in line[1:].split():
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    meta[k] = v
        r = csv.reader([line] + fh.readlines())
        next(r)                                     # the header row we just read
        rows = [row for row in r if row]
    return meta, rows


def _bands(counts):
    """Frequency tercile per relation, by count rank (ties keep csv order)."""
    n = len(counts)
    edges = (n + 2) // 3, (2 * n + 1) // 3
    return [BAND[0 if i < edges[0] else (1 if i < edges[1] else 2)] for i in range(n)]


def _colours(bands):
    return [{'high': HIGH, 'mid': MID, 'rare': RARE}[b] for b in bands]


def figure(dataset, in_dir, out_dir, width=7.0, height=4.6):
    fmeta, frows = _read(os.path.join(in_dir, f'obs_freq_{dataset}.csv'))
    names = [r[0] for r in frows]
    counts = np.array([int(r[1]) for r in frows], dtype=np.int64)
    bands = _bands(counts)
    colours = _colours(bands)

    epath = os.path.join(in_dir, f'obs_errors_{dataset}.csv')
    emeta, erows = _read(epath) if os.path.exists(epath) else ({}, [])
    by_rel = {n: [] for n in names}
    for rel, val in erows:
        if rel in by_rel:
            by_rel[rel].append(float(val))
    p95 = float(emeta.get('global_p95', 'nan'))

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(width, height), sharex=True,
                                     gridspec_kw={'height_ratios': [1.0, 1.35],
                                                  'hspace': 0.12})
    x = np.arange(len(names))
    ax_a.bar(x, np.maximum(counts, 1), color=colours, width=0.78, linewidth=0)
    ax_a.set_yscale('log')
    ax_a.set_ylabel('training events')
    ax_a.grid(axis='y', which='major', lw=0.4, alpha=0.35)
    ax_a.set_axisbelow(True)
    ax_a.text(0.004, 0.93, '(a)', transform=ax_a.transAxes, va='top', fontweight='bold')
    span = counts.max() / max(counts[counts > 0].min(), 1) if counts.max() else 1
    ax_a.text(0.99, 0.93, f'{len(names)} relations, {span:,.0f}$\\times$ span',
              transform=ax_a.transAxes, va='top', ha='right', fontsize=8, color='0.35')

    pos = [i for i, n in enumerate(names) if len(by_rel[n]) > 1]
    data = [by_rel[names[i]] for i in pos]
    if data:
        parts = ax_b.violinplot(data, positions=pos, widths=0.82, showextrema=False)
        for body, i in zip(parts['bodies'], pos):
            body.set_facecolor(colours[i])
            body.set_alpha(0.55)
            body.set_edgecolor('none')
        bp = ax_b.boxplot(data, positions=pos, widths=0.16, showfliers=False,
                          patch_artist=True, medianprops=dict(color='w', lw=1.0),
                          whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
        for patch, i in zip(bp['boxes'], pos):
            patch.set_facecolor(colours[i])
            patch.set_edgecolor('0.25')
            patch.set_linewidth(0.6)
    if np.isfinite(p95):
        ax_b.axhline(p95, ls='--', lw=0.9, color='0.25', zorder=3)
        ax_b.text(-0.45, p95, ' global 95th pct', va='bottom', ha='left',
                  fontsize=8, color='0.25')
    ax_b.set_ylabel('benign reconstruction error')
    ax_b.grid(axis='y', lw=0.4, alpha=0.35)
    ax_b.set_axisbelow(True)
    ax_b.text(0.004, 0.96, '(b)', transform=ax_b.transAxes, va='top', fontweight='bold')
    ax_b.set_xlim(-0.7, len(names) - 0.3)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([n.replace('EVENT_', '') for n in names], rotation=90,
                         fontsize=7)

    handles = [Patch(facecolor=HIGH, label='frequent'),
               Patch(facecolor=MID, label='medium'),
               Patch(facecolor=RARE, label='rare')]
    if np.isfinite(p95):
        handles.append(Line2D([], [], ls='--', color='0.25',
                              label='relation-agnostic threshold'))
    ax_a.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 0.86),
                frameon=False, fontsize=8, ncol=2, handlelength=1.3,
                columnspacing=1.0)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'fig2_observations_{dataset}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out}  ({len(names)} relations, '
          f'{sum(len(v) for v in by_rel.values())} errors, '
          f'calib={fmeta.get("calib", "?")})')
    return out


def main():
    ap = argparse.ArgumentParser(description='Fig.2 renderer')
    ap.add_argument('--datasets', nargs='+', default=['cadets', 'theia', 'trace'])
    ap.add_argument('--in_dir', default=os.path.join('runs', 'obs'))
    ap.add_argument('--out_dir', default=os.path.join('runs', 'obs'))
    a = ap.parse_args()

    plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.7,
                         'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'savefig.bbox': 'tight'})
    ab = lambda p: p if os.path.isabs(p) else os.path.join(HERE, p)   # noqa: E731
    for ds in a.datasets:
        if not os.path.exists(os.path.join(ab(a.in_dir), f'obs_freq_{ds}.csv')):
            print(f'{ds}: skip, no obs_freq csv (run scripts.export_obs_data first)')
            continue
        print(ds)
        figure(ds, ab(a.in_dir), ab(a.out_dir))


if __name__ == '__main__':
    main()
