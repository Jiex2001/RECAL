"""Inference + calibration + detection + metrics (§6.5, §7).

    python -m scripts.eval --config configs/cadets.yaml

Writes runs/{exp}/{errors_calib.pkl, errors_test.pkl, calib_table.pkl,
metrics.json, diagnostics.json}.
"""

import argparse
import json
import os
import pickle as pkl
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import detect, graphio, metrics as M                 # noqa: E402
from src.calibrate import CalibrationTable                    # noqa: E402
from src.inference import ErrorTable, compute_embeddings, compute_errors  # noqa: E402
from src.model.autoencoder import RECALModel                  # noqa: E402
from src.utils import (config_hash, dump_json, get_logger, load_config,  # noqa: E402
                       pick_device, run_dir, set_random_seed)


def main():
    ap = argparse.ArgumentParser(description='RECAL evaluation')
    ap.add_argument('--config', required=True)
    ap.add_argument('--override', default=None, help='json dict of config overrides')
    a = ap.parse_args()

    cfg = load_config(a.config, json.loads(a.override) if a.override else None)
    set_random_seed(cfg['train']['seed'])
    device = pick_device(cfg)
    dataset = cfg['data']['dataset']
    out = run_dir(cfg)
    log = get_logger(cfg, 'eval')
    t0 = time.time()

    store = graphio.GraphStore(dataset, cfg, device)
    train_tags, calib_tag, test_tags = graphio.split_tags(
        store.meta, cfg['data']['calib_index'])
    n_rel = store.n_rel
    rel_names = graphio.relation_names(dataset)

    ck = torch.load(os.path.join(out, 'model.pt'), map_location=device, weights_only=False)
    assert ck['n_feat'] == store.n_feat and ck['n_rel'] == n_rel
    model = RECALModel(store.n_feat, n_rel, cfg).to(device)
    model.load_state_dict(ck['model'])
    model.eval()

    dcfg = cfg['detect']
    n_rounds, mode = dcfg['n_rounds'], dcfg['mode']

    # ---- stage 1 reference: training-graph embeddings (F11 / §7.2)
    embs = [compute_embeddings(model, store.get(t)) for t in train_tags]
    x_train = torch.cat(embs, dim=0)
    del embs
    scorer = detect.KnnScorer(detect.knn_k(dataset, cfg), dcfg['knn_ref_max'],
                             cfg['train']['seed']).fit(x_train)
    log.info(f'knn k={scorer.k} ref={tuple(scorer.ref.shape)} '
             f'mean_dist={scorer.mean_distance:.4f} ({time.time() - t0:.0f}s)')
    del x_train
    store.drop_gpu()

    # ---- calibration graph (§7.1): benign errors + knn scores
    g = store.get(calib_tag)
    knn_calib = scorer.score(compute_embeddings(model, g))
    err_calib = compute_errors(model, g, n_rounds, cfg['train']['seed'])
    with open(os.path.join(out, 'errors_calib.pkl'), 'wb') as f:
        pkl.dump(err_calib, f)
    table = CalibrationTable.build(err_calib, n_rel, dcfg['calibration'],
                                   dcfg['min_calib_samples'])
    table.save(os.path.join(out, 'calib_table.pkl'))
    log.info(f'calib {calib_tag}: {len(err_calib)} errors, '
             f'{len(table.per_rel)}/{n_rel} relations with own bucket '
             f'({time.time() - t0:.0f}s)')
    del g
    store.drop_gpu()

    # ---- test graphs
    knn_parts, err_parts, offsets, counts, edge_parts = [], [], [], [], []
    participation = {r: [] for r in range(n_rel)}
    cur = 0
    for tag in test_tags:
        g = store.get(tag)
        knn_parts.append(scorer.score(compute_embeddings(model, g)))
        err_parts.append(compute_errors(model, g, n_rounds, cfg['train']['seed']))
        for r in range(n_rel):
            p = g.rel_part[r]
            if p.numel():
                participation[r].append(p.cpu().numpy() + cur)
        edge_parts.append(g.edge_index.cpu().numpy() + cur)   # for the 2-hop protocol
        offsets.append(cur)
        counts.append(g.num_nodes)
        cur += g.num_nodes
        log.info(f'test {tag}: {g.num_nodes} nodes, {len(err_parts[-1])} errors '
                 f'({time.time() - t0:.0f}s)')
        del g
        store.drop_gpu()
    knn_test = np.concatenate(knn_parts)
    err_test = ErrorTable.concat(err_parts, offsets)
    edge_test = np.concatenate(edge_parts, axis=1)
    n_test_nodes = cur
    del knn_parts, err_parts, edge_parts
    with open(os.path.join(out, 'errors_test.pkl'), 'wb') as f:
        pkl.dump(err_test, f)
    participation = {r: np.concatenate(v) for r, v in participation.items() if v}

    # ---- stage 2 (Eq.5/Eq.6)
    agg = dcfg['aggregation']
    cand_thr = detect.candidate_threshold(knn_calib, dcfg['alpha_cand'])
    if mode == 'knn_only':
        s_test, s_calib = knn_test, knn_calib
    else:
        q_calib = table.transform(err_calib, n_rel)
        q_test = table.transform(err_test, n_rel)
        # `fisher`/`sidak` map any table onto [0, 1]; `max`/`mean` inherit its range,
        # so only they need the below-everything sentinel of a z-score table.
        qtab = dcfg['calibration'] != 'zscore'
        bd = qtab or agg in ('fisher', 'sidak')
        sentinel = 0.0 if bd else float(min(q_calib.min(), q_test.min())) - 1.0
        s_calib = detect.aggregate_scores(err_calib, q_calib, err_calib.num_nodes,
                                          agg, sentinel, qtab)
        s_test = detect.aggregate_scores(err_test, q_test, n_test_nodes, agg,
                                         sentinel, qtab)
        if mode == 'two_stage':
            fb = dcfg.get('fallback', 'zero')
            s_calib = detect.apply_candidates(s_calib, knn_calib, cand_thr,
                                              sentinel, fb, bd)
            s_test = detect.apply_candidates(s_test, knn_test, cand_thr,
                                             sentinel, fb, bd)
    n_cand = int((knn_test >= cand_thr).sum())
    log.info(f'cand_thr={cand_thr:.4f} candidates={n_cand}/{n_test_nodes} mode={mode} '
             f'agg={agg} fallback={dcfg.get("fallback", "zero")}')

    # ---- metrics (the only place test labels are read)
    mt, diag = M.evaluate(cfg, dataset, s_test, s_calib, counts, participation,
                          rel_names, edge_index=edge_test,
                          extra={'config_hash': config_hash(cfg),
                                 'n_candidates': n_cand,
                                 'cand_thr': cand_thr})
    dump_json(mt, os.path.join(out, 'metrics.json'))
    dump_json(diag, os.path.join(out, 'diagnostics.json'))
    log.info(json.dumps({k: v for k, v in mt.items()
                         if k != 'per_relation_recall'}, indent=1))
    log.info(f'done in {time.time() - t0:.0f}s -> {out}/metrics.json')


if __name__ == '__main__':
    main()
