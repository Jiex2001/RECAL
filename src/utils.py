"""Shared helpers: seeding, activations/norms, config loading, logging.

`set_random_seed` / `create_activation` / `create_norm` are copied from
MAGIC/utils/utils.py with one fix noted below.
"""

import hashlib
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # RECAL/


def set_random_seed(seed: int = 0):
    """Copied from MAGIC/utils/utils.py."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_activation(name):
    """Copied from MAGIC/utils/utils.py."""
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None or name == "none":
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    raise NotImplementedError(name)


def create_norm(name):
    """Copied from MAGIC/utils/utils.py, case-insensitive.

    Upstream matches lowercase only, so a `norm='BatchNorm'` setting silently
    resolves to None; we lower() first, which enables normalization as configured.
    """
    if name is None:
        return None
    name = str(name).lower()
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "none":
        return None
    raise NotImplementedError(name)


# ---------------------------------------------------------------- config ----

def _deep_update(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str, extra_overrides: dict | None = None) -> dict:
    """base.yaml <- <path> <- extra_overrides.

    A config may declare `inherit: <relative path>` to chain (ablation yamls
    inherit from a dataset yaml).  base.yaml is always the root.
    """
    cfg_dir = os.path.join(HERE, "configs")
    with open(os.path.join(cfg_dir, "base.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    chain = []
    cur = os.path.abspath(path)
    while cur is not None:
        with open(cur, "r", encoding="utf-8") as f:
            node = yaml.safe_load(f) or {}
        parent = node.pop("inherit", None)
        if os.path.basename(cur) != "base.yaml":
            chain.append(node)
        cur = os.path.abspath(os.path.join(os.path.dirname(cur), parent)) if parent else None
    for node in reversed(chain):  # root-most override applied first
        _deep_update(cfg, node)
    if extra_overrides:
        _deep_update(cfg, extra_overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict):
    """A shared decoder cannot feed M2 per-relation errors."""
    if cfg["model"]["decoder"] == "shared_gat" and cfg["detect"]["mode"] != "knn_only":
        raise ValueError(
            "model.decoder=shared_gat produces no per-relation errors; "
            "detect.mode must be knn_only."
        )
    if cfg["model"]["decoder"] not in ("per_relation", "shared_gat"):
        raise ValueError(f"unknown model.decoder {cfg['model']['decoder']}")
    if cfg["mask"]["mode"] not in ("powerlaw", "uniform"):
        raise ValueError(f"unknown mask.mode {cfg['mask']['mode']}")
    if cfg["detect"]["mode"] not in ("two_stage", "knn_only", "quantile_only"):
        raise ValueError(f"unknown detect.mode {cfg['detect']['mode']}")
    if cfg["detect"]["calibration"] not in ("quantile", "zscore"):
        raise ValueError(f"unknown detect.calibration {cfg['detect']['calibration']}")
    if cfg["detect"]["aggregation"] not in ("fisher", "max", "mean", "sidak"):
        raise ValueError(f"unknown detect.aggregation {cfg['detect']['aggregation']}")
    if cfg["eval"]["operating_point"] not in ("best_f1", "fixed_calib"):
        raise ValueError(f"unknown eval.operating_point {cfg['eval']['operating_point']}")
    if cfg["eval"].get("protocol", "threatrace_2hop") not in ("threatrace_2hop",
                                                              "strict"):
        raise ValueError(f"unknown eval.protocol {cfg['eval']['protocol']}")


def config_hash(cfg: dict) -> str:
    """§13.2: sha256 of the canonicalized yaml, first 8 hex chars."""
    canon = yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False, allow_unicode=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]


def run_dir(cfg: dict, create: bool = True) -> str:
    d = os.path.join(HERE, "runs", cfg["exp_name"])
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def snapshot_config(cfg: dict):
    """§13.4: every run keeps its own config.yaml."""
    with open(os.path.join(run_dir(cfg), "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True, allow_unicode=True)


def proc_dir(dataset: str) -> str:
    return os.path.join(HERE, "proc", dataset)


def get_logger(cfg: dict, name: str = "train") -> logging.Logger:
    logger = logging.getLogger(f"recal.{cfg['exp_name']}.{name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(run_dir(cfg), f"{name}.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def pick_device(cfg: dict) -> torch.device:
    want = cfg.get("device", "cuda")
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
