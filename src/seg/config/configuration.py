"""
src/seg/config/configuration.py
--------------------------------
Loads config.yaml and (optionally) an experiment override YAML,
then builds a fully-typed ExperimentConfig object.

Deep-merge logic:
  base  = config/config.yaml          (project defaults)
  exp   = config/experiments/exp1.yaml  (experiment overrides — optional)
  final = deep_merge(base, exp)

Any key present in the experiment yaml overwrites the base value.
Keys only in base are kept as-is.

Usage:
    from src.seg.config.configuration import get_config
    cfg = get_config("config/config.yaml", "config/experiments/exp1.yaml")
    print(cfg.training.lr)
"""

from pathlib import Path
from typing import Optional
import yaml

from src.seg.entity.config_entity import (
    DataConfig, ModelConfig, TrainingConfig, LossConfig,
    TrackingConfig, CheckpointConfig, ExperimentConfig,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return as a plain Python dict."""
    #print(f"Loading YAML config from {path}...")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge `override` into `base`.
    - Dicts are merged key by key.
    - All other types: override wins.
    Returns a new dict (base and override are not modified).
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    #print(f"Deep merged config with override. Example key 'training.lr': {result.get('training', {}).get('lr')}")
    return result


# ── Public entry point ────────────────────────────────────────────────

def get_config(
    base_config_path: str | Path = "config/config.yaml",
    exp_config_path: Optional[str | Path] = None,
) -> ExperimentConfig:
    """
    Load base config and optional experiment override, then build
    a fully-typed ExperimentConfig.

    Args:
        base_config_path : path to config/config.yaml
        exp_config_path  : path to an experiment yaml, or None

    Returns:
        ExperimentConfig — pass this object around everywhere.
    """
    #print(f"Loading base config from {base_config_path}...")
    cfg = _load_yaml(base_config_path)

    if exp_config_path is not None:
        exp_cfg = _load_yaml(exp_config_path)
        #print(f"Experiment override config loaded from {exp_config_path} printing 1 example key: {exp_cfg.get('training', {}).get('lr')}")
        cfg = _deep_merge(cfg, exp_cfg)

    return _build_config(cfg)


def _build_config(cfg: dict) -> ExperimentConfig:
    """Convert the merged dict into typed dataclass objects."""

    # ── experiment id ──
    exp_section = cfg.get("experiment", {})
    experiment_id = exp_section.get("id", "default_run")

    # ── data ──
    d = cfg["data"]
    data = DataConfig(
        root        = d["root"],
        dataset     = d["dataset"],
        num_classes = d["num_classes"],
        ignore_index= d["ignore_index"],
        image_size  = d["image_size"],
        batch_size  = d["batch_size"],
        num_workers = d["num_workers"],
        max_train_samples  = d.get("max_train_samples"),
        max_val_samples = d.get("max_val_samples"),
        max_test_samples = d.get("max_test_samples"),
    )

    # ── model ──
    m = cfg["model"]
    model = ModelConfig(
        name               = m["name"],
        backbone           = m["backbone"],
        output_stride      = m["output_stride"],
        use_pretrained_backbone= m["use_pretrained_backbone"],
        backbone_weights_path = m.get("backbone_weights_path"),  
        use_jpu            = m.get("use_jpu", False),  # default to False if not specified
    )

    # ── training ──
    t = cfg["training"]
    training = TrainingConfig(
        epochs            = t["epochs"],
        lr                = t["lr"],
        lr_scheduler      = t["lr_scheduler"],
        momentum          = t.get("momentum", 0.9),
        weight_decay      = t.get("weight_decay", 1e-4),
        optimizer         = t.get("optimizer", "sgd"),
        aux_loss          = t.get("aux_loss", True),
        aux_weight        = t.get("aux_weight", 0.4),
        amp               = t.get("amp", True),
        accumulation_steps= t.get("accumulation_steps", 1),
        grad_clip         = t.get("grad_clip"),
    )

    # ── loss ──
    l = cfg.get("loss", {"type": "ce"})

    loss_type = l.get("type", "ce")

    loss_kwargs = {
        k: v for k, v in l.items()
        if k != "type"
    }
    loss = LossConfig(
        type=loss_type,
        kwargs=loss_kwargs,
    )

    # ── tracking ──
    tr = cfg.get("tracking", {})
    mlf = tr.get("mlflow", {})
    tb  = tr.get("tensorboard", {})
    tracking = TrackingConfig(
        mlflow_enabled    = mlf.get("enabled", False),
        mlflow_uri        = mlf.get("tracking_uri", "outputs/mlruns"),
        mlflow_experiment = mlf.get("experiment_name", experiment_id),
        tb_enabled        = tb.get("enabled", False),
        tb_log_dir        = tb.get("log_dir", "outputs/tensorboard"),
    )

    # ── checkpoint ──
    ck = cfg.get("checkpoint", {})
    checkpoint = CheckpointConfig(
        dir       = ck.get("dir", "outputs/checkpoints"),
        save_top_k= ck.get("save_top_k", 3),
        resume    = ck.get("resume"),
    )

    return ExperimentConfig(
        experiment_id = experiment_id,
        data          = data,
        model         = model,
        training      = training,
        loss          = loss,
        tracking      = tracking,
        checkpoint    = checkpoint,
    )