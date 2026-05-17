"""
main.py
-------
Main training entry point.

Usage:
    python main.py --exp_config config/experiments/exp1.yaml
    python main.py --exp_config config/experiments/exp2.yaml
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path

from src.seg.config.configuration import get_config
from src.seg.constants import CONFIG_PATH
from src.seg.utils.common import set_seed, get_device, count_parameters
from src.seg.datasets.dataloader import build_dataloader
from src.seg.losses.losses import build_loss
from src.seg.training.trainer import Trainer


def build_model(cfg):
    """Build DeepLabV3+ model."""
    from src.seg.models.deeplabv3_plus import get_segmentation_model

    model = get_segmentation_model(
        num_classes=cfg.data.num_classes,
        backbone=cfg.model.backbone,
        output_stride=cfg.model.output_stride,
        aux=cfg.training.aux_loss,
        pretrained_base=cfg.model.pretrained_backbone,
        backbone_weights_path=cfg.model.get('backbone_weights_path'),  # ✅ ADD
    )
    return model


def build_optimizer(model, cfg):
    """Build optimizer based on config."""
    opt_name = getattr(cfg.training, "optimizer", "sgd").lower()

    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.lr,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
        )
    elif opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
    elif opt_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, cfg, num_iters_per_epoch: int):
    """Build learning rate scheduler based on config."""
    sched_name = cfg.training.lr_scheduler.lower()

    if sched_name == "poly":
        # Polynomial decay: lr = initial_lr * (1 - iter / total_iter)^0.9
        total_iters = cfg.training.epochs * num_iters_per_epoch
        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda iter: (1 - iter / total_iters) ** 0.9,
        )
    elif sched_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs
        )
    elif sched_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")

    return None


def main(args):
    """Main training pipeline."""

    # ── Load config ──
    cfg = get_config(CONFIG_PATH, args.exp_config)

    # ── Setup ──
    seed = cfg.project.seed if hasattr(cfg, "project") and hasattr(cfg.project, "seed") else 42
    set_seed(seed)
    device = get_device()

    print(f"\n{'='*70}")
    print(f"TRAINING: {cfg.experiment_id}")
    print(f"{'='*70}\n")

    # ── Data ──
    print("[Data] Loading train/val sets...")
    train_loader = build_dataloader(cfg.data, split="train")
    val_loader = build_dataloader(cfg.data, split="val")

    # ── Model ──
    print("[Model] Building DeepLabV3+...")
    model = build_model(cfg)
    print(f"        {count_parameters(model)}")

    # ── Optimizer ──
    print(f"[Optimizer] {cfg.training.optimizer.upper()}")
    optimizer = build_optimizer(model, cfg)

    # ── Scheduler ──
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    print(f"[Scheduler] {cfg.training.lr_scheduler.upper()}")

    # ── Loss ──
    print(f"[Loss] {cfg.loss.type.upper()}")
    loss_fn = build_loss(
        cfg.loss.type,
        ignore_index=cfg.data.ignore_index,
        **cfg.loss.kwargs,
    )

    # ── Trainer ──
    print("[Trainer] Initializing...\n")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )

    # ── Train ──
    trainer.train()

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on Cityscapes.")
    parser.add_argument(
        "--exp_config",
        type=str,
        default=None,
        help="Path to experiment config yaml (e.g., config/experiments/exp_01.yaml)",
    )
    args = parser.parse_args()

    if args.exp_config is None:
        raise ValueError("--exp_config is required. Example: config/experiments/exp_01.yaml")

    main(args)