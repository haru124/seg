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

def build_model(cfg, checkpoint_path=None):
    """
    Build DeepLabV3+ model.

    checkpoint_path=None : fresh model (normal training start)
    checkpoint_path=str  : load weights only, no optimizer state
                           (stage-2 fine-tuning with more layers unfrozen)
    """
    from src.seg.models.deeplabv3_plus import get_segmentation_model

    model = get_segmentation_model(
        num_classes             = cfg.data.num_classes,
        backbone                = cfg.model.backbone,
        output_stride           = cfg.model.output_stride,
        aux                     = cfg.training.aux_loss,
        use_pretrained_backbone = cfg.model.use_pretrained_backbone,
        backbone_weights_path   = cfg.model.backbone_weights_path,
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        print(f"[Resume] Model weights loaded from: {checkpoint_path}")
        print("[Resume] Optimizer NOT loaded — rebuild with new param groups.")

    return model



def build_optimizer_with_groups(model, training_cfg, base_lr: float):
    """
    Build optimizer with two parameter groups for backbone-aware LR control.

    Group 1 — head, FPN, layer4 (already trained 1+ rounds): base_lr
    Group 2 — layer1, layer2, layer3 (newly unfrozen):        base_lr * 0.1

    If no params fall into the backbone group (e.g. only layer4 is unfrozen),
    group 2 will be empty and is automatically removed.

    Supports: sgd, adam, adamw — reads from training_cfg.optimizer
    """
    import torch

    backbone_new_params = []
    other_params        = []

    _new_layer_keys = [
        "backbone.body.layer1",
        "backbone.body.layer2",
        "backbone.body.layer3",
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name for key in _new_layer_keys):
            backbone_new_params.append(param)
        else:
            other_params.append(param)

    # Build groups — skip empty groups to avoid optimizer warnings
    param_groups = []
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr":     base_lr,
            "name":   "head_fpn_layer4",
        })
    if backbone_new_params:
        param_groups.append({
            "params": backbone_new_params,
            "lr":     base_lr * 0.1,
            "name":   "backbone_new_layers",
        })

    if not param_groups:
        raise ValueError("[Optimizer] No trainable parameters found in model.")

    for g in param_groups:
        print(
            f"[Optimizer] Group '{g['name']}': "
            f"{len(g['params'])} params, lr={g['lr']:.2e}"
        )

    opt_name = getattr(training_cfg, "optimizer", "sgd").lower()
    wd       = getattr(training_cfg, "weight_decay", 1e-4)

    if opt_name == "sgd":
        momentum = getattr(training_cfg, "momentum", 0.9)
        nesterov = getattr(training_cfg, "nesterov", False)
        return torch.optim.SGD(
            param_groups,
            momentum=momentum,
            weight_decay=wd,
            nesterov=nesterov,
        )
    elif opt_name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=wd)

    elif opt_name == "adam":
        return torch.optim.Adam(param_groups, weight_decay=wd)

    else:
        raise ValueError(
            f"[Optimizer] Unknown optimizer: '{opt_name}'. "
            "Choose: sgd | adam | adamw"
        )

def build_scheduler(optimizer, cfg, num_iters_per_epoch: int):
    """
    Build LR scheduler from config.

    Scheduler types and their step location:
      poly           → LambdaLR  → batch-level  → stepped in _train_epoch
      cosine         → CosineAnnealingLR → epoch-level → stepped in train()
      cosine_warmup  → SequentialLR(Linear+Cosine) → epoch-level → stepped in train()
      step           → StepLR    → epoch-level  → stepped in train()

    num_iters_per_epoch is only needed for poly (counts total batches).
    
    """
    from torch.optim.lr_scheduler import (
        LambdaLR,
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
        StepLR,
    )

    sched_name = cfg.training.lr_scheduler.lower()
    warmup_epochs = getattr(cfg.training, "warmup_epochs", 0)
    total_epochs  = cfg.training.epochs

    if sched_name == "poly":
        # Batch-level. total_iters = total batches across all epochs.
        # lr_lambda receives the current iteration count (not epoch).
        # _train_epoch steps this every accumulated batch.
        # Polynomial decay: lr = initial_lr * (1 - iter / total_iter)^0.9
        total_iters = cfg.training.epochs * num_iters_per_epoch
        _total_iters = total_iters
        _power       = getattr(cfg.training, "lr_poly_power", 0.9)
        _floor       = 1e-4   # multiplier floor: actual LR never below base_lr * 1e-4

        return LambdaLR(
            optimizer,
            lr_lambda=lambda current_iter: max(
                (1 - current_iter / _total_iters) ** _power,
                _floor,
            ),
        )
        
    elif sched_name == "cosine":
        # Epoch-level. No warmup.
        return CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=cfg.training.lr * 0.001,
        )
    elif sched_name == "cosine_warmup":
        # Epoch-level. Linear warmup then cosine decay.
        # warmup_epochs=0 → falls back to plain cosine.
        if warmup_epochs <= 0:
            print(
                f"[Scheduler] cosine_warmup requested but warmup_epochs=0. "
                f"Using plain cosine."
            )
            return CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=cfg.training.lr * 0.001,
            )

        cosine_epochs = total_epochs - warmup_epochs
        if cosine_epochs <= 0:
            raise ValueError(
                f"warmup_epochs={warmup_epochs} >= total epochs={total_epochs}. "
                "Reduce warmup_epochs."
            )

        warmup_sched = LinearLR(
            optimizer,
            start_factor=0.1,      # starts at 10% of base_lr
            end_factor=1.0,        # reaches 100% of base_lr after warmup_epochs
            total_iters=warmup_epochs,
        )
        cosine_sched = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=cfg.training.lr * 0.001,
        )
        print(
            f"[Scheduler] Cosine+Warmup | warmup={warmup_epochs} epochs | "
            f"cosine={cosine_epochs} epochs | "
            f"eta_min={cfg.training.lr * 0.001:.2e}"
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
    elif sched_name == "step":
        step_size = getattr(cfg.training, "lr_step_size", 30)
        gamma     = getattr(cfg.training, "lr_gamma",     0.1)
        print(f"[Scheduler] StepLR | step_size={step_size} | gamma={gamma}")
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    else:
        raise ValueError(
            f"Unknown scheduler: '{sched_name}'. "
            "Choose: poly | cosine | cosine_warmup | step"
        )




def main(args):
    """Main training pipeline."""

    # ── Load config ──
    cfg = get_config(CONFIG_PATH, args.exp_config)
    ckpt_path = args.ckpt_path

    # ── Setup ──
    seed = cfg.project.seed if hasattr(cfg, "project") and hasattr(cfg.project, "seed") else 42
    set_seed(seed)
    device = get_device()

    # ── Determinism + memory stability ──────────────────────────────
    torch.backends.cudnn.benchmark     = False   # ← ADD HERE
    # benchmark=True makes cuDNN allocate/free workspace memory to find
    # the fastest conv algorithm per shape. With variable crop sizes
    # (Cityscapes augmentation), this fragments GPU memory every batch.
    # False uses a fixed heuristic — slightly slower per conv, but flat
    # predictable memory usage. Prevents OOM crashes on laptops.
    torch.backends.cudnn.deterministic = True    # ← ADD HERE (pairs with set_seed)
    # Without this, even with fixed seed, cuDNN non-deterministic ops
    # produce different results across runs. No speed cost for inference;
    # small cost during training (acceptable for reproducibility).

    print(f"\n{'='*70}")
    print(f"TRAINING: {cfg.experiment_id}")
    print(f"{'='*70}\n")

    # ── Data ──
    print("[Data] Loading train/val sets...")
    train_loader = build_dataloader(cfg.data, split="train")
    val_loader = build_dataloader(cfg.data, split="val")

    # ── Model ──
    print("[Model] Building DeepLabV3+...")
    model = build_model(cfg, ckpt_path)
    #model = build_model(cfg, checkpoint_path="outputs/checkpoints/exp3/best.pth")
    print(f"        {count_parameters(model)}")

    # ── Optimizer ──
    print(f"[Optimizer] {cfg.training.optimizer.upper()}")

    # Build optimizer with parameter groups — NOT the standard single-group version
    optimizer = build_optimizer_with_groups(
        model       = model,
        training_cfg = cfg.training,
        base_lr     = cfg.training.lr,
    )

    # ── Scheduler ──
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    print(f"[Scheduler] {cfg.training.lr_scheduler.upper()}")

    # ── Loss ──
    print(f"[Loss] {cfg.loss.type.upper()}")
    
    loss_kwargs = cfg.loss.kwargs if cfg.loss.kwargs else {}
    loss_fn = build_loss(
    cfg.loss.type,
    ignore_index=cfg.data.ignore_index,
    **loss_kwargs,
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

    #--- TRAIN---#
    try:
        trainer.train()
    except Exception as e:
        import traceback
        print("\n[ERROR] Training crashed:\n")
        traceback.print_exc()


    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on Cityscapes.")
    parser.add_argument(
        "--exp_config",
        type=str,
        default=None,
        help="Path to experiment config yaml (e.g., config/experiments/exp1.yaml)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint directory",
    )
    args = parser.parse_args()

    if args.exp_config is None:
        raise ValueError("--exp_config is required. Example: config/experiments/exp1.yaml")

    main(args)