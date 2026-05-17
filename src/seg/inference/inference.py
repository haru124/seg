"""
inference.py
-----------------
Comprehensive evaluation on the test split:
  1. Load best checkpoint
  2. Compute test loss + all metrics
  3. Log to MLflow and TensorBoard
  4. Visualize predictions on 5 random samples (side-by-side GT + Pred)

Usage:
    python inference.py --exp_config config/experiments/exp_01.yaml
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.seg.config.configuration import get_config
from src.seg.constants import CONFIG_PATH, CITYSCAPES_CLASSES
from src.seg.utils.common import set_seed, get_device, count_parameters
from src.seg.datasets.dataloader import build_dataloader
from src.seg.losses.losses import build_loss
from src.seg.evaluation.metrics import SegmentationMetrics
from src.seg.utils.checkpoint import load_checkpoint
from src.seg.utils.visualization import save_batch_grid, plot_class_iou
from src.seg.tracking.mlflow_logger import MLflowLogger
from src.seg.tracking.tensorboard_logger import TensorboardLogger


def build_model(cfg):
    """Build the model from config."""
    from src.seg.models.deeplabv3_plus import get_segmentation_model
    model = get_segmentation_model(
        num_classes=cfg.data.num_classes,
        backbone=cfg.model.backbone,
        output_stride=cfg.model.output_stride,
        aux=cfg.training.aux_loss,
        pretrained_base=False,  # we'll load from checkpoint
    )
    return model


@torch.no_grad()
def evaluate_test_set(model, test_loader, loss_fn, metrics, device, cfg):
    """
    Run evaluation on the entire test set.
    Returns metrics dict.
    """
    model.eval()
    metrics.reset()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_images = []

    pbar = tqdm(test_loader, desc="[Test] Evaluating")

    for images, targets in pbar:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()

        # Forward pass (no AMP for test)
        outputs = model(images)
        loss    = loss_fn(outputs, targets)

        # Extract predictions
        pred_logits = outputs["out"] if isinstance(outputs, dict) else outputs
        preds = pred_logits.argmax(dim=1)  # (B, H, W)

        # Update metrics
        metrics.update(preds, targets)
        total_loss += loss.item()

        # Store for visualization
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_images.append(images.cpu())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Compute all metrics
    result = metrics.compute()
    result["test_loss"] = total_loss / len(test_loader)

    return result, all_preds, all_targets, all_images


def save_visualizations(all_preds, all_targets, all_images, cfg):
    """
    Save visualization for 5 random samples: input | GT | pred side-by-side.
    """
    num_samples = min(5, len(all_preds))
    indices = np.random.choice(len(all_preds), num_samples, replace=False)

    vis_dir = Path(cfg.checkpoint.dir).parent / "inference" / cfg.experiment_id
    vis_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx, batch_idx in enumerate(indices):
        preds = torch.from_numpy(all_preds[batch_idx])
        targets = torch.from_numpy(all_targets[batch_idx])
        images = all_images[batch_idx]

        # Save the batch grid (will show up to 4 samples from this batch)
        save_path = vis_dir / f"test_samples_{sample_idx}.png"
        save_batch_grid(images, preds, targets, str(save_path), max_images=1)

    print(f"[Visualization] Saved to {vis_dir}")


def save_metrics_plots(result, cfg):
    """Save per-class IoU bar chart."""
    plot_dir = Path(cfg.checkpoint.dir).parent / "metrics" / cfg.experiment_id
    plot_dir.mkdir(parents=True, exist_ok=True)

    per_class_iou = result["per_class_iou"]
    plot_path = plot_dir / "test_per_class_iou.png"
    plot_class_iou(per_class_iou, str(plot_path),
                   title="Test Set: Per-Class IoU",
                   class_names=CITYSCAPES_CLASSES)

    print(f"[Plots] Saved to {plot_dir}")


def log_results(result, cfg):
    """Log test results to MLflow and TensorBoard."""
    # Find the best checkpoint to extract run info
    ckpt_dir = Path(cfg.checkpoint.dir)
    ckpts = sorted(ckpt_dir.glob(f"{cfg.experiment_id}_epoch*.pth"),
                   key=lambda p: float(p.stem.split("mIoU")[-1]), reverse=True)

    if not ckpts:
        print("[Log] No checkpoints found. Skipping MLflow/TensorBoard logging.")
        return

    best_ckpt = ckpts[0]
    print(f"[Log] Best checkpoint: {best_ckpt.name}")

    # ── MLflow ──
    if cfg.tracking.mlflow_enabled:
        try:
            # Create a new run for test evaluation
            mlf = MLflowLogger(
                cfg.tracking.mlflow_uri,
                cfg.tracking.mlflow_experiment,
                run_name=f"{cfg.experiment_id}_test_eval",
            )

            # Log test metrics
            test_metrics = {
                "test_loss": float(result["test_loss"]),
                "test_mIoU": float(result["mIoU"]),
                "test_fw_iou": float(result["fw_iou"]),
                "test_pixel_acc": float(result["mean_pixel_acc"]),
                "test_class_acc": float(result["mean_class_acc"]),
                "test_boundary_iou": float(result["boundary_iou"]),
                "test_boundary_fscore": float(result["boundary_fscore"]),
            }
            mlf.log_metrics(test_metrics, step=0)

            # Log per-class IoU
            per_class_iou = result["per_class_iou"]
            for cls_idx, class_name in enumerate(CITYSCAPES_CLASSES):
                iou = per_class_iou[cls_idx]
                iou_val = float(iou) if not np.isnan(iou) else 0.0
                mlf.log_metrics({f"test_iou_{class_name}": iou_val}, step=0)

            mlf.finish()
            print("[MLflow] Test results logged.")
        except Exception as e:
            print(f"[MLflow] Error: {e}")

    # ── TensorBoard ──
    if cfg.tracking.tb_enabled:
        try:
            tb = TensorboardLogger(cfg.tracking.tb_log_dir, f"{cfg.experiment_id}_test")

            # Log scalars
            test_metrics = {
                "loss": result["test_loss"],
                "mIoU": result["mIoU"],
                "fw_iou": result["fw_iou"],
                "pixel_acc": result["mean_pixel_acc"],
                "class_acc": result["mean_class_acc"],
                "boundary_iou": result["boundary_iou"],
                "boundary_fscore": result["boundary_fscore"],
            }
            tb.log_scalars(test_metrics, step=0, prefix="Test")

            tb.close()
            print("[TensorBoard] Test results logged.")
        except Exception as e:
            print(f"[TensorBoard] Error: {e}")


def main(args):
    """Main test evaluation pipeline."""
    cfg = get_config(CONFIG_PATH, args.exp_config)
    set_seed(cfg.project.seed if hasattr(cfg, 'project') else 42)
    device = get_device()

    print(f"\n{'='*70}")
    print(f"TEST EVALUATION: {cfg.experiment_id}")
    print(f"{'='*70}\n")

    # ── Load test data ──
    test_loader = build_dataloader(cfg.data, split="test")

    # ── Build model ──
    model = build_model(cfg)
    print(f"[Model] {count_parameters(model)}")

    # ── Load best checkpoint ──
    ckpt_dir = Path(cfg.checkpoint.dir)
    ckpts = sorted(ckpt_dir.glob(f"{cfg.experiment_id}_epoch*.pth"),
                   key=lambda p: float(p.stem.split("mIoU")[-1]), reverse=True)

    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    best_ckpt = ckpts[0]
    load_checkpoint(str(best_ckpt), model, device=str(device))
    print(f"[Checkpoint] Loaded best: {best_ckpt.name}\n")

    # ── Build loss and metrics ──
    loss_fn = build_loss(
        cfg.loss.type,
        ignore_index=cfg.data.ignore_index,
        **cfg.loss.kwargs
    )
    metrics = SegmentationMetrics(
        num_classes=cfg.data.num_classes,
        ignore_index=cfg.data.ignore_index,
    )

    # ── Evaluate ──
    print("Running test evaluation...\n")
    result, all_preds, all_targets, all_images = evaluate_test_set(
        model, test_loader, loss_fn, metrics, device, cfg
    )

    # ── Print results ──
    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Loss               : {result['test_loss']:.4f}")
    print(f"  mIoU               : {result['mIoU']:.4f}")
    print(f"  Frequency-weighted IoU : {result['fw_iou']:.4f}")
    print(f"  Mean Pixel Accuracy: {result['mean_pixel_acc']:.4f}")
    print(f"  Mean Class Accuracy: {result['mean_class_acc']:.4f}")
    print(f"  Boundary IoU       : {result['boundary_iou']:.4f}")
    print(f"  Boundary F-Score   : {result['boundary_fscore']:.4f}")
    print(f"{'='*70}\n")

    # ── Save visualizations ──
    save_visualizations(all_preds, all_targets, all_images, cfg)

    # ── Save plots ──
    save_metrics_plots(result, cfg)

    # ── Log to MLflow + TensorBoard ──
    log_results(result, cfg)

    print("\n✓ Test evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on test set with metrics and visualizations."
    )
    parser.add_argument("--exp_config", type=str, default=None,
                        help="Path to experiment config yaml")
    args = parser.parse_args()

    if args.exp_config is None:
        raise ValueError("--exp_config is required")

    main(args)