"""
src/seg/inference/domain_check.py

Domain shift evaluation: run a trained Cityscapes segmentation model
on images from a different dataset.

Two modes:
  1. ImageFolderDataset  — flat folder of images, no annotations
     Use for: infimgs, random road images — visualization only

  2. MaskFolderDataset   — images + labelTrainIds PNG masks
     Use for: BDD100K segmentation, KITTI semantic, Cityscapes-format datasets

Usage:
    # Mode 1 — images only (visualization only)
    python -m src.seg.inference.domain_check --exp_config config/experiments/exp3.yaml --img_dir    data/images/infimgs --n_samples  15
    
    # Mode 2 — with GT masks (mIoU computation)
    python -m src.seg.inference.domain_check \
        --exp_config config/experiments/exp3_adamw_focal.yaml \
        --img_dir    data/bdd100k/images/val \
        --mask_dir   data/bdd100k/labels/val \
        --n_samples  10
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.seg.config.configuration import get_config
from src.seg.constants import CONFIG_PATH, CITYSCAPES_CLASSES
from src.seg.models.deeplabv3_plus import get_segmentation_model
from src.seg.utils.checkpoint import load_checkpoint
from src.seg.utils.common import get_device
from src.seg.evaluation.metrics import SegmentationMetrics
from src.seg.utils.visualization import save_batch_grid, plot_class_iou, plot_metrics_bar

VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ══════════════════════════════════════════════════════════════════════
# PREPROCESSING — matches final_inference.py exactly
# ══════════════════════════════════════════════════════════════════════

def _preprocess_image(path: Path, image_size: tuple) -> torch.Tensor:
    """
    Load and preprocess image to match training pipeline.
    Returns float32 tensor [3, H, W].
    """
    from src.seg.constants import IMAGENET_MEAN, IMAGENET_STD

    img  = Image.open(path).convert("RGB")
    img  = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
    img  = np.array(img, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    img  = (img - mean) / std
    img  = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)


def _preprocess_mask(path: Path, image_size: tuple) -> torch.Tensor:
    """
    Load labelTrainIds PNG mask, resize with nearest neighbour.
    Returns int64 tensor [H, W] with values 0-18 or 255 (ignore).
    """
    mask = Image.open(path)
    mask = mask.resize((image_size[1], image_size[0]), Image.NEAREST)
    return torch.from_numpy(np.array(mask, dtype=np.int64))


# ══════════════════════════════════════════════════════════════════════
# DATASET 1 — Flat image folder, no annotations
# ══════════════════════════════════════════════════════════════════════

class ImageFolderDataset(Dataset):
    """Images only — visualization mode, no mIoU computation."""

    def __init__(self, img_dir: str, image_size: tuple):
        self.image_size = image_size
        self.paths = sorted(
            p for p in Path(img_dir).rglob("*")
            if p.suffix.lower() in VALID_IMG_EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found under {img_dir}")
        self.has_annotations = False
        print(f"[DomainCheck] ImageFolder — {len(self.paths)} images in {img_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        image = _preprocess_image(path, self.image_size)
        return image, torch.zeros(self.image_size, dtype=torch.int64), path.name


# ══════════════════════════════════════════════════════════════════════
# DATASET 2 — Image + GT mask pairs (Cityscapes labelTrainIds format)
# ══════════════════════════════════════════════════════════════════════

class MaskFolderDataset(Dataset):
    """
    Image + GT mask pairs. Supports:
      - BDD100K:         stem_train_id.png
      - Cityscapes-style: stem_gtFine_labelTrainIds.png
      - KITTI:           same stem .png
    """

    def __init__(self, img_dir: str, mask_dir: str, image_size: tuple):
        self.image_size = image_size
        img_dir  = Path(img_dir)
        mask_dir = Path(mask_dir)

        all_images = sorted(
            p for p in img_dir.rglob("*")
            if p.suffix.lower() in VALID_IMG_EXTS
        )

        self.pairs = []
        for img_path in all_images:
            mask_path = self._find_mask(img_path, mask_dir)
            if mask_path is not None:
                self.pairs.append((img_path, mask_path))

        if not self.pairs:
            raise FileNotFoundError(
                f"No matched image/mask pairs found.\n"
                f"  img_dir : {img_dir}\n"
                f"  mask_dir: {mask_dir}"
            )

        self.has_annotations = True
        print(
            f"[DomainCheck] MaskFolder — {len(self.pairs)} matched pairs "
            f"({len(all_images) - len(self.pairs)} skipped, no mask found)"
        )

    @staticmethod
    def _find_mask(img_path: Path, mask_dir: Path):
        stem = img_path.stem
        for sfx in ("_leftImg8bit",):
            if stem.endswith(sfx):
                stem = stem[: -len(sfx)]
                break
        candidates = [
            mask_dir / f"{stem}.png",
            mask_dir / f"{stem}_gtFine_labelTrainIds.png",
            mask_dir / f"{stem}_train_id.png",
            mask_dir / f"{stem}_labelTrainIds.png",
            mask_dir / f"{stem}_seg.png",
        ]
        for c in candidates:
            if c.exists():
                return c
        for p in mask_dir.rglob(f"{stem}*.png"):
            return p
        return None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = _preprocess_image(img_path, self.image_size)
        mask  = _preprocess_mask(mask_path, self.image_size)
        return image, mask, img_path.name


# ══════════════════════════════════════════════════════════════════════
# COLLATE
# ══════════════════════════════════════════════════════════════════════

def _collate_fn(batch):
    images    = torch.stack([b[0] for b in batch], 0)
    masks     = torch.stack([b[1] for b in batch], 0)
    filenames = [b[2] for b in batch]
    return images, masks, filenames


# ══════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_seg_metrics(model, loader, device, num_classes, ignore_index=255):
    """Compute full segmentation metrics. Only called when GT masks available."""
    metrics_obj = SegmentationMetrics(
        num_classes=num_classes,
        ignore_index=ignore_index,
        compute_boundary=False,
    )
    metrics_obj.reset()
    model.eval()

    total_loss = 0.0
    n_batches  = 0

    for images, masks, _ in tqdm(loader, desc="[DomainCheck] Computing metrics"):
        images = images.to(device)
        masks  = masks.to(device)

        out    = model(images)
        logits = out["out"] if isinstance(out, dict) else out
        preds  = logits.argmax(dim=1)

        # Resize GT if spatial dims don't match
        if masks.shape != preds.shape:
            masks = F.interpolate(
                masks.float().unsqueeze(1),
                size=preds.shape[-2:],
                mode="nearest",
            ).squeeze(1).long()

        metrics_obj.update(preds, masks)
        n_batches += 1

    return metrics_obj.compute()


# ══════════════════════════════════════════════════════════════════════
# VISUALISATION — uses save_batch_grid from your existing viz utils
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def _visualize_samples(model, dataset, device, n_samples, vis_dir, seed, has_gt):
    import matplotlib.pyplot as plt
    from src.seg.utils.visualization import colorize_mask
    from src.seg.constants import IMAGENET_MEAN, IMAGENET_STD

    vis_dir.mkdir(parents=True, exist_ok=True)
    n_samples = min(n_samples, len(dataset))
    indices   = np.random.default_rng(seed).choice(
        len(dataset), size=n_samples, replace=False
    )

    def _denorm(tensor):
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)
        img  = tensor.permute(1, 2, 0).cpu().numpy()
        img  = np.clip(img * std + mean, 0, 1)
        return (img * 255).astype(np.uint8)

    model.eval()
    for i, idx in enumerate(indices, start=1):
        image, mask, fname = dataset[int(idx)]

        out    = model(image.unsqueeze(0).to(device))
        logits = out["out"] if isinstance(out, dict) else out
        pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        img_rgb      = _denorm(image)
        pred_colored = colorize_mask(pred)

        # Overlay prediction on original image (alpha blend)
        overlay = (img_rgb * 0.5 + pred_colored * 0.5).astype(np.uint8)

        if has_gt:
            # Mode 2: image | GT mask | prediction overlay
            gt_colored = colorize_mask(mask.numpy().astype(np.uint8))
            fig, axes  = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].imshow(img_rgb);      axes[0].set_title("Image")
            axes[1].imshow(gt_colored);   axes[1].set_title("GT Mask")
            axes[2].imshow(overlay);      axes[2].set_title("Prediction")
        else:
            # Mode 1: image | prediction overlay only
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(img_rgb);  axes[0].set_title("Image")
            axes[1].imshow(overlay);  axes[1].set_title("Prediction")

        for ax in axes:
            ax.axis("off")

        plt.suptitle(fname, fontsize=10, y=1.01)
        plt.tight_layout()

        save_path = vis_dir / f"sample_{i:03d}_{Path(fname).stem}.png"
        plt.savefig(str(save_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Viz] {i}/{n_samples} → {save_path.name}")
# ══════════════════════════════════════════════════════════════════════
# PLOTS — uses plot_class_iou and plot_metrics_bar from your viz utils
# ══════════════════════════════════════════════════════════════════════

def _save_plots(metrics: dict, num_classes: int, plot_dir: Path, exp_id: str):
    """
    Saves same plots as inference.py:
      - per_class_iou.png   (via plot_class_iou)
      - metrics_summary.png (via plot_metrics_bar)
    """
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Per-class IoU — same call as inference.py
    plot_class_iou(
        metrics["per_class_iou"],
        str(plot_dir / "domain_per_class_iou.png"),
        title=f"Domain Shift: Per-Class IoU — {exp_id}",
        class_names=CITYSCAPES_CLASSES[:num_classes],
    )

    # Summary metrics bar — same call as inference.py
    plot_metrics_bar(
        {
            "mIoU":      float(metrics.get("mIoU",           0)),
            "Pixel Acc": float(metrics.get("mean_pixel_acc", 0)),
            "Precision": float(metrics.get("mean_precision", 0)),
            "Recall":    float(metrics.get("mean_recall",    0)),
            "F1":        float(metrics.get("mean_f1",        0)),
        },
        str(plot_dir / "domain_metrics_summary.png"),
        title=f"Domain Shift: Metrics — {exp_id}",
    )

    print(f"[Plots] Saved to {plot_dir}")


# ══════════════════════════════════════════════════════════════════════
# RESULTS PRINT + SAVE — matches inference.py style
# ══════════════════════════════════════════════════════════════════════

def _print_metrics(metrics: dict, num_classes: int):
    print(f"\n{'='*70}")
    print("DOMAIN SHIFT RESULTS")
    print(f"{'='*70}")
    print(f"  mIoU               : {metrics.get('mIoU',           0):.4f}")
    print(f"  Mean Pixel Accuracy: {metrics.get('mean_pixel_acc', 0):.4f}")
    print(f"  Precision          : {metrics.get('mean_precision', 0):.4f}")
    print(f"  Recall             : {metrics.get('mean_recall',    0):.4f}")
    print(f"  F1                 : {metrics.get('mean_f1',        0):.4f}")
    print(f"\nPer-Class IoU:")
    for cls_name, iou in zip(CITYSCAPES_CLASSES[:num_classes], metrics.get("per_class_iou", [])):
        flag = " ←" if (not np.isnan(iou) and iou < 0.3) else ""
        val  = "N/A" if np.isnan(iou) else f"{iou:.4f}"
        print(f"  {cls_name:<20}: {val}{flag}")
    print(f"{'='*70}\n")


def _save_results_json(metrics: dict, output_dir: Path, ckpt_path: str, data_path: str, num_classes: int):
    def _to_py(v):
        if isinstance(v, (np.floating, np.integer)): return float(v)
        if isinstance(v, np.ndarray): return v.tolist()
        return v

    results = {
        "checkpoint":    ckpt_path,
        "data_path":     data_path,
        "metrics": {
            k: _to_py(metrics.get(k, 0))
            for k in ["mIoU", "fw_iou", "mean_pixel_acc",
                      "mean_class_acc", "mean_precision", "mean_recall", "mean_f1"]
        },
        "per_class_iou": {
            cls: _to_py(iou)
            for cls, iou in zip(
                CITYSCAPES_CLASSES[:num_classes],
                metrics.get("per_class_iou", [])
            )
        },
    }
    out = output_dir / "domain_check_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[DomainCheck] Results saved → {out}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def run_domain_check(
    exp_config:   str,
    img_dir:      str,
    mask_dir:     str  = None,
    checkpoint:   str  = None,
    n_samples:    int  = 10,
    output_dir:   str  = None,
    sample_seed:  int  = 0,
    batch_size:   int  = 4,
    max_eval:     int  = 300,
):
    cfg    = get_config(CONFIG_PATH, exp_config)
    device = get_device()

    # ── Output dirs — mirrors inference.py layout ──────────────────────
    # inference.py uses:
    #   outputs/inference/{exp_id}/inference_imgs/
    #   outputs/inference/{exp_id}/metrics/
    # We mirror as:
    #   outputs/inference/{exp_id}/domain_check/visualizations/
    #   outputs/inference/{exp_id}/domain_check/metrics/
    base_dir = (
        Path(output_dir)
        if output_dir
        else Path(cfg.checkpoint.dir).parent.parent
          / "inference" / cfg.experiment_id / "domain_check"
    )
    vis_dir  = base_dir / "visualizations"
    plot_dir = base_dir / "metrics"
    for d in (base_dir, vis_dir, plot_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Checkpoint — auto-load best if not specified ────────────────────
    if checkpoint is None:
        ckpt_dir = Path(cfg.checkpoint.dir)
        ckpts = sorted(
            ckpt_dir.rglob(f"{cfg.experiment_id}_epoch*.pth"),
            key=lambda p: float(p.stem.split("mIoU")[-1]),
            reverse=True,
        )
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        checkpoint = str(ckpts[0])

    mode_str = f"img+masks ({mask_dir})" if mask_dir else f"images only ({img_dir})"
    print(f"\n{'='*65}")
    print(f"  DOMAIN SHIFT CHECK — SEGMENTATION — {cfg.experiment_id}")
    print(f"{'='*65}")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Mode       : {mode_str}")
    print(f"  Device     : {device}")
    print(f"  Output dir : {base_dir}")
    print(f"{'='*65}\n")

    # ── Model — same as inference.py ──────────────────────────────────
    model = get_segmentation_model(
        num_classes             = cfg.data.num_classes,
        backbone                = cfg.model.backbone,
        output_stride           = cfg.model.output_stride,
        aux                     = cfg.training.aux_loss,
        use_pretrained_backbone = False,
        backbone_weights_path   = None,
    )
    load_checkpoint(checkpoint, model, device=str(device))
    model.to(device).eval()

    image_size = tuple(cfg.data.image_size)

    # ── Dataset ───────────────────────────────────────────────────────
    if mask_dir is not None:
        dataset = MaskFolderDataset(img_dir, mask_dir, image_size)
    else:
        dataset = ImageFolderDataset(img_dir, image_size)

    has_gt = getattr(dataset, "has_annotations", False)

    # Limit eval size
    if len(dataset) > max_eval:
        orig = len(dataset)
        idxs = np.random.default_rng(sample_seed).choice(orig, size=max_eval, replace=False)
        dataset = torch.utils.data.Subset(dataset, idxs)
        print(f"[DomainCheck] Restricted {orig} → {max_eval} samples")

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = _collate_fn,
        num_workers = 0,
        pin_memory  = True,
    )
    print(f"[DomainCheck] Batches: {len(loader)}\n")

    # ── Metrics ───────────────────────────────────────────────────────
    metrics = {}
    if has_gt:
        print("─" * 45)
        print("Computing segmentation metrics ...")
        print("─" * 45)
        metrics = compute_seg_metrics(
            model        = model,
            loader       = loader,
            device       = device,
            num_classes  = cfg.data.num_classes,
            ignore_index = cfg.data.ignore_index,
        )
        _print_metrics(metrics, cfg.data.num_classes)
        _save_plots(metrics, cfg.data.num_classes, plot_dir, cfg.experiment_id)
        _save_results_json(metrics, base_dir, checkpoint, img_dir, cfg.data.num_classes)
    else:
        print("[DomainCheck] No GT masks — visualization only.\n")

    # ── Visualize — uses save_batch_grid ──────────────────────────────
    print(f"{'─'*45}")
    print(f"Visualising {n_samples} samples ...")
    print(f"{'─'*45}")
    _visualize_samples(
        model     = model,
        dataset   = dataset,
        device    = device,
        n_samples = n_samples,
        vis_dir   = vis_dir,
        seed      = sample_seed,
        has_gt    = has_gt
    )

    print(f"\n[DomainCheck] Done. Outputs → {base_dir}\n")
    return metrics


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="Segmentation domain shift check")
    p.add_argument("--exp_config",   required=True,
                   help="Path to experiment config yaml")
    p.add_argument("--img_dir",      required=True,
                   help="Folder of images (flat or nested)")
    p.add_argument("--mask_dir",     default=None,
                   help="Folder of labelTrainIds PNG masks (optional, enables mIoU)")
    p.add_argument("--checkpoint",   default=None,
                   help="Path to .pth checkpoint. Auto-loads best if omitted.")
    p.add_argument("--n_samples",    type=int,   default=10)
    p.add_argument("--output_dir",   default=None,
                   help="Override output dir. Default mirrors inference.py layout.")
    p.add_argument("--sample_seed",  type=int,   default=0)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--max_eval",     type=int,   default=300)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_domain_check(
        exp_config  = args.exp_config,
        img_dir     = args.img_dir,
        mask_dir    = args.mask_dir,
        checkpoint  = args.checkpoint,
        n_samples   = args.n_samples,
        output_dir  = args.output_dir,
        sample_seed = args.sample_seed,
        batch_size  = args.batch_size,
        max_eval    = args.max_eval,
    )