"""
src/seg/utils/visualization.py
--------------------------------
Helpers for visualising segmentation predictions.

Functions:
    colorize_mask      — class-index mask (H,W) → RGB image (H,W,3)
    save_prediction    — save side-by-side: input | GT | prediction
    save_batch_grid    — save a grid of predictions for a whole batch
    plot_class_iou     — bar chart of per-class IoU
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch

from src.seg.constants import CITYSCAPES_PALETTE, CITYSCAPES_CLASSES, IMAGENET_MEAN, IMAGENET_STD


# ── Colour mask helper ────────────────────────────────────────────────

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert a class-index mask to an RGB image using the Cityscapes palette.

    Args:
        mask : (H, W) int array with values 0–18 (or 255 = ignore)

    Returns:
        (H, W, 3) uint8 RGB array
    """
    h, w   = mask.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in enumerate(CITYSCAPES_PALETTE):
        colour[mask == cls_id] = rgb
    # Ignore pixels (255) remain black
    return colour


# ── Denormalise for display ───────────────────────────────────────────

def _denorm(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation and convert (C,H,W) tensor → (H,W,3) uint8.
    """
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    img  = tensor.cpu().numpy().transpose(1, 2, 0)   # (H,W,3)
    img  = img * std + mean
    img  = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


# ── Save one prediction ───────────────────────────────────────────────

def save_prediction(
    image_tensor: torch.Tensor,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    save_path: str,
    title: str = "",
):
    """
    Save a three-panel figure: input image | ground truth | prediction.

    Args:
        image_tensor : (3, H, W) normalised float tensor
        pred_mask    : (H, W) int array — argmax of model output
        gt_mask      : (H, W) int array — ground truth
        save_path    : where to save the PNG
        title        : optional figure title
    """
    img     = _denorm(image_tensor)
    gt_rgb  = colorize_mask(gt_mask)
    pred_rgb= colorize_mask(pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(img);       axes[0].set_title("Input Image")
    axes[1].imshow(gt_rgb);    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_rgb);  axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=13)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Save a batch grid ─────────────────────────────────────────────────

def save_batch_grid(
    images: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    save_path: str,
    max_images: int = 4,
):
    """
    Save a grid of [image | GT | pred] panels for up to max_images from a batch.

    Args:
        images   : (B, 3, H, W) normalised float tensors
        preds    : (B, H, W)    predicted class IDs
        targets  : (B, H, W)    GT class IDs
        save_path: output PNG path
        max_images: how many samples from the batch to show
    """
    n = min(images.shape[0], max_images)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))

    if n == 1:
        axes = axes.reshape(1, -1)   # ensure 2D array for indexing

    for row, i in enumerate(range(n)):
        img     = _denorm(images[i])
        gt_rgb  = colorize_mask(targets[i].cpu().numpy() if isinstance(targets[i], torch.Tensor) else targets[i])
        pred_rgb= colorize_mask(preds[i].cpu().numpy() if isinstance(preds[i], torch.Tensor) else preds[i])

        axes[row, 0].imshow(img);       axes[row, 0].set_title("Image")
        axes[row, 1].imshow(gt_rgb);    axes[row, 1].set_title("GT Mask")
        axes[row, 2].imshow(pred_rgb);  axes[row, 2].set_title("Prediction")
        for ax in axes[row]:
            ax.axis("off")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved batch grid → {save_path}")


# ── Per-class IoU bar chart ───────────────────────────────────────────

def plot_class_iou(
    per_class_iou: list,
    save_path: str,
    title: str = "Per-Class IoU",
    class_names: list = CITYSCAPES_CLASSES,
):
    """
    Save a horizontal bar chart of per-class IoU values.

    Args:
        per_class_iou : list of float (length = num_classes), NaN for absent classes
        save_path     : output PNG path
        title         : chart title
        class_names   : class name labels
    """
    iou_arr = np.array(per_class_iou)
    names   = class_names[:len(iou_arr)]

    # Sort by IoU descending for readability
    order  = np.argsort(iou_arr)[::-1]   # descending
    sorted_iou   = iou_arr[order]
    sorted_names = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.4)))
    colours = ["#4CAF50" if v >= 0.5 else "#FF9800" if v >= 0.3 else "#F44336"
               for v in sorted_iou]
    bars = ax.barh(sorted_names, sorted_iou, color=colours)

    # Add value labels
    for bar, v in zip(bars, sorted_iou):
        label = f"{v:.3f}" if not np.isnan(v) else "N/A"
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=8,
        )

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("IoU")
    ax.set_title(title)
    ax.invert_yaxis()   # highest IoU at the top

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved IoU chart → {save_path}")