"""
src/seg/utils/visualization.py
--------------------------------
Helpers for visualising segmentation predictions.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from src.seg.constants import CITYSCAPES_PALETTE, CITYSCAPES_CLASSES, IMAGENET_MEAN, IMAGENET_STD


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """(H, W) class-index mask → (H, W, 3) RGB uint8."""
    h, w   = mask.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in enumerate(CITYSCAPES_PALETTE):
        colour[mask == cls_id] = rgb
    return colour


def _denorm(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation: (C,H,W) tensor → (H,W,3) uint8."""
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = img * std + mean
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def save_prediction(image_tensor, pred_mask, gt_mask, save_path, title=""):
    """Save three-panel figure: input | GT | prediction."""
    img      = _denorm(image_tensor)
    gt_rgb   = colorize_mask(gt_mask)
    pred_rgb = colorize_mask(pred_mask)

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


def save_batch_grid(images, preds, targets, save_path, max_images=4):
    """Save grid of [image | GT | pred] for up to max_images samples."""
    n    = min(images.shape[0], max_images)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row in range(n):
        img      = _denorm(images[row])
        gt_np    = targets[row].cpu().numpy() if isinstance(targets[row], torch.Tensor) else targets[row]
        pred_np  = preds[row].cpu().numpy()   if isinstance(preds[row],   torch.Tensor) else preds[row]
        gt_rgb   = colorize_mask(gt_np)
        pred_rgb = colorize_mask(pred_np)

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


def plot_class_iou(per_class_iou, save_path, title="Per-Class IoU",
                   class_names=CITYSCAPES_CLASSES):
    """Horizontal bar chart of per-class IoU, sorted descending."""
    iou_arr = np.array(per_class_iou, dtype=float)
    names   = class_names[:len(iou_arr)]

    # Sort descending, put NaN at bottom
    nan_mask     = np.isnan(iou_arr)
    iou_filled   = np.where(nan_mask, -1, iou_arr)
    order        = np.argsort(iou_filled)[::-1]
    sorted_iou   = iou_arr[order]
    sorted_names = [names[i] for i in order]

    colours = []
    for v in sorted_iou:
        if np.isnan(v):    colours.append("#9E9E9E")
        elif v >= 0.5:     colours.append("#4CAF50")
        elif v >= 0.3:     colours.append("#FF9800")
        else:              colours.append("#F44336")

    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.45)))
    bars = ax.barh(sorted_names, np.nan_to_num(sorted_iou), color=colours)

    for bar, v in zip(bars, sorted_iou):
        label = f"{v:.3f}" if not np.isnan(v) else "N/A"
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8)

    ax.set_xlim(0, 1.1)
    ax.set_xlabel("IoU")
    ax.set_title(title)
    ax.invert_yaxis()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved IoU chart → {save_path}")


def save_confusion_matrix(confusion_matrix, save_path,
                           class_names=CITYSCAPES_CLASSES,
                           normalize=True):
    """
    Save confusion matrix as a heatmap PNG.

    normalize=True  → row-normalized (shows per-class recall, easier to read)
    normalize=False → raw pixel counts
    """
    try:
        import seaborn as sns
    except ImportError:
        print("[Viz] seaborn not installed — skipping confusion matrix. pip install seaborn")
        return

    cm = np.array(confusion_matrix, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=False,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names[:cm.shape[1]],
        yticklabels=class_names[:cm.shape[0]],
        cbar_kws={"label": "Recall (row-normalized)" if normalize else "Pixel Count"},
        ax=ax,
        vmin=0, vmax=1 if normalize else None,
    )
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title(
        f"Confusion Matrix{' (Normalized by GT class)' if normalize else ''}",
        fontsize=14, fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved confusion matrix → {save_path}")