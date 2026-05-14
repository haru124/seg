"""
src/seg/evaluation/metrics.py
------------------------------
Segmentation metrics for Cityscapes / DeepLabV3+ evaluation.

All metrics are computed from a running confusion matrix that is
accumulated across batches via update(), then read with compute().

Metrics computed:
    mIoU           — mean Intersection over Union across all classes
    per_class_iou  — IoU for each individual class
    fw_iou         — frequency-weighted IoU (rare classes count less)
    mean_pixel_acc — overall pixel accuracy (correct px / total px)
    mean_class_acc — mean per-class accuracy (accounts for class imbalance)
    boundary_iou   — IoU computed only on boundary pixels (±d pixels from edges)
    boundary_fscore— F-score (harmonic mean of precision & recall) at boundaries

Usage:
    metrics = SegmentationMetrics(num_classes=19, ignore_index=255)

    for images, targets in val_loader:
        preds = model(images)["out"].argmax(dim=1)   # (B, H, W) long
        metrics.update(preds, targets)

    results = metrics.compute()
    print(results["mIoU"], results["boundary_iou"])
    metrics.reset()   # call before next epoch
"""

import numpy as np
import torch
from scipy.ndimage import binary_dilation


class SegmentationMetrics:
    """
    Accumulates predictions and labels, computes all metrics on demand.

    Args:
        num_classes  : number of foreground classes (19 for Cityscapes)
        ignore_index : label value to exclude from all metrics (255)
        boundary_dilation : number of pixels to dilate edges for boundary metrics
                            Typical values: 2–5. Default 3.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255,
                 boundary_dilation: int = 3):
        self.num_classes        = num_classes
        self.ignore_index       = ignore_index
        self.boundary_dilation  = boundary_dilation

        # Confusion matrix: rows = true class, cols = predicted class
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        # Boundary accumulators: boundary_tp[c], boundary_pred[c], boundary_gt[c]
        self.boundary_tp   = np.zeros(num_classes, dtype=np.int64)
        self.boundary_pred = np.zeros(num_classes, dtype=np.int64)
        self.boundary_gt   = np.zeros(num_classes, dtype=np.int64)

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self):
        """Call at the start of each evaluation epoch."""
        self.confusion_matrix[:] = 0
        self.boundary_tp[:]      = 0
        self.boundary_pred[:]    = 0
        self.boundary_gt[:]      = 0

    # ── Update (per batch) ────────────────────────────────────────────

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Accumulate one batch of predictions.

        Args:
            preds   : (B, H, W) long tensor — argmax applied already
            targets : (B, H, W) long tensor — ground-truth class IDs
        """
        preds_np   = preds.cpu().numpy()    # (B, H, W)
        targets_np = targets.cpu().numpy()  # (B, H, W)

        for pred, target in zip(preds_np, targets_np):
            # ── confusion matrix ──
            valid = target != self.ignore_index
            p = pred[valid].astype(np.int64)
            t = target[valid].astype(np.int64)

            # Only count pixels with valid class IDs in both arrays
            in_range = (p >= 0) & (p < self.num_classes) & \
                       (t >= 0) & (t < self.num_classes)
            p, t = p[in_range], t[in_range]

            hist = np.bincount(
                self.num_classes * t + p,
                minlength=self.num_classes ** 2,
            ).reshape(self.num_classes, self.num_classes)
            self.confusion_matrix += hist

            # ── boundary metrics ──
            self._update_boundary(pred, target)

    # ── Boundary helpers ──────────────────────────────────────────────

    @staticmethod
    def _get_boundary(mask: np.ndarray, dilation: int) -> np.ndarray:
        """
        Return a boolean array that is True where the mask has an
        object boundary (class change within `dilation` pixels).

        Strategy: erode the binary class mask, then XOR with original
        to find the border ring.
        A dilation kernel of size (2d+1)×(2d+1) is used.
        """
        struct = np.ones((2 * dilation + 1, 2 * dilation + 1), dtype=bool)
        # boundary = pixels that change class within `dilation` distance
        dilated = binary_dilation(mask, structure=struct)
        return dilated ^ mask   # True only at the boundary ring

    def _update_boundary(self, pred: np.ndarray, target: np.ndarray):
        """
        For each class, compute TP/pred_count/gt_count at boundary pixels.
        Only boundaries of the GT mask are considered.
        """
        d = self.boundary_dilation
        ignore_mask = (target == self.ignore_index)

        for c in range(self.num_classes):
            gt_c   = (target == c)
            pred_c = (pred == c)

            # Skip completely absent classes
            if gt_c.sum() == 0:
                continue

            # Boundary ring pixels for this class
            gt_boundary   = self._get_boundary(gt_c, d)
            pred_boundary = self._get_boundary(pred_c, d)

            # Exclude ignore pixels
            gt_boundary   = gt_boundary   & ~ignore_mask
            pred_boundary = pred_boundary & ~ignore_mask

            self.boundary_tp[c]   += (gt_boundary & pred_boundary).sum()
            self.boundary_pred[c] += pred_boundary.sum()
            self.boundary_gt[c]   += gt_boundary.sum()

    # ── Compute (end of epoch) ────────────────────────────────────────

    def compute(self) -> dict:
        """
        Compute all metrics from the accumulated confusion matrix and
        boundary counts.

        Returns a dict with keys:
            mIoU, per_class_iou, fw_iou,
            mean_pixel_acc, mean_class_acc,
            boundary_iou, boundary_fscore,
            per_class_boundary_iou, per_class_boundary_fscore
        """
        hist = self.confusion_matrix.astype(np.float64)

        # ── Standard confusion-matrix metrics ─────────────────────────

        # Per-class IoU:  TP / (TP + FP + FN) for each class
        tp     = np.diag(hist)                         # (C,)
        fp     = hist.sum(axis=0) - tp                 # false positives per class
        fn     = hist.sum(axis=1) - tp                 # false negatives per class
        denom  = tp + fp + fn
        per_class_iou = np.where(denom > 0, tp / denom, np.nan)

        # mIoU: mean over classes that appear in GT (ignore NaN)
        miou = float(np.nanmean(per_class_iou))

        # Frequency-weighted IoU: weight each class by its pixel frequency
        freq   = hist.sum(axis=1) / hist.sum()         # (C,)
        valid  = freq > 0
        fw_iou = float((freq[valid] * per_class_iou[valid]).sum())

        # Mean pixel accuracy: fraction of correctly classified pixels
        total_px      = hist.sum()
        correct_px    = tp.sum()
        mean_pixel_acc = float(correct_px / total_px) if total_px > 0 else 0.0

        # Mean class accuracy: average recall per class
        gt_per_class = hist.sum(axis=1)                # true positives + false negatives
        cls_acc = np.where(gt_per_class > 0, tp / gt_per_class, np.nan)
        mean_class_acc = float(np.nanmean(cls_acc))

        # ── Boundary metrics ──────────────────────────────────────────

        # Per-class boundary IoU: TP_b / (TP_b + FP_b + FN_b)
        b_tp   = self.boundary_tp.astype(np.float64)
        b_pred = self.boundary_pred.astype(np.float64)
        b_gt   = self.boundary_gt.astype(np.float64)

        b_fp = b_pred - b_tp
        b_fn = b_gt   - b_tp
        b_denom = b_tp + b_fp + b_fn
        per_class_biou = np.where(b_denom > 0, b_tp / b_denom, np.nan)
        boundary_iou   = float(np.nanmean(per_class_biou))

        # Per-class boundary F-score: harmonic mean of precision & recall
        b_prec = np.where(b_pred > 0, b_tp / b_pred, np.nan)   # precision
        b_rec  = np.where(b_gt   > 0, b_tp / b_gt,   np.nan)   # recall
        b_fscore_denom = b_prec + b_rec
        per_class_bfscore = np.where(
            b_fscore_denom > 0,
            2.0 * b_prec * b_rec / b_fscore_denom,
            np.nan,
        )
        boundary_fscore = float(np.nanmean(per_class_bfscore))

        return {
            "mIoU"                     : miou,
            "per_class_iou"            : per_class_iou.tolist(),
            "fw_iou"                   : fw_iou,
            "mean_pixel_acc"           : mean_pixel_acc,
            "mean_class_acc"           : mean_class_acc,
            "boundary_iou"             : boundary_iou,
            "boundary_fscore"          : boundary_fscore,
            "per_class_boundary_iou"   : per_class_biou.tolist(),
            "per_class_boundary_fscore": per_class_bfscore.tolist(),
        }

    # ── Pretty print helper ───────────────────────────────────────────

    def summary(self, class_names: list = None) -> str:
        """Return a readable summary string of compute() results."""
        r = self.compute()
        lines = [
            f"  mIoU           : {r['mIoU']:.4f}",
            f"  fw_iou         : {r['fw_iou']:.4f}",
            f"  mean_pixel_acc : {r['mean_pixel_acc']:.4f}",
            f"  mean_class_acc : {r['mean_class_acc']:.4f}",
            f"  boundary_iou   : {r['boundary_iou']:.4f}",
            f"  boundary_fscore: {r['boundary_fscore']:.4f}",
        ]
        if class_names:
            lines.append("\n  Per-class IoU:")
            for name, iou in zip(class_names, r["per_class_iou"]):
                iou_str = f"{iou:.4f}" if not np.isnan(iou) else "  N/A"
                lines.append(f"    {name:<20} {iou_str}")
        return "\n".join(lines)
