"""
src/seg/evaluation/metrics.py
------------------------------
Segmentation metrics for Cityscapes / DeepLabV3+ evaluation.

Metrics computed:
    mIoU               — mean Intersection over Union (primary metric)
    fw_iou             — frequency-weighted IoU
    mean_pixel_acc     — overall pixel accuracy
    weighted_pixel_acc — class-frequency-weighted pixel accuracy
    mean_class_acc     — mean per-class recall
    mean_precision     — mean per-class precision
    mean_recall        — mean per-class recall
    mean_f1            — mean per-class F1 score
    boundary_iou       — IoU at boundary pixels only
    boundary_fscore    — F-score at boundary pixels

Usage:
    metrics = SegmentationMetrics(num_classes=19, ignore_index=255)
    for images, targets in val_loader:
        preds = model(images)["out"].argmax(dim=1)
        metrics.update(preds, targets)
    results = metrics.compute()
    metrics.reset()
"""

import numpy as np
import torch
from scipy.ndimage import binary_dilation


# Cityscapes class names for pretty printing
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]


class SegmentationMetrics:
    """
    Accumulates predictions and labels across batches,
    computes all metrics at end of epoch via compute().

    Args:
        num_classes       : number of classes (19 for Cityscapes)
        ignore_index      : label to exclude (255 for Cityscapes)
        boundary_dilation : pixel radius for boundary metrics (default 3)
    """

    def __init__(self, num_classes: int, ignore_index: int = 255,
                 boundary_dilation: int = 3, compute_boundary: bool = False):
        self.num_classes       = num_classes
        self.ignore_index      = ignore_index
        self.boundary_dilation = boundary_dilation

        # confusion_matrix[true_class, pred_class]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        # boundary accumulators per class
        self.compute_boundary = compute_boundary
        self.boundary_tp   = np.zeros(num_classes, dtype=np.int64)
        self.boundary_pred = np.zeros(num_classes, dtype=np.int64)
        self.boundary_gt   = np.zeros(num_classes, dtype=np.int64)

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self):
        """Call at the start of each evaluation epoch."""
        self.confusion_matrix[:] = 0
        self.boundary_tp[:]      = 0
        self.boundary_pred[:]    = 0
        self.boundary_gt[:]      = 0

    # ── Update (per batch) ─────────────────────────────────────────────
    """
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        '''
        Accumulate one batch.

        Args:
            preds   : (B, H, W) long tensor — argmax of logits
            targets : (B, H, W) long tensor — ground truth labels
        '''
        preds_np   = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        for pred, target in zip(preds_np, targets_np):
            # ── confusion matrix ──────────────────────────────────────
            valid    = target != self.ignore_index
            p        = pred[valid].astype(np.int64)
            t        = target[valid].astype(np.int64)
            in_range = (p >= 0) & (p < self.num_classes) & \
                       (t >= 0) & (t < self.num_classes)
            p, t     = p[in_range], t[in_range]

            hist = np.bincount(
                self.num_classes * t + p,
                minlength=self.num_classes ** 2,
            ).reshape(self.num_classes, self.num_classes)
            self.confusion_matrix += hist

            # ── boundary metrics ──────────────────────────────────────
            if self.compute_boundary:
                self._update_boundary(pred, target)
    """
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds_np   = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Process entire batch at once instead of per-image loop
        valid      = targets_np != self.ignore_index
        p          = preds_np[valid].astype(np.int64)
        t          = targets_np[valid].astype(np.int64)
        in_range   = (p >= 0) & (p < self.num_classes) & \
                    (t >= 0) & (t < self.num_classes)
        p, t       = p[in_range], t[in_range]

        hist = np.bincount(
            self.num_classes * t + p,
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += hist

        if self.compute_boundary:
            for pred_img, target_img in zip(preds_np, targets_np):
                self._update_boundary(pred_img, target_img)

    # ── Boundary helpers ───────────────────────────────────────────────

    @staticmethod
    def _get_boundary(mask: np.ndarray, dilation: int) -> np.ndarray:
        """
        Return boolean array True at boundary pixels of a binary mask.
        Boundary = pixels that change class within `dilation` pixels.
        """
        struct  = np.ones((2 * dilation + 1, 2 * dilation + 1), dtype=bool)
        dilated = binary_dilation(mask, structure=struct)
        return dilated ^ mask

    def _update_boundary(self, pred: np.ndarray, target: np.ndarray):
        """Accumulate boundary TP/pred/gt counts per class."""
        d           = self.boundary_dilation
        ignore_mask = (target == self.ignore_index)

        for c in range(self.num_classes):
            gt_c   = (target == c)
            pred_c = (pred == c)

            if gt_c.sum() == 0:
                continue

            gt_boundary   = self._get_boundary(gt_c, d)   & ~ignore_mask
            pred_boundary = self._get_boundary(pred_c, d) & ~ignore_mask

            self.boundary_tp[c]   += (gt_boundary & pred_boundary).sum()
            self.boundary_pred[c] += pred_boundary.sum()
            self.boundary_gt[c]   += gt_boundary.sum()

    # ── Compute (end of epoch) ─────────────────────────────────────────

    def compute(self) -> dict:
        """
        Compute all metrics from accumulated confusion matrix.

        All NaN divisions are suppressed cleanly with np.errstate —
        this is expected behaviour when a class has zero pixels in the
        validation set (common for rare classes like train/motorcycle
        especially early in training or with small datasets).
        NaN values are excluded from mean computations via np.nanmean.
        """
        hist = self.confusion_matrix.astype(np.float64)

        with np.errstate(divide='ignore', invalid='ignore'):

            # ── Standard metrics from confusion matrix ─────────────────

            tp    = np.diag(hist)                   # (C,) true positives per class
            fp    = hist.sum(axis=0) - tp           # false positives per class
            fn    = hist.sum(axis=1) - tp           # false negatives per class

            # IoU per class: TP / (TP + FP + FN)
            denom         = tp + fp + fn
            per_class_iou = np.where(denom > 0, tp / denom, np.nan)
            miou          = float(np.nanmean(per_class_iou))

            # Frequency-weighted IoU
            freq   = hist.sum(axis=1) / hist.sum()
            valid  = freq > 0
            fw_iou = float((freq[valid] * per_class_iou[valid]).sum())

            # Overall pixel accuracy
            total_px       = hist.sum()
            correct_px     = tp.sum()
            mean_pixel_acc = float(correct_px / total_px) if total_px > 0 else 0.0

            # Per-class recall = TP / (TP + FN) = TP / GT_pixels
            gt_per_class   = hist.sum(axis=1)
            per_class_recall = np.where(gt_per_class > 0,
                                        tp / gt_per_class, np.nan)
            mean_class_acc   = float(np.nanmean(per_class_recall))

            # Per-class precision = TP / (TP + FP) = TP / predicted_pixels
            pred_per_class     = hist.sum(axis=0)
            per_class_precision = np.where(pred_per_class > 0,
                                           tp / pred_per_class, np.nan)

            # Per-class F1 = 2 * P * R / (P + R)
            pr_sum       = per_class_precision + per_class_recall
            per_class_f1 = np.where(pr_sum > 0,
                                    2 * per_class_precision * per_class_recall / pr_sum,
                                    np.nan)

            mean_precision = float(np.nanmean(per_class_precision))
            mean_recall    = float(np.nanmean(per_class_recall))
            mean_f1        = float(np.nanmean(per_class_f1))

            # Class-frequency weighted pixel accuracy
            class_weights      = gt_per_class / hist.sum()
            cls_acc            = per_class_recall   # same as per-class recall
            weighted_pixel_acc = float(np.nansum(class_weights * cls_acc))

            # ── Boundary metrics ───────────────────────────────────────
            boundary_iou    = 0.0
            boundary_fscore = 0.0
            per_class_biou = np.zeros(self.num_classes, dtype=np.float64)
            per_class_bfscore = np.zeros(self.num_classes, dtype=np.float64)
            
            if self.compute_boundary:

                b_tp   = self.boundary_tp.astype(np.float64)
                b_pred = self.boundary_pred.astype(np.float64)
                b_gt   = self.boundary_gt.astype(np.float64)

                b_fp    = b_pred - b_tp
                b_fn    = b_gt   - b_tp
                b_denom = b_tp + b_fp + b_fn

                # Boundary IoU per class
                per_class_biou = np.where(b_denom > 0, b_tp / b_denom, np.nan)
                boundary_iou   = float(np.nanmean(per_class_biou))

                # Boundary precision and recall
                b_prec = np.where(b_pred > 0, b_tp / b_pred, np.nan)
                b_rec  = np.where(b_gt   > 0, b_tp / b_gt,   np.nan)

                # Boundary F-score
                b_pr_sum          = b_prec + b_rec
                per_class_bfscore = np.where(b_pr_sum > 0,
                                            2.0 * b_prec * b_rec / b_pr_sum,
                                            np.nan)
                boundary_fscore   = float(np.nanmean(per_class_bfscore)) 

        return {
            # ── Primary metric ──────────────────────────────────────────
            "mIoU"                     : miou,

            # ── Pixel-level metrics ────────────────────────────────────
            "fw_iou"                   : fw_iou,
            "mean_pixel_acc"           : mean_pixel_acc,
            "weighted_pixel_acc"       : weighted_pixel_acc,
            "mean_class_acc"           : mean_class_acc,

            # ── Precision / Recall / F1 ────────────────────────────────
            "mean_precision"           : mean_precision,
            "mean_recall"              : mean_recall,
            "mean_f1"                  : mean_f1,

            # ── Boundary metrics ───────────────────────────────────────
            "boundary_iou"             : boundary_iou,
            "boundary_fscore"          : boundary_fscore,

            # ── Per-class arrays (for logging / analysis) ──────────────
            "per_class_iou"            : per_class_iou.tolist(),
            "per_class_precision"      : per_class_precision.tolist(),
            "per_class_recall"         : per_class_recall.tolist(),
            "per_class_f1"             : per_class_f1.tolist(),
            "per_class_support"        : gt_per_class.astype(np.int64).tolist(),
            "per_class_boundary_iou"   : per_class_biou.tolist(),
            "per_class_boundary_fscore": per_class_bfscore.tolist(),

            # ── Confusion matrix (for visualization) ───────────────────
            #"confusion_matrix"         : self.confusion_matrix.tolist(),
        }

    # ── Pretty print ───────────────────────────────────────────────────

    def summary(self, class_names: list = None) -> str:
        """Return a readable summary string."""
        r     = self.compute()
        names = class_names or CITYSCAPES_CLASSES

        lines = [
            f"  mIoU               : {r['mIoU']:.4f}",
            f"  fw_iou             : {r['fw_iou']:.4f}",
            f"  mean_pixel_acc     : {r['mean_pixel_acc']:.4f}",
            f"  weighted_pixel_acc : {r['weighted_pixel_acc']:.4f}",
            f"  mean_class_acc     : {r['mean_class_acc']:.4f}",
            f"  mean_precision     : {r['mean_precision']:.4f}",
            f"  mean_recall        : {r['mean_recall']:.4f}",
            f"  mean_f1            : {r['mean_f1']:.4f}",
            f"  boundary_iou       : {r['boundary_iou']:.4f}",
            f"  boundary_fscore    : {r['boundary_fscore']:.4f}",
            "",
            f"  {'Class':<22} {'IoU':>7} {'Prec':>7} {'Recall':>7} "
            f"{'F1':>7} {'Support':>10}",
            "  " + "-" * 64,
        ]

        for i, name in enumerate(names[:self.num_classes]):
            iou  = r['per_class_iou'][i]
            prec = r['per_class_precision'][i]
            rec  = r['per_class_recall'][i]
            f1   = r['per_class_f1'][i]
            sup  = r['per_class_support'][i]

            def fmt(v):
                return f"{v:.4f}" if not np.isnan(v) else "   N/A"

            lines.append(
                f"  {name:<22} {fmt(iou):>7} {fmt(prec):>7} "
                f"{fmt(rec):>7} {fmt(f1):>7} {sup:>10,}"
            )

        return "\n".join(lines)