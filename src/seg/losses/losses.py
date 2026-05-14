# src/seg/losses/losses.py
"""
Segmentation loss functions for Cityscapes / DeepLabV3+.

All losses accept preds as either:
  - torch.Tensor  (N, C, H, W)  — raw logits
  - dict          {"out": Tensor, "aux": Tensor}  — DeepLabV3+ output

Ignore index 255 is masked out before all computations.

Available via build_loss(name):
  Distribution-based : "ce", "weighted_ce", "ohem", "focal"
  Region-based       : "dice", "generalized_dice", "tversky",
                       "focal_tversky", "iou", "lovasz"
  Compound           : "ce_dice", "ce_generalized_dice",
                       "focal_dice", "focal_tversky_ce",
                       "ohem_dice", "iou_ce"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _get_pred(preds):
    """Extract main logit tensor from dict or raw tensor."""
    return preds["out"] if isinstance(preds, dict) else preds


def _get_aux(preds):
    """Return aux tensor or None."""
    return preds.get("aux") if isinstance(preds, dict) else None


def _aux_ce(preds, target, aux_weight: float, ignore_index: int) -> torch.Tensor:
    """Add auxiliary branch CE loss if present."""
    aux = _get_aux(preds)
    if aux is not None:
        return aux_weight * F.cross_entropy(aux, target, ignore_index=ignore_index)
    return torch.tensor(0.0, device=_get_pred(preds).device)


def _valid_pixels(pred, target, ignore_index: int):
    """
    Returns flattened valid (non-ignore) pixels.
    pred_flat : (V, C)
    target_flat: (V,)
    """
    n, c, h, w = pred.shape
    pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, c)
    target_flat = target.view(-1)
    valid = target_flat != ignore_index
    return pred_flat[valid], target_flat[valid]


def _one_hot_smooth(target, num_classes: int, ignore_index: int, device):
    """
    Returns one-hot (N, C, H, W) float with ignore positions zeroed.
    """
    valid_mask = (target != ignore_index).unsqueeze(1).float()   # (N,1,H,W)
    t = target.clone()
    t[target == ignore_index] = 0
    one_hot = F.one_hot(t, num_classes).permute(0, 3, 1, 2).float()  # (N,C,H,W)
    return one_hot * valid_mask, valid_mask


# ══════════════════════════════════════════════════════════════════════
# 1. DISTRIBUTION-BASED LOSSES
# ══════════════════════════════════════════════════════════════════════

class MixCELoss(nn.Module):
    """
    Standard Cross-Entropy with auxiliary branch support.
    Direct equivalent of Tramac repo's MixSoftmaxCrossEntropyLoss.

    aux_weight : weight for the auxiliary decoder output (default 0.4
                 follows DeepLabV3+ paper)
    weight     : optional (C,) class frequency weights for imbalance
    """
    def __init__(self, aux_weight: float = 0.4, ignore_index: int = 255,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_index)

    def forward(self, preds, target):
        pred = _get_pred(preds)
        loss = self.criterion(pred, target)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


class WeightedCELoss(nn.Module):
    """
    Cross-Entropy where class weights are computed from inverse
    pixel frequency.  Pass precomputed (C,) weight tensor, or let
    the loss compute it lazily on the first batch.

    Usage:
        # precompute once from train set (recommended):
        weights = compute_class_weights(train_loader, num_classes=19)
        loss_fn = WeightedCELoss(class_weights=weights)

        # or let it estimate from the first batch (less accurate):
        loss_fn = WeightedCELoss()
    """
    def __init__(self, class_weights: Optional[torch.Tensor] = None,
                 aux_weight: float = 0.4, ignore_index: int = 255):
        super().__init__()
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights)

    def forward(self, preds, target):
        pred = _get_pred(preds)
        w = self.class_weights.to(pred.device) if self.class_weights is not None else None
        loss = F.cross_entropy(pred, target, weight=w,
                               ignore_index=self.ignore_index)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


def compute_class_weights(loader, num_classes: int,
                          ignore_index: int = 255) -> torch.Tensor:
    """
    Utility: iterate loader once and return inverse-frequency
    class weights as a (num_classes,) float tensor.
    Call this once before training, save the result.
    """
    counts = torch.zeros(num_classes)
    for _, masks in loader:
        for c in range(num_classes):
            counts[c] += (masks == c).sum().item()
    counts = counts.clamp(min=1)
    freq = counts / counts.sum()
    weights = 1.0 / freq
    weights = weights / weights.sum() * num_classes   # normalise
    return weights


class OHEMCELoss(nn.Module):
    """
    Online Hard Example Mining Cross-Entropy.

    Only the hardest pixels (those where predicted probability of the
    true class is lowest) contribute to the gradient.

    thresh    : pixels with p_gt > thresh are discarded as 'easy'
    min_kept  : always keep at least this many pixels per batch
                (prevents degenerate case where nothing is kept)

    Paper: "Training Region-based Object Detectors with Online
            Hard Example Mining" — same principle applied pixel-wise.
    Widely used in DeepLabV3+ Cityscapes training.
    """
    def __init__(self, thresh: float = 0.7, min_kept: int = 100_000,
                 ignore_index: int = 255, aux_weight: float = 0.4):
        super().__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight

    def _ohem(self, pred, target):
        n, c, h, w = pred.shape
        target_flat = target.view(-1)
        valid = target_flat != self.ignore_index
        target_safe = target_flat.clone()
        target_safe[~valid] = 0

        with torch.no_grad():
            prob = F.softmax(pred.permute(0, 2, 3, 1).reshape(-1, c), dim=1)
            p_gt = prob[torch.arange(n * h * w, device=pred.device), target_safe]
            p_gt[~valid] = 1.0   # push easy ignore pixels out of selection

        kept = max(self.min_kept, int(valid.sum().item() * 0.5))
        if p_gt.numel() > kept:
            threshold = p_gt.kthvalue(min(kept, p_gt.numel()))[0]
            keep = p_gt < max(threshold.item(), self.thresh)
            valid = valid & keep

        target_out = target_flat.clone()
        target_out[~valid] = self.ignore_index
        return F.cross_entropy(pred, target_out.view(n, h, w),
                               ignore_index=self.ignore_index)

    def forward(self, preds, target):
        pred = _get_pred(preds)
        loss = self._ohem(pred, target)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for dense prediction.

    L = -α_t * (1 - p_t)^γ * log(p_t)

    gamma : focusing parameter.  0 → standard CE.
            Typical values: 0.5, 1.0, 2.0 (default)
    alpha : optional scalar or (C,) tensor for class balance weight.
            If None, no alpha weighting.

    Reference: "Focal Loss for Dense Object Detection" (Lin et al.)
    """
    def __init__(self, gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None,
                 ignore_index: int = 255, aux_weight: float = 0.4):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight
        self.register_buffer("alpha", alpha)

    def forward(self, preds, target):
        pred = _get_pred(preds)
        pred_flat, target_flat = _valid_pixels(pred, target, self.ignore_index)

        log_p = F.log_softmax(pred_flat, dim=1)
        p = torch.exp(log_p)
        log_p_gt = log_p[torch.arange(len(target_flat)), target_flat]
        p_gt = p[torch.arange(len(target_flat)), target_flat]
        focal_w = (1.0 - p_gt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(pred.device)
            focal_w = focal_w * alpha[target_flat]

        loss = -(focal_w * log_p_gt).mean()
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


# ══════════════════════════════════════════════════════════════════════
# 2. REGION-BASED LOSSES
# ══════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    Soft multiclass Dice Loss.

    Dice = 2 * |P ∩ G| / (|P| + |G|)
    Loss = 1 - mean_over_classes(Dice)

    smooth : Laplace smoothing; prevents division by zero on empty
             classes (common in Cityscapes for rare classes like train)
    """
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255,
                 aux_weight: float = 0.4):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight

    def _dice(self, pred, target):
        n, c, h, w = pred.shape
        prob = F.softmax(pred, dim=1)
        one_hot, valid_mask = _one_hot_smooth(target, c, self.ignore_index,
                                              pred.device)
        prob = prob * valid_mask
        dims = (0, 2, 3)
        inter = (prob * one_hot).sum(dims)
        card = (prob + one_hot).sum(dims)
        dice = (2.0 * inter + self.smooth) / (card + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, preds, target):
        pred = _get_pred(preds)
        loss = self._dice(pred, target)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss (GDL).

    Like standard Dice but each class is weighted by the INVERSE of
    its label frequency squared — so rare classes (bicycle, train,
    motorcycle in Cityscapes) contribute much more to the loss.

    Weight w_c = 1 / (sum_of_gt_pixels_for_class_c)^2

    Reference: "Generalised Dice overlap as a deep learning loss
                function for highly unbalanced segmentations"
                (Sudre et al., 2017)
    """
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255,
                 aux_weight: float = 0.4):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight

    def _gdl(self, pred, target):
        n, c, h, w = pred.shape
        prob = F.softmax(pred, dim=1)
        one_hot, valid_mask = _one_hot_smooth(target, c, self.ignore_index,
                                              pred.device)
        prob = prob * valid_mask

        # Class frequency weights: w_c = 1 / (freq_c^2 + eps)
        gt_freq = one_hot.sum(dim=(0, 2, 3))                  # (C,)
        w = 1.0 / (gt_freq ** 2 + 1e-6)                       # (C,)

        dims = (0, 2, 3)
        inter = (w * (prob * one_hot).sum(dims)).sum()
        card  = (w * (prob + one_hot).sum(dims)).sum()
        gdl   = 1.0 - (2.0 * inter + self.smooth) / (card + self.smooth)
        return gdl

    def forward(self, preds, target):
        pred = _get_pred(preds)
        loss = self._gdl(pred, target)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss — asymmetric generalisation of Dice.

    Tversky = TP / (TP + α*FP + β*FN)

    α controls false-positive penalty.
    β controls false-negative penalty.

    Setting α=β=0.5 → Dice Loss.
    Setting α=0.3, β=0.7 → penalises missed detections (FN) more.
    Good for: rare thin structures (bicycle lanes, poles in Cityscapes).

    Reference: "Tversky loss function for image segmentation using 3D
                fully convolutional deep networks" (Salehi et al., 2017)
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1.0, ignore_index: int = 255,
                 aux_weight: float = 0.4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight

    def _tversky(self, pred, target):
        n, c, h, w = pred.shape
        prob = F.softmax(pred, dim=1)
        one_hot, valid_mask = _one_hot_smooth(target, c, self.ignore_index,
                                              pred.device)
        prob = prob * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        tp = (prob * one_hot).sum(dims)
        fp = (prob * (1.0 - one_hot)).sum(dims)
        fn = ((1.0 - prob) * one_hot).sum(dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky.mean()

    def forward(self, preds, target):
        pred = _get_pred(preds)
        loss = self._tversky(pred, target)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss.

    Applies a focal-style exponent γ to Tversky:
    FTL = (1 - Tversky)^γ

    Low γ (< 1): emphasises large errors.
    High γ (> 1): focuses on hard, small-region classes.
    Typical γ range: 0.75 – 1.33

    Reference: "A novel focal Tversky loss function with improved
                attention U-Net for lesion segmentation"
                (Abraham & Khan, 2019)
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 0.75, smooth: float = 1.0,
                 ignore_index: int = 255, aux_weight: float = 0.4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight

    def _focal_tversky(self, pred, target):
        n, c, h, w = pred.shape
        prob = F.softmax(pred, dim=1)
        one_hot, valid_mask = _one_hot_smooth(target, c, self.ignore_index,
                                              pred.device)
        prob = prob * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        tp = (prob * one_hot).sum(dims)
        fp = (prob * (1.0 - one_hot)).sum(dims)
        fn = ((1.0 - prob) * one_hot).sum(dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        ftl = (1.0 - tversky) ** self.gamma
        return ftl.mean()

    def forward(self, preds, target):
        pred = _get_pred(preds)
        loss = self._focal_tversky(pred, target)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


class SoftIoULoss(nn.Module):
    """
    Soft (differentiable) IoU Loss.

    IoU = TP / (TP + FP + FN)
    Loss = 1 - mean_over_classes(IoU)

    Directly optimises the IoU metric, unlike CE which optimises
    per-pixel accuracy.  Smooth version avoids the non-differentiable
    argmax by using softmax probabilities directly.

    Note: the Lovász-Softmax loss (below) is theoretically superior
          for optimising mIoU, but this is simpler and dependency-free.
    """
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255,
                 aux_weight: float = 0.4):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight

    def _siou(self, pred, target):
        n, c, h, w = pred.shape
        prob = F.softmax(pred, dim=1)
        one_hot, valid_mask = _one_hot_smooth(target, c, self.ignore_index,
                                              pred.device)
        prob = prob * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        inter = (prob * one_hot).sum(dims)
        union = (prob + one_hot - prob * one_hot).sum(dims)
        iou   = (inter + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()

    def forward(self, preds, target):
        pred = _get_pred(preds)
        loss = self._siou(pred, target)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax Loss — the theoretically optimal surrogate for mIoU.

    Constructs a convex extension of the IoU loss using the Lovász
    extension from combinatorial optimisation.  Shown to outperform
    CE + Dice on Cityscapes in the original paper.

    mode:
        "multiclass" : per-class Lovász loss, averaged (use this)
        "binary"     : only for single-class problems

    Reference: "The Lovász-Softmax loss: A tractable surrogate for the
                optimisation of the IoU measure in neural networks"
                (Berman et al., CVPR 2018)

    No external library needed — implemented from scratch below.
    """
    def __init__(self, ignore_index: int = 255, aux_weight: float = 0.4,
                 per_image: bool = False):
        super().__init__()
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight
        self.per_image = per_image

    @staticmethod
    def _lovasz_grad(gt_sorted):
        """Lovász extension gradient for a sorted binary GT vector."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        iou = 1.0 - intersection / union
        if p > 1:
            iou[1:] = iou[1:] - iou[:-1]
        return iou

    def _lovasz_softmax_flat(self, probs_flat, labels_flat):
        """
        probs_flat : (V, C) softmax probabilities, V = valid pixels
        labels_flat: (V,)   integer class labels
        """
        c = probs_flat.shape[1]
        losses = []
        for cl in range(c):
            fg = (labels_flat == cl).float()
            if fg.sum() == 0:
                continue
            errors = (fg - probs_flat[:, cl]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = self._lovasz_grad(fg_sorted)
            losses.append((errors_sorted * grad).sum())
        if not losses:
            return torch.tensor(0.0, device=probs_flat.device,
                                requires_grad=True)
        return torch.stack(losses).mean()

    def _forward_single(self, pred, target):
        n, c, h, w = pred.shape
        probs = F.softmax(pred, dim=1)
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, c)
        target_flat = target.view(-1)
        valid = target_flat != self.ignore_index
        return self._lovasz_softmax_flat(probs_flat[valid], target_flat[valid])

    def forward(self, preds, target):
        pred = _get_pred(preds)
        if self.per_image:
            # Average Lovász over images in batch independently
            loss = torch.stack([
                self._forward_single(pred[i:i+1], target[i:i+1])
                for i in range(pred.shape[0])
            ]).mean()
        else:
            loss = self._forward_single(pred, target)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


# ══════════════════════════════════════════════════════════════════════
# 3. COMPOUND LOSSES
# ══════════════════════════════════════════════════════════════════════

class CEDiceLoss(nn.Module):
    """
    CE + λ * Dice.
    The most widely used compound loss in segmentation.
    CE handles per-pixel accuracy; Dice handles region overlap.
    """
    def __init__(self, dice_weight: float = 0.5, aux_weight: float = 0.4,
                 ignore_index: int = 255):
        super().__init__()
        self.ce   = MixCELoss(aux_weight=0.0, ignore_index=ignore_index)
        self.dice = DiceLoss(aux_weight=0.0,  ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.aux_weight  = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        loss = self.ce(preds, target) + self.dice_weight * self.dice(preds, target)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


class CEGeneralizedDiceLoss(nn.Module):
    """
    CE + λ * Generalized Dice.
    Better than CE+Dice when class imbalance is extreme (Cityscapes).
    """
    def __init__(self, gdl_weight: float = 0.5, aux_weight: float = 0.4,
                 ignore_index: int = 255):
        super().__init__()
        self.ce  = MixCELoss(aux_weight=0.0, ignore_index=ignore_index)
        self.gdl = GeneralizedDiceLoss(aux_weight=0.0, ignore_index=ignore_index)
        self.gdl_weight  = gdl_weight
        self.aux_weight  = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        loss = self.ce(preds, target) + self.gdl_weight * self.gdl(preds, target)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


class FocalDiceLoss(nn.Module):
    """
    Focal + λ * Dice.
    Focal handles hard pixels; Dice handles region overlap.
    """
    def __init__(self, gamma: float = 2.0, dice_weight: float = 0.5,
                 aux_weight: float = 0.4, ignore_index: int = 255):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, aux_weight=0.0, ignore_index=ignore_index)
        self.dice  = DiceLoss(aux_weight=0.0, ignore_index=ignore_index)
        self.dice_weight  = dice_weight
        self.aux_weight   = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        loss = self.focal(preds, target) + self.dice_weight * self.dice(preds, target)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


class FocalTverskyCELoss(nn.Module):
    """
    Focal Tversky + λ * CE.
    Focal Tversky pushes rare-class recall; CE keeps overall accuracy.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 0.75, ce_weight: float = 0.5,
                 aux_weight: float = 0.4, ignore_index: int = 255):
        super().__init__()
        self.ftl = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma,
                                    aux_weight=0.0, ignore_index=ignore_index)
        self.ce  = MixCELoss(aux_weight=0.0, ignore_index=ignore_index)
        self.ce_weight    = ce_weight
        self.aux_weight   = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        loss = self.ftl(preds, target) + self.ce_weight * self.ce(preds, target)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


class OHEMDiceLoss(nn.Module):
    """
    OHEM CE + λ * Dice.
    Hard pixel mining from OHEM + region awareness from Dice.
    Particularly strong for Cityscapes.
    """
    def __init__(self, thresh: float = 0.7, min_kept: int = 100_000,
                 dice_weight: float = 0.4, aux_weight: float = 0.4,
                 ignore_index: int = 255):
        super().__init__()
        self.ohem = OHEMCELoss(thresh=thresh, min_kept=min_kept,
                               aux_weight=0.0, ignore_index=ignore_index)
        self.dice = DiceLoss(aux_weight=0.0, ignore_index=ignore_index)
        self.dice_weight  = dice_weight
        self.aux_weight   = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        loss = self.ohem(preds, target) + self.dice_weight * self.dice(preds, target)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


class IoUCELoss(nn.Module):
    """
    Soft IoU + λ * CE.
    Directly optimises IoU metric while CE stabilises early training.
    """
    def __init__(self, iou_weight: float = 0.5, aux_weight: float = 0.4,
                 ignore_index: int = 255):
        super().__init__()
        self.iou = SoftIoULoss(aux_weight=0.0, ignore_index=ignore_index)
        self.ce  = MixCELoss(aux_weight=0.0,   ignore_index=ignore_index)
        self.iou_weight   = iou_weight
        self.aux_weight   = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        loss = self.iou_weight * self.iou(preds, target) + self.ce(preds, target)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


class LovaszCELoss(nn.Module):
    """
    Lovász-Softmax + λ * CE.
    Lovász directly optimises mIoU; CE prevents training instability.
    This is one of the strongest combinations for Cityscapes mIoU.
    """
    def __init__(self, ce_weight: float = 0.5, aux_weight: float = 0.4,
                 ignore_index: int = 255):
        super().__init__()
        self.lovasz = LovaszSoftmaxLoss(aux_weight=0.0, ignore_index=ignore_index)
        self.ce     = MixCELoss(aux_weight=0.0, ignore_index=ignore_index)
        self.ce_weight    = ce_weight
        self.aux_weight   = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        loss = self.lovasz(preds, target) + self.ce_weight * self.ce(preds, target)
        loss = loss + _aux_ce(preds, target, self.aux_weight, self.ignore_index)
        return loss


# ══════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════

_LOSS_REGISTRY = {
    # Distribution
    "ce":                  MixCELoss,
    "weighted_ce":         WeightedCELoss,
    "ohem":                OHEMCELoss,
    "focal":               FocalLoss,
    # Region
    "dice":                DiceLoss,
    "generalized_dice":    GeneralizedDiceLoss,
    "tversky":             TverskyLoss,
    "focal_tversky":       FocalTverskyLoss,
    "iou":                 SoftIoULoss,
    "lovasz":              LovaszSoftmaxLoss,
    # Compound
    "ce_dice":             CEDiceLoss,
    "ce_generalized_dice": CEGeneralizedDiceLoss,
    "focal_dice":          FocalDiceLoss,
    "focal_tversky_ce":    FocalTverskyCELoss,
    "ohem_dice":           OHEMDiceLoss,
    "iou_ce":              IoUCELoss,
    "lovasz_ce":           LovaszCELoss,
}


def build_loss(loss_type: str, ignore_index: int = 255, **kwargs) -> nn.Module:
    """
    Factory function.  Pass loss_type string + any constructor kwargs.

    Examples:
        build_loss("ce")
        build_loss("ohem", thresh=0.6, min_kept=80000)
        build_loss("focal", gamma=1.5)
        build_loss("tversky", alpha=0.3, beta=0.7)
        build_loss("focal_tversky", alpha=0.3, beta=0.7, gamma=0.75)
        build_loss("ce_dice", dice_weight=0.5)
        build_loss("lovasz_ce", ce_weight=0.5)
    """
    if loss_type not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{loss_type}'.\n"
            f"Available: {sorted(_LOSS_REGISTRY.keys())}"
        )
    return _LOSS_REGISTRY[loss_type](ignore_index=ignore_index, **kwargs)


def list_losses() -> list:
    """Convenience: print all available loss names."""
    return sorted(_LOSS_REGISTRY.keys())
