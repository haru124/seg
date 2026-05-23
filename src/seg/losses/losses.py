"""
src/seg/losses/losses.py
-------------------------
Clean, readable segmentation losses for DeepLabV3+.

HOW TO USE:
    from src.seg.losses.losses import build_loss

    loss_fn = build_loss("ce")                        # standard cross-entropy
    loss_fn = build_loss("ce_dice")                   # CE + Dice (most popular)
    loss_fn = build_loss("focal")                     # focal loss
    loss_fn = build_loss("ohem")                      # hard example mining
    loss_fn = build_loss("lovasz_ce")                 # best for mIoU

HOW MODEL OUTPUT WORKS:
    DeepLabV3+ returns a dict: {"out": main_pred, "aux": aux_pred}
    All losses handle this automatically.
    aux_weight=0.4 means: total_loss = main_loss + 0.4 * aux_loss

AVAILABLE LOSSES:
    Single   : "ce", "weighted_ce", "focal", "ohem"
    Region   : "dice", "iou", "lovasz"
    Compound : "ce_dice", "ce_iou", "focal_dice", "ohem_dice", "lovasz_ce"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS  (not for direct use)
# ══════════════════════════════════════════════════════════════════════

def _extract_main(preds):
    """Get the main prediction tensor from model output dict or raw tensor."""
    return preds["out"] if isinstance(preds, dict) else preds


def _extract_aux(preds):
    """Get the auxiliary prediction tensor, or None if not present."""
    return preds.get("aux") if isinstance(preds, dict) else None


def _aux_loss(preds, target, aux_weight, ignore_index):
    """Compute auxiliary branch CE loss. Returns 0 if no aux branch."""
    aux = _extract_aux(preds)
    if aux is not None:
        return aux_weight * F.cross_entropy(aux, target, ignore_index=ignore_index)
    return 0.0


def _to_one_hot(target, num_classes, ignore_index):
    """
    Convert integer label map to one-hot float tensor.
    Ignore pixels are zeroed out in all channels.

    Returns:
        one_hot   : (N, C, H, W) float
        valid_mask: (N, 1, H, W) float — 1 for valid pixels, 0 for ignored
    """
    valid_mask = (target != ignore_index).unsqueeze(1).float()
    safe_target = target.clone()
    safe_target[target == ignore_index] = 0          # avoid index out of range
    one_hot = F.one_hot(safe_target, num_classes)    # (N, H, W, C)
    one_hot = one_hot.permute(0, 3, 1, 2).float()   # (N, C, H, W)
    one_hot = one_hot * valid_mask                    # zero out ignored pixels
    return one_hot, valid_mask


# ══════════════════════════════════════════════════════════════════════
# CROSS-ENTROPY LOSSES
# ══════════════════════════════════════════════════════════════════════

class CrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy loss.
    Simple and effective — good default choice.

    Args:
        aux_weight   : weight for auxiliary decoder branch (0.4 is standard)
        ignore_index : pixel value to ignore (255 for Cityscapes)
    """
    def __init__(self, aux_weight=0.4, ignore_index=255):
        super().__init__()
        self.aux_weight   = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        pred = _extract_main(preds)
        loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index)
        loss = loss + _aux_loss(preds, target, self.aux_weight, self.ignore_index)
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy with per-class weights.
    Use when some classes are rare (e.g. bicycle, motorcycle in Cityscapes).

    Args:
        class_weights: (num_classes,) tensor — higher = more attention to that class
                       Compute with: 1 / class_pixel_frequency
        aux_weight   : weight for aux branch
        ignore_index : pixel to ignore
    
    Example:
        weights = torch.tensor([1.0, 2.5, 1.0, ...])  # 19 values for Cityscapes
        loss_fn = WeightedCrossEntropyLoss(class_weights=weights)
    """
    def __init__(self, class_weights=None, aux_weight=0.4, ignore_index=255):
        super().__init__()
        self.aux_weight   = aux_weight
        self.ignore_index = ignore_index
        # register_buffer: moves weights to GPU automatically with model.to(device)
        self.register_buffer("class_weights", class_weights)

    def forward(self, preds, target):
        pred    = _extract_main(preds)
        weights = self.class_weights.to(pred.device) if self.class_weights is not None else None
        loss    = F.cross_entropy(pred, target, weight=weights,
                                  ignore_index=self.ignore_index)
        loss    = loss + _aux_loss(preds, target, self.aux_weight, self.ignore_index)
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss — down-weights easy pixels, focuses on hard ones.
    Good when you have class imbalance.

    Formula: FL = -(1 - p_correct)^gamma * log(p_correct)
    
    Args:
        gamma: focusing strength. 0 = standard CE. Higher = more focus on hard pixels.
               Typical values: 1.0, 2.0 (default), 3.0
        aux_weight   : weight for aux branch
        ignore_index : pixel to ignore
    """
    def __init__(self, gamma=2.0, aux_weight=0.4, ignore_index=255):
        super().__init__()
        self.gamma        = gamma
        self.aux_weight   = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        pred = _extract_main(preds)

        # Flatten and remove ignored pixels
        N, C, H, W    = pred.shape
        pred_flat      = pred.permute(0, 2, 3, 1).reshape(-1, C)   # (N*H*W, C)
        target_flat    = target.view(-1)                             # (N*H*W,)
        valid          = target_flat != self.ignore_index
        pred_flat      = pred_flat[valid]
        target_flat    = target_flat[valid]

        # Compute focal weight: (1 - p_correct)^gamma
        log_p          = F.log_softmax(pred_flat, dim=1)
        p              = torch.exp(log_p)
        p_correct      = p[torch.arange(len(target_flat)), target_flat]  ###p[[0,1,2], [1,0,2]] means: take p[0][1] take p[1][0] take p[2][2]
        log_p_correct  = log_p[torch.arange(len(target_flat)), target_flat]
        focal_weight   = (1.0 - p_correct) ** self.gamma

        loss = -(focal_weight * log_p_correct).mean()
        loss = loss + _aux_loss(preds, target, self.aux_weight, self.ignore_index)
        return loss


class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining (OHEM) loss.
    Only trains on the hardest pixels — where the model is most confused.
    Very effective for Cityscapes.

    Args:
        thresh    : pixels where p_correct > thresh are ignored as "easy"
        min_kept  : always keep at least this many pixels (prevents empty batches)
        aux_weight: weight for aux branch
        ignore_index: pixel to ignore
    """
    def __init__(self, thresh=0.7, min_kept=100_000,
                 aux_weight=0.4, ignore_index=255):
        super().__init__()
        self.thresh       = thresh
        self.min_kept     = min_kept
        self.aux_weight   = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        pred = _extract_main(preds)
        N, C, H, W = pred.shape

        # Get probability of the correct class for each pixel
        target_flat = target.view(-1)
        safe_target = target_flat.clone()
        safe_target[target_flat == self.ignore_index] = 0

        with torch.no_grad():
            prob        = F.softmax(pred.permute(0, 2, 3, 1).reshape(-1, C), dim=1)
            p_correct   = prob[torch.arange(N * H * W, device=pred.device), safe_target]
            p_correct[target_flat == self.ignore_index] = 1.0  # mark ignored as easy

        # Keep the hardest pixels (lowest p_correct)
        num_valid = (target_flat != self.ignore_index).sum().item()
        num_keep  = max(self.min_kept, int(num_valid * 0.5))
        if p_correct.numel() > num_keep:
            threshold = p_correct.kthvalue(min(num_keep, p_correct.numel()))[0]
            hard_mask = p_correct < max(threshold.item(), self.thresh)
        else:
            hard_mask = target_flat != self.ignore_index

        # Set easy pixels to ignore_index so CE skips them
        hard_target = target_flat.clone()
        hard_target[~hard_mask] = self.ignore_index
        loss = F.cross_entropy(pred, hard_target.view(N, H, W),
                               ignore_index=self.ignore_index)
        loss = loss + _aux_loss(preds, target, self.aux_weight, self.ignore_index)
        return loss


# ══════════════════════════════════════════════════════════════════════
# REGION-BASED LOSSES
# ══════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    Dice loss — measures overlap between prediction and ground truth.
    
    Formula: Dice = 2*|P∩G| / (|P|+|G|),   Loss = 1 - Dice
    
    Good complement to CE: CE is pixel-wise, Dice is region-wise.
    Use ce_dice compound loss for best results.

    Args:
        smooth      : small value to avoid division by zero
        aux_weight  : weight for aux branch
        ignore_index: pixel to ignore
    """
    def __init__(self, smooth=1.0, aux_weight=0.4, ignore_index=255):
        super().__init__()
        self.smooth       = smooth
        self.aux_weight   = aux_weight
        self.ignore_index = ignore_index

    def _dice(self, pred, target):
        N, C, H, W        = pred.shape
        prob              = F.softmax(pred, dim=1)               # (N, C, H, W)  
        #apply softmax across the CLASS dimension (C) NOT across height or width
        one_hot, mask     = _to_one_hot(target, C, self.ignore_index)
        prob              = prob * mask                           # zero ignored pixels

        # Sum over batch + spatial dims, keep class dim
        intersection      = (prob * one_hot).sum(dim=(0, 2, 3))  # (C,)
        cardinality       = (prob + one_hot).sum(dim=(0, 2, 3))  # (C,)
        dice_per_class    = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()

    def forward(self, preds, target):
        pred = _extract_main(preds)
        loss = self._dice(pred, target)
        loss = loss + _aux_loss(preds, target, self.aux_weight, self.ignore_index)
        return loss


class SoftIoULoss(nn.Module):
    """
    Soft IoU loss — directly optimises the IoU metric.
    
    Formula: IoU = |P∩G| / |P∪G|,   Loss = 1 - IoU
    
    Args:
        smooth      : small value to avoid division by zero
        aux_weight  : weight for aux branch
        ignore_index: pixel to ignore
    """
    def __init__(self, smooth=1.0, aux_weight=0.4, ignore_index=255):
        super().__init__()
        self.smooth       = smooth
        self.aux_weight   = aux_weight
        self.ignore_index = ignore_index

    def _iou(self, pred, target):
        N, C, H, W     = pred.shape
        prob           = F.softmax(pred, dim=1)
        one_hot, mask  = _to_one_hot(target, C, self.ignore_index)
        prob           = prob * mask
        one_hot        = one_hot * mask

        intersection   = (prob * one_hot).sum(dim=(0, 2, 3))
        union          = (prob + one_hot - prob * one_hot).sum(dim=(0, 2, 3))
        iou_per_class  = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou_per_class.mean()

    def forward(self, preds, target):
        pred = _extract_main(preds)
        loss = self._iou(pred, target)
        loss = loss + _aux_loss(preds, target, self.aux_weight, self.ignore_index)
        return loss


class LovaszLoss(nn.Module):
    """
    Lovász-Softmax loss — theoretically the best surrogate for mIoU.
    Shown to outperform CE+Dice on Cityscapes benchmarks.
    No external library needed.

    Args:
        aux_weight  : weight for aux branch
        ignore_index: pixel to ignore
    
    Reference: Berman et al., CVPR 2018
    """
    def __init__(self, aux_weight=0.4, ignore_index=255):
        super().__init__()
        self.aux_weight   = aux_weight
        self.ignore_index = ignore_index

    @staticmethod
    def _lovasz_gradient(gt_sorted):
        """Compute Lovász extension gradient for sorted binary labels."""
        n             = len(gt_sorted)
        total_pos     = gt_sorted.sum()
        tp_cumsum     = total_pos - gt_sorted.float().cumsum(0)
        fp_cumsum     = (1 - gt_sorted).float().cumsum(0)
        iou_values    = 1.0 - tp_cumsum / (total_pos + fp_cumsum)
        # Gradient = difference of successive IoU values
        if n > 1:
            iou_values[1:] = iou_values[1:] - iou_values[:-1]
        return iou_values

    def _lovasz_per_class(self, probs, labels):
        """
        probs : (V, C) softmax probabilities for V valid pixels
        labels: (V,)   integer class labels
        """
        C      = probs.shape[1]
        losses = []
        for c in range(C):
            fg     = (labels == c).float()       # binary: is this pixel class c?
            if fg.sum() == 0:
                continue
            errors          = (fg - probs[:, c]).abs()       # prediction error per pixel
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted       = fg[perm]
            grad            = self._lovasz_gradient(fg_sorted)
            losses.append((errors_sorted * grad).sum())

        return torch.stack(losses).mean() if losses else torch.tensor(0.0,
               device=probs.device, requires_grad=True)

    def forward(self, preds, target):
        pred   = _extract_main(preds)
        N, C, H, W = pred.shape
        probs  = F.softmax(pred, dim=1)

        # Flatten and filter ignored pixels
        probs_flat  = probs.permute(0, 2, 3, 1).reshape(-1, C)   # (N*H*W, C)
        target_flat = target.view(-1)
        valid       = target_flat != self.ignore_index
        probs_flat  = probs_flat[valid]
        target_flat = target_flat[valid]

        loss = self._lovasz_per_class(probs_flat, target_flat)
        loss = loss + _aux_loss(preds, target, self.aux_weight, self.ignore_index)
        return loss


# ══════════════════════════════════════════════════════════════════════
# COMPOUND LOSSES  (mix and match above)
# ══════════════════════════════════════════════════════════════════════

class CEDiceLoss(nn.Module):
    """
    Cross-Entropy + Dice.   total = CE + weight * Dice
    Most popular combination. CE for pixel accuracy, Dice for region overlap.
    
    Args:
        dice_weight : how much Dice contributes (0.5 = equal weight)
        aux_weight  : aux branch weight
        ignore_index: pixel to ignore
    """
    def __init__(self, dice_weight=0.5, aux_weight=0.4, ignore_index=255):
        super().__init__()
        # aux_weight=0 in sub-losses — we add aux once at the end
        self.ce          = CrossEntropyLoss(aux_weight=0.0, ignore_index=ignore_index)
        self.dice        = DiceLoss(aux_weight=0.0, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.aux_weight  = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        loss = self.ce(preds, target) + self.dice_weight * self.dice(preds, target)
        loss = loss + _aux_loss(preds, target, self.aux_weight, self.ignore_index)
        return loss

class WeightedCEDiceLoss(nn.Module):
    """
    Weighted Cross-Entropy + Dice.

    total = WeightedCE + dice_weight * Dice

    Good for:
        - class imbalance
        - small/thin objects
        - segmentation overlap quality

    Args:
        class_weights : tensor of class weights
        dice_weight   : Dice contribution strength
        aux_weight    : aux branch weight
        ignore_index  : ignored label
    """

    def __init__(
        self,
        class_weights=None,
        dice_weight=0.5,
        aux_weight=0.4,
        ignore_index=255,
    ):
        super().__init__()

        self.wce = WeightedCrossEntropyLoss(
            class_weights=class_weights,
            aux_weight=0.0,
            ignore_index=ignore_index,
        )

        self.dice = DiceLoss(
            aux_weight=0.0,
            ignore_index=ignore_index,
        )

        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):

        loss = (
            self.wce(preds, target)
            + self.dice_weight * self.dice(preds, target)
        )

        loss = loss + _aux_loss(
            preds,
            target,
            self.aux_weight,
            self.ignore_index,
        )

        return loss

class CEIoULoss(nn.Module):
    """
    Cross-Entropy + Soft IoU.   total = CE + weight * IoU
    Directly optimises IoU metric while CE stabilises training.

    Args:
        iou_weight  : how much IoU loss contributes
        aux_weight  : aux branch weight
        ignore_index: pixel to ignore
    """
    def __init__(self, iou_weight=0.5, aux_weight=0.4, ignore_index=255):
        super().__init__()
        self.ce          = CrossEntropyLoss(aux_weight=0.0, ignore_index=ignore_index)
        self.iou         = SoftIoULoss(aux_weight=0.0, ignore_index=ignore_index)
        self.iou_weight  = iou_weight
        self.aux_weight  = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        loss = self.ce(preds, target) + self.iou_weight * self.iou(preds, target)
        loss = loss + _aux_loss(preds, target, self.aux_weight, self.ignore_index)
        return loss


class FocalDiceLoss(nn.Module):
    """
    Focal + Dice.   total = Focal + weight * Dice
    Focal handles hard pixels + Dice handles region overlap.
    Good for imbalanced datasets.

    Args:
        gamma       : focal loss focusing strength (2.0 default)
        dice_weight : how much Dice contributes
        aux_weight  : aux branch weight
        ignore_index: pixel to ignore
    """
    def __init__(self, gamma=2.0, dice_weight=0.5, aux_weight=0.4, ignore_index=255):
        super().__init__()
        self.focal       = FocalLoss(gamma=gamma, aux_weight=0.0, ignore_index=ignore_index)
        self.dice        = DiceLoss(aux_weight=0.0, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.aux_weight  = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        loss = self.focal(preds, target) + self.dice_weight * self.dice(preds, target)
        loss = loss + _aux_loss(preds, target, self.aux_weight, self.ignore_index)
        return loss


class OHEMDiceLoss(nn.Module):
    """
    OHEM + Dice.   total = OHEM_CE + weight * Dice
    Hard example mining + region overlap. Strong for Cityscapes.

    Args:
        thresh      : easy pixel threshold for OHEM
        min_kept    : minimum pixels kept in OHEM
        dice_weight : how much Dice contributes
        aux_weight  : aux branch weight
        ignore_index: pixel to ignore
    """
    def __init__(self, thresh=0.7, min_kept=100_000,
                 dice_weight=0.4, aux_weight=0.4, ignore_index=255):
        super().__init__()
        self.ohem        = OHEMLoss(thresh=thresh, min_kept=min_kept,
                                    aux_weight=0.0, ignore_index=ignore_index)
        self.dice        = DiceLoss(aux_weight=0.0, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.aux_weight  = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        loss = self.ohem(preds, target) + self.dice_weight * self.dice(preds, target)
        loss = loss + _aux_loss(preds, target, self.aux_weight, self.ignore_index)
        return loss


class LovaszCELoss(nn.Module):
    """
    Lovász + CE.   total = Lovász + weight * CE
    Best combination for maximising mIoU on Cityscapes.
    Lovász directly optimises mIoU; CE prevents training instability.

    Args:
        ce_weight   : how much CE contributes (0.5 default)
        aux_weight  : aux branch weight
        ignore_index: pixel to ignore
    """
    def __init__(self, ce_weight=0.5, aux_weight=0.4, ignore_index=255):
        super().__init__()
        self.lovasz      = LovaszLoss(aux_weight=0.0, ignore_index=ignore_index)
        self.ce          = CrossEntropyLoss(aux_weight=0.0, ignore_index=ignore_index)
        self.ce_weight   = ce_weight
        self.aux_weight  = aux_weight
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        loss = self.lovasz(preds, target) + self.ce_weight * self.ce(preds, target)
        loss = loss + _aux_loss(preds, target, self.aux_weight, self.ignore_index)
        return loss


# ══════════════════════════════════════════════════════════════════════
# FACTORY — use this in your config/code
# ══════════════════════════════════════════════════════════════════════

_REGISTRY = {
    # Single losses
    "ce"          : CrossEntropyLoss,
    "weighted_ce" : WeightedCrossEntropyLoss,
    "focal"       : FocalLoss,
    "ohem"        : OHEMLoss,
    "dice"        : DiceLoss,
    "iou"         : SoftIoULoss,
    "lovasz"      : LovaszLoss,
    # Compound losses
    "ce_dice"     : CEDiceLoss,
    "ce_iou"      : CEIoULoss,
    "focal_dice"  : FocalDiceLoss,
    "ohem_dice"   : OHEMDiceLoss,
    "lovasz_ce"   : LovaszCELoss,
    "wce_dice"    : WeightedCEDiceLoss
}


def build_loss(loss_type: str, ignore_index: int = 255, **kwargs) -> nn.Module:
    """
    Build a loss function by name.

    Usage in config yaml:
        loss:
          type: ce_dice
          kwargs:
            dice_weight: 0.5
            aux_weight: 0.4

    Usage in code:
        loss_fn = build_loss("ce_dice", dice_weight=0.5)
        loss_fn = build_loss("focal", gamma=1.5)
        loss_fn = build_loss("ohem_dice", thresh=0.6, dice_weight=0.3)
        loss_fn = build_loss("lovasz_ce", ce_weight=0.5)
    """
    kwargs.pop("kwargs", None)
    if loss_type not in _REGISTRY:
        raise ValueError(
            f"Unknown loss '{loss_type}'.\n"
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[loss_type](ignore_index=ignore_index, **kwargs)