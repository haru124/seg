# src/seg/evaluation/metrics.py
# Based on core/utils/score.py from Tramac/awesome-semantic-segmentation-pytorch
import numpy as np
import torch


class SegmentationMetrics:
    """Computes mIoU and pixel accuracy from confusion matrix."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @staticmethod
    def _fast_hist(label_pred, label_true, num_classes):
        mask = (label_true >= 0) & (label_true < num_classes)
        hist = np.bincount(
            num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)
        return hist

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds:   (B, H, W) long — argmax already applied
        targets: (B, H, W) long
        """
        preds_np = preds.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()

        # mask ignore
        valid = targets_np != self.ignore_index
        preds_np = preds_np[valid]
        targets_np = targets_np[valid]

        self.confusion_matrix += self._fast_hist(preds_np, targets_np, self.num_classes)

    def compute(self) -> dict:
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()                      # pixel accuracy
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)                               # mean class accuracy

        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iu)

        freq = hist.sum(axis=1) / hist.sum()
        fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()             # frequency-weighted IoU

        return {
            "pixAcc": float(acc),
            "mclsAcc": float(acc_cls),
            "mIoU": float(mean_iou),
            "fwIoU": float(fw_iou),
            "class_iou": iu.tolist(),
        }
