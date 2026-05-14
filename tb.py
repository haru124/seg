"""
src/seg/tracking/tensorboard_logger.py
----------------------------------------
Thin wrapper around PyTorch's SummaryWriter for TensorBoard logging.

Usage:
    tb = TensorboardLogger(log_dir="outputs/tensorboard", exp_id="exp01")
    tb.log_scalar("Loss/train", 0.45, step=1)
    tb.log_scalars({"mIoU": 0.43, "fw_iou": 0.51}, step=1, prefix="Metrics")
    tb.log_image("Predictions/batch_0", tensor_image, step=1)
    tb.close()

View in browser:
    tensorboard --logdir outputs/tensorboard
"""

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    """
    Wraps SummaryWriter with helper methods.

    Args:
        log_dir : base directory for all TensorBoard logs
        exp_id  : experiment ID; logs go into log_dir/exp_id/
    """

    def __init__(self, log_dir: str, exp_id: str):
        run_dir = Path(log_dir) / exp_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(run_dir))
        print(f"[TensorBoard] Logging to: {run_dir}")
        print(f"[TensorBoard] View with:  tensorboard --logdir {log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: dict, step: int, prefix: str = ""):
        """
        Log multiple scalars at once.

        Args:
            metrics : {name: value} dict
            step    : epoch / global step
            prefix  : optional tag prefix, e.g. "Metrics" → "Metrics/mIoU"
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                tag = f"{prefix}/{name}" if prefix else name
                self.writer.add_scalar(tag, value, step)

    def log_image(self, tag: str, image_tensor, step: int):
        """
        Log an image tensor.

        Args:
            image_tensor : (3, H, W) or (H, W) float tensor in [0, 1]
        """
        self.writer.add_image(tag, image_tensor, step)

    def log_figure(self, tag: str, figure, step: int):
        """
        Log a matplotlib figure directly to TensorBoard.

        Args:
            figure : matplotlib.figure.Figure object
        """
        self.writer.add_figure(tag, figure, step)

    def close(self):
        """Flush and close the SummaryWriter. Call at end of training."""
        self.writer.flush()
        self.writer.close()
