# src/ods/tracking/tensorboard_logger.py
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


class TensorboardLogger:
    def __init__(self, log_dir: str, exp_id: str):
        run_dir = Path(log_dir) / exp_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(run_dir))
        print(f"[TensorBoard] Logging to: {run_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: dict, step: int, prefix: str = ""):
        for k, v in metrics.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.writer.add_scalar(tag, v, step)

    def log_image(self, tag: str, image_tensor, step: int):
        """image_tensor: (3, H, W) or (H, W) float [0,1]"""
        self.writer.add_image(tag, image_tensor, step)

    def log_figure(self, tag: str, figure, step: int):
        self.writer.add_figure(tag, figure, step)

    def close(self):
        self.writer.close()