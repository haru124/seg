"""
src/seg/utils/common.py
------------------------
Shared utility functions used across the project.

Functions:
    set_seed         — reproducible random state
    get_device       — auto-select GPU or CPU with info print
    setup_logger     — file + console logger
    count_parameters — model parameter count string
"""

import random
import logging
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


def set_seed(seed: int = 42):
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU + all GPUs).
    Also disables non-deterministic cuDNN ops for full reproducibility.

    Note: deterministic mode can slow training slightly.
    If speed is critical and you don't need exact reproducibility,
    set torch.backends.cudnn.benchmark = True instead.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Auto-select the best available device and print GPU info.

    Returns:
        torch.device("cuda") if a GPU is available, else torch.device("cpu")
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Device] GPU: {name} | VRAM: {mem_gb:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[Device] CPU — no CUDA GPU found")
    return device


def setup_logger(name: str, log_dir: str, exp_id: str) -> logging.Logger:
    """
    Create a logger that writes to both a timestamped log file and stdout.

    Args:
        name    : logger name (e.g. "trainer")
        log_dir : directory where log files are saved
        exp_id  : experiment ID used in the filename

    Returns:
        logging.Logger
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir_path / f"{exp_id}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if setup_logger is called more than once
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(str(log_file))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger


def count_parameters(model: torch.nn.Module) -> str:
    """
    Return a human-readable string with total and trainable parameter counts.

    Example output:
        "Total: 39.64M | Trainable: 39.64M"
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total / 1e6:.2f}M | Trainable: {trainable / 1e6:.2f}M"
