"""
src/seg/datasets/dataloader.py
--------------------------------
Builds PyTorch DataLoader objects for train / val / test splits.

Usage:
    from src.seg.datasets.dataloader import build_dataloader
    train_loader = build_dataloader(cfg.data, split="train")
    val_loader   = build_dataloader(cfg.data, split="val")
"""

from torch.utils.data import DataLoader

from src.seg.entity.config_entity import DataConfig
from src.seg.datasets.cityscapes_dataset import CityscapesDataset
from src.seg.datasets.transforms import get_train_transforms, get_val_transforms


def build_dataloader(cfg: DataConfig, split: str) -> DataLoader:
    """
    Build a DataLoader for the requested split.

    Args:
        cfg   : DataConfig — pulled from ExperimentConfig.data
        split : "train" | "val" | "test"

    Returns:
        torch.utils.data.DataLoader
    """
    assert split in ("train", "val", "test"), \
        f"split must be train/val/test, got '{split}'"

    image_size = tuple(cfg.image_size)   # convert list → tuple (H, W)

    # Select the appropriate transform pipeline
    if split == "train":
        transforms = get_train_transforms(image_size)
    else:
        # val and test use the same deterministic pipeline
        transforms = get_val_transforms(image_size)

    dataset = CityscapesDataset(
        root       = cfg.root,
        split      = split,
        transforms = transforms,
    )

    # shuffle=True only for training; deterministic order for val/test
    shuffle = (split == "train")

    # drop_last=True for training keeps batch sizes consistent,
    # which matters for BatchNorm stability
    drop_last = (split == "train")

    loader = DataLoader(
        dataset,
        batch_size  = cfg.batch_size,
        shuffle     = shuffle,
        num_workers = cfg.num_workers,
        pin_memory  = True,    # faster GPU transfer
        drop_last   = drop_last,
    )

    print(f"[DataLoader] {split}: {len(dataset)} samples, "
          f"{len(loader)} batches (batch_size={cfg.batch_size})")

    return loader
