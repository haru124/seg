"""
src/seg/datasets/cityscapes_dataset.py
---------------------------------------
PyTorch Dataset for the Cityscapes semantic segmentation benchmark.

Expected folder layout (set via data.root in config):
    data/
      images/
        train/  <city>/  *.png
        val/    <city>/  *.png
        test/   <city>/  *.png
      gtFine/
        train/  <city>/  *_gtFine_labelIds.png
        val/    <city>/  *_gtFine_labelIds.png
        test/   <city>/  *_gtFine_labelIds.png   (no labels — inference only)

The dataset converts raw Cityscapes label IDs (0–33 + misc) to the
19 training IDs used by all benchmarks.
Pixels that don't correspond to a training class are set to 255
(the ignore_index).
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

from src.ods.constants import LABEL_ID_TO_TRAIN_ID, IGNORE_INDEX


class CityscapesDataset(Dataset):
    """
    Cityscapes segmentation dataset.

    Args:
        root      : path to the data/ folder
        split     : "train" | "val" | "test"
        transforms: callable that takes (PIL image, PIL mask) and returns
                    (tensor image, tensor mask).  Pass None to get raw PIL.
    """

    def __init__(self, root: str, split: str = "train", transforms=None):
        assert split in ("train", "val", "test"), \
            f"split must be train/val/test, got '{split}'"

        self.root       = Path(root)
        self.split      = split
        self.transforms = transforms

        # Build a lookup array: raw_label_id → train_id
        # Array index = raw label ID, value = train ID (or 255 = ignore)
        self._label_map = self._build_label_map()

        # Discover all image/mask pairs
        self.image_paths, self.mask_paths = self._find_pairs()

        print(f"[Dataset] Cityscapes {split}: {len(self.image_paths)} images")

    # ── label ID remapping ─────────────────────────────────────────────

    def _build_label_map(self) -> np.ndarray:
        """
        Build a 256-element lookup array.
        label_map[raw_id] = train_id   (or 255 for ignored classes)
        """
        label_map = np.full(256, IGNORE_INDEX, dtype=np.uint8)
        for raw_id, train_id in LABEL_ID_TO_TRAIN_ID.items():
            label_map[raw_id] = train_id
        return label_map

    # ── file discovery ─────────────────────────────────────────────────

    def _find_pairs(self):
        """
        Walk the images/<split>/ tree and pair each image with its
        corresponding gtFine label file.

        Image filename example:
            aachen_000000_000019_leftImg8bit.png
        Corresponding mask:
            aachen_000000_000019_gtFine_labelIds.png
        """
        img_dir  = self.root / "images"  / self.split
        mask_dir = self.root / "gtFine"  / self.split

        image_paths = []
        mask_paths  = []

        # Images live inside city sub-folders
        for city_dir in sorted(img_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                # Derive the mask filename from the image filename
                stem = img_path.stem.replace("_leftImg8bit", "")
                mask_name = f"{stem}_gtFine_labelIds.png"
                mask_path = mask_dir / city_dir.name / mask_name

                if self.split == "test":
                    # Test set has no ground-truth labels
                    image_paths.append(img_path)
                    mask_paths.append(None)
                elif mask_path.exists():
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
                else:
                    # Log missing masks instead of crashing
                    print(f"[Dataset] WARNING: mask not found for {img_path.name}")

        if len(image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {img_dir}. "
                "Check that data/ is populated and paths are correct."
            )

        return image_paths, mask_paths

    # ── core Dataset interface ─────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # Load and remap mask (test set has no mask)
        if self.mask_paths[idx] is not None:
            raw_mask = np.array(Image.open(self.mask_paths[idx]), dtype=np.uint8)
            # Vectorised ID remapping using the lookup array
            train_mask = self._label_map[raw_mask]
            mask = Image.fromarray(train_mask)
        else:
            # Test set: return a dummy all-ignore mask
            w, h = image.size
            mask = Image.fromarray(
                np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
            )

        # Apply joint augmentation / normalisation transforms
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask

    # ── convenience ───────────────────────────────────────────────────

    def get_image_path(self, idx: int) -> str:
        """Return the original file path for an index (useful for saving predictions)."""
        return str(self.image_paths[idx])
