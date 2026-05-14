"""
src/seg/datasets/transforms.py
--------------------------------
Joint augmentation transforms that operate on (PIL image, PIL mask) pairs.

All spatial transforms must be applied identically to both image and mask.
Colour / normalisation transforms are applied only to the image.

Provided transforms:
    JointResize          — resize both to a fixed (H, W)
    JointRandomCrop      — random crop to (H, W)
    JointRandomHFlip     — random horizontal flip
    JointRandomScale     — random scale within a range, then crop
    JointColorJitter     — colour jitter on image only
    JointToTensor        — PIL → float tensor (image) / long tensor (mask)
    JointNormalize       — ImageNet normalisation on image only
    JointCompose         — chains a list of joint transforms

Typical train pipeline:
    transforms = JointCompose([
        JointRandomScale(scale_range=(0.5, 2.0), crop_size=(512, 1024)),
        JointRandomHFlip(p=0.5),
        JointColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        JointToTensor(),
        JointNormalize(),
    ])

Typical val/test pipeline:
    transforms = JointCompose([
        JointResize((512, 1024)),
        JointToTensor(),
        JointNormalize(),
    ])
"""

import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from src.ods.constants import IMAGENET_MEAN, IMAGENET_STD, IGNORE_INDEX


class JointCompose:
    """Chain multiple joint transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class JointResize:
    """
    Resize both image and mask to (height, width).
    Image uses bilinear interpolation; mask uses nearest (no interpolation
    between class IDs).
    """

    def __init__(self, size: tuple):
        # size = (H, W)
        self.size = size  # (H, W)

    def __call__(self, image, mask):
        h, w = self.size
        image = TF.resize(image, (h, w), interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  (h, w), interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


class JointRandomHFlip:
    """Randomly flip both image and mask horizontally."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        return image, mask


class JointRandomScale:
    """
    Randomly scale the image (and mask) by a factor in scale_range,
    then take a random crop of crop_size.

    This is the standard DeepLabV3+ training augmentation.
    If the scaled image is smaller than crop_size, it is padded:
      - image : padded with ImageNet mean (neutral colour)
      - mask  : padded with IGNORE_INDEX
    """

    def __init__(self, scale_range=(0.5, 2.0), crop_size=(512, 1024)):
        self.scale_range = scale_range
        self.crop_size   = crop_size  # (H, W)

    def __call__(self, image, mask):
        scale = random.uniform(*self.scale_range)
        w, h  = image.size
        new_w = int(w * scale)
        new_h = int(h * scale)

        image = TF.resize(image, (new_h, new_w), TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  (new_h, new_w), TF.InterpolationMode.NEAREST)

        crop_h, crop_w = self.crop_size

        # Pad if needed
        pad_h = max(crop_h - new_h, 0)
        pad_w = max(crop_w - new_w, 0)
        if pad_h > 0 or pad_w > 0:
            # torchvision pads as (left, top, right, bottom)
            image = TF.pad(image, (0, 0, pad_w, pad_h),
                           fill=tuple(int(m * 255) for m in IMAGENET_MEAN))
            mask  = TF.pad(mask,  (0, 0, pad_w, pad_h), fill=IGNORE_INDEX)

        # Random crop
        i, j, h, w = torch.randint(0, max(image.size[1] - crop_h + 1, 1), (1,)).item(), \
                     torch.randint(0, max(image.size[0] - crop_w + 1, 1), (1,)).item(), \
                     crop_h, crop_w
        image = TF.crop(image, i, j, h, w)
        mask  = TF.crop(mask,  i, j, h, w)

        return image, mask


class JointRandomCrop:
    """
    Simple random crop to crop_size without scaling.
    Pads if image is smaller.
    """

    def __init__(self, crop_size=(512, 1024)):
        self.crop_size = crop_size  # (H, W)

    def __call__(self, image, mask):
        crop_h, crop_w = self.crop_size
        w, h = image.size

        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, (0, 0, pad_w, pad_h),
                           fill=tuple(int(m * 255) for m in IMAGENET_MEAN))
            mask  = TF.pad(mask,  (0, 0, pad_w, pad_h), fill=IGNORE_INDEX)
            w, h = image.size

        # Pick a random top-left corner
        top  = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        image = TF.crop(image, top, left, crop_h, crop_w)
        mask  = TF.crop(mask,  top, left, crop_h, crop_w)

        return image, mask


class JointColorJitter:
    """
    Apply random colour jitter to the image only.
    Mask is returned unchanged.
    """

    def __init__(self, brightness=0.3, contrast=0.3,
                 saturation=0.3, hue=0.1):
        from torchvision.transforms import ColorJitter
        self.jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image, mask):
        image = self.jitter(image)
        return image, mask


class JointToTensor:
    """
    Convert PIL image → float32 tensor (C, H, W) in [0, 1].
    Convert PIL mask  → int64  tensor (H, W).
    """

    def __call__(self, image, mask):
        image = TF.to_tensor(image)                                # float32 [0,1]
        mask  = torch.from_numpy(np.array(mask, dtype=np.int64))  # int64
        return image, mask


class JointNormalize:
    """
    Normalise image tensor with ImageNet mean/std.
    Mask is unchanged.
    """

    def __init__(self,
                 mean=IMAGENET_MEAN,
                 std=IMAGENET_STD):
        self.mean = mean
        self.std  = std

    def __call__(self, image, mask):
        image = TF.normalize(image, self.mean, self.std)
        return image, mask


# ── Convenience builders ──────────────────────────────────────────────

def get_train_transforms(image_size: tuple) -> JointCompose:
    """
    Standard training augmentation pipeline for DeepLabV3+ on Cityscapes.
    image_size: (H, W)
    """
    return JointCompose([
        JointRandomScale(scale_range=(0.5, 2.0), crop_size=image_size),
        JointRandomHFlip(p=0.5),
        JointColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        JointToTensor(),
        JointNormalize(),
    ])


def get_val_transforms(image_size: tuple) -> JointCompose:
    """
    Validation / test pipeline — just resize and normalise, no augmentation.
    image_size: (H, W)
    """
    return JointCompose([
        JointResize(image_size),
        JointToTensor(),
        JointNormalize(),
    ])
