"""
compute_class_weights.py

Run ONCE before training to compute per-class pixel weights for
weighted cross-entropy loss.

Usage:
    python compute_class_weights.py --gt_dir /path/to/cityscapes/gtFine/train

Output:
    Prints weights — copy the list into your config yaml under loss.class_weights
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Cityscapes 19 train-id classes
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic_light", "traffic_sign", "vegetation",
    "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]

# ─── Tier thresholds ──────────────────────────────────────────────────────────
# Based on percentage of ALL pixels (including ignore=255)
TIER_1_THRESH = 20.0   # >= 20%  → weight 1.0  (dominant: road, vegetation, sky)
TIER_2_THRESH = 1.0    # 1–20%   → weight 2.0  (moderate: sidewalk, building, car)
                        # < 1%    → weight 3.0  (rare: rider, truck, bus, train, motorcycle)


def compute_weights_tier(pixel_counts: dict, total_pixels: int, num_classes: int = 19) -> np.ndarray:
    """
    Scan all *_gtFine_labelTrainIds.png files in gt_dir (recursively).
    Count pixels per class, including ignore (255) in the denominator
    so rare classes get the full benefit of their true scarcity.
    """
    # ── Percentage of each class out of ALL pixels ────────────────────────
    percentages = pixel_counts / total_pixels * 100.0

    # ── Tiered weights ────────────────────────────────────────────────────
    weights = np.ones(num_classes, dtype=np.float32)
    for c in range(num_classes):
        p = percentages[c]
        if p >= TIER_1_THRESH:
            weights[c] = 1.0
        elif p >= TIER_2_THRESH:
            weights[c] = 2.0
        else:
            weights[c] = 3.0           # < 1% of all pixels

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'Class':<20} {'Pixels':>15} {'%Total':>9} {'Weight':>8} {'Tier':>6}")
    print("─" * 62)
    for c in range(num_classes):
        tier = "T1" if weights[c] == 1.0 else ("T2" if weights[c] == 2.0 else "T3")
        print(
            f"{CLASS_NAMES[c]:<20} "
            f"{pixel_counts[c]:>15,} "
            f"{percentages[c]:>8.3f}% "
            f"{weights[c]:>8.1f} "
            f"{tier:>6}"
        )
    ignore_px = total_pixels - int(pixel_counts.sum())
    print(f"\n{'ignore(255)':<20} {ignore_px:>15,} {ignore_px/total_pixels*100:>8.3f}%")
    print(f"{'TOTAL':<20} {total_pixels:>15,}")

    print("\n" + "=" * 62)
    print("Copy this into your config yaml under loss.class_weights:")
    print(f"class_weights: {weights.tolist()}")

    return weights


"""
Method: Inverse-frequency weighting, median-frequency normalized, with clipping.

Why median-frequency normalization?
    - Raw inverse frequency: rare classes get weights like 50–200x — unstable gradients.
    - Divide by median frequency → weights are centered around 1.0 for typical classes,
      rare classes get 3–8x, dominant classes get ~0.3–0.5x.
    - This is the standard method from Eigen & Fergus (2015) and used in many
      Cityscapes baselines.

Clipping at max_weight:
    - Even after median normalization, extreme outliers (train, rider) can reach 10–15x.
    - Clipping at 8.0 keeps gradients stable without losing the imbalance correction.

Formula:
    freq_c     = pixels_c / total_valid_pixels  (only labeled pixels, not ignore=255)
    median_f   = median(freq_c for all c)
    weight_c   = clip(median_f / freq_c, min=0.1, max=8.0)

"""
MAX_WEIGHT = 8.0   # clip ceiling — prevents unstable gradients for ultra-rare classes
MIN_WEIGHT = 0.1   # floor — dominant classes still contribute


def compute_weights(pixel_counts: dict, total_valid: int, num_classes: int = 19) -> np.ndarray:
    # ── Frequencies (only over valid/labeled pixels) ──────────────────────
    frequencies_valid = pixel_counts / total_valid  # shape: [num_classes]

    # ── Median-frequency normalization ────────────────────────────────────
    median_freq = np.median(frequencies_valid)
    raw_weights = median_freq / (frequencies_valid + 1e-10)

    # ── Clip to stable range ──────────────────────────────────────────────
    weights = np.clip(raw_weights, MIN_WEIGHT, MAX_WEIGHT).astype(np.float32)

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'Class':<20} {'Pixels':>15} {'Freq%':>9} {'RawW':>8} {'ClipW':>8}")
    print("─" * 65)
    for c in range(num_classes):
        clipped = "✂" if raw_weights[c] > MAX_WEIGHT or raw_weights[c] < MIN_WEIGHT else " "
        print(
            f"{CLASS_NAMES[c]:<20} "
            f"{pixel_counts[c]:>15,} "
            f"{frequencies_valid[c]*100:>8.3f}% "
            f"{raw_weights[c]:>8.3f} "
            f"{weights[c]:>8.3f} {clipped}"
        )
    print(f"\nMedian frequency : {median_freq*100:.3f}%")
    print(f"Weight range     : [{weights.min():.3f}, {weights.max():.3f}]")
    print("\n" + "=" * 65)
    print("Copy into your config yaml under loss.class_weights:")
    print(f"class_weights: {weights.tolist()}")
    return weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Cityscapes class weights")
    parser.add_argument(
        "--gt_dir",
        #required=True,
        help="Path to gtFine/train directory",
        default = "data/gtFine/train",
    )
    parser.add_argument("--num_classes", type=int, default=19)
    args = parser.parse_args()

    num_classes = 19
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    total_valid_pixels  = 0   # only labeled pixels (NOT ignore=255)
    total_pixels = 0
    gt_paths = sorted(Path(args.gt_dir).rglob("*_gtFine_labelTrainIds.png"))
    if not gt_paths:
        raise FileNotFoundError(
            f"No labelTrainIds files found in {args.gt_dir}. "
            "Check the path points to gtFine/train."
        )
    print(f"Found {len(gt_paths)} annotation files in {args.gt_dir}")

    for gt_path in tqdm(gt_paths, desc="Scanning pixels"):
        gt = np.array(Image.open(gt_path), dtype=np.int32)
        total_pixels += gt.size
        valid_mask = gt != 255
        total_valid_pixels += int(valid_mask.sum())

        for c in range(num_classes):
            pixel_counts[c] += int(np.sum(gt == c))

    weights_tier = compute_weights_tier(pixel_counts, total_pixels, args.num_classes)
    weights_mdif = compute_weights(pixel_counts, total_valid_pixels, args.num_classes)