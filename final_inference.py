"""
src/seg/final_inference.py
---------------------------
Production inference script with comprehensive metrics and visualization.

Features:
  - Loads trained model from checkpoint
  - Processes images from inference folder
  - Measures inference time
  - Visualizes predictions with side-by-side comparison
  - Saves detailed metrics

Usage:
    python final_inference --checkpoint outputs/checkpoints/exp1/exp1_epoch080_mIoU0.79.pth \
                                       --image_dir data/inference \
                                       --output_dir outputs/inference_newDataset \
                                       --config config/experiments/exp1.yaml
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.seg.config.configuration import get_config
from src.seg.constants import CONFIG_PATH, CITYSCAPES_PALETTE, CITYSCAPES_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from src.seg.models.deeplabv3_plus import get_segmentation_model
from src.seg.utils.checkpoint import load_checkpoint
from src.seg.utils.common import get_device


# ═══════════════════════════════════════════════════════════════════════
# IMAGE LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

def load_and_preprocess_image(
    image_path: Path,
    image_size: Tuple[int, int],
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
    """
    Load image, preprocess for model, keep original for visualization.
    
    Returns:
        tensor:       (1, 3, H, W) normalized tensor ready for model
        original_img: PIL Image (original size, for saving)
        original_size: (H, W) tuple of original dimensions
    """
    # Load original
    original_img = Image.open(image_path).convert("RGB")
    original_size = (original_img.height, original_img.width)  # (H, W)
    
    # Resize to model input size
    img_resized = original_img.resize((image_size[1], image_size[0]), Image.BILINEAR)
    
    # To numpy array
    img_np = np.array(img_resized, dtype=np.float32) / 255.0  # [0, 1]
    
    # Normalize with ImageNet stats
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img_np = (img_np - mean) / std
    
    # To tensor: (H, W, 3) → (3, H, W) → (1, 3, H, W)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    
    return tensor, original_img, original_size


# ═══════════════════════════════════════════════════════════════════════
# COLORIZATION & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert class indices (H, W) to RGB visualization (H, W, 3).
    """
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls_id, rgb in enumerate(CITYSCAPES_PALETTE):
        color[mask == cls_id] = rgb
    
    return color


def create_side_by_side_visualization(
    original_img: Image.Image,
    pred_mask: np.ndarray,
    save_path: Path,
):
    """
    Create visualization: Original | Prediction
    """
    # Resize pred_mask to match original size
    pred_colored = colorize_mask(pred_mask)
    pred_pil = Image.fromarray(pred_colored)
    pred_pil = pred_pil.resize(original_img.size, Image.NEAREST)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(pred_pil)
    axes[1].set_title("Segmentation Prediction", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight')
    plt.close(fig)


def create_class_distribution_plot(
    pred_mask: np.ndarray,
    save_path: Path,
):
    """
    Create bar chart showing pixel distribution per class.
    """
    unique, counts = np.unique(pred_mask, return_counts=True)
    total_pixels = pred_mask.size
    
    # Create dict of class -> percentage
    class_percentages = {}
    for cls_id, count in zip(unique, counts):
        if cls_id < len(CITYSCAPES_CLASSES):
            class_name = CITYSCAPES_CLASSES[cls_id]
            percentage = (count / total_pixels) * 100
            class_percentages[class_name] = percentage
    
    # Sort by percentage
    sorted_classes = sorted(class_percentages.items(), key=lambda x: x[1], reverse=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    classes, percentages = zip(*sorted_classes)
    
    colors = [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in 
              [CITYSCAPES_PALETTE[CITYSCAPES_CLASSES.index(cls)] for cls in classes]]
    
    ax.barh(classes, percentages, color=colors)
    ax.set_xlabel("Percentage of Image (%)", fontsize=12)
    ax.set_title("Class Distribution in Prediction", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add percentage labels
    for i, (cls, pct) in enumerate(sorted_classes):
        ax.text(pct + 0.5, i, f'{pct:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference_on_image(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    image_size: Tuple[int, int],
    output_dir: Path,
    save_visualizations: bool = True,
) -> dict:
    """
    Run inference on a single image.
    
    Returns:
        dict with keys: inference_time_ms, pred_mask, class_distribution
    """
    # Load and preprocess
    load_start = time.time()
    tensor, original_img, original_size = load_and_preprocess_image(image_path, image_size)
    load_time = (time.time() - load_start) * 1000
    
    # Move to device
    tensor = tensor.to(device)
    
    # Inference
    inference_start = time.time()
    output = model(tensor)
    inference_time = (time.time() - inference_start) * 1000
    
    # Extract prediction
    pred_logits = output["out"] if isinstance(output, dict) else output
    pred_mask = pred_logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
    
    # Post-process
    post_start = time.time()
    
    # Resize prediction to original image size
    pred_mask_resized = Image.fromarray(pred_mask.astype(np.uint8))
    pred_mask_resized = pred_mask_resized.resize(
        (original_size[1], original_size[0]),  # (W, H) for PIL
        Image.NEAREST
    )
    pred_mask_final = np.array(pred_mask_resized)
    
    # Class distribution
    unique, counts = np.unique(pred_mask_final, return_counts=True)
    class_dist = {
        CITYSCAPES_CLASSES[cls_id]: int(count)
        for cls_id, count in zip(unique, counts)
        if cls_id < len(CITYSCAPES_CLASSES)
    }
    
    post_time = (time.time() - post_start) * 1000
    
    # Save visualizations
    if save_visualizations:
        vis_path = output_dir / "visualizations" / f"{image_path.stem}_prediction.png"
        create_side_by_side_visualization(original_img, pred_mask_final, vis_path)
        
        dist_path = output_dir / "distributions" / f"{image_path.stem}_distribution.png"
        create_class_distribution_plot(pred_mask_final, dist_path)
        
        # Save colored mask
        mask_path = output_dir / "masks" / f"{image_path.stem}_mask.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        colored = colorize_mask(pred_mask_final)
        Image.fromarray(colored).save(str(mask_path))
    
    return {
        "image_name": image_path.name,
        "original_size": original_size,
        "model_input_size": image_size,
        "load_time_ms": load_time,
        "inference_time_ms": inference_time,
        "post_process_time_ms": post_time,
        "total_time_ms": load_time + inference_time + post_time,
        "class_distribution": class_dist,
        "pred_mask": pred_mask_final,
    }


def run_inference_on_folder(
    model: torch.nn.Module,
    image_dir: Path,
    device: torch.device,
    image_size: Tuple[int, int],
    output_dir: Path,
) -> List[dict]:
    """
    Run inference on all images in a folder.
    """
    # Find all images
    image_paths = sorted(list(image_dir.glob("*.png")) + 
                        list(image_dir.glob("*.jpg")) + 
                        list(image_dir.glob("*.jpeg")))
    
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    
    print(f"[Inference] Found {len(image_paths)} images")
    print(f"[Inference] Output directory: {output_dir}")
    
    results = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        result = run_inference_on_image(
            model, img_path, device, image_size, output_dir, save_visualizations=True
        )
        results.append(result)
    
    return results


# ═══════════════════════════════════════════════════════════════════════
# METRICS SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def print_summary_statistics(results: List[dict]):
    """
    Print comprehensive summary of inference results.
    """
    print("\n" + "="*80)
    print("INFERENCE SUMMARY")
    print("="*80)
    
    # Timing statistics
    total_images = len(results)
    load_times = [r["load_time_ms"] for r in results]
    inf_times = [r["inference_time_ms"] for r in results]
    post_times = [r["post_process_time_ms"] for r in results]
    total_times = [r["total_time_ms"] for r in results]
    
    print(f"\nProcessed: {total_images} images")
    print(f"\n{'Metric':<25} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    print(f"{'Load Time (ms)':<25} {np.mean(load_times):>10.2f} {np.min(load_times):>10.2f} {np.max(load_times):>10.2f}")
    print(f"{'Inference Time (ms)':<25} {np.mean(inf_times):>10.2f} {np.min(inf_times):>10.2f} {np.max(inf_times):>10.2f}")
    print(f"{'Post-process Time (ms)':<25} {np.mean(post_times):>10.2f} {np.min(post_times):>10.2f} {np.max(post_times):>10.2f}")
    print(f"{'Total Time (ms)':<25} {np.mean(total_times):>10.2f} {np.min(total_times):>10.2f} {np.max(total_times):>10.2f}")
    
    print(f"\nThroughput: {1000 / np.mean(inf_times):.2f} images/second")
    print(f"Total processing time: {sum(total_times) / 1000:.2f} seconds")
    
    # Class distribution across all images
    print("\n" + "-"*80)
    print("AGGREGATED CLASS DISTRIBUTION")
    print("-"*80)
    
    all_class_counts = {}
    for result in results:
        for cls_name, count in result["class_distribution"].items():
            all_class_counts[cls_name] = all_class_counts.get(cls_name, 0) + count
    
    total_pixels = sum(all_class_counts.values())
    sorted_classes = sorted(all_class_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Class':<20} {'Pixel Count':>15} {'Percentage':>12}")
    print("-" * 50)
    for cls_name, count in sorted_classes[:10]:  # Top 10 classes
        percentage = (count / total_pixels) * 100
        print(f"{cls_name:<20} {count:>15,} {percentage:>11.2f}%")
    
    print("\n" + "="*80)


def save_results_to_file(results: List[dict], output_dir: Path):
    """
    Save results to text file for later reference.
    """
    report_path = output_dir / "inference_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED INFERENCE REPORT\n")
        f.write("="*80 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"\n{'─'*80}\n")
            f.write(f"Image {i}: {result['image_name']}\n")
            f.write(f"{'─'*80}\n")
            f.write(f"Original size: {result['original_size']}\n")
            f.write(f"Model input size: {result['model_input_size']}\n")
            f.write(f"Load time: {result['load_time_ms']:.2f} ms\n")
            f.write(f"Inference time: {result['inference_time_ms']:.2f} ms\n")
            f.write(f"Post-process time: {result['post_process_time_ms']:.2f} ms\n")
            f.write(f"Total time: {result['total_time_ms']:.2f} ms\n")
            f.write(f"\nClass distribution:\n")
            for cls_name, count in sorted(result['class_distribution'].items(), 
                                         key=lambda x: x[1], reverse=True):
                f.write(f"  {cls_name:<20} {count:>10,} pixels\n")
    
    print(f"\n[Report] Saved detailed report to: {report_path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main(args):
    """Main inference pipeline."""
    
    # Load config
    cfg = get_config(CONFIG_PATH, args.config)
    device = get_device()
    
    print(f"\n{'='*80}")
    print(f"FINAL INFERENCE")
    print(f"{'='*80}\n")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}\n")
    
    # Build model
    print("[Model] Building DeepLabV3+...")
    model = get_segmentation_model(
        num_classes=cfg.data.num_classes,
        backbone=cfg.model.backbone,
        output_stride=cfg.model.output_stride,
        aux=False,  # No aux head for inference
        use_pretrained_backbone=False,
        backbone_weights_path=None,
    )
    
    # Load checkpoint
    print(f"[Model] Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(args.checkpoint, model, device=str(device))
    model = model.to(device)
    model.eval()
    
    print("[Model] ✓ Ready for inference\n")
    
    # Run inference
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    
    results = run_inference_on_folder(
        model=model,
        image_dir=image_dir,
        device=device,
        image_size=tuple(cfg.data.image_size),
        output_dir=output_dir,
    )
    
    # Print summary
    print_summary_statistics(results)
    
    # Save report
    save_results_to_file(results, output_dir)
    
    print(f"\n✓ Inference complete. Results saved to: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on images from different datasets."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/inference",
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/inference_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config yaml",
    )
    
    args = parser.parse_args()
    main(args)