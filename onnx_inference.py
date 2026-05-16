"""
src/ods/onnx/inference.py
=========================
Run inference using ONNX Runtime (much faster on CPU).

Key differences:
- PyTorch inference: Loads full framework (~1 GB memory)
- ONNX inference: Just the model (~100-300 MB)
- ONNX on CPU: Often 2-5x faster than PyTorch CPU
- ONNX on GPU: Same speed as PyTorch, smaller file size

This module provides CPU inference optimized for production.
"""

import onnxruntime as ort
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
import time

from src.ods.constants import IMAGENET_MEAN, IMAGENET_STD, CITYSCAPES_PALETTE


class ONNXInference:
    """ONNX Runtime inference wrapper."""

    def __init__(self, onnx_path: str, providers: list = None):
        """
        Initialize ONNX inference session.

        Args:
            onnx_path : path to .onnx model file
            providers : ["CPUExecutionProvider"] or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        """
        if providers is None:
            providers = ["CPUExecutionProvider"]

        print(f"[ONNX Runtime] Loading model from {onnx_path}")
        print(f"[ONNX Runtime] Providers: {providers}")

        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers,
            sess_options=self._get_session_options(),
        )
        self.onnx_path = onnx_path

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        print(f"[ONNX Runtime] Input name: {self.input_name}")
        print(f"[ONNX Runtime] Input shape: {self.input_shape}")
        print(f"[ONNX Runtime] Output name: {self.output_name}")

    def _get_session_options(self):
        """Configure session for CPU inference."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # CPU threads
        return sess_options

    def preprocess(self, image_path: str, image_size: tuple = (512, 1024)) -> np.ndarray:
        """
        Load and preprocess image for ONNX inference.

        Args:
            image_path : path to image
            image_size : (height, width)

        Returns:
            Preprocessed image as numpy array (1, 3, H, W)
        """
        # ── Load image ──
        img = Image.open(image_path).convert("RGB")
        img = np.array(img, dtype=np.float32)

        # ── Resize ──
        h, w = image_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── Normalize ──
        img = img / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        img = (img - mean) / std

        # ── To tensor format (C, H, W) → (1, C, H, W) ──
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img.astype(np.float32)

    def predict(self, image_input: np.ndarray) -> np.ndarray:
        """
        Run inference.

        Args:
            image_input : preprocessed image (1, 3, H, W) as numpy array

        Returns:
            Logits (1, 19, H, W)
        """
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: image_input},
        )
        return outputs[0]  # (1, 19, H, W)

    def postprocess(self, logits: np.ndarray) -> np.ndarray:
        """
        Convert logits to class indices.

        Args:
            logits : (1, 19, H, W)

        Returns:
            Mask (H, W) with class indices 0-18
        """
        mask = np.argmax(logits[0], axis=0)  # (H, W)
        return mask.astype(np.uint8)


def colorize_mask_onnx(mask: np.ndarray) -> np.ndarray:
    """Convert class indices to RGB using Cityscapes palette."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_idx in range(19):
        r, g, b = CITYSCAPES_PALETTE[cls_idx]
        rgb[mask == cls_idx] = [b, g, r]  # OpenCV uses BGR

    return rgb


def run_onnx_inference_single(
    onnx_path: str,
    image_path: str,
    output_dir: str = "outputs/onnx_inference",
    image_size: tuple = (512, 1024),
    use_gpu: bool = False,
) -> tuple:
    """
    Run inference on a single image using ONNX.

    Args:
        onnx_path  : path to .onnx model
        image_path : path to input image
        output_dir : where to save output
        image_size : (height, width)
        use_gpu    : use GPU if available

    Returns:
        (prediction_mask, inference_time_ms, file_size_mb)
    """
    # ── Setup providers ──
    providers = []
    if use_gpu:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    # ── Initialize ──
    onnx_inf = ONNXInference(onnx_path, providers=providers)

    # ── Get model file size ──
    model_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)

    # ── Preprocess ──
    print(f"\n[ONNX Inference] Loading image: {image_path}")
    image_input = onnx_inf.preprocess(image_path, image_size)

    # ── Inference (with timing) ──
    print("[ONNX Inference] Running inference...")
    start = time.time()
    logits = onnx_inf.predict(image_input)
    inference_time = (time.time() - start) * 1000  # ms

    # ── Postprocess ──
    mask = onnx_inf.postprocess(logits)

    # ── Save ──
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_rgb = colorize_mask_onnx(mask)
    output_path = output_dir / f"onnx_{Path(image_path).stem}_pred.png"
    cv2.imwrite(str(output_path), mask_rgb)

    print(f"[ONNX Inference] ✓ Saved to {output_path}")
    print(f"[ONNX Inference] Inference time: {inference_time:.2f} ms")
    print(f"[ONNX Inference] Model size: {model_size_mb:.2f} MB")

    return mask, inference_time, model_size_mb


def run_onnx_inference_folder(
    onnx_path: str,
    image_dir: str,
    output_dir: str = "outputs/onnx_inference",
    image_size: tuple = (512, 1024),
    use_gpu: bool = False,
    max_images: int = None,
) -> list:
    """
    Batch inference on folder using ONNX.

    Args:
        onnx_path  : path to .onnx model
        image_dir  : folder with images
        output_dir : where to save outputs
        image_size : (height, width)
        use_gpu    : use GPU if available
        max_images : limit number of images (for testing)

    Returns:
        List of (image_path, mask, inference_time_ms)
    """
    image_dir = Path(image_dir)
    image_paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))

    if max_images:
        image_paths = image_paths[:max_images]

    print(f"\n[ONNX Inference] Found {len(image_paths)} images")

    results = []
    total_time = 0

    for idx, img_path in enumerate(image_paths):
        print(f"\n[{idx + 1}/{len(image_paths)}] {img_path.name}")

        mask, inf_time, model_size = run_onnx_inference_single(
            onnx_path,
            str(img_path),
            output_dir,
            image_size,
            use_gpu,
        )
        results.append((str(img_path), mask, inf_time))
        total_time += inf_time

    print(f"\n[ONNX Inference Summary]")
    print(f"  Total images: {len(results)}")
    print(f"  Total time: {total_time:.2f} ms")
    print(f"  Average per image: {total_time / len(results):.2f} ms")

    return results
