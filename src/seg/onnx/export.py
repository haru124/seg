"""
src/seg/onnx/export.py
======================
Export trained DeepLabV3+ model to ONNX format.

ONNX (Open Neural Network Exchange):
- Universal model format (PyTorch → ONNX → TensorFlow, TensorRT, ONNX Runtime, etc.)
- Enables inference on any platform with ONNX Runtime
- CPU-optimized inference (faster than PyTorch on CPU)
- Model compression & quantization tools available
- Hardware acceleration support (GPU, TPU, mobile)

Why ONNX?
1. Inference speed: Often 2-5x faster on CPU than PyTorch
2. Portability: Single model file works everywhere
3. Deployment: No need to ship entire PyTorch library (smaller, lighter)
4. Production: Industry standard for model serving

Usage:
    python onnx_benchmark.py --exp_config config/experiments/exp_01.yaml
"""

import torch
import onnx
import onnxruntime as ort
from pathlib import Path
import numpy as np

from src.seg.utils.checkpoint import load_checkpoint
from src.seg.utils.common import count_parameters


def export_to_onnx(
    model,
    checkpoint_path: str,
    output_dir: str,
    input_shape: tuple = (1, 3, 512, 1024),
    opset_version: int = 17,
    use_external_data_format: bool = False,
    device: str = "cpu",
) -> str:
    """
    Export PyTorch model to ONNX format.

    Args:
        model              : PyTorch model (nn.Module)
        checkpoint_path    : path to saved checkpoint
        output_dir         : where to save .onnx file
        input_shape        : (batch, channels, height, width)
        opset_version      : ONNX opset (17 = recent, good compatibility)
        use_external_data_format : for models > 2GB
        device             : "cpu" or "cuda"

    Returns:
        Path to exported .onnx file
    """
    # ── Load checkpoint ──
    print("[ONNX] Loading checkpoint...")
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()

    # ── Create dummy input ──
    dummy_input = torch.randn(input_shape, device=device)

    # ── Export ──
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "deeplabv3plus.onnx"

    print(f"[ONNX] Exporting to {onnx_path}...")
    print(f"       Input shape: {input_shape}")
    print(f"       Opset version: {opset_version}")

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
        export_params=True,
        verbose=False,
    )

    print(f"[ONNX] ✓ Exported successfully to {onnx_path}")

    # ── Verify ONNX model ──
    print("[ONNX] Verifying model...")
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("[ONNX] ✓ Model is valid")
    except Exception as e:
        print(f"[ONNX] ✗ Verification failed: {e}")
        return None

    # ── Print model info ──
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"[ONNX] File size: {file_size_mb:.2f} MB")

    return str(onnx_path)


def get_onnx_model_info(onnx_path: str) -> dict:
    """
    Load ONNX model and print info.
    """
    onnx_model = onnx.load(onnx_path)

    # Get input/output info
    graph = onnx_model.graph
    inputs = graph.input
    outputs = graph.output

    info = {
        "inputs": [(inp.name, [d.dim_value for d in inp.type.tensor_type.shape.dim]) for inp in inputs],
        "outputs": [(out.name, [d.dim_value for d in out.type.tensor_type.shape.dim]) for out in outputs],
        "num_nodes": len(graph.node),
    }

    print("\n[ONNX Model Info]")
    print(f"  Inputs:     {info['inputs']}")
    print(f"  Outputs:    {info['outputs']}")
    print(f"  Num nodes:  {info['num_nodes']}")

    return info