"""
src/seg/onnx/export.py
======================
Experiment-aware ONNX exporter for DeepLabV3+

Saves to:
    outputs/onnx_checkpoints/<exp_id>/deeplabv3plus.onnx
"""

import torch
import onnx
from pathlib import Path
from src.seg.utils.checkpoint import load_checkpoint


def find_best_checkpoint(ckpt_dir: str) -> str:
    """
    Pick the best checkpoint from a directory.

    Strategy (in order):
    1. File with highest mIoU parsed from filename  (e.g. exp1_epoch020_mIoU0.4231.pth)
    2. Fallback: most recently modified .pth file
    """
    ckpt_dir = Path(ckpt_dir)
    ckpts = list(ckpt_dir.glob("*.pth"))

    if not ckpts:
        raise FileNotFoundError(
            f"No .pth checkpoints found in {ckpt_dir}\n"
            f"Make sure training has completed and checkpoints saved there."
        )

    def get_miou(p: Path) -> float:
        name = p.stem  # e.g. "exp1_epoch020_mIoU0.4231"
        if "mIoU" in name:
            try:
                return float(name.split("mIoU")[-1])
            except ValueError:
                pass
        # fallback: use last-modified time as tiebreaker
        return p.stat().st_mtime * 1e-12  # tiny so real mIoU always wins

    best = max(ckpts, key=get_miou)
    return str(best)


def export_to_onnx(
    model,
    exp_name: str,
    ckpt_dir: str,
    output_root: str = "outputs",
    input_shape: tuple = (1, 3, 512, 1024),
    opset_version: int = 17,
    device: str = "cpu",
) -> str:
    """
    Load best checkpoint, export to ONNX, validate, return path.

    Output:
        outputs/onnx_checkpoints/<exp_name>/deeplabv3plus.onnx
    """
    print("\n================ ONNX EXPORT ================")
    print(f"  Experiment : {exp_name}")
    print(f"  Ckpt dir   : {ckpt_dir}")

    # 1. find best checkpoint
    best_ckpt = find_best_checkpoint(ckpt_dir)
    print(f"  Checkpoint : {best_ckpt}")

    # 2. load weights into model
    print("[ONNX] Loading weights...")
    load_checkpoint(best_ckpt, model, device=device)
    model = model.to(device)
    model.eval()

    # 3. dummy input
    dummy = torch.randn(input_shape, device=device)

    # 4. output path
    output_dir = Path(output_root) / "onnx_checkpoints" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "deeplabv3plus.onnx"
    print(f"[ONNX] Saving to: {onnx_path}")

    # 5. export
    # dynamic_axes: batch + spatial so the same .onnx works for any size
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
        export_params=True,
        verbose=False,
    )

    # 6. validate
    print("[ONNX] Validating...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    size_mb = onnx_path.stat().st_size / (1024 * 1024)

    print(f"[ONNX] ✓ Valid  |  Size: {size_mb:.2f} MB")
    print(f"[ONNX] Saved → {onnx_path}")
    print("=============================================\n")

    return str(onnx_path)