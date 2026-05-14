# src/seg/onnx/export.py
import torch
from pathlib import Path


def export_to_onnx(
    model: torch.nn.Module,
    checkpoint_path: str,
    export_path: str,
    input_shape=(1, 3, 512, 1024),
    opset: int = 17,
    device: str = "cpu",
):
    from src.ods.utils.checkpoint import load_checkpoint
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval().to(device)

    dummy = torch.randn(*input_shape).to(device)
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(export_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"[ONNX] Exported → {export_path}")