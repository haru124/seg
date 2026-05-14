# src/ods/onnx/inference.py
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from src.ods.inference.inference import colorize_mask
from pathlib import Path


def run_onnx_inference(
    onnx_path: str,
    image_path: str,
    image_size=(512, 1024),
    save_dir: str = "outputs/inference",
) -> np.ndarray:
    session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = tf(img).unsqueeze(0).numpy()

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]
    pred_mask = output.argmax(axis=1).squeeze(0)

    color = colorize_mask(pred_mask)
    save_path = Path(save_dir) / (Path(image_path).stem + "_onnx_pred.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color).save(str(save_path))
    print(f"[ONNX Inference] Saved → {save_path}")
    return pred_mask