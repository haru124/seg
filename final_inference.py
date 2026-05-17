# src/seg/inference/inference.py
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms

from src.seg.constants import CITYSCAPES_PALETTE, IGNORE_INDEX


def load_image(image_path: str, image_size=(512, 1024)) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)   # (1, 3, H, W)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class index mask (H, W) → RGB image (H, W, 3)."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in enumerate(CITYSCAPES_PALETTE):
        color[mask == cls_id] = rgb
    return color


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    image_size=(512, 1024),
    save_dir: str = "outputs/inference",
) -> np.ndarray:
    model.eval()
    tensor = load_image(image_path, image_size).to(device)

    output = model(tensor)
    pred = output["out"] if isinstance(output, dict) else output
    pred_mask = pred.argmax(dim=1).squeeze(0).cpu().numpy()   # (H, W)

    color_mask = colorize_mask(pred_mask)
    save_path = Path(save_dir) / (Path(image_path).stem + "_pred.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color_mask).save(str(save_path))
    print(f"[Inference] Saved → {save_path}")
    return pred_mask