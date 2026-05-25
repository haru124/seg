"""
benchmark.py
============
Compare PyTorch CPU / GPU vs ONNX CPU / GPU on a single test image.

Usage:
    python benchmark.py \
        --exp_config  config/experiments/exp_01.yaml \
        --onnx_path   outputs/onnx_checkpoints/exp1/deeplabv3plus.onnx \
        --image_dir   data/images/test \
        --runs        20

Output:
    Backend         Latency (ms)    Throughput (FPS)    Speedup
    PyTorch CPU     94.76           10.6                reference
    PyTorch GPU     9.34            107.1               10.15x ★
    ONNX CPU        40.42           24.7                2.34x
"""

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image

from src.seg.constants import IMAGENET_MEAN, IMAGENET_STD
from src.seg.utils.common import count_parameters
from src.seg.models.deeplabv3_plus import get_segmentation_model
from src.seg.onnx.export import find_best_checkpoint
from src.seg.utils.checkpoint import load_checkpoint
from src.seg.utils.visualization import colorize_mask, _denorm

# ──────────────────────────────────────────────
# Config loader  (same as run_export.py)
# ──────────────────────────────────────────────

from src.seg.config.configuration import get_config
from src.seg.constants import CONFIG_PATH

def load_config(yaml_path: str):
    return get_config(CONFIG_PATH, yaml_path)


# ──────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────

def grab_one_image(image_dir: str):
    """Grab one random image from data/images/test (any city subfolder)."""
    image_dir = Path(image_dir)
    images = list(image_dir.rglob("*.png")) + list(image_dir.rglob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images found under {image_dir}")
    return str(random.choice(images))


def preprocess(image_path: str, size=(512, 1024)) -> tuple:
    """Returns (torch_tensor [1,3,H,W], numpy_array [1,3,H,W] float32)."""
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size          # (W, H)
    img = np.array(img, dtype=np.float32)

    h, w = size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    img  = (img - mean) / std
    img  = np.transpose(img, (2, 0, 1))          # HWC → CHW
    img  = np.expand_dims(img, 0)                # → 1CHW

    tensor = torch.from_numpy(img.copy())
    return tensor, img.astype(np.float32), orig_size


# ──────────────────────────────────────────────
# Timed inference helpers
# ──────────────────────────────────────────────

def time_pytorch(model, x_tensor, device, warmup=5, runs=20):
    """Returns (avg_latency_ms, pred_mask HxW numpy)."""
    model = model.to(device).eval()
    x     = x_tensor.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        last_out = None
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            last_out = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    # extract mask from last run
    logits = last_out["out"] if isinstance(last_out, dict) else last_out
    mask   = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    return float(np.mean(times)), mask


def time_onnx(onnx_path, x_numpy, use_gpu=False, warmup=5, runs=20):
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu else
        ["CPUExecutionProvider"]
    )
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # let ONNX Runtime decide optimal thread count for your CPU
    # hardcoding 4 can be slower than auto on many machines
    sess_opts.intra_op_num_threads  = 0   # 0 = auto
    sess_opts.inter_op_num_threads  = 0   # 0 = auto
    sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    session    = ort.InferenceSession(onnx_path, providers=providers,
                                      sess_options=sess_opts)
    input_name = session.get_inputs()[0].name

    for _ in range(warmup):
        session.run(None, {input_name: x_numpy})

    times    = []
    last_out = None
    for _ in range(runs):
        t0       = time.perf_counter()
        last_out = session.run(None, {input_name: x_numpy})
        times.append((time.perf_counter() - t0) * 1000)

    mask = np.argmax(last_out[0][0], axis=0).astype(np.uint8)
    return float(np.mean(times)), mask


# ──────────────────────────────────────────────
# Table printer
# ──────────────────────────────────────────────

def print_table(results: list, base_latency: float):
    header = f"{'Backend':<18} {'Latency (ms)':<16} {'Throughput (FPS)':<20} {'Speedup'}"
    sep    = "─" * 70
    print("\n" + sep)
    print(header)
    print(sep)

    max_speedup = max(base_latency / lat for _, lat in results)

    for name, lat in results:
        fps     = 1000.0 / lat
        speedup = base_latency / lat
        tag     = "★" if speedup == max_speedup and speedup > 1.0 else ""

        if name == results[0][0]:
            sp_str = "reference"
        else:
            sp_str = f"{speedup:.2f}×  {tag}"

        print(f"{name:<18} {lat:<16.2f} {fps:<20.1f} {sp_str}")

    print(sep + "\n")

def save_comparison(image_path, preds_dict, image_size, save_dir="outputs/benchmark_viz"):
    """
    Save side-by-side figure:
        Input Image | PyTorch CPU | PyTorch GPU | ONNX CPU | ONNX GPU

    preds_dict: OrderedDict of backend_name -> mask (H,W numpy uint8)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    # load and resize original for display
    img = PILImage.open(image_path).convert("RGB")
    img = np.array(img.resize((image_size[1], image_size[0]),
                               PILImage.BILINEAR), dtype=np.uint8)

    panels      = [("Input Image", img)] + \
                  [(name, colorize_mask(mask)) for name, mask in preds_dict.items()]
    n           = len(panels)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (title, rgb) in zip(axes, panels):
        ax.imshow(rgb)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Segmentation Prediction Comparison", fontsize=13, y=1.02)
    plt.tight_layout()

    save_dir  = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem      = Path(image_path).stem
    save_path = save_dir / f"benchmark_{stem}.png"

    plt.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Viz] Saved comparison → {save_path}")


def get_pytorch_model_size_mb(model) -> float:
    """
    Measure PyTorch model size in MB by writing to an in-memory buffer.
    More accurate than checking the .pth file (which includes optimizer state).
    """
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 * 1024)


def get_onnx_param_count(onnx_path: str) -> int:
    """
    Count total parameters in ONNX model by summing all initializer tensors.
    Initializers = the stored weights/biases inside the .onnx graph.
    """
    import onnx
    model  = onnx.load(onnx_path)
    total  = 0
    for init in model.graph.initializer:
        count = 1
        for dim in init.dims:
            count *= dim
        total += count
    return total

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inference speed benchmark")
    parser.add_argument("--exp_config",  required=True)
    parser.add_argument("--onnx_path",   required=True,
                        help="Path to exported .onnx file")
    parser.add_argument("--image_dir",   default="data/images/test",
                        help="Root folder of test images (any city subdirs ok)")
    parser.add_argument("--image_size",  default="512,1024",
                        help="HxW used during training, e.g. 512,1024")
    parser.add_argument("--warmup",      type=int, default=5)
    parser.add_argument("--runs",        type=int, default=20,
                        help="Number of timed runs to average")
    args = parser.parse_args()

    cfg        = load_config(args.exp_config)
    image_size = tuple(int(x) for x in args.image_size.split(","))  # (H, W)

    # ── Build & load model ───────────────────────────────────────────
    model = get_segmentation_model(
        num_classes             = cfg.data.num_classes,
        backbone                = cfg.model.backbone,
        output_stride           = cfg.model.output_stride,
        aux                     = cfg.training.aux_loss,
        use_pretrained_backbone = False,
        use_jpu                 = getattr(cfg.model, "use_jpu", False),
    )

    best_ckpt = find_best_checkpoint(cfg.checkpoint.dir)
    load_checkpoint(best_ckpt, model, device="cpu")
    model.eval()

    # ── Image ────────────────────────────────────────────────────────
    img_path = grab_one_image(args.image_dir)
    x_torch, x_numpy, orig_size = preprocess(img_path, image_size)

    pt_size_mb   = get_pytorch_model_size_mb(model)
    onnx_size_mb = Path(args.onnx_path).stat().st_size / (1024 * 1024)
     
    n_params_onnx  = get_onnx_param_count(args.onnx_path)

    # count raw int ourselves — don't use count_parameters() since it returns a string
    n_params_pt    = sum(p.numel() for p in model.parameters())
    param_diff     = n_params_pt - n_params_onnx

    print("\n" + "═" * 70)
    print("  INFERENCE BENCHMARK — DeepLabV3+")
    print("═" * 70)
    print(f"  Image path      : {img_path}")
    print(f"  Orig size       : {orig_size[0]}×{orig_size[1]}  (W×H)")
    print(f"  Input tensor    : {list(x_torch.shape)}")
    print(f"  Backbone        : {cfg.model.backbone}  |  OS={cfg.model.output_stride}")
    print(f"  Checkpoint      : {best_ckpt}")
    print()
    print(f"  ── Model Size ──────────────────────────────────")
    print(f"  PyTorch (.pth)  : {pt_size_mb:.2f} MB   (weights only, no optimizer state)")
    print(f"  ONNX    (.onnx) : {onnx_size_mb:.2f} MB")
    print(f"  Size reduction  : {pt_size_mb - onnx_size_mb:.2f} MB  "
          f"({(1 - onnx_size_mb/pt_size_mb)*100:.1f}% smaller)")
    print()
    print(f"  ── Parameters ──────────────────────────────────")
    print(f"  PyTorch params  : {n_params_pt:,}  ({n_params_pt/1e6:.2f} M)")
    print(f"  ONNX    params  : {n_params_onnx:,}  ({n_params_onnx/1e6:.2f} M)")
    if param_diff == 0:
        print(f"  Difference      : 0  ✓ exact match")
    else:
        print(f"  Difference      : {param_diff:,}  ← investigate if large")
    print()
    print(f"  Warmup runs     : {args.warmup}  |  Timed runs: {args.runs}")
    print("═" * 70)

    results    = []   # (backend_name, latency_ms)
    preds_dict = {}   # backend_name -> mask for visualization

    # ── PyTorch CPU ──────────────────────────────────────────────────
    print("\n[1/4] PyTorch CPU ...")
    cpu_lat, cpu_mask = time_pytorch(model, x_torch, torch.device("cpu"),
                                     warmup=args.warmup, runs=args.runs)
    results.append(("PyTorch CPU", cpu_lat))
    preds_dict["PyTorch CPU"] = cpu_mask
    print(f"      avg {cpu_lat:.2f} ms")

    # ── PyTorch GPU ──────────────────────────────────────────────────
    if torch.cuda.is_available():
        print("[2/4] PyTorch GPU ...")
        gpu_lat, gpu_mask = time_pytorch(model, x_torch, torch.device("cuda"),
                                         warmup=args.warmup, runs=args.runs)
        results.append(("PyTorch GPU", gpu_lat))
        preds_dict["PyTorch GPU"] = gpu_mask
        print(f"      avg {gpu_lat:.2f} ms")
    else:
        print("[2/4] PyTorch GPU — CUDA not available, skipping")

    # ── ONNX CPU ─────────────────────────────────────────────────────
    print("[3/4] ONNX CPU ...")
    onnx_cpu_lat, onnx_cpu_mask = time_onnx(args.onnx_path, x_numpy, use_gpu=False,
                                             warmup=args.warmup, runs=args.runs)
    results.append(("ONNX CPU", onnx_cpu_lat))
    preds_dict["ONNX CPU"] = onnx_cpu_mask
    print(f"      avg {onnx_cpu_lat:.2f} ms")

    # ── ONNX GPU ─────────────────────────────────────────────────────
    if "CUDAExecutionProvider" in ort.get_available_providers():
        print("[4/4] ONNX GPU ...")
        onnx_gpu_lat, onnx_gpu_mask = time_onnx(args.onnx_path, x_numpy, use_gpu=True,
                                                  warmup=args.warmup, runs=args.runs)
        results.append(("ONNX GPU", onnx_gpu_lat))
        preds_dict["ONNX GPU"] = onnx_gpu_mask
        print(f"      avg {onnx_gpu_lat:.2f} ms")
    else:
        print("[4/4] ONNX GPU — CUDAExecutionProvider not available, skipping")

    # ── Print table ───────────────────────────────────────────────────
    base_latency = results[0][1]
    print_table(results, base_latency)

    # ── Visualization ─────────────────────────────────────────────────
    save_comparison(
        image_path  = img_path,
        preds_dict  = preds_dict,
        image_size  = image_size,
        save_dir    = "outputs/benchmark_viz",
    )

if __name__ == "__main__":
    main()