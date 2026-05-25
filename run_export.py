"""
run_export.py
=============
CLI wrapper: load exp config → build model → export to ONNX.

Usage:
    python run_export.py --exp_config config/experiments/exp_01.yaml
"""

import argparse
import torch.nn as nn

from src.seg.entity.config_entity import ExperimentConfig   # adjust import if needed
from src.seg.models.deeplabv3_plus import get_segmentation_model
from src.seg.onnx.export import export_to_onnx
from src.seg.config.configuration import get_config
from src.seg.constants import CONFIG_PATH

def load_config(yaml_path: str):
    return get_config(CONFIG_PATH, yaml_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--device", default="cpu",
                        help="cpu or cuda (for export step only)")
    args = parser.parse_args()

    cfg = load_config(args.exp_config)

    # Build model (no pretrained backbone needed — weights come from checkpoint)
    model = get_segmentation_model(
        num_classes            = cfg.data.num_classes,
        backbone               = cfg.model.backbone,
        output_stride          = cfg.model.output_stride,
        aux                    = cfg.training.aux_loss,          # aux head not needed for inference
        use_pretrained_backbone= False,
        norm_layer             = __import__('torch').nn.BatchNorm2d,
        use_jpu                = getattr(cfg.model, "use_jpu", False),
    )

    onnx_path = export_to_onnx(
        model       = model,
        exp_name    = cfg.experiment_id,
        ckpt_dir    = cfg.checkpoint.dir,
        output_root = "outputs",
        input_shape = (1, 3, *cfg.data.image_size),   # (1,3,H,W)
        device      = args.device,
    )

    print(f"\nDone. ONNX model at:\n  {onnx_path}")
    print("\nNext step — run the benchmark:")
    print(f"  python benchmark.py --exp_config {args.exp_config} --onnx_path {onnx_path}")


if __name__ == "__main__":
    main()