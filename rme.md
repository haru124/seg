# DeepLabV3+ Cityscapes Semantic Segmentation

A complete PyTorch implementation of DeepLabV3+ for semantic segmentation on the Cityscapes dataset. Optimized for training on 4 GB GPU memory.

## Features

✅ **Architecture**: DeepLabV3+ with Encoder-Decoder  
✅ **Backbones**: ResNet50/34, ResNet V1b, MobileNetV2  
✅ **Loss Functions**: CE, Focal, OHEM, Dice, Lovász, and compound losses  
✅ **Metrics**: mIoU, Frequency-weighted IoU, Per-class IoU, Boundary IoU, Boundary F-score  
✅ **Experiment Tracking**: MLflow + TensorBoard  
✅ **Mixed Precision**: AMP for 4 GB VRAM GPUs  
✅ **Test Evaluation**: Full metrics + visualization of predictions  

---

## Quick Start

### 1. Setup

```bash
# Clone repo
git clone <repo>
cd seg

# Install dependencies
pip install -r requirements.txt

# Create folders
python template.py
```

### 2. Download Cityscapes Dataset

Download from: https://www.cityscapes-dataset.com/

Extract to `seg/data/`:
```
seg/data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

### 3. Train

```bash
# Train exp_01 (ResNet50 + OHEM CE)
python main.py --exp_config config/experiments/exp_01.yaml

# Or modify and train your own experiment
python main.py --exp_config config/experiments/exp_02.yaml
```

Monitor with TensorBoard:
```bash
tensorboard --logdir outputs/tensorboard
```

### 4. Evaluate on Test Set

```bash
python test_inference.py --exp_config config/experiments/exp_01.yaml
```

This will:
- Load the best checkpoint
- Compute all metrics on test split
- Log to MLflow & TensorBoard
- Visualize 5 random samples (GT + prediction side-by-side)
- Save per-class IoU bar chart

---

## Project Structure

```
seg/
├── data/                          # Dataset (add after download)
│   ├── images/  {train, val, test}
│   └── gtFine/  {train, val, test}
├── weights/                       # Pretrained weights (optional)
├── src/ods/
│   ├── constants/                 # Project constants
│   ├── entity/                    # Config dataclasses
│   ├── config/                    # Config loader
│   ├── datasets/                  # Dataset & transforms
│   ├── models/
│   │   ├── backbones/            # ResNet, ResNet V1b, MobileNetV2
│   │   ├── nn/                    # Helper modules (JPU, basic layers)
│   │   └── deeplabv3_plus.py     # Main architecture
│   ├── losses/                    # 15+ loss functions
│   ├── evaluation/                # Metrics (mIoU, boundary IoU, etc.)
│   ├── training/                  # Trainer with AMP + checkpointing
│   ├── inference/                 # Single-image inference
│   ├── tracking/                  # MLflow + TensorBoard loggers
│   └── utils/                     # Visualization, checkpoints, logging
├── config/
│   ├── config.yaml               # Base config
│   └── experiments/
│       ├── exp_01.yaml           # ResNet50 + OHEM CE (recommended)
│       ├── exp_02.yaml           # ResNet34 + CE+Dice
│       └── exp_03.yaml           # MobileNetV2 + Focal+Dice
├── outputs/                       # Generated during training
│   ├── checkpoints/
│   ├── logs/
│   ├── tensorboard/
│   ├── mlruns/
│   └── inference/
├── main.py                        # Training entry point
├── test_inference.py              # Test evaluation entry point
├── template.py                    # Create folder structure
├── requirements.txt
├── .gitignore
└── README.md                      # This file
```

---

## Available Backbones

| Backbone | Params | Speed | Accuracy | Best For |
|----------|--------|-------|----------|----------|
| ResNet34 | 21 M   | Fast  | 78-79%   | Quick experiments |
| ResNet50 | 39 M   | Good  | 79-80%   | **Recommended** |
| ResNet101 | 60 M  | Slow  | 80-81%   | High accuracy |
| ResNet50 V1b | 39 M | Good | 79-80%  | With dilation |
| MobileNetV2 | 3.5 M | Very fast | 75-76% | Edge devices |

---

## Loss Functions

### Distribution-based
- `ce` - Standard Cross-Entropy
- `focal` - Focal Loss (harder examples)
- `ohem` - Online Hard Example Mining
- `weighted_ce` - Class-weighted CE

### Region-based
- `dice` - Dice Loss
- `generalized_dice` - Weighted Dice for imbalance
- `tversky` - Asymmetric generalization of Dice
- `iou` - Soft IoU Loss
- `lovasz` - Lovász-Softmax (direct mIoU optimization)

### Compound
- `ce_dice` - CE + λ*Dice ✅ Most popular
- `focal_dice` - Focal + Dice
- `ohem_dice` - OHEM + Dice ✅ Paper baseline
- `lovasz_ce` - Lovász + CE ✅ Best for mIoU

---

## Metrics

| Metric | Meaning |
|--------|---------|
| **mIoU** | Mean Intersection-over-Union (primary metric) |
| **fw_iou** | Frequency-weighted IoU (accounts for class imbalance) |
| **pixel_acc** | Overall pixel accuracy |
| **class_acc** | Mean per-class accuracy |
| **boundary_iou** | IoU computed only on boundary pixels (±3px) |
| **boundary_fscore** | F-score at boundaries |
| **per_class_iou** | IoU for each individual class |

---

## Training Tips

### For 4 GB GPU
```yaml
batch_size: 4              # Critical
accumulation_steps: 4      # Effective batch = 16
amp: true                  # Mixed precision
image_size: [512, 1024]   # Safe resolution
output_stride: 16         # Lighter than 8
```

### Learning Rates by Optimizer
- **SGD**: `lr: 0.007` (with poly scheduler)
- **AdamW**: `lr: 0.0003` (with cosine scheduler)
- **Adam**: `lr: 0.001` (with cosine scheduler)

### Best Loss for Cityscapes
Ranked by typical mIoU:
1. `lovasz_ce` - Direct mIoU optimization
2. `ohem_dice` - Paper baseline (recommended)
3. `ce_dice` - Stable and fast
4. `focal_dice` - Good for class imbalance

---

## Experiment Management

### Create a new experiment

1. Copy an experiment yaml:
```bash
cp config/experiments/exp_01.yaml config/experiments/exp_XX.yaml
```

2. Modify parameters:
```yaml
experiment:
  id: "exp_XX_your_name"
  description: "..."
model:
  backbone: "resnet101"     # try different
  output_stride: 8          # try 8 for better accuracy
loss:
  type: "lovasz_ce"         # try different loss
```

3. Train:
```bash
python main.py --exp_config config/experiments/exp_XX.yaml
```

### MLflow Dashboard
```bash
mlflow ui --backend-store-uri ./outputs/mlruns
```
Open http://localhost:5000

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `batch_size` to 2, increase `accumulation_steps` to 8 |
| Poor validation loss | Try `lovasz_ce` loss or reduce `lr` |
| Training diverges | Enable `grad_clip: 0.35` |
| Slow data loading | Increase `num_workers` (0 = disabled) |

---

## References

- [DeepLabV3+ Paper](https://arxiv.org/abs/1802.02611)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [Awesome Semantic Segmentation Pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)
- [Lovász-Softmax Loss](https://arxiv.org/abs/1805.08318)

---

## License

This project is for educational purposes. Follow Cityscapes dataset license.

---

**Happy training! 🚀**
