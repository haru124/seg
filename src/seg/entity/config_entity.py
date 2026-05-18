"""
src/seg/entity/config_entity.py
--------------------------------
Typed dataclasses that hold every configuration section.

After loading config.yaml (+ optional experiment override yaml),
the configuration.py module builds these objects.
Using dataclasses gives you:
  - IDE autocompletion on cfg.training.lr  etc.
  - Easy to print / log
  - No dict key typos at runtime
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    """Everything related to data loading."""
    root: str                    # path to data/ folder
    dataset: str                 # "cityscapes"
    num_classes: int             # 19 for Cityscapes
    ignore_index: int            # 255 — ignored in loss & metrics
    image_size: List[int]        # [H, W] e.g. [512, 1024]
    batch_size: int              # per-GPU batch size
    num_workers: int             # DataLoader worker processes
    max_train_samples: Optional[int]    # for debugging; None = use full dataset
    max_val_samples: Optional[int]
    max_test_samples: Optional[int]

@dataclass
class ModelConfig:
    """Model architecture settings."""
    name: str                           # "deeplabv3_plus"
    backbone: str                       # "resnet50" | "resnet34" | "mobilenet_v2"
    output_stride: int                  # 16 (lighter) or 8 (better accuracy)
    use_pretrained_backbone: bool          
    # True:
    #   - load local backbone weights if path exists
    #   - otherwise download torchvision pretrained weights
    # False:
    #   - random initialization
    backbone_weights_path: Optional[str] # path to backbone weights or None
    use_jpu: bool                       # whether to use Joint Pyramid Upsampling module (adds memory overhead)


@dataclass
class TrainingConfig:
    """Optimiser, scheduler, and training loop settings."""
    epochs: int
    lr: float
    lr_scheduler: str          # "poly" | "cosine" | "step"
    momentum: float            # for SGD
    weight_decay: float
    optimizer: str             # "sgd" | "adamw" | "adam"
    aux_loss: bool             # use auxiliary decoder loss
    aux_weight: float          # weight for auxiliary loss (paper: 0.4)
    amp: bool                  # mixed precision (FP16) — essential for 4 GB GPU
    accumulation_steps: int    # gradient accumulation; effective batch = batch*steps
    grad_clip: Optional[float] # max gradient norm; None = no clipping


@dataclass
class LossConfig:
    """Loss function selection and its hyperparameters."""
    type: str                        # "ce" | "ohem" | "focal" | "ce_dice" | ...
    # The remaining keys are loss-specific kwargs (thresh, gamma, dice_weight …)
    # We store them as a plain dict so build_loss() can forward them easily.
    kwargs: dict = field(default_factory=dict)


@dataclass
class TrackingConfig:
    """MLflow and TensorBoard settings."""
    mlflow_enabled: bool
    mlflow_uri: str
    mlflow_experiment: str
    tb_enabled: bool
    tb_log_dir: str


@dataclass
class CheckpointConfig:
    """Where to save checkpoints and how many to keep."""
    dir: str
    save_top_k: int            # keep only top-k checkpoints ranked by val mIoU
    resume: Optional[str]      # path to a .pth file to resume from, or None


@dataclass
class ExperimentConfig:
    """
    Top-level config object passed everywhere in the project.
    Built by src/ods/config/configuration.py from YAML files.
    """
    experiment_id: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    tracking: TrackingConfig
    checkpoint: CheckpointConfig