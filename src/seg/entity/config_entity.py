# src/ods/entity/config_entity.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class DataConfig:
    root: str
    dataset: str
    num_classes: int
    ignore_index: int
    image_size: List[int]
    batch_size: int
    num_workers: int


@dataclass
class ModelConfig:
    name: str
    backbone: str
    output_stride: int
    pretrained_backbone: bool
    pretrained_weights: Optional[str]


@dataclass
class TrainingConfig:
    epochs: int
    lr: float
    lr_scheduler: str
    momentum: float
    weight_decay: float
    optimizer: str          
    aux_loss: bool
    aux_weight: float
    grad_clip: Optional[float]
    amp: bool
    accumulation_steps: int

@dataclass
class TrackingConfig:
    mlflow_enabled: bool
    mlflow_uri: str
    mlflow_experiment: str
    tb_enabled: bool
    tb_log_dir: str


@dataclass
class CheckpointConfig:
    dir: str
    save_top_k: int
    resume: Optional[str]


@dataclass
class ExperimentConfig:
    experiment_id: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    tracking: TrackingConfig
    checkpoint: CheckpointConfig