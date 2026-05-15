"""
src/seg/constants/__init__.py
------------------------------
Central place for every project-wide constant.
Import from here everywhere else — never hardcode strings/paths in code.

Usage:
    from src.ods.constants import NUM_CLASSES, CITYSCAPES_CLASSES, CONFIG_PATH
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
# Root of the project (two levels up from this file: constants/ → ods/ → src/ → seg/)
PROJECT_ROOT = Path(__file__).resolve().parents[4]

CONFIG_PATH       = PROJECT_ROOT / "config" / "config.yaml"
EXPERIMENTS_DIR   = PROJECT_ROOT / "config" / "experiments"
DATA_ROOT         = PROJECT_ROOT / "data"
WEIGHTS_DIR       = PROJECT_ROOT / "weights"
OUTPUTS_DIR       = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR   = OUTPUTS_DIR / "checkpoints"
LOGS_DIR          = OUTPUTS_DIR / "logs"
TENSORBOARD_DIR   = OUTPUTS_DIR / "tensorboard"
MLRUNS_DIR        = OUTPUTS_DIR / "mlruns"
INFERENCE_DIR     = OUTPUTS_DIR / "inference"

# ── Dataset constants ─────────────────────────────────────────────────
NUM_CLASSES   = 19      # Cityscapes has 19 trainable classes
IGNORE_INDEX  = 255     # Pixels with this label are ignored in loss & metrics

# Class names in the order used by training IDs (0–18)
CITYSCAPES_CLASSES = [
    "road",           # 0
    "sidewalk",       # 1
    "building",       # 2
    "wall",           # 3
    "fence",          # 4
    "pole",           # 5
    "traffic light",  # 6
    "traffic sign",   # 7
    "vegetation",     # 8
    "terrain",        # 9
    "sky",            # 10
    "person",         # 11
    "rider",          # 12
    "car",            # 13
    "truck",          # 14
    "bus",            # 15
    "train",          # 16
    "motorcycle",     # 17
    "bicycle",        # 18
]

# RGB colours for each class — used to colourise prediction masks
# Order matches CITYSCAPES_CLASSES (index = train ID)
CITYSCAPES_PALETTE = [
    (128,  64, 128),   # road
    (244,  35, 232),   # sidewalk
    ( 70,  70,  70),   # building
    (102, 102, 156),   # wall
    (190, 153, 153),   # fence
    (153, 153, 153),   # pole
    (250, 170,  30),   # traffic light
    (220, 220,   0),   # traffic sign
    (107, 142,  35),   # vegetation
    (152, 251, 152),   # terrain
    ( 70, 130, 180),   # sky
    (220,  20,  60),   # person
    (255,   0,   0),   # rider
    (  0,   0, 142),   # car
    (  0,   0,  70),   # truck
    (  0,  60, 100),   # bus
    (  0,  80, 100),   # train
    (  0,   0, 230),   # motorcycle
    (119,  11,  32),   # bicycle
]

# ImageNet normalisation — used for all backbone pre-training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Cityscapes raw label ID → training ID mapping
# Labels not listed here map to IGNORE_INDEX (255)
# Source: https://github.com/mcordts/cityscapesScripts
LABEL_ID_TO_TRAIN_ID = {
    # raw_id : train_id
    7:  0,   # road
    8:  1,   # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}