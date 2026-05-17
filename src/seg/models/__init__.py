"""
src/seg/models/__init__.py
"""
from .deeplabv3_plus import get_segmentation_model, DeepLabV3Plus

__all__ = ['get_segmentation_model', 'DeepLabV3Plus']