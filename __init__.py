"""
UrbanSense AI - AI Models Package
Standalone model architectures for each of the 8 computer vision capabilities.
"""

from .base_detector import BaseDetector, CATEGORY_COLORS, get_device, draw_detection_box

__all__ = [
    "BaseDetector",
    "CATEGORY_COLORS",
    "get_device",
    "draw_detection_box"
]
