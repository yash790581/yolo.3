"""
UrbanSense AI - Base Detector & Utilities for AI Models
Standard interface and color maps across all 8 detection modules.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torch

CATEGORY_COLORS = {
    "vehicle": (46, 204, 113),       # Emerald Green
    "number_plate": (52, 152, 219),   # Sky Blue
    "pothole": (41, 128, 185),        # Amber/Crimson
    "person": (155, 89, 182),         # Amethyst Purple
    "traffic_light": (241, 196, 15),   # Yellow
    "traffic_sign": (230, 126, 34),    # Carrot Orange
    "zebra_crossing": (26, 188, 156),  # Turquoise
    "waterlogging": (0, 168, 255),     # Deep Water Blue
    "hazard": (231, 76, 60),          # Bright Red
}


def get_device() -> str:
    """Returns 'cuda' if GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def draw_detection_box(
    image: np.ndarray,
    bbox: List[float],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 2
) -> None:
    """Draws a styled bounding box with label tag on the image in-place."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    tag_y1 = max(0, y1 - text_h - baseline - 4)
    tag_y2 = y1
    tag_x2 = min(w - 1, x1 + text_w + 8)

    cv2.rectangle(image, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 4, tag_y2 - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        lineType=cv2.LINE_AA
    )


class BaseDetector(ABC):
    """
    Standardized Base Detector for all UrbanSense AI models.
    """

    def __init__(self, name: str, model_path: Optional[str] = None, confidence: float = 0.3):
        self.name = name
        self.model_path = model_path
        self.confidence = confidence
        self.device = get_device()

    @abstractmethod
    def detect(self, frame: np.ndarray, conf: Optional[float] = None, **kwargs) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Runs inference on a single BGR frame.

        Returns:
            detections: List of detection dictionaries:
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": float,
                    "class_name": str,
                    "category": str,
                    "label": str,
                    "track_id": Optional[int],
                    "metadata": dict
                }
            summary: Summary dictionary for the model inference.
        """
        pass
