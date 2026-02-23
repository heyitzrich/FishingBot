"""
Pixel color classifier using HSV color space.

Improvement over the original C# RGB approach:
  - HSV isolates hue from brightness/saturation, making detection
    far more robust across different lighting conditions and game settings.
  - Two HSV ranges handle the red hue wraparound at 0°/360° in OpenCV
    (which uses 0-180 scale).
"""

from __future__ import annotations

from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger()

HsvRange = Tuple[List[int], List[int]]  # (lower, upper) in HSV


class ClassifierMode(Enum):
    RED  = "red"
    BLUE = "blue"


class PixelClassifier:
    """
    Classifies pixels by color to locate the fishing bobber feather.

    Modes:
        RED  — standard feather (most fishing)
        BLUE — lava fishing feather

    Usage:
        mask = classifier.apply_mask(bgr_frame)
        # mask is a binary 2D array; white pixels matched the feather color
    """

    def __init__(self, mode: ClassifierMode, cfg: dict):
        """
        Args:
            mode: ClassifierMode.RED or ClassifierMode.BLUE
            cfg:  The detection config dict from config.yaml
        """
        self.mode = mode
        self._ranges: List[HsvRange] = self._build_ranges(cfg)
        logger.debug(f"PixelClassifier mode={mode.value}, ranges={self._ranges}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_mask(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Convert frame to HSV and return a binary mask where feather
        pixels are 255 and everything else is 0.

        Args:
            bgr_frame: OpenCV BGR image (H x W x 3).

        Returns:
            Grayscale mask (H x W), dtype uint8.
        """
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in self._ranges:
            lo = np.array(lower, dtype=np.uint8)
            hi = np.array(upper, dtype=np.uint8)
            combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))

        return combined

    def is_match(self, r: int, g: int, b: int) -> bool:
        """
        Single-pixel check (used for per-pixel iteration if needed).
        Converts one pixel to HSV and tests against all ranges.
        """
        pixel = np.array([[[b, g, r]]], dtype=np.uint8)  # OpenCV is BGR
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        for lower, upper in self._ranges:
            if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
                return True
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_ranges(self, cfg: dict) -> List[HsvRange]:
        if self.mode == ClassifierMode.RED:
            red_cfg = cfg["red"]
            return [
                (red_cfg["lower1"], red_cfg["upper1"]),
                (red_cfg["lower2"], red_cfg["upper2"]),
            ]
        else:
            blue_cfg = cfg["blue"]
            return [
                (blue_cfg["lower"], blue_cfg["upper"]),
            ]
