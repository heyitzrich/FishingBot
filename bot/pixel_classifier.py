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
from typing import List, Optional, Tuple

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
        self._cfg = cfg
        self._ranges: List[HsvRange] = self._build_ranges(cfg)
        self._blue_min_delta: int = int(cfg.get("blue", {}).get("blue_channel_min_delta", 20))
        red_cfg = cfg.get("red", {}) if isinstance(cfg.get("red"), dict) else {}
        # Optional red-mode hardening inspired by FishingFun:
        #   - (r * colour_multiplier) > g and (r * colour_multiplier) > b
        #   - g and b are "reasonably close"
        # Set multipliers to 0 to disable (HSV-only behavior).
        self._red_colour_multiplier: float = float(red_cfg.get("colour_multiplier", 0.0) or 0.0)
        self._red_closeness_multiplier: float = float(
            red_cfg.get("colour_closeness_multiplier", 0.0) or 0.0
        )
        self._red_closeness_offset: int = int(red_cfg.get("colour_closeness_offset", 20))
        self._red_ranges: Optional[List[HsvRange]] = None
        if self.mode == ClassifierMode.BLUE:
            red_cfg = cfg["red"]
            self._red_ranges = [
                (red_cfg["lower1"], red_cfg["upper1"]),
                (red_cfg["lower2"], red_cfg["upper2"]),
            ]
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

        # Blue mode hardening:
        #  1) Remove pixels matching red ranges
        #  2) Keep only pixels where blue channel is meaningfully dominant
        if self.mode == ClassifierMode.BLUE:
            if self._red_ranges:
                red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                for lower, upper in self._red_ranges:
                    lo = np.array(lower, dtype=np.uint8)
                    hi = np.array(upper, dtype=np.uint8)
                    red_mask = cv2.bitwise_or(red_mask, cv2.inRange(hsv, lo, hi))
                combined = cv2.bitwise_and(combined, cv2.bitwise_not(red_mask))

            b = bgr_frame[:, :, 0].astype(np.int16)
            g = bgr_frame[:, :, 1].astype(np.int16)
            r = bgr_frame[:, :, 2].astype(np.int16)
            max_rg = np.maximum(r, g)
            dominant_blue = (b - max_rg) >= self._blue_min_delta
            combined = combined.copy()
            combined[~dominant_blue] = 0

        # Red mode hardening (optional):
        # Helps reject warm UI/landscape reds that pass the hue threshold.
        if self.mode == ClassifierMode.RED and (
            self._red_colour_multiplier > 0.0 or self._red_closeness_multiplier > 0.0
        ):
            b = bgr_frame[:, :, 0].astype(np.int16)
            g = bgr_frame[:, :, 1].astype(np.int16)
            r = bgr_frame[:, :, 2].astype(np.int16)

            keep = np.ones(bgr_frame.shape[:2], dtype=bool)
            if self._red_colour_multiplier > 0.0:
                rm = float(self._red_colour_multiplier)
                keep &= (r.astype(np.float32) * rm) > g
                keep &= (r.astype(np.float32) * rm) > b

            if self._red_closeness_multiplier > 0.0:
                cm = float(self._red_closeness_multiplier)
                offset = int(self._red_closeness_offset)
                max_gb = np.maximum(g, b).astype(np.float32)
                min_gb = np.minimum(g, b).astype(np.float32)
                keep &= (min_gb * cm) > (max_gb - float(offset))

            combined = combined.copy()
            combined[~keep] = 0

        return combined

    def is_match(self, r: int, g: int, b: int) -> bool:
        """
        Single-pixel check (used for per-pixel iteration if needed).
        Converts one pixel to HSV and tests against all ranges.
        """
        pixel = np.array([[[b, g, r]]], dtype=np.uint8)  # OpenCV is BGR
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        if self.mode == ClassifierMode.BLUE:
            # Require blue channel dominance to reject red-ish or neutral pixels.
            if (int(b) - max(int(r), int(g))) < self._blue_min_delta:
                return False
            if self._red_ranges:
                for lower, upper in self._red_ranges:
                    if (
                        lower[0] <= h <= upper[0]
                        and lower[1] <= s <= upper[1]
                        and lower[2] <= v <= upper[2]
                    ):
                        return False

        if self.mode == ClassifierMode.RED and (
            self._red_colour_multiplier > 0.0 or self._red_closeness_multiplier > 0.0
        ):
            if self._red_colour_multiplier > 0.0:
                if (float(r) * float(self._red_colour_multiplier)) <= float(g):
                    return False
                if (float(r) * float(self._red_colour_multiplier)) <= float(b):
                    return False
            if self._red_closeness_multiplier > 0.0:
                mx = max(int(g), int(b))
                mn = min(int(g), int(b))
                if (float(mn) * float(self._red_closeness_multiplier)) <= (
                    float(mx) - float(self._red_closeness_offset)
                ):
                    return False

        for lower, upper in self._ranges:
            if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
                return True
        return False

    def update_ranges(self, cfg: dict) -> None:
        """
        Rebuild HSV ranges and hardening parameters from an updated config.

        Allows live tuning from the GUI without recreating the classifier.
        """
        self._cfg = cfg
        self._ranges = self._build_ranges(cfg)
        self._blue_min_delta = int(cfg.get("blue", {}).get("blue_channel_min_delta", 20))
        red_cfg = cfg.get("red", {}) if isinstance(cfg.get("red"), dict) else {}
        self._red_colour_multiplier = float(red_cfg.get("colour_multiplier", 0.0) or 0.0)
        self._red_closeness_multiplier = float(
            red_cfg.get("colour_closeness_multiplier", 0.0) or 0.0
        )
        self._red_closeness_offset = int(red_cfg.get("colour_closeness_offset", 20))
        if self.mode == ClassifierMode.BLUE:
            full_cfg_red = cfg.get("red", {})
            if "lower1" in full_cfg_red:
                self._red_ranges = [
                    (full_cfg_red["lower1"], full_cfg_red["upper1"]),
                    (full_cfg_red["lower2"], full_cfg_red["upper2"]),
                ]
        logger.debug(f"PixelClassifier ranges updated: {self._ranges}")

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
