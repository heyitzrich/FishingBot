"""
Bobber finder using OpenCV connected-components analysis.

Improvement over the original C# approach:
  - cv2.connectedComponentsWithStats replaces a manual pixel-scoring loop
  - Accurate centroid via component statistics, not just "first match"
  - Restricts search area to a tight radius around the last known position
    for faster tracking after the initial lock-on
  - Logs a warning when a scan exceeds 200ms (same threshold as the original)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from bot.pixel_classifier import PixelClassifier
from utils.screen import capture_region, bitmap_to_screen_coords
from utils.logger import get_logger

logger = get_logger()


class BobberFinder:
    """
    Finds the fishing bobber in the captured screen region.

    Algorithm:
        1. Capture the configured screen region with mss
        2. Apply HSV color mask via PixelClassifier
        3. Optionally restrict search to a radius around the last known position
        4. Run cv2.connectedComponentsWithStats to find pixel clusters
        5. Return screen coords of the centroid of the largest qualifying cluster
    """

    def __init__(self, classifier: PixelClassifier, region: dict, cfg: dict):
        """
        Args:
            classifier: Configured PixelClassifier (RED or BLUE mode).
            region:     mss capture region dict (top/left/width/height).
            cfg:        The detection section of config.yaml.
        """
        self._classifier = classifier
        self._region = region
        self._min_pixels: int = cfg["min_cluster_pixels"]
        self._max_pixels: int = cfg["max_total_pixels"]
        # Use 4× cluster_radius for the tracking window — wide enough to
        # follow the bobber as it bobs, tight enough to exclude false positives
        self._track_radius: int = cfg["cluster_radius"] * 4
        self._template_enabled: bool = bool(
            cfg.get("template_fallback", {}).get("enabled", False)
        )
        self._template_threshold: float = float(
            cfg.get("template_fallback", {}).get("match_threshold", 0.84)
        )
        self._template_bgr: Optional[np.ndarray] = None
        self._template_shape: Optional[Tuple[int, int]] = None  # (h, w)
        self._template_warned = False
        self._load_template(cfg)

        self._last_bmp_pos: Optional[Tuple[int, int]] = None
        self._last_frame:   Optional[np.ndarray] = None
        self._last_mask:    Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find(self) -> Optional[Tuple[int, int]]:
        """
        Capture the screen region and locate the bobber.

        Returns:
            (screen_x, screen_y) of the bobber centroid, or None if not found.
        """
        t0 = time.perf_counter()

        frame = capture_region(self._region)
        mask  = self._classifier.apply_mask(frame)

        # Cache frames for the debug overlay in main.py
        self._last_frame = frame
        self._last_mask  = mask

        search_mask = self._restrict_to_last(mask)

        total = cv2.countNonZero(search_mask)

        if total == 0:
            fallback = self._find_with_template(frame)
            if fallback is not None:
                self._last_bmp_pos = fallback
                return bitmap_to_screen_coords(fallback[0], fallback[1], self._region)
            self._last_bmp_pos = None
            return None

        if total > self._max_pixels:
            logger.warning(
                f"Too many matching pixels ({total}) — color thresholds may be "
                "too broad. Narrow the HSV ranges in config.yaml."
            )
            fallback = self._find_with_template(frame)
            if fallback is not None:
                self._last_bmp_pos = fallback
                return bitmap_to_screen_coords(fallback[0], fallback[1], self._region)
            self._last_bmp_pos = None
            return None

        result = self._find_best_component(search_mask)
        if result is None:
            fallback = self._find_with_template(frame)
            if fallback is not None:
                self._last_bmp_pos = fallback
                return bitmap_to_screen_coords(fallback[0], fallback[1], self._region)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 200:
            logger.warning(f"BobberFinder scan took {elapsed_ms:.0f}ms (threshold: 200ms)")

        return result

    def reset(self) -> None:
        """Clear the cached position. Call after each cast so the finder
        performs a full-frame search for the new bobber position."""
        self._last_bmp_pos = None

    @property
    def last_bitmap_pos(self) -> Optional[Tuple[int, int]]:
        """Last bobber position in bitmap (capture-region) coordinates.
        Used by the debug overlay to draw the reticle."""
        return self._last_bmp_pos

    @property
    def last_frame(self) -> Optional[np.ndarray]:
        """Most recently captured BGR frame (for debug display)."""
        return self._last_frame

    @property
    def last_mask(self) -> Optional[np.ndarray]:
        """Most recently computed binary mask (for debug display)."""
        return self._last_mask

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_template(self, cfg: dict) -> None:
        """Load template image for optional matchTemplate fallback."""
        if not self._template_enabled:
            return

        template_cfg = cfg.get("template_fallback", {})
        template_path_raw = str(template_cfg.get("template_path", "")).strip()
        if not template_path_raw:
            logger.warning(
                "Template fallback enabled but no template_path was provided; "
                "fallback is disabled."
            )
            self._template_enabled = False
            return

        template_path = Path(template_path_raw)
        if not template_path.is_absolute():
            template_path = Path.cwd() / template_path

        if not template_path.exists():
            logger.warning(
                f"Template fallback image not found: {template_path}. "
                "Fallback is disabled."
            )
            self._template_enabled = False
            return

        tpl = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if tpl is None or tpl.size == 0:
            logger.warning(
                f"Template fallback failed to load image: {template_path}. "
                "Fallback is disabled."
            )
            self._template_enabled = False
            return

        self._template_bgr = tpl
        self._template_shape = tpl.shape[:2]
        logger.info(
            "Template fallback loaded: %s (threshold=%.2f)",
            template_path,
            self._template_threshold,
        )

    def _restrict_to_last(self, mask: np.ndarray) -> np.ndarray:
        """
        If we have a cached bobber position, zero out everything outside a
        radius around it. Falls back to the full mask if the restricted
        region has fewer than min_cluster_pixels hits.
        """
        if self._last_bmp_pos is None:
            return mask

        lx, ly = self._last_bmp_pos
        r = self._track_radius
        h, w = mask.shape

        roi = np.zeros_like(mask)
        x1, y1 = max(0, lx - r), max(0, ly - r)
        x2, y2 = min(w, lx + r), min(h, ly + r)
        roi[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

        if cv2.countNonZero(roi) < self._min_pixels:
            return mask  # fall back to full-frame search

        return roi

    def _find_with_template(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Fallback detection using template matching.
        Returns bitmap coordinates of match center, or None on no confident match.
        """
        if not self._template_enabled or self._template_bgr is None or self._template_shape is None:
            return None

        t_h, t_w = self._template_shape
        f_h, f_w = frame_bgr.shape[:2]
        if t_h > f_h or t_w > f_w:
            if not self._template_warned:
                logger.warning(
                    "Template fallback image is larger than capture region; "
                    "fallback will be skipped."
                )
                self._template_warned = True
            return None

        # Full-color template match so feather+bob colors contribute to confidence.
        result = cv2.matchTemplate(frame_bgr, self._template_bgr, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < self._template_threshold:
            return None

        center_x = int(max_loc[0] + t_w / 2)
        center_y = int(max_loc[1] + t_h / 2)
        logger.debug(
            "Template fallback matched at bitmap (%s, %s), score=%.3f",
            center_x,
            center_y,
            max_val,
        )
        return center_x, center_y

    def _find_best_component(self, mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Run connected-components analysis and return the screen coordinates
        of the centroid of the largest cluster that meets the minimum size.
        """
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        best_label = -1
        best_area  = 0

        for label in range(1, num_labels):  # label 0 = background
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= self._min_pixels and area > best_area:
                best_area  = area
                best_label = label

        if best_label == -1:
            self._last_bmp_pos = None
            return None

        bmp_x = int(centroids[best_label][0])
        bmp_y = int(centroids[best_label][1])
        self._last_bmp_pos = (bmp_x, bmp_y)

        logger.debug(f"Bobber at bitmap ({bmp_x}, {bmp_y}), cluster area={best_area}px")

        return bitmap_to_screen_coords(bmp_x, bmp_y, self._region)
