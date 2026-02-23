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
            self._last_bmp_pos = None
            return None

        if total > self._max_pixels:
            logger.warning(
                f"Too many matching pixels ({total}) — color thresholds may be "
                "too broad. Narrow the HSV ranges in config.yaml."
            )
            self._last_bmp_pos = None
            return None

        result = self._find_best_component(search_mask)

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
