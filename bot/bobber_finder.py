"""
Bobber finder using OpenCV connected-components analysis.

Improvements:
  - cv2.connectedComponentsWithStats replaces a manual pixel-scoring loop
  - Accurate centroid via component statistics, not just "first match"
  - Restricts search area around the last known position for faster tracking
  - Optional automatic red/blue mode fallback when the configured mode misses
  - Logs a warning when a scan exceeds 200ms
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from bot.pixel_classifier import ClassifierMode, PixelClassifier
from utils.logger import get_logger
from utils.screen import bitmap_to_screen_coords, capture_region

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
        self._detector_order: str = str(
            cfg.get("detector_order", "template_first")
        ).strip().lower()

        self._primary_mode: str = classifier.mode.value
        self._auto_mode_fallback: bool = bool(cfg.get("auto_mode_fallback", True))
        self._auto_mode_warned = False
        self._alt_classifier: Optional[PixelClassifier] = None
        self._alt_mode: Optional[str] = None
        if self._auto_mode_fallback:
            alt_mode = (
                ClassifierMode.BLUE
                if classifier.mode == ClassifierMode.RED
                else ClassifierMode.RED
            )
            self._alt_classifier = PixelClassifier(alt_mode, cfg)
            self._alt_mode = alt_mode.value

        # Initial lock hardening to avoid selecting tiny nameplate fragments.
        self._initial_min_pixels: int = int(
            cfg.get("initial_lock_min_cluster_pixels", max(self._min_pixels, 8))
        )
        self._initial_center_penalty: float = float(
            cfg.get("initial_lock_center_penalty", 12.0)
        )
        self._initial_target_y_frac: float = float(
            cfg.get("initial_lock_target_y_frac", 0.42)
        )
        self._initial_text_aspect_threshold: float = float(
            cfg.get("initial_lock_text_aspect_threshold", 2.6)
        )

        # Use 4x cluster_radius for the tracking window.
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
        logger.info("Detection order: %s", self._detector_order)

        self._last_bmp_pos: Optional[Tuple[int, int]] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_mask: Optional[np.ndarray] = None

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

        had_previous_lock = self._last_bmp_pos is not None
        frame = capture_region(self._region)
        self._last_frame = frame
        self._last_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Tracking phase favors color first for more reliable movement/bite updates.
        if had_previous_lock:
            color_first = self._find_with_color_modes(frame)
            if color_first is not None:
                self._warn_if_slow(t0)
                return color_first

            template_fallback = self._find_with_template(frame)
            if template_fallback is not None:
                self._last_bmp_pos = template_fallback
                self._warn_if_slow(t0)
                return bitmap_to_screen_coords(
                    template_fallback[0], template_fallback[1], self._region
                )
        elif self._detector_order == "template_first":
            template_first = self._find_with_template(frame)
            if template_first is not None:
                self._last_bmp_pos = template_first
                self._warn_if_slow(t0)
                return bitmap_to_screen_coords(
                    template_first[0], template_first[1], self._region
                )

            color_fallback = self._find_with_color_modes(frame)
            if color_fallback is not None:
                self._warn_if_slow(t0)
                return color_fallback
        else:
            color_first = self._find_with_color_modes(frame)
            if color_first is not None:
                self._warn_if_slow(t0)
                return color_first

            template_fallback = self._find_with_template(frame)
            if template_fallback is not None:
                self._last_bmp_pos = template_fallback
                self._warn_if_slow(t0)
                return bitmap_to_screen_coords(
                    template_fallback[0], template_fallback[1], self._region
                )

        self._last_bmp_pos = None
        self._warn_if_slow(t0)
        return None

    def reset(self) -> None:
        """Clear the cached position before a new cast."""
        self._last_bmp_pos = None

    @property
    def last_bitmap_pos(self) -> Optional[Tuple[int, int]]:
        """Last bobber position in capture-region coordinates."""
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

    def _warn_if_slow(self, started_at: float) -> None:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if elapsed_ms > 200:
            logger.warning("BobberFinder scan took %.0fms (threshold: 200ms)", elapsed_ms)

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
                "Template fallback image not found: %s. Fallback is disabled.",
                template_path,
            )
            self._template_enabled = False
            return

        tpl = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if tpl is None or tpl.size == 0:
            logger.warning(
                "Template fallback failed to load image: %s. Fallback is disabled.",
                template_path,
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
        radius around it. Falls back to full mask when too few pixels remain.
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
            return mask

        return roi

    def _find_with_color_modes(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int]]:
        """Try primary color mode, then optional opposite color fallback mode."""
        mask = self._classifier.apply_mask(frame_bgr)
        self._last_mask = mask

        result = self._find_with_color_mask(mask, self._primary_mode)
        if result is not None:
            return result

        if self._auto_mode_fallback and self._alt_classifier is not None and self._alt_mode:
            alt_mask = self._alt_classifier.apply_mask(frame_bgr)
            alt_result = self._find_with_color_mask(alt_mask, self._alt_mode)
            if alt_result is not None:
                self._last_mask = alt_mask
                if not self._auto_mode_warned:
                    logger.warning(
                        "Configured detection.mode='%s' did not lock bobber, but '%s' did. "
                        "Consider switching detection.mode to '%s'.",
                        self._primary_mode,
                        self._alt_mode,
                        self._alt_mode,
                    )
                    self._auto_mode_warned = True
                return alt_result

        return None

    def _find_with_color_mask(self, mask: np.ndarray, mode_name: str) -> Optional[Tuple[int, int]]:
        """Find bobber from a classifier mask for one mode."""
        search_mask = self._restrict_to_last(mask)
        total = cv2.countNonZero(search_mask)

        if total == 0:
            return None

        if total > self._max_pixels:
            logger.warning(
                "Too many '%s' matching pixels (%s). Narrow HSV thresholds.",
                mode_name,
                total,
            )
            return None

        return self._find_best_component(search_mask)

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
        Run connected-components analysis and return screen coordinates
        of the centroid of the largest cluster that meets the minimum size.
        """
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        best_label = -1
        best_area = 0
        best_score = float("-inf")
        h, w = mask.shape[:2]

        # Initial acquisition (after cast) is where false locks happen most.
        is_initial_lock = self._last_bmp_pos is None
        min_area = self._initial_min_pixels if is_initial_lock else self._min_pixels
        target_x = w * 0.5
        target_y = h * self._initial_target_y_frac

        for label in range(1, num_labels):  # label 0 = background
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            score = float(area)
            if is_initial_lock:
                comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
                comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
                if comp_h > 0:
                    aspect = comp_w / float(comp_h)
                    if aspect >= self._initial_text_aspect_threshold and area < 50:
                        continue

                cx = float(centroids[label][0])
                cy = float(centroids[label][1])
                dx = (cx - target_x) / max(float(w), 1.0)
                dy = (cy - target_y) / max(float(h), 1.0)
                norm_dist = (dx * dx + dy * dy) ** 0.5
                score -= norm_dist * self._initial_center_penalty

            if score > best_score:
                best_score = score
                best_area = area
                best_label = label

        if best_label == -1:
            self._last_bmp_pos = None
            return None

        bmp_x = int(centroids[best_label][0])
        bmp_y = int(centroids[best_label][1])
        self._last_bmp_pos = (bmp_x, bmp_y)

        logger.debug("Bobber at bitmap (%s, %s), cluster area=%spx", bmp_x, bmp_y, best_area)

        return bitmap_to_screen_coords(bmp_x, bmp_y, self._region)
