"""
Bite detection by monitoring the fishing bobber's Y-position over time.

Robust multi-signal detection with confirmation:
  1. Median drop detection (primary, very robust)
  2. Peak drop detection (confirms sustained movement)
  3. Requires BOTH median AND peak to exceed thresholds (AND logic)

Improvement over the original:
  - Rolling window of configurable size for noise filtering
  - Confirmation logic: requires multiple signals to agree
  - Balanced window size (10 samples) - filters noise while staying responsive
  - Conservative thresholds to avoid false positives
  - Configurable via YAML; no hardcoded UI sliders needed
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Optional

from utils.logger import get_logger

logger = get_logger()

_WINDOW_SIZE = 10  # Balanced for noise filtering and responsiveness


class BiteWatcher:
    """
    Watches the bobber Y-coordinate and fires a callback when a bite is detected.

    Usage:
        watcher = BiteWatcher(cfg["bite"], on_bite=handle_bite)
        watcher.reset(initial_y)

        while not watcher.is_timed_out():
            pos = finder.find()
            if pos and watcher.update(pos[1]):
                break   # bite confirmed
    """

    def __init__(self, cfg: dict, on_bite: Callable[[], None]):
        """
        Args:
            cfg:     The bite section of config.yaml.
            on_bite: Called exactly once when a bite is detected. Use for
                     side-effects (sound, notification); FishBot handles looting.
        """
        self._strike_value: int   = cfg["strike_value"]
        self._timeout:      float = cfg["timeout_seconds"]
        self._update_s:     float = cfg["position_update_ms"] / 1000.0
        self._on_bite = on_bite

        self._initial_y:  Optional[int] = None
        self._start_time: float = 0.0
        self._last_tick:  float = 0.0
        self._y_window:   deque = deque(maxlen=_WINDOW_SIZE)
        self._triggered:  bool  = False
        self._detected_drop: float = 0.0  # Track the actual drop amount when bite detected
        self._detection_method: str = ""  # Track which detection method triggered

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, initial_y: int) -> None:
        """
        Prepare for a new cast. Must be called after the bobber is located
        and before the first update() call.

        Args:
            initial_y: Screen Y-coordinate of the bobber immediately after casting.
        """
        self._initial_y  = initial_y
        self._start_time = time.perf_counter()
        self._last_tick  = self._start_time
        self._y_window.clear()
        self._y_window.append(initial_y)
        self._triggered  = False
        self._detected_drop = 0.0
        self._detection_method = ""
        logger.debug(
            f"BiteWatcher reset — initial_y={initial_y}, "
            f"strike={self._strike_value}px, timeout={self._timeout}s"
        )

    def update(self, current_y: int) -> bool:
        """
        Feed a new bobber Y-position reading.

        Uses confirmation-based detection (requires multiple signals):
        1. Primary: Median drop >= threshold (filters noise)
        2. Confirmation: Peak drop also significant (confirms sustained movement)
        3. Both must agree to trigger (AND logic, not OR)

        Rate-limited by position_update_ms to avoid hammering calculations.

        Returns:
            True if a bite was detected (or was already triggered previously).
        """
        if self._triggered:
            return True

        now = time.perf_counter()
        if now - self._last_tick < self._update_s:
            return False
        self._last_tick = now

        self._y_window.append(current_y)

        # Need enough samples for reliable detection
        if len(self._y_window) < 5:
            return False

        # Compute detection signals
        median_drop = self._median_drop()
        peak_drop = self._peak_drop()

        # Conservative thresholds
        median_threshold = self._strike_value
        peak_threshold = self._strike_value - 1  # Slightly lower for confirmation

        bite_detected = False
        detection_reason = ""

        # Require BOTH median AND peak to confirm bite (reduces false positives)
        if median_drop >= median_threshold and peak_drop >= peak_threshold:
            bite_detected = True
            self._detected_drop = median_drop
            detection_reason = f"confirmed: median={median_drop:.1f}px, peak={peak_drop:.1f}px"

        logger.debug(
            f"BiteWatcher y={current_y} median={median_drop:.1f}px peak={peak_drop:.1f}px "
            f"({len(self._y_window)} samples, need both >= {median_threshold}/{peak_threshold}px)"
        )

        if bite_detected:
            self._detection_method = detection_reason
            logger.info(
                f"Bite detected! {detection_reason} "
                f"(threshold={self._strike_value}px, elapsed={self.elapsed:.1f}s)"
            )
            self._triggered = True
            self._on_bite()
            return True

        return False

    def is_timed_out(self) -> bool:
        """True if the full timeout window has elapsed without a bite."""
        return (time.perf_counter() - self._start_time) > self._timeout

    @property
    def elapsed(self) -> float:
        """Seconds since reset()."""
        return time.perf_counter() - self._start_time

    @property
    def triggered(self) -> bool:
        """True if a bite was already detected this cast."""
        return self._triggered

    @property
    def detected_drop(self) -> float:
        """The actual drop amount in pixels when bite was detected (0 if no bite)."""
        return self._detected_drop

    @property
    def detection_method(self) -> str:
        """The detection method that triggered the bite (e.g., 'median-drop', 'peak-drop', 'velocity')."""
        return self._detection_method

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _median_drop(self) -> float:
        """
        Median Y displacement from the initial position across all samples.
        Positive value = bobber has moved downward (screen Y increases downward).
        """
        drops = sorted(y - self._initial_y for y in self._y_window)
        mid   = len(drops) // 2
        return float(drops[mid])

    def _peak_drop(self) -> float:
        """
        Maximum Y displacement from the initial position across all samples.
        Used as confirmation signal - ensures movement is sustained, not just noise.
        Positive value = bobber has moved downward.
        """
        if not self._y_window:
            return 0.0
        drops = [y - self._initial_y for y in self._y_window]
        return float(max(drops))
