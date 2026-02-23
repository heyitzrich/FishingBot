"""
Bite detection by monitoring the fishing bobber's Y-position over time.

Uses the median of recent Y-position deltas to declare a bite, which makes
detection robust against single-frame noise or brief camera jitter — the same
strategy as the original C# project.

Improvement over the original:
  - Rolling window of configurable size (not just a list that grows forever)
  - Median computed over deltas from the initial position (not adjacent pairs),
    so a sustained drop reads correctly even if the bobber oscillates slightly
  - Configurable via YAML; no hardcoded UI sliders needed
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Optional

from utils.logger import get_logger

logger = get_logger()

_WINDOW_SIZE = 12  # number of Y samples to keep in the rolling median window


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
        logger.debug(
            f"BiteWatcher reset — initial_y={initial_y}, "
            f"strike={self._strike_value}px, timeout={self._timeout}s"
        )

    def update(self, current_y: int) -> bool:
        """
        Feed a new bobber Y-position reading.

        Rate-limited by position_update_ms to avoid hammering the median calc.

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

        if len(self._y_window) < 2:
            return False

        drop = self._median_drop()
        logger.debug(
            f"BiteWatcher y={current_y} drop={drop:.1f}px "
            f"({len(self._y_window)} samples, threshold={self._strike_value}px)"
        )

        if drop >= self._strike_value:
            logger.info(
                f"Bite detected! Median drop={drop:.1f}px "
                f"(threshold={self._strike_value}px, "
                f"elapsed={self.elapsed:.1f}s)"
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
