"""
Main fishing bot state machine.

Loop:
    IDLE → CASTING → SEARCHING → WATCHING → LOOTING → (back to CASTING)

  - On timeout (no bite in time):   recast immediately, increment timeouts stat
  - On bobber not found:            recast immediately, increment no_bobber stat
  - Every ~10 minutes (if configured): press maintenance keys between casts
    (e.g. apply fishing lure, delete junk from bag via WoW macro)
"""

from __future__ import annotations

import random
import threading
import time
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

from bot.bobber_finder import BobberFinder
from bot.bite_watcher  import BiteWatcher
from utils.input       import send_key, right_click
from utils.logger      import get_logger

logger = get_logger()


class BotState(Enum):
    IDLE      = auto()
    CASTING   = auto()
    SEARCHING = auto()
    WATCHING  = auto()
    LOOTING   = auto()
    STOPPED   = auto()


class SessionStats:
    """Tracks runtime statistics for the current fishing session."""

    def __init__(self) -> None:
        self.start_time = time.perf_counter()
        self.casts      = 0
        self.catches    = 0
        self.timeouts   = 0
        self.no_bobber  = 0

    @property
    def catch_rate(self) -> float:
        return (self.catches / self.casts * 100.0) if self.casts else 0.0

    @property
    def elapsed_minutes(self) -> float:
        return (time.perf_counter() - self.start_time) / 60.0

    @property
    def casts_per_hour(self) -> float:
        h = self.elapsed_minutes / 60.0
        return (self.casts / h) if h > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"Casts: {self.casts} | "
            f"Catches: {self.catches} ({self.catch_rate:.1f}%) | "
            f"Timeouts: {self.timeouts} | "
            f"No-bobber: {self.no_bobber} | "
            f"CPH: {self.casts_per_hour:.0f} | "
            f"Up: {self.elapsed_minutes:.1f}min"
        )


class FishBot:
    """
    Orchestrates the complete fishing loop.

    Args:
        hwnd:    WoW window handle (from utils.input.find_wow_window).
        finder:  BobberFinder instance.
        watcher: BiteWatcher instance.
        cfg:     Full parsed config dict (all sections).
    """

    # Hard limit for the bobber-search phase before giving up and recasting
    _SEARCH_TIMEOUT_S = 10.0

    def __init__(
        self,
        hwnd:    int,
        finder:  BobberFinder,
        watcher: BiteWatcher,
        cfg:     dict,
    ) -> None:
        self._hwnd    = hwnd
        self._finder  = finder
        self._watcher = watcher

        game_cfg   = cfg["game"]
        timing_cfg = cfg["timing"]

        self._cast_key:        str       = game_cfg["cast_key"]
        self._ten_keys:        List[str] = game_cfg.get("ten_min_keys") or []
        self._ten_min_s:       float     = timing_cfg["ten_min_interval_min"] * 60.0
        self._macro_1_wait_s:  float     = float(timing_cfg.get("macro_1_wait_s", 6.0))
        self._macro_2_wait_s:  float     = float(timing_cfg.get("macro_2_wait_s", 6.0))
        self._macro_default_wait_s: float = float(timing_cfg.get("macro_default_wait_s", 1.0))
        self._post_cast_s:     float     = timing_cfg["post_cast_wait_ms"] / 1000.0
        self._loot_delay_s:    float     = timing_cfg["loot_delay_ms"] / 1000.0
        self._post_loot_s:    float     = timing_cfg["post_loot_wait_ms"] / 1000.0
        self._jitter_ms:       int       = timing_cfg["random_jitter_ms"]

        self._state:           BotState                    = BotState.IDLE
        self._bobber_pos:      Optional[Tuple[int, int]]   = None
        self._last_ten_min:    float                       = 0.0  # trigger immediately on first loop
        self._stop_event                                   = threading.Event()

        self.stats:            SessionStats                = SessionStats()

        # Optional callback — set by main.py to trigger UI/overlay refresh
        self.on_state_change:  Optional[Callable[[BotState], None]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Run the bot loop in the calling thread (blocking). Catches Ctrl+C."""
        logger.info(
            f"FishBot starting — key={self._cast_key!r}, "
            f"ten-min-keys={self._ten_keys or 'disabled'}"
        )
        self._stop_event.clear()
        try:
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("Stopped by keyboard interrupt.")
        finally:
            self._set_state(BotState.STOPPED)
            logger.info(f"Session summary — {self.stats}")

    def stop(self) -> None:
        """Signal the bot to stop cleanly after the current iteration."""
        self._stop_event.set()

    @property
    def state(self) -> BotState:
        return self._state

    @property
    def bobber_pos(self) -> Optional[Tuple[int, int]]:
        """Most recent screen coordinates of the detected bobber."""
        return self._bobber_pos

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._maybe_do_ten_min_keys()
            self._do_cast()

            if self._stop_event.is_set():
                break

            bobber = self._do_search()
            if bobber is None:
                self.stats.no_bobber += 1
                logger.warning("Bobber not found after searching — recasting")
                continue

            bitten = self._do_watch()
            if not bitten:
                self.stats.timeouts += 1
                logger.info(
                    f"Timeout after {self._watcher._timeout:.0f}s — recasting | {self.stats}"
                )
                continue

            self._do_loot()

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _do_cast(self) -> None:
        self._set_state(BotState.CASTING)
        self._finder.reset()
        self._bobber_pos = None
        self.stats.casts += 1
        logger.info(f"[Cast #{self.stats.casts}] Pressing {self._cast_key!r}")
        send_key(self._hwnd, self._cast_key, jitter_ms=0)
        self._jittered_sleep(self._post_cast_s)

    def _do_search(self) -> Optional[Tuple[int, int]]:
        self._set_state(BotState.SEARCHING)
        deadline = time.perf_counter() + self._SEARCH_TIMEOUT_S

        while time.perf_counter() < deadline:
            if self._stop_event.is_set():
                return None
            pos = self._finder.find()
            if pos is not None:
                self._bobber_pos = pos
                logger.info(f"Bobber found at screen {pos}")
                return pos
            time.sleep(0.05)

        return None

    def _do_watch(self) -> bool:
        """
        Monitor the bobber until a bite is detected or the timeout expires.

        Returns True if a bite was confirmed, False on timeout.
        """
        self._set_state(BotState.WATCHING)

        if self._bobber_pos is None:
            return False

        self._watcher.reset(self._bobber_pos[1])

        while not self._watcher.is_timed_out():
            if self._stop_event.is_set():
                return False

            pos = self._finder.find()

            if pos is None:
                # Bobber vanished — treat as bite (it sank or was looted by lag)
                logger.debug("Bobber lost during watch — treating as bite")
                return True

            self._bobber_pos = pos

            if self._watcher.update(pos[1]):
                return True

            time.sleep(0.033)   # ~30 fps polling rate

        return False

    def _do_loot(self) -> None:
        self._set_state(BotState.LOOTING)
        self.stats.catches += 1

        pos = self._bobber_pos
        logger.info(f"Looting catch #{self.stats.catches} at {pos} | {self.stats}")

        self._jittered_sleep(self._loot_delay_s)

        if pos is not None:
            right_click(self._hwnd, pos[0], pos[1], jitter_ms=self._jitter_ms)

        # Wait for loot animation/pickup to finish before next cast
        self._jittered_sleep(self._post_loot_s)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_do_ten_min_keys(self) -> None:
        """Press maintenance keys if the 10-minute interval has elapsed."""
        if not self._ten_keys:
            return
        now = time.perf_counter()
        if now - self._last_ten_min < self._ten_min_s:
            return

        logger.info(f"10-min maintenance keys: {self._ten_keys}")
        for idx, key in enumerate(self._ten_keys):
            send_key(self._hwnd, key, jitter_ms=self._jitter_ms)
            wait_s = self._get_macro_wait_s(idx)
            if wait_s > 0:
                logger.info(
                    "Waiting %.1fs after Macro %s (%r).",
                    wait_s,
                    idx + 1,
                    key,
                )
                time.sleep(wait_s)

        # Start the next interval after maintenance sequence fully completes.
        self._last_ten_min = time.perf_counter()

    def _get_macro_wait_s(self, idx: int) -> float:
        """Return per-macro delay in seconds after pressing each maintenance key."""
        if idx == 0:
            return max(0.0, self._macro_1_wait_s)
        if idx == 1:
            return max(0.0, self._macro_2_wait_s)
        return max(0.0, self._macro_default_wait_s)

    def _jittered_sleep(self, base_s: float) -> None:
        """Sleep for base_s plus a random anti-pattern jitter."""
        jitter = random.uniform(0.0, self._jitter_ms / 1000.0)
        time.sleep(base_s + jitter)

    def _set_state(self, state: BotState) -> None:
        self._state = state
        if self.on_state_change:
            self.on_state_change(state)
        logger.debug(f"State → {state.name}")
