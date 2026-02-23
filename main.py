"""
FishingBot — entry point.

Usage:
    python main.py
    python main.py --mode blue          # override color detection mode
    python main.py --config custom.yaml # use a different config file
    python main.py --no-window          # headless mode, no OpenCV window

The debug window (enabled via config.yaml debug.show_window) displays:
  - The captured screen region with the color mask overlaid in red
  - A white corner-bracket reticle around the detected bobber
  - Current bot state (color-coded)
  - Live session statistics (casts, catches, catch rate, CPH, uptime)

The bot runs in a background thread so the OpenCV window stays responsive.
WoW must be in Windowed Fullscreen (borderless) mode for background operation.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from bot.bobber_finder  import BobberFinder
from bot.bite_watcher   import BiteWatcher
from bot.fish_bot       import FishBot, BotState
from bot.pixel_classifier import ClassifierMode, PixelClassifier
from utils.input        import find_wow_window
from utils.logger       import get_logger, setup_logger
from utils.screen       import get_search_region

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with p.open("r") as f:
        return yaml.safe_load(f)

# ------------------------------------------------------------------
# Debug window
# ------------------------------------------------------------------

# State label colors in BGR
_STATE_COLORS = {
    BotState.IDLE:      (200, 200, 200),
    BotState.CASTING:   (  0, 200, 255),
    BotState.SEARCHING: (255, 200,   0),
    BotState.WATCHING:  (  0, 220,   0),
    BotState.LOOTING:   (  0, 100, 255),
    BotState.STOPPED:   (100, 100, 100),
}


def build_debug_frame(bot: FishBot, finder: BobberFinder) -> Optional[np.ndarray]:
    """
    Compose a debug display frame from the bot's most recent capture.

    Layers (bottom to top):
      1. Raw captured BGR frame
      2. Semi-transparent red tint where the color mask fired
      3. White L-bracket reticle around the bobber centroid
      4. State badge (top-left)
      5. Session stats (bottom)

    Returns None if no frame has been captured yet.
    """
    frame = finder.last_frame
    mask  = finder.last_mask
    if frame is None or mask is None:
        return None

    display = frame.copy()

    # -- Color mask overlay ------------------------------------------
    overlay = np.zeros_like(display)
    overlay[mask > 0] = (0, 0, 180)   # red in BGR
    cv2.addWeighted(overlay, 0.35, display, 1.0, 0, display)

    # -- Bobber reticle ----------------------------------------------
    bmp_pos = finder.last_bitmap_pos
    if bmp_pos is not None:
        cx, cy = bmp_pos
        arm = 20    # length of each reticle arm
        stub = 8    # length of the short perpendicular leg
        t    = 2    # line thickness
        col  = (255, 255, 255)

        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            x, y = cx + dx * arm, cy + dy * arm
            cv2.line(display, (x, y), (x - dx * stub, y),        col, t, cv2.LINE_AA)
            cv2.line(display, (x, y), (x,              y - dy * stub), col, t, cv2.LINE_AA)

    # -- State badge (top-left) --------------------------------------
    state       = bot.state
    state_color = _STATE_COLORS.get(state, (255, 255, 255))
    label       = f"  {state.name}"
    cv2.rectangle(display, (0, 0), (200, 26), (20, 20, 20), -1)
    cv2.putText(display, label, (4, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 1, cv2.LINE_AA)

    # -- Session stats (bottom) --------------------------------------
    stats_lines = str(bot.stats).split(" | ")
    bar_h = 20
    for i, line in enumerate(stats_lines):
        y = display.shape[0] - 8 - i * bar_h
        cv2.rectangle(display, (0, y - bar_h + 4), (display.shape[1], y + 4),
                      (20, 20, 20), -1)
        cv2.putText(display, line, (6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)

    return display

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Python WoW Fishing Bot")
    parser.add_argument("--config",    default="config.yaml",
                        help="Path to config YAML (default: config.yaml)")
    parser.add_argument("--mode",      choices=["red", "blue"],
                        help="Override detection.mode from config")
    parser.add_argument("--no-window", action="store_true",
                        help="Disable the debug OpenCV window")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logger(level=cfg["debug"]["log_level"])
    log = get_logger()

    # Resolve detection mode
    mode_str = args.mode or cfg["detection"]["mode"]
    mode     = ClassifierMode.RED if mode_str == "red" else ClassifierMode.BLUE
    log.info(f"Detection mode: {mode.value.upper()}")

    # Locate WoW window
    try:
        hwnd = find_wow_window(cfg["game"]["process_name"])
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(1)

    # Build component graph
    region     = get_search_region(cfg["detection"]["search_region"])
    classifier = PixelClassifier(mode, cfg["detection"])
    finder     = BobberFinder(classifier, region, cfg["detection"])
    watcher    = BiteWatcher(cfg["bite"], on_bite=lambda: None)
    bot        = FishBot(hwnd, finder, watcher, cfg)

    log.info(
        f"Search region: left={region['left']} top={region['top']} "
        f"w={region['width']} h={region['height']}"
    )
    log.info("WoW must be in Windowed Fullscreen mode for background operation.")

    # Run bot in a daemon thread
    bot_thread = threading.Thread(target=bot.start, daemon=True, name="FishBot")
    bot_thread.start()

    # Debug window (must stay in the main thread on Windows)
    show_window = cfg["debug"]["show_window"] and not args.no_window
    refresh_ms  = cfg["debug"]["window_refresh_ms"]

    if show_window:
        log.info("Debug window open. Press 'q' inside the window to stop.")
        while bot_thread.is_alive():
            frame = build_debug_frame(bot, finder)
            if frame is not None:
                cv2.imshow("FishingBot", frame)
            key = cv2.waitKey(refresh_ms) & 0xFF
            if key == ord("q"):
                log.info("'q' pressed — stopping bot.")
                bot.stop()
                break
        cv2.destroyAllWindows()
    else:
        log.info("Headless mode. Press Ctrl+C to stop.")
        try:
            bot_thread.join()
        except KeyboardInterrupt:
            bot.stop()

    bot_thread.join(timeout=5)
    log.info("FishingBot exited.")


if __name__ == "__main__":
    main()
