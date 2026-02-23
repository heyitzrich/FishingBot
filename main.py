"""
FishingBot — entry point.

Usage:
    python main.py                      # auto-uses %APPDATA%\\FishingBot\\config.yaml
    python main.py --mode blue          # override color detection mode
    python main.py --config custom.yaml # use a different config file
    python main.py --gui                # launch CustomTkinter GUI
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

import cv2

from bot.bobber_finder  import BobberFinder
from bot.bite_watcher   import BiteWatcher
from bot.fish_bot       import FishBot
from bot.pixel_classifier import ClassifierMode, PixelClassifier
from gui.debug_overlay  import build_debug_frame
from utils.app_paths     import get_session_log_path, is_frozen
from utils.config_manager import ConfigError, load_runtime_config
from utils.input        import find_wow_window
from utils.logger       import get_logger, setup_logger
from utils.screen       import get_search_region, init as init_screen

def main() -> None:
    parser = argparse.ArgumentParser(description="Python WoW Fishing Bot")
    parser.add_argument("--config",    default=None,
                        help="Path to config YAML (default: %%APPDATA%%\\FishingBot\\config.yaml)")
    parser.add_argument("--mode",      choices=["red", "blue"],
                        help="Override detection.mode from config")
    parser.add_argument("--gui",       action="store_true",
                        help="Launch the CustomTkinter GUI")
    parser.add_argument("--cli",       action="store_true",
                        help="Force legacy non-GUI mode (useful for debugging executable builds)")
    parser.add_argument("--no-window", action="store_true",
                        help="Disable the debug OpenCV window")
    args = parser.parse_args()

    try:
        config_path, cfg = load_runtime_config(args.config)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    log_file = get_session_log_path()
    setup_logger(level=cfg["debug"]["log_level"], log_file=log_file)
    log = get_logger()
    log.info("Config path: %s", config_path)
    log.info("Log file: %s", log_file)

    launch_gui = args.gui or (is_frozen() and not args.cli and not args.no_window)

    if launch_gui:
        from gui.app import App

        try:
            app = App(config_path=str(config_path), mode_override=args.mode, log_file=log_file)
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(1)
        app.mainloop()
        return

    # Resolve detection mode
    mode_str = (args.mode or cfg["detection"]["mode"]).strip().lower()
    mode     = ClassifierMode.RED if mode_str == "red" else ClassifierMode.BLUE
    log.info(f"Detection mode: {mode.value.upper()}")

    # Locate WoW window
    try:
        hwnd = find_wow_window(cfg["game"]["process_name"])
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(1)

    # Build component graph
    init_screen(hwnd)
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
