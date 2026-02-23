"""CustomTkinter application for FishingBot."""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Optional

import cv2
import customtkinter as ctk
import yaml

from bot.bite_watcher import BiteWatcher
from bot.bobber_finder import BobberFinder
from bot.fish_bot import FishBot
from bot.pixel_classifier import ClassifierMode, PixelClassifier
from gui.frames import (
    ConfigEditorFrame,
    ControlsFrame,
    LogViewerFrame,
    PreviewFrame,
    StatsFrame,
)
from gui.log_handler import QueueHandler
from utils.input import find_wow_window
from utils.logger import get_logger, setup_logger
from utils.screen import get_search_region, init as init_screen


class App(ctk.CTk):
    """Main GUI application."""

    def __init__(self, config_path: str = "config.yaml", mode_override: Optional[str] = None):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("FishingBot")
        self.geometry("1100x700")
        self.minsize(980, 620)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._config_path = Path(config_path)
        self._cfg = self._load_config(self._config_path)
        if mode_override:
            self._cfg["detection"]["mode"] = mode_override

        setup_logger(level=self._cfg["debug"]["log_level"])
        self._log = get_logger()

        self._log_queue: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=5000)
        self._queue_handler = QueueHandler(self._log_queue)
        self._queue_handler.setLevel(logging.DEBUG)
        self._queue_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        self._log.addHandler(self._queue_handler)

        self._bot: Optional[FishBot] = None
        self._bot_thread: Optional[threading.Thread] = None
        self._finder: Optional[BobberFinder] = None
        self._preview_enabled = True
        self._closing = False

        self._build_layout()
        self.config_editor.load_config(self._cfg)
        self.controls.set_state("IDLE")

        self.after(100, self._poll)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(self)
        right = ctk.CTkFrame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)

        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=3)
        left.grid_rowconfigure(1, weight=2)

        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)

        self.preview = PreviewFrame(left)
        self.preview.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        self.log_viewer = LogViewerFrame(left)
        self.log_viewer.grid(row=1, column=0, sticky="nsew")

        self.controls = ControlsFrame(
            right,
            on_start=self.start_bot,
            on_stop=self.stop_bot,
            on_preview_toggle=self._set_preview_enabled,
            preview_enabled=self._preview_enabled,
        )
        self.controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.stats = StatsFrame(right)
        self.stats.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        self.config_editor = ConfigEditorFrame(right, on_save=self._save_config)
        self.config_editor.grid(row=2, column=0, sticky="nsew")

    def start_bot(self) -> None:
        if self._bot_thread and self._bot_thread.is_alive():
            return
        try:
            mode_str = str(self._cfg["detection"]["mode"]).strip().lower()
            mode = ClassifierMode.RED if mode_str == "red" else ClassifierMode.BLUE

            hwnd = find_wow_window(self._cfg["game"]["process_name"])
            init_screen(hwnd)
            region = get_search_region(self._cfg["detection"]["search_region"])

            classifier = PixelClassifier(mode, self._cfg["detection"])
            finder = BobberFinder(classifier, region, self._cfg["detection"])
            watcher = BiteWatcher(self._cfg["bite"], on_bite=lambda: None)
            bot = FishBot(hwnd, finder, watcher, self._cfg)

            self._finder = finder
            self._bot = bot
            self._bot_thread = threading.Thread(target=bot.start, daemon=True, name="FishBot")
            self._bot_thread.start()

            self.controls.set_running(True)
            self._log.info("Detection mode: %s", mode.value.upper())
            self._log.info(
                "Started bot. Search region: left=%s top=%s w=%s h=%s",
                region["left"],
                region["top"],
                region["width"],
                region["height"],
            )
        except Exception as exc:
            self._log.error("Failed to start bot: %s", exc)
            self.controls.set_running(False)

    def stop_bot(self) -> None:
        if self._bot is None:
            return
        self._bot.stop()
        self._log.info("Stop requested.")

    def capture_template(self) -> None:
        if self._finder is None:
            self._log.warning("Cannot capture template: bot is not initialized.")
            return

        frame = self._finder.last_frame
        bmp_pos = self._finder.last_bitmap_pos
        if frame is None or bmp_pos is None:
            self._log.warning(
                "Cannot capture template yet: wait until the bobber is detected."
            )
            return

        template_cfg = self._cfg.setdefault("detection", {}).setdefault(
            "template_fallback", {}
        )
        capture_size = int(template_cfg.get("capture_size_px", 84))
        capture_size = max(24, min(capture_size, 220))
        half_size = capture_size // 2

        h, w = frame.shape[:2]
        cx, cy = bmp_pos
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)
        if (x2 - x1) < 24 or (y2 - y1) < 24:
            self._log.warning("Template crop is too small; try again when bobber is centered.")
            return

        crop = frame[y1:y2, x1:x2].copy()
        template_path_raw = str(
            template_cfg.get("template_path", "templates/bobber_blue.png")
        ).strip()
        out_path = Path(template_path_raw)
        if not out_path.is_absolute():
            out_path = self._config_path.parent / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(out_path), crop):
            self._log.error("Failed to save template image: %s", out_path)
            return

        template_cfg["enabled"] = True
        self._save_config(self._cfg)
        self._log.info(
            "Template captured (%sx%s): %s. Restart bot to load updated template fallback.",
            crop.shape[1],
            crop.shape[0],
            out_path,
        )

    def _poll(self) -> None:
        if self._closing:
            return

        self._drain_logs()

        if self._bot is not None:
            self.controls.set_state(self._bot.state)
            self.stats.update_stats(self._bot.stats)

        if self._preview_enabled and self._bot is not None and self._finder is not None:
            frame = self._get_preview_frame()
            self.preview.update_frame(frame)

        running = bool(self._bot_thread and self._bot_thread.is_alive())
        self.controls.set_running(running)

        self.after(100, self._poll)

    def _refresh_preview_once(self) -> None:
        if self._bot is None or self._finder is None:
            return
        frame = self._get_preview_frame()
        self.preview.update_frame(frame)

    def _get_preview_frame(self):
        if self._finder is None or self._finder.last_frame is None:
            return None
        # Raw captured preview with only a minimal detection marker.
        frame = self._finder.last_frame.copy()
        bmp_pos = self._finder.last_bitmap_pos
        if bmp_pos is not None:
            cx, cy = bmp_pos
            half = 14
            h, w = frame.shape[:2]
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w - 1, cx + half)
            y2 = min(h - 1, cy + half)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def _drain_logs(self) -> None:
        for _ in range(50):
            try:
                level, line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_viewer.append(line, level)

    def _save_config(self, cfg: dict) -> None:
        with self._config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self._cfg = cfg
        setup_logger(level=self._cfg["debug"]["log_level"])
        self._log.info("Config saved to %s. Changes apply on next start.", self._config_path)

    def _set_preview_enabled(self, enabled: bool) -> None:
        self._preview_enabled = enabled
        self.preview.set_paused(not enabled)
        if enabled:
            self._refresh_preview_once()
        self._log.info("Live preview %s.", "enabled" if enabled else "disabled")

    def _on_close(self) -> None:
        self._closing = True
        self.stop_bot()

        if self._bot_thread and self._bot_thread.is_alive():
            self._bot_thread.join(timeout=5)

        if self._queue_handler in self._log.handlers:
            self._log.removeHandler(self._queue_handler)

        self.destroy()

    @staticmethod
    def _load_config(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
