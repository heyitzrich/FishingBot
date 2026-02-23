"""Application path helpers for source and frozen (PyInstaller) runtimes."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

APP_NAME = "FishingBot"


def is_frozen() -> bool:
    """True when running from a PyInstaller executable."""
    return bool(getattr(sys, "frozen", False))


def get_resource_root() -> Path:
    """
    Return directory that contains bundled resources.

    - Source run: repository root.
    - Frozen run: PyInstaller extraction directory.
    """
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[1]


def get_user_data_dir() -> Path:
    """Per-user writable data directory."""
    appdata = os.getenv("APPDATA")
    if appdata:
        return Path(appdata) / APP_NAME
    return Path.home() / f".{APP_NAME.lower()}"


def get_user_config_path() -> Path:
    return get_user_data_dir() / "config.yaml"


def get_logs_dir() -> Path:
    return get_user_data_dir() / "logs"


def get_session_log_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return get_logs_dir() / f"fishingbot_{stamp}.log"


def get_default_config_path() -> Path:
    return get_resource_root() / "config.default.yaml"


def get_default_templates_dir() -> Path:
    return get_resource_root() / "templates"
