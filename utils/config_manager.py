"""Config bootstrap, migration, validation, and save/load helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from utils.app_paths import (
    get_default_config_path,
    get_default_templates_dir,
    get_user_config_path,
    get_user_data_dir,
)


class ConfigError(RuntimeError):
    """Raised when a config file is missing, invalid, or cannot be parsed."""


def resolve_config_path(config_arg: Optional[str]) -> Path:
    """Resolve runtime config path. Defaults to %APPDATA%/FishingBot/config.yaml."""
    if config_arg:
        return Path(config_arg).expanduser().resolve()
    return get_user_config_path()


def ensure_runtime_files(config_path: Path) -> None:
    """
    Ensure runtime config exists.

    If a default config is available and runtime config is missing, copy defaults
    and seed a templates directory for first-run users.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        return

    default_cfg_path = get_default_config_path()
    if not default_cfg_path.exists():
        raise ConfigError(
            f"Default config not found at {default_cfg_path}. "
            "Cannot initialize runtime config."
        )
    shutil.copyfile(default_cfg_path, config_path)

    default_templates = get_default_templates_dir()
    user_templates = get_user_data_dir() / "templates"
    if default_templates.exists():
        user_templates.mkdir(parents=True, exist_ok=True)
        for src in default_templates.glob("*"):
            if src.is_file():
                dst = user_templates / src.name
                if not dst.exists():
                    shutil.copyfile(src, dst)


def load_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file {path}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return parsed


def load_default_config() -> Dict[str, Any]:
    default_path = get_default_config_path()
    return load_config_file(default_path)


def _deep_merge(defaults: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    for key, value in incoming.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _require_section(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    value = cfg.get(section)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid '{section}' section in config.")
    return value


def _require_str(cfg: Dict[str, Any], key: str) -> str:
    value = cfg.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing or invalid '{key}' value.")
    return value.strip()


def _require_number(cfg: Dict[str, Any], key: str, min_value: Optional[float] = None) -> float:
    value = cfg.get(key)
    if not isinstance(value, (int, float)):
        raise ConfigError(f"Missing or invalid numeric '{key}' value.")
    num = float(value)
    if min_value is not None and num < min_value:
        raise ConfigError(f"'{key}' must be >= {min_value}.")
    return num


def _require_bool(cfg: Dict[str, Any], key: str) -> bool:
    value = cfg.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"Missing or invalid boolean '{key}' value.")
    return value


def validate_config(cfg: Dict[str, Any]) -> None:
    game = _require_section(cfg, "game")
    detection = _require_section(cfg, "detection")
    bite = _require_section(cfg, "bite")
    timing = _require_section(cfg, "timing")
    debug = _require_section(cfg, "debug")

    _require_str(game, "process_name")
    _require_str(game, "cast_key")

    keys = game.get("ten_min_keys") or []
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        raise ConfigError("'game.ten_min_keys' must be a list of strings (or null).")

    mode = _require_str(detection, "mode").lower()
    if mode not in {"red", "blue"}:
        raise ConfigError("'detection.mode' must be 'red' or 'blue'.")
    detector_order = _require_str(detection, "detector_order").lower()
    if detector_order not in {"template_first", "color_first"}:
        raise ConfigError(
            "'detection.detector_order' must be 'template_first' or 'color_first'."
        )
    _require_bool(detection, "auto_mode_fallback")

    search_region = _require_section(detection, "search_region")
    for name in ("left_frac", "top_frac", "width_frac", "height_frac"):
        val = _require_number(search_region, name, min_value=0.0)
        if val > 1.0:
            raise ConfigError(f"'detection.search_region.{name}' must be <= 1.0.")

    _require_number(detection, "min_cluster_pixels", min_value=1.0)
    _require_number(detection, "max_total_pixels", min_value=1.0)
    _require_number(detection, "cluster_radius", min_value=1.0)

    _require_number(bite, "strike_value", min_value=1.0)
    _require_number(bite, "timeout_seconds", min_value=1.0)
    _require_number(bite, "position_update_ms", min_value=1.0)

    _require_number(timing, "post_cast_wait_ms", min_value=0.0)
    _require_number(timing, "loot_delay_ms", min_value=0.0)
    _require_number(timing, "post_loot_wait_ms", min_value=0.0)
    _require_number(timing, "ten_min_interval_min", min_value=0.0)
    _require_number(timing, "macro_1_wait_s", min_value=0.0)
    _require_number(timing, "macro_2_wait_s", min_value=0.0)
    _require_number(timing, "macro_default_wait_s", min_value=0.0)
    _require_number(timing, "random_jitter_ms", min_value=0.0)

    log_level = _require_str(debug, "log_level").upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError("debug.log_level must be one of DEBUG/INFO/WARNING/ERROR/CRITICAL.")


def resolve_runtime_paths(cfg: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """
    Resolve path-like config values for runtime use only.

    Current behavior:
      - detection.template_fallback.template_path is made absolute using the
        config directory when it is relative.
    """
    detection = cfg.get("detection")
    if not isinstance(detection, dict):
        return cfg

    template_cfg = detection.get("template_fallback")
    if not isinstance(template_cfg, dict):
        return cfg

    template_path_raw = template_cfg.get("template_path")
    if not isinstance(template_path_raw, str) or not template_path_raw.strip():
        return cfg

    template_path = Path(template_path_raw)
    if not template_path.is_absolute():
        template_path = (config_path.parent / template_path).resolve()
    template_cfg["template_path"] = str(template_path)
    return cfg


def load_runtime_config(config_arg: Optional[str]) -> tuple[Path, Dict[str, Any]]:
    """
    Resolve config path, bootstrap default file if needed, then validate config.

    Missing keys are auto-filled from bundled defaults and written back to disk.
    """
    config_path = resolve_config_path(config_arg)
    ensure_runtime_files(config_path)

    defaults = load_default_config()
    current = load_config_file(config_path)
    merged = _deep_merge(defaults, current)
    validate_config(merged)
    save_config(config_path, merged)
    resolved = resolve_runtime_paths(merged, config_path)
    return config_path, resolved


def save_config(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
