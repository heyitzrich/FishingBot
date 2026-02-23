"""
Logging setup with colored console output.
"""

import logging
from pathlib import Path
import sys
from typing import Optional

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


def setup_logger(
    name: str = "FishingBot",
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Create and configure the root logger.

    Args:
        name:  Logger name.
        level: Verbosity — DEBUG, INFO, WARNING, ERROR.

    Returns:
        Configured Logger instance.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    if HAS_COLORLOG:
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(message)s",
            datefmt="%H:%M:%S",
        )

    # Console handler (single instance)
    has_console = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "_fishingbot_console", False)
        for h in logger.handlers
    )
    if not has_console:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler._fishingbot_console = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    # Optional file handler (single instance per file path)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        normalized = str(log_file.resolve())
        has_file = any(
            isinstance(h, logging.FileHandler)
            and str(Path(h.baseFilename).resolve()) == normalized
            for h in logger.handlers
        )
        if not has_file:
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "FishingBot") -> logging.Logger:
    """Return the named logger (must call setup_logger first)."""
    return logging.getLogger(name)
