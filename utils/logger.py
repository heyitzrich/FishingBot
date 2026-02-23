"""
Logging setup with colored console output.
"""

import logging
import sys

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


def setup_logger(name: str = "FishingBot", level: str = "INFO") -> logging.Logger:
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

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

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

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger(name: str = "FishingBot") -> logging.Logger:
    """Return the named logger (must call setup_logger first)."""
    return logging.getLogger(name)
