"""
Screen capture utilities using mss (fast multi-monitor screenshot library).

The capture region is the center portion of the primary monitor — matching
the original FishingFun C# approach — to avoid scanning irrelevant UI edges.
"""

from __future__ import annotations

import numpy as np
import mss
import mss.tools
from typing import Tuple, Dict

from utils.logger import get_logger

logger = get_logger()

# Shared mss instance (thread-safe for reads)
_sct = mss.mss()


def get_primary_monitor() -> Dict:
    """Return the primary monitor geometry dict from mss."""
    return _sct.monitors[1]  # monitors[0] = all monitors combined, [1] = primary


def get_search_region(cfg: dict) -> Dict[str, int]:
    """
    Compute the absolute pixel region to scan based on config fractions.

    Config keys (all floats 0-1):
        left_frac, top_frac, width_frac, height_frac

    Returns:
        mss monitor dict: {"top": int, "left": int, "width": int, "height": int}
    """
    mon = get_primary_monitor()
    sw, sh = mon["width"], mon["height"]

    left   = int(sw * cfg["left_frac"])
    top    = int(sh * cfg["top_frac"])
    width  = int(sw * cfg["width_frac"])
    height = int(sh * cfg["height_frac"])

    return {"top": top, "left": left, "width": width, "height": height}


def capture_region(region: Dict[str, int]) -> np.ndarray:
    """
    Capture the given screen region and return it as a BGR numpy array
    (OpenCV-compatible format).

    Args:
        region: Dict with top/left/width/height keys.

    Returns:
        numpy array of shape (height, width, 3) in BGR.
    """
    screenshot = _sct.grab(region)
    # mss returns BGRA; drop the alpha channel → BGR
    frame = np.array(screenshot)[:, :, :3]
    return frame


def bitmap_to_screen_coords(bmp_x: int, bmp_y: int, region: Dict[str, int]) -> Tuple[int, int]:
    """
    Convert pixel coordinates within the captured bitmap back to absolute
    screen coordinates.

    Args:
        bmp_x:  X within the captured image.
        bmp_y:  Y within the captured image.
        region: The capture region dict used when grabbing the frame.

    Returns:
        (screen_x, screen_y) absolute coordinates.
    """
    return bmp_x + region["left"], bmp_y + region["top"]
