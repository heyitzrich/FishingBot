"""
Screen capture utilities using Win32 PrintWindow API.

Captures the WoW window directly via its device context, so the bot works
even when WoW is behind other windows (but NOT minimized — the GPU stops
rendering minimized windows).
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes

import numpy as np
import win32gui
import win32ui
import win32con
from typing import Tuple, Dict

from utils.logger import get_logger

logger = get_logger()

# Module-level hwnd — set once via init() from main.py
_hwnd: int = 0

# PrintWindow flags
PW_CLIENTONLY = 0x1
PW_RENDERFULLCONTENT = 0x2


def init(hwnd: int) -> None:
    """Store the WoW window handle for capture functions."""
    global _hwnd
    _hwnd = hwnd


def get_search_region(cfg: dict) -> Dict[str, int]:
    """
    Compute the pixel region to scan based on config fractions of the
    WoW window's client area.

    Config keys (all floats 0-1):
        left_frac, top_frac, width_frac, height_frac

    Returns:
        Dict: {"top": int, "left": int, "width": int, "height": int}
        Values are relative to the window client area.
    """
    rect = win32gui.GetClientRect(_hwnd)
    sw, sh = rect[2], rect[3]

    left   = int(sw * cfg["left_frac"])
    top    = int(sh * cfg["top_frac"])
    width  = int(sw * cfg["width_frac"])
    height = int(sh * cfg["height_frac"])

    return {"top": top, "left": left, "width": width, "height": height}


def capture_region(region: Dict[str, int]) -> np.ndarray:
    """
    Capture the WoW window client area via PrintWindow and crop to region.

    Args:
        region: Dict with top/left/width/height keys (client-area relative).

    Returns:
        numpy array of shape (height, width, 3) in BGR.
    """
    rect = win32gui.GetClientRect(_hwnd)
    cw, ch = rect[2], rect[3]

    if cw == 0 or ch == 0:
        return np.zeros((region["height"], region["width"], 3), dtype=np.uint8)

    # Create device contexts and bitmap
    hwnd_dc = win32gui.GetWindowDC(_hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()

    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, cw, ch)
    save_dc.SelectObject(bitmap)

    # Capture via PrintWindow (works for background windows)
    ctypes.windll.user32.PrintWindow(
        _hwnd, save_dc.GetSafeHdc(),
        PW_CLIENTONLY | PW_RENDERFULLCONTENT
    )

    # Convert bitmap to numpy array
    bmp_bits = bitmap.GetBitmapBits(True)
    frame = np.frombuffer(bmp_bits, dtype=np.uint8).reshape((ch, cw, 4))

    # Cleanup GDI resources
    win32gui.DeleteObject(bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(_hwnd, hwnd_dc)

    # Crop to the search region and convert BGRA → BGR
    top = region["top"]
    left = region["left"]
    bottom = top + region["height"]
    right = left + region["width"]
    cropped = frame[top:bottom, left:right, :3]

    return cropped.copy()


def bitmap_to_screen_coords(bmp_x: int, bmp_y: int, region: Dict[str, int]) -> Tuple[int, int]:
    """
    Convert pixel coordinates within the captured bitmap back to absolute
    screen coordinates.

    Args:
        bmp_x:  X within the captured image.
        bmp_y:  Y within the captured image.
        region: The capture region dict (client-area relative).

    Returns:
        (screen_x, screen_y) absolute screen coordinates.
    """
    # bitmap coords → client-area coords
    client_x = bmp_x + region["left"]
    client_y = bmp_y + region["top"]
    # client-area coords → screen coords
    screen_x, screen_y = win32gui.ClientToScreen(_hwnd, (client_x, client_y))
    return screen_x, screen_y
