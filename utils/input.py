"""
Background-capable input simulation using Win32 PostMessage.

PostMessage queues messages directly to the window's message queue without
requiring the window to be focused. This allows the bot to run while WoW
is behind other windows.

Requirements:
    - WoW must run in Windowed Fullscreen (borderless) mode
    - WoW must NOT be minimized (screen capture will fail if minimized)
"""

from __future__ import annotations

import random
import time
from typing import Optional

import win32api
import win32con
import win32gui

from utils.logger import get_logger

logger = get_logger()

# Virtual key code map: string → VK constant
_VK_MAP: dict = {
    # Number row
    "0": 0x30, "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34,
    "5": 0x35, "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39,
    # Function keys
    "F1":  win32con.VK_F1,  "F2":  win32con.VK_F2,  "F3":  win32con.VK_F3,
    "F4":  win32con.VK_F4,  "F5":  win32con.VK_F5,  "F6":  win32con.VK_F6,
    "F7":  win32con.VK_F7,  "F8":  win32con.VK_F8,  "F9":  win32con.VK_F9,
    "F10": win32con.VK_F10, "F11": win32con.VK_F11, "F12": win32con.VK_F12,
    # Letters A-Z
    **{chr(c): c for c in range(ord("A"), ord("Z") + 1)},
}


def find_wow_window(process_name: str) -> int:
    """
    Find the WoW window handle by enumerating visible windows.

    Args:
        process_name: From config (e.g. "Wow", "WowClassic"). Used as hint
                      but we match against known WoW window titles.

    Returns:
        Window handle (hwnd).

    Raises:
        RuntimeError: If no WoW window is found.
    """
    known_titles = ["World of Warcraft", "WoW"]
    found: list = []

    def _cb(hwnd: int, _) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if any(t in title for t in known_titles):
            found.append(hwnd)

    win32gui.EnumWindows(_cb, None)

    if not found:
        raise RuntimeError(
            "WoW window not found. Make sure the game is running "
            "in Windowed Fullscreen (borderless) mode."
        )

    hwnd = found[0]
    title = win32gui.GetWindowText(hwnd)
    logger.info(f"Found WoW window: hwnd={hwnd:#010x}  title={title!r}")
    return hwnd


def send_key(hwnd: int, key: str, jitter_ms: int = 225) -> None:
    """
    Press and release a key, posting directly to the WoW window handle.
    Does not require WoW to be in focus.

    Args:
        hwnd:      WoW window handle from find_wow_window().
        key:       Key string — "2", "F1", "T", etc.
        jitter_ms: Max random extra delay after the key press (anti-pattern).
    """
    vk = _VK_MAP.get(key.upper())
    if vk is None:
        raise ValueError(f"Unknown key {key!r}. Valid keys: {sorted(_VK_MAP)}")

    win32api.PostMessage(hwnd, win32con.WM_KEYDOWN, vk, 0)
    time.sleep(random.uniform(0.05, 0.075))   # realistic key-hold duration
    win32api.PostMessage(hwnd, win32con.WM_KEYUP, vk, 0)

    if jitter_ms > 0:
        time.sleep(random.uniform(0.0, jitter_ms / 1000.0))

    logger.debug(f"Key {key!r} (vk={vk:#04x}) → hwnd={hwnd:#010x}")


def right_click(hwnd: int, screen_x: int, screen_y: int, jitter_ms: int = 225) -> None:
    """
    Right-click at coordinates purely via PostMessage — no real cursor movement.

    Sends WM_MOUSEMOVE first so WoW updates its internal hover target to the
    bobber, then sends WM_RBUTTONDOWN/UP. The user's real cursor is never
    touched.

    Args:
        hwnd:      WoW window handle.
        screen_x:  Absolute screen X coordinate.
        screen_y:  Absolute screen Y coordinate.
        jitter_ms: Max random post-click delay.
    """
    client_x, client_y = win32gui.ScreenToClient(hwnd, (screen_x, screen_y))
    l_param = win32api.MAKELONG(client_x, client_y)

    # Move WoW's internal cursor to the bobber position.
    # Use SendMessage (synchronous) so WoW processes the hover before the click,
    # preventing the user's real mouse movement from interfering.
    win32gui.SendMessage(hwnd, win32con.WM_MOUSEMOVE, 0, l_param)
    time.sleep(0.05)

    # Right-click at that position (synchronous to avoid race with real mouse)
    win32gui.SendMessage(hwnd, win32con.WM_RBUTTONDOWN, win32con.MK_RBUTTON, l_param)
    time.sleep(random.uniform(0.05, 0.1))
    win32gui.SendMessage(hwnd, win32con.WM_RBUTTONUP, 0, l_param)

    if jitter_ms > 0:
        time.sleep(random.uniform(0.0, jitter_ms / 1000.0))

    logger.debug(
        f"Right-click screen=({screen_x},{screen_y}) "
        f"client=({client_x},{client_y})"
    )
