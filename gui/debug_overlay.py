"""Debug frame composition shared by OpenCV window and GUI preview."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from bot.bobber_finder import BobberFinder
from bot.fish_bot import BotState, FishBot

# State label colors in BGR.
_STATE_COLORS = {
    BotState.IDLE: (200, 200, 200),
    BotState.CASTING: (0, 200, 255),
    BotState.SEARCHING: (255, 200, 0),
    BotState.WATCHING: (0, 220, 0),
    BotState.LOOTING: (0, 100, 255),
    BotState.STOPPED: (100, 100, 100),
}


def build_debug_frame(bot: FishBot, finder: BobberFinder) -> Optional[np.ndarray]:
    """
    Compose a debug display frame from the bot's most recent capture.

    Returns None if no frame has been captured yet.
    """
    frame = finder.last_frame
    mask = finder.last_mask
    if frame is None or mask is None:
        return None

    display = frame.copy()

    # Color mask overlay.
    overlay = np.zeros_like(display)
    overlay[mask > 0] = (0, 0, 180)  # red in BGR
    cv2.addWeighted(overlay, 0.35, display, 1.0, 0, display)

    # Bobber reticle with color-coded detection method.
    bmp_pos = finder.last_bitmap_pos
    detection_method = finder.last_detection_method

    if bmp_pos is not None:
        cx, cy = bmp_pos
        arm = 20
        stub = 8
        thickness = 2

        # Color-code based on detection method (BGR format)
        if detection_method == 'template':
            color = (0, 255, 0)  # Green for template matching
        elif detection_method == 'red':
            color = (0, 0, 255)  # Red for red pixel detection
        elif detection_method == 'blue':
            color = (255, 0, 0)  # Blue for blue pixel detection
        else:
            color = (255, 255, 255)  # White as fallback

        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            x, y = cx + dx * arm, cy + dy * arm
            cv2.line(display, (x, y), (x - dx * stub, y), color, thickness, cv2.LINE_AA)
            cv2.line(display, (x, y), (x, y - dy * stub), color, thickness, cv2.LINE_AA)

    # State badge.
    state = bot.state
    state_color = _STATE_COLORS.get(state, (255, 255, 255))
    label = f"  {state.name}"
    cv2.rectangle(display, (0, 0), (200, 26), (20, 20, 20), -1)
    cv2.putText(
        display,
        label,
        (4, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        state_color,
        1,
        cv2.LINE_AA,
    )

    # Session stats bar.
    stats_lines = str(bot.stats).split(" | ")
    bar_h = 20
    for i, line in enumerate(stats_lines):
        y = display.shape[0] - 8 - i * bar_h
        cv2.rectangle(
            display,
            (0, y - bar_h + 4),
            (display.shape[1], y + 4),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            display,
            line,
            (6, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    return display
