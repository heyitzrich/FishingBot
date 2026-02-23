"""Live debug preview panel."""

from __future__ import annotations

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image


class PreviewFrame(ctk.CTkFrame):
    """Displays a resized live debug frame from OpenCV capture."""

    def __init__(self, master, width: int = 640, height: int = 360):
        super().__init__(master)
        self._width = width
        self._height = height
        blank = Image.new("RGB", (self._width, self._height), (21, 21, 21))
        self._image = ctk.CTkImage(
            light_image=blank,
            dark_image=blank,
            size=(self._width, self._height),
        )
        self._has_frame = False
        self._paused = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self,
            text="LIVE DEBUG PREVIEW",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))

        self._image_label = ctk.CTkLabel(
            self,
            image=self._image,
            text="Waiting for frames...",
            width=self._width,
            height=self._height,
            fg_color=("#151515", "#151515"),
            corner_radius=10,
        )
        self._image_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

    def update_frame(self, bgr_frame: Optional[np.ndarray]) -> None:
        if self._paused or bgr_frame is None:
            return

        resized = cv2.resize(
            bgr_frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        self._image.configure(
            light_image=pil_image,
            dark_image=pil_image,
            size=(self._width, self._height),
        )
        self._has_frame = True
        self._image_label.configure(image=self._image, text="")

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        if paused:
            self._image_label.configure(image=self._image, text="Live preview paused.")
            return

        if self._has_frame:
            self._image_label.configure(image=self._image, text="")
        else:
            self._image_label.configure(image=self._image, text="Waiting for frames...")
