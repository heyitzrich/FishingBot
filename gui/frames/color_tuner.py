"""Color detection tuner window with sliders, presets, and static mask preview.

Opens as a separate settings window. Captures a screenshot on open and
lets users adjust detection parameters against that frozen frame.

Presets allow saving/loading named color profiles for different zones.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

from bot.pixel_classifier import ClassifierMode, PixelClassifier


def _extract_color_settings(det: dict, mode: str) -> dict:
    """Extract only the color-related keys from a detection config."""
    settings: dict = {"mode": mode}
    if mode == "red":
        settings["red"] = copy.deepcopy(det.get("red", {}))
    else:
        settings["blue"] = copy.deepcopy(det.get("blue", {}))
    return settings


def _apply_color_settings(det: dict, settings: dict) -> dict:
    """Merge saved color settings back into a full detection config."""
    det = copy.deepcopy(det)
    mode = settings.get("mode", "red")
    if mode == "red" and "red" in settings:
        det["red"] = copy.deepcopy(settings["red"])
    elif mode == "blue" and "blue" in settings:
        det["blue"] = copy.deepcopy(settings["blue"])
    return det


class ColorTunerWindow(ctk.CTkToplevel):
    """Popup settings window for tuning color detection."""

    PREVIEW_W = 480
    PREVIEW_H = 270

    def __init__(
        self,
        master,
        cfg: Dict[str, Any],
        on_save: Callable[[dict], None],
        on_slider_change: Callable[[dict], None],
        screenshot: Optional[np.ndarray] = None,
    ):
        super().__init__(master)
        self.title("Color Configuration")
        self.geometry("560x780")
        self.minsize(480, 600)
        self.resizable(True, True)
        self.after(50, self._bring_to_front)

        self._on_save = on_save
        self._on_slider_change = on_slider_change
        self._current_cfg: Dict[str, Any] = copy.deepcopy(cfg)
        self._mode: str = str(cfg.get("detection", {}).get("mode", "red")).strip().lower()
        self._screenshot: Optional[np.ndarray] = screenshot
        self._slider_widgets: list = []
        self._sliders: Dict[str, ctk.CTkSlider] = {}
        self._slider_labels: Dict[str, ctk.CTkLabel] = {}
        self._suppress_callback = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # --- Mask preview (static screenshot) ---
        blank = Image.new("RGB", (self.PREVIEW_W, self.PREVIEW_H), (21, 21, 21))
        self._preview_image = ctk.CTkImage(
            light_image=blank, dark_image=blank,
            size=(self.PREVIEW_W, self.PREVIEW_H),
        )
        self._preview_label = ctk.CTkLabel(
            self,
            image=self._preview_image,
            text="No screenshot captured" if screenshot is None else "",
            width=self.PREVIEW_W,
            height=self.PREVIEW_H,
            fg_color=("#151515", "#151515"),
            corner_radius=8,
        )
        self._preview_label.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))

        # --- Recapture button ---
        self._recapture_btn = ctk.CTkButton(
            self, text="Recapture Screenshot", command=self._request_recapture,
        )
        self._recapture_btn.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        self._recapture_callback: Optional[Callable[[], Optional[np.ndarray]]] = None

        # --- Presets bar ---
        self._build_preset_bar()

        # --- Scrollable slider area ---
        self._scroll = ctk.CTkScrollableFrame(self)
        self._scroll.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 6))
        self._scroll.grid_columnconfigure(0, weight=1)

        # --- Bottom bar ---
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 12))
        bottom.grid_columnconfigure(0, weight=1)

        self._status = ctk.CTkLabel(bottom, text="", anchor="w")
        self._status.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self._save_btn = ctk.CTkButton(
            bottom, text="Apply", command=self._handle_save,
        )
        self._save_btn.grid(row=1, column=0, sticky="ew")

        self._rebuild_sliders()
        if self._screenshot is not None:
            self._refresh_preview()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_recapture_callback(self, cb: Callable[[], Optional[np.ndarray]]) -> None:
        self._recapture_callback = cb

    def set_screenshot(self, frame: np.ndarray) -> None:
        self._screenshot = frame
        self._refresh_preview()

    def _bring_to_front(self) -> None:
        self.attributes("-topmost", True)
        self.after(100, lambda: self.attributes("-topmost", False))
        self.focus_force()

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def _build_preset_bar(self) -> None:
        preset_frame = ctk.CTkFrame(self)
        preset_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
        preset_frame.grid_columnconfigure(0, weight=1)

        # Top row: dropdown + load
        top_row = ctk.CTkFrame(preset_frame, fg_color="transparent")
        top_row.grid(row=0, column=0, sticky="ew", pady=(6, 4), padx=8)
        top_row.grid_columnconfigure(0, weight=1)

        self._preset_var = ctk.StringVar(value="")
        self._preset_dropdown = ctk.CTkOptionMenu(
            top_row, variable=self._preset_var, values=self._get_preset_names(),
            command=self._on_preset_selected, dynamic_resizing=False,
        )
        self._preset_dropdown.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        if not self._get_preset_names():
            self._preset_dropdown.set("No presets")

        load_btn = ctk.CTkButton(
            top_row, text="Load", width=60, command=self._load_preset,
        )
        load_btn.grid(row=0, column=1, padx=(0, 4))

        # Bottom row: save as / rename / delete
        bot_row = ctk.CTkFrame(preset_frame, fg_color="transparent")
        bot_row.grid(row=1, column=0, sticky="ew", pady=(0, 6), padx=8)
        bot_row.grid_columnconfigure(0, weight=1)

        self._preset_name_entry = ctk.CTkEntry(
            bot_row, placeholder_text="Preset name...",
        )
        self._preset_name_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        save_as_btn = ctk.CTkButton(
            bot_row, text="Save As", width=65, command=self._save_preset,
        )
        save_as_btn.grid(row=0, column=1, padx=(0, 4))

        rename_btn = ctk.CTkButton(
            bot_row, text="Rename", width=65, command=self._rename_preset,
        )
        rename_btn.grid(row=0, column=2, padx=(0, 4))

        delete_btn = ctk.CTkButton(
            bot_row, text="Delete", width=65, fg_color="#8B0000",
            hover_color="#a52a2a", command=self._delete_preset,
        )
        delete_btn.grid(row=0, column=3)

    def _get_presets(self) -> dict:
        return self._current_cfg.get("detection", {}).get("color_presets", {})

    def _get_preset_names(self) -> List[str]:
        return sorted(self._get_presets().keys())

    def _refresh_preset_dropdown(self) -> None:
        names = self._get_preset_names()
        self._preset_dropdown.configure(values=names if names else ["No presets"])
        if not names:
            self._preset_dropdown.set("No presets")

    def _on_preset_selected(self, name: str) -> None:
        self._preset_name_entry.delete(0, "end")
        self._preset_name_entry.insert(0, name)

    def _save_preset(self) -> None:
        name = self._preset_name_entry.get().strip()
        if not name:
            self._status.configure(text="Enter a preset name.", text_color="#ff7b7b")
            return

        det = self._build_detection_cfg()
        settings = _extract_color_settings(det, self._mode)

        presets = self._current_cfg.setdefault("detection", {}).setdefault("color_presets", {})
        presets[name] = settings

        self._persist_config()
        self._refresh_preset_dropdown()
        self._preset_var.set(name)
        self._status.configure(text=f"Preset '{name}' saved.", text_color="#86d993")

    def _load_preset(self) -> None:
        name = self._preset_var.get()
        presets = self._get_presets()
        if name not in presets:
            self._status.configure(text="Select a preset to load.", text_color="#ff7b7b")
            return

        settings = presets[name]
        det = self._current_cfg.get("detection", {})
        updated_det = _apply_color_settings(det, settings)
        self._current_cfg["detection"] = updated_det

        self._suppress_callback = True
        self._rebuild_sliders()
        self._suppress_callback = False

        detection_cfg = self._build_detection_cfg()
        self._current_cfg["detection"] = detection_cfg
        self._on_slider_change(detection_cfg)
        self._persist_config()
        self._refresh_preview()
        self._status.configure(text=f"Preset '{name}' loaded & saved.", text_color="#86d993")

    def _rename_preset(self) -> None:
        old_name = self._preset_var.get()
        new_name = self._preset_name_entry.get().strip()
        presets = self._get_presets()

        if old_name not in presets:
            self._status.configure(text="Select a preset to rename.", text_color="#ff7b7b")
            return
        if not new_name:
            self._status.configure(text="Enter a new name.", text_color="#ff7b7b")
            return
        if new_name == old_name:
            return

        presets[new_name] = presets.pop(old_name)
        self._persist_config()
        self._refresh_preset_dropdown()
        self._preset_var.set(new_name)
        self._status.configure(
            text=f"Renamed '{old_name}' to '{new_name}'.", text_color="#86d993",
        )

    def _delete_preset(self) -> None:
        name = self._preset_var.get()
        presets = self._get_presets()
        if name not in presets:
            self._status.configure(text="Select a preset to delete.", text_color="#ff7b7b")
            return

        del presets[name]
        self._persist_config()
        self._refresh_preset_dropdown()
        self._preset_name_entry.delete(0, "end")
        self._status.configure(text=f"Preset '{name}' deleted.", text_color="#86d993")

    def _persist_config(self) -> None:
        """Save current cfg to disk without applying as active config."""
        try:
            updated = copy.deepcopy(self._current_cfg)
            self._on_save(updated)
        except Exception as exc:
            self._status.configure(text=f"Save failed: {exc}", text_color="#ff7b7b")

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _refresh_preview(self) -> None:
        if self._screenshot is None:
            return

        det = self._build_detection_cfg()
        mode = ClassifierMode.RED if self._mode == "red" else ClassifierMode.BLUE
        classifier = PixelClassifier(mode, det)
        mask = classifier.apply_mask(self._screenshot)

        display = self._screenshot.copy()
        overlay_color = (0, 255, 0) if self._mode == "red" else (255, 100, 0)
        colored = np.zeros_like(display)
        colored[:] = overlay_color
        mask_bool = mask > 0
        display[mask_bool] = cv2.addWeighted(
            display, 0.4, colored, 0.6, 0,
        )[mask_bool]

        resized = cv2.resize(
            display, (self.PREVIEW_W, self.PREVIEW_H),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        self._preview_image.configure(
            light_image=pil_image, dark_image=pil_image,
            size=(self.PREVIEW_W, self.PREVIEW_H),
        )
        self._preview_label.configure(image=self._preview_image, text="")

    def _request_recapture(self) -> None:
        if self._recapture_callback is not None:
            frame = self._recapture_callback()
            if frame is not None:
                self.set_screenshot(frame)

    # ------------------------------------------------------------------
    # Slider management
    # ------------------------------------------------------------------

    def _rebuild_sliders(self) -> None:
        for w in self._slider_widgets:
            w.destroy()
        self._slider_widgets.clear()
        self._sliders.clear()
        self._slider_labels.clear()

        det = self._current_cfg.get("detection", {})
        row = 0

        if self._mode == "red":
            red = det.get("red", {})
            l1 = red.get("lower1", [0, 120, 100])
            u1 = red.get("upper1", [10, 255, 255])
            l2 = red.get("lower2", [170, 120, 100])
            u2 = red.get("upper2", [180, 255, 255])

            row = self._add_section_label(row, "Hue Range")
            row = self._add_slider_row(row, "Hue Min (Low)", "r_h1_lo", 0, 180, l1[0])
            row = self._add_slider_row(row, "Hue Max (Low)", "r_h1_hi", 0, 180, u1[0])
            row = self._add_slider_row(row, "Hue Min (High)", "r_h2_lo", 0, 180, l2[0])
            row = self._add_slider_row(row, "Hue Max (High)", "r_h2_hi", 0, 180, u2[0])

            row = self._add_section_label(row, "Saturation / Value")
            row = self._add_slider_row(row, "Saturation Min", "r_s_lo", 0, 255, l1[1])
            row = self._add_slider_row(row, "Value Min", "r_v_lo", 0, 255, l1[2])

            row = self._add_section_label(row, "Hardening")
            row = self._add_slider_row(
                row, "Colour Multiplier", "cm", 0, 300,
                int(float(red.get("colour_multiplier", 0)) * 100),
            )
            row = self._add_slider_row(
                row, "Closeness Multiplier", "ccm", 0, 500,
                int(float(red.get("colour_closeness_multiplier", 0)) * 100),
            )
            row = self._add_slider_row(
                row, "Closeness Offset", "cco", 0, 100,
                int(red.get("colour_closeness_offset", 20)),
            )
        else:
            blue = det.get("blue", {})
            lo = blue.get("lower", [100, 120, 100])
            hi = blue.get("upper", [130, 255, 255])

            row = self._add_section_label(row, "Hue Range")
            row = self._add_slider_row(row, "Hue Min", "b_h_lo", 0, 180, lo[0])
            row = self._add_slider_row(row, "Hue Max", "b_h_hi", 0, 180, hi[0])

            row = self._add_section_label(row, "Saturation / Value")
            row = self._add_slider_row(row, "Saturation Min", "b_s_lo", 0, 255, lo[1])
            row = self._add_slider_row(row, "Value Min", "b_v_lo", 0, 255, lo[2])

            row = self._add_section_label(row, "Hardening")
            row = self._add_slider_row(
                row, "Blue Ch. Min Delta", "delta", 0, 100,
                int(blue.get("blue_channel_min_delta", 20)),
            )

    def _add_section_label(self, row: int, text: str) -> int:
        label = ctk.CTkLabel(
            self._scroll, text=text,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        )
        label.grid(row=row, column=0, sticky="w", padx=4, pady=(8, 2))
        self._slider_widgets.append(label)
        return row + 1

    def _add_slider_row(
        self, row: int, label: str, key: str,
        from_: int, to: int, value: int,
    ) -> int:
        frame = ctk.CTkFrame(self._scroll, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text=label, anchor="w", width=150).grid(
            row=0, column=0, sticky="w", padx=(4, 8),
        )

        if key in ("cm", "ccm"):
            display_text = f"{value / 100:.2f}"
        else:
            display_text = str(value)

        val_label = ctk.CTkLabel(frame, text=display_text, anchor="e", width=40)
        val_label.grid(row=0, column=2, sticky="e", padx=(4, 4))

        slider = ctk.CTkSlider(
            frame, from_=from_, to=to, number_of_steps=to - from_,
            command=lambda v, k=key, lbl=val_label: self._on_slide(k, v, lbl),
        )
        slider.set(value)
        slider.grid(row=0, column=1, sticky="ew", padx=(0, 4))

        self._sliders[key] = slider
        self._slider_labels[key] = val_label
        self._slider_widgets.append(frame)
        return row + 1

    def _on_slide(self, key: str, value: float, label: ctk.CTkLabel) -> None:
        if key in ("cm", "ccm"):
            label.configure(text=f"{value / 100:.2f}")
        else:
            label.configure(text=str(int(value)))

        if self._suppress_callback:
            return

        detection_cfg = self._build_detection_cfg()
        self._on_slider_change(detection_cfg)
        self._refresh_preview()

    def _build_detection_cfg(self) -> dict:
        det = copy.deepcopy(self._current_cfg.get("detection", {}))
        s = self._sliders

        if self._mode == "red":
            h1_lo = int(s["r_h1_lo"].get())
            h1_hi = int(s["r_h1_hi"].get())
            h2_lo = int(s["r_h2_lo"].get())
            h2_hi = int(s["r_h2_hi"].get())
            s_lo = int(s["r_s_lo"].get())
            v_lo = int(s["r_v_lo"].get())

            det["red"]["lower1"] = [h1_lo, s_lo, v_lo]
            det["red"]["upper1"] = [h1_hi, 255, 255]
            det["red"]["lower2"] = [h2_lo, s_lo, v_lo]
            det["red"]["upper2"] = [h2_hi, 255, 255]
            det["red"]["colour_multiplier"] = round(s["cm"].get() / 100, 2)
            det["red"]["colour_closeness_multiplier"] = round(s["ccm"].get() / 100, 2)
            det["red"]["colour_closeness_offset"] = int(s["cco"].get())
        else:
            h_lo = int(s["b_h_lo"].get())
            h_hi = int(s["b_h_hi"].get())
            s_lo = int(s["b_s_lo"].get())
            v_lo = int(s["b_v_lo"].get())

            det["blue"]["lower"] = [h_lo, s_lo, v_lo]
            det["blue"]["upper"] = [h_hi, 255, 255]
            det["blue"]["blue_channel_min_delta"] = int(s["delta"].get())

        return det

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _handle_save(self) -> None:
        if not self._current_cfg:
            return
        try:
            updated = copy.deepcopy(self._current_cfg)
            det = self._build_detection_cfg()
            updated["detection"] = det
            self._on_save(updated)
            self._current_cfg = updated
            self._status.configure(text="Color config saved.", text_color="#86d993")
        except Exception as exc:
            self._status.configure(
                text=f"Save failed: {exc}", text_color="#ff7b7b",
            )
