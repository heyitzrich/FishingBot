"""Start/stop controls and state badge."""

from __future__ import annotations

import customtkinter as ctk

_STATE_COLORS = {
    "IDLE": "#a9a9a9",
    "CASTING": "#48b2ff",
    "SEARCHING": "#f0c15f",
    "WATCHING": "#49d867",
    "LOOTING": "#ff9b4f",
    "STOPPED": "#7a7a7a",
}


class ControlsFrame(ctk.CTkFrame):
    def __init__(
        self,
        master,
        on_start,
        on_stop,
        on_preview_toggle,
        preview_enabled: bool = True,
    ):
        super().__init__(master)
        self._on_preview_toggle = on_preview_toggle

        self.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(
            self,
            text="CONTROLS",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 8))

        self.start_button = ctk.CTkButton(
            self, text="Start Bot", command=on_start, height=34
        )
        self.start_button.grid(row=1, column=0, sticky="ew", padx=(12, 6), pady=(0, 8))

        self.stop_button = ctk.CTkButton(
            self,
            text="Stop Bot",
            command=on_stop,
            fg_color="#4a4a4a",
            hover_color="#555555",
            height=34,
        )
        self.stop_button.grid(row=1, column=1, sticky="ew", padx=(6, 12), pady=(0, 8))

        ctk.CTkLabel(self, text="State:", anchor="w").grid(
            row=2, column=0, sticky="w", padx=12, pady=(0, 10)
        )
        self.state_value = ctk.CTkLabel(
            self,
            text="IDLE",
            text_color=_STATE_COLORS["IDLE"],
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        self.state_value.grid(row=2, column=1, sticky="w", padx=12, pady=(0, 10))

        ctk.CTkLabel(self, text="Live Preview:", anchor="w").grid(
            row=3, column=0, sticky="w", padx=12, pady=(0, 10)
        )
        self._preview_var = ctk.BooleanVar(value=preview_enabled)
        self.preview_switch = ctk.CTkSwitch(
            self,
            text="",
            variable=self._preview_var,
            command=self._handle_preview_toggle,
            onvalue=True,
            offvalue=False,
        )
        self.preview_switch.grid(row=3, column=1, sticky="w", padx=12, pady=(0, 10))

        self.set_running(False)

    def set_running(self, running: bool) -> None:
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def set_state(self, state) -> None:
        state_name = getattr(state, "name", str(state)).upper()
        self.state_value.configure(
            text=state_name,
            text_color=_STATE_COLORS.get(state_name, "#d7d7d7"),
        )

    def _handle_preview_toggle(self) -> None:
        self._on_preview_toggle(bool(self._preview_var.get()))
