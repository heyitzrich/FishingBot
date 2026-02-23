"""Color-coded log text viewer."""

from __future__ import annotations

import customtkinter as ctk


class LogViewerFrame(ctk.CTkFrame):
    def __init__(self, master, max_lines: int = 3000):
        super().__init__(master)
        self._max_lines = max_lines

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self,
            text="LOG VIEWER",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))

        self.textbox = ctk.CTkTextbox(self, wrap="word")
        self.textbox.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.textbox.configure(state="disabled")

        # Text tags for per-level coloring.
        self.textbox.tag_config("DEBUG", foreground="#74d9ff")
        self.textbox.tag_config("INFO", foreground="#88e18f")
        self.textbox.tag_config("WARNING", foreground="#ffd166")
        self.textbox.tag_config("ERROR", foreground="#ff7b7b")
        self.textbox.tag_config("CRITICAL", foreground="#ff4d4d")

    def append(self, line: str, level: str) -> None:
        self.textbox.configure(state="normal")
        self.textbox.insert("end", f"{line}\n", level.upper())
        self.textbox.see("end")
        self._trim_if_needed()
        self.textbox.configure(state="disabled")

    def _trim_if_needed(self) -> None:
        total_lines = int(float(self.textbox.index("end-1c").split(".")[0]))
        extra = total_lines - self._max_lines
        if extra > 0:
            self.textbox.delete("1.0", f"{extra + 1}.0")
