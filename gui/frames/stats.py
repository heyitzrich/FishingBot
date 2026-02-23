"""Live session statistics grid."""

from __future__ import annotations

import customtkinter as ctk


class StatsFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self,
            text="STATISTICS",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 8))

        self._values = {}
        labels = [
            ("casts", "Casts"),
            ("catches", "Catches"),
            ("catch_rate", "Catch Rate"),
            ("casts_per_hour", "CPH"),
            ("timeouts", "Timeouts"),
            ("no_bobber", "No Bobber"),
            ("uptime", "Uptime"),
        ]

        for idx, (key, label) in enumerate(labels, start=1):
            ctk.CTkLabel(self, text=f"{label}:", anchor="w").grid(
                row=idx, column=0, sticky="w", padx=(12, 8), pady=2
            )
            value = ctk.CTkLabel(
                self,
                text="-",
                anchor="w",
                font=ctk.CTkFont(size=13, weight="bold"),
            )
            value.grid(row=idx, column=1, sticky="w", padx=(0, 12), pady=2)
            self._values[key] = value

    def update_stats(self, stats) -> None:
        if stats is None:
            return

        self._values["casts"].configure(text=str(stats.casts))
        self._values["catches"].configure(text=str(stats.catches))
        self._values["catch_rate"].configure(text=f"{stats.catch_rate:.1f}%")
        self._values["casts_per_hour"].configure(text=f"{stats.casts_per_hour:.0f}")
        self._values["timeouts"].configure(text=str(stats.timeouts))
        self._values["no_bobber"].configure(text=str(stats.no_bobber))
        self._values["uptime"].configure(text=f"{stats.elapsed_minutes:.1f} min")
