"""Simplified editor for core FishingBot config fields."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional

import customtkinter as ctk


def _parse_str(value: str) -> str:
    return value


def _parse_int(value: str) -> int:
    return int(value)


def _parse_float(value: str) -> float:
    return float(value)


def _get_path(root: Dict[str, Any], path: str) -> Any:
    cur: Any = root
    for key in path.split("."):
        cur = cur[key]
    return cur


def _set_path(root: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur: Dict[str, Any] = root
    for key in keys[:-1]:
        cur = cur[key]
    cur[keys[-1]] = value


class ConfigEditorFrame(ctk.CTkFrame):
    def __init__(self, master, on_save):
        super().__init__(master)
        self._on_save = on_save
        self._current_cfg: Dict[str, Any] = {}
        self._parsers: Dict[str, Callable[[str], Any]] = {
            "game.cast_key": _parse_str,
            "game.macro_1": _parse_str,
            "game.macro_2": _parse_str,
            "detection.mode": _parse_str,
            "timing.loot_delay_ms": _parse_int,
            "timing.ten_min_interval_min": _parse_float,
        }
        self._vars: Dict[str, ctk.StringVar] = {
            "game.cast_key": ctk.StringVar(value=""),
            "game.macro_1": ctk.StringVar(value=""),
            "game.macro_2": ctk.StringVar(value=""),
            "detection.mode": ctk.StringVar(value="red"),
            "timing.loot_delay_ms": ctk.StringVar(value=""),
            "timing.ten_min_interval_min": ctk.StringVar(value=""),
        }

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self,
            text="CONFIGURATION",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 8))

        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        self.scroll.grid_columnconfigure(1, weight=1)

        row = 0
        self._add_entry(row, "Cast Key", "game.cast_key")
        row += 1
        self._add_entry(row, "Macro 1", "game.macro_1")
        row += 1
        self._add_entry(row, "Macro 2", "game.macro_2")
        row += 1
        self._add_option_menu(
            row, "Detection Mode", "detection.mode", values=["red", "blue"]
        )
        row += 1
        self._add_entry(row, "Loot Delay (ms)", "timing.loot_delay_ms")
        row += 1
        self._add_entry(row, "Macro Interval (min)", "timing.ten_min_interval_min")

        self._status = ctk.CTkLabel(self, text="", anchor="w")
        self._status.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 6))

        self.save_button = ctk.CTkButton(self, text="Save Config", command=self._handle_save)
        self.save_button.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))

    def _add_entry(self, row: int, label: str, key: str) -> None:
        ctk.CTkLabel(self.scroll, text=label, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(4, 8), pady=4
        )
        widget = ctk.CTkEntry(self.scroll, textvariable=self._vars[key])
        widget.grid(row=row, column=1, sticky="ew", padx=(0, 4), pady=4)

    def _add_option_menu(self, row: int, label: str, key: str, values: list[str]) -> None:
        ctk.CTkLabel(self.scroll, text=label, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(4, 8), pady=4
        )
        widget = ctk.CTkOptionMenu(self.scroll, values=values, variable=self._vars[key])
        widget.grid(row=row, column=1, sticky="ew", padx=(0, 4), pady=4)

    def load_config(self, cfg: Dict[str, Any]) -> None:
        self._current_cfg = copy.deepcopy(cfg)
        ten_min_keys = cfg["game"].get("ten_min_keys") or []

        self._vars["game.cast_key"].set(str(_get_path(cfg, "game.cast_key")))
        self._vars["game.macro_1"].set(str(ten_min_keys[0]) if len(ten_min_keys) > 0 else "")
        self._vars["game.macro_2"].set(str(ten_min_keys[1]) if len(ten_min_keys) > 1 else "")
        self._vars["detection.mode"].set(str(_get_path(cfg, "detection.mode")))
        self._vars["timing.loot_delay_ms"].set(str(_get_path(cfg, "timing.loot_delay_ms")))
        self._vars["timing.ten_min_interval_min"].set(
            str(_get_path(cfg, "timing.ten_min_interval_min"))
        )
        self._status.configure(text="")

    def _handle_save(self) -> None:
        if not self._current_cfg:
            return
        try:
            updated = copy.deepcopy(self._current_cfg)
            for key in ("game.cast_key", "detection.mode", "timing.loot_delay_ms", "timing.ten_min_interval_min"):
                raw = self._vars[key].get().strip()
                parsed = self._parsers[key](raw)
                _set_path(updated, key, parsed)

            macro_1 = self._parsers["game.macro_1"](self._vars["game.macro_1"].get().strip())
            macro_2 = self._parsers["game.macro_2"](self._vars["game.macro_2"].get().strip())
            keys = []
            if macro_1:
                keys.append(macro_1)
            if macro_2:
                keys.append(macro_2)
            _set_path(updated, "game.ten_min_keys", keys)

            self._on_save(updated)
            self._current_cfg = updated
            self._status.configure(text="Config saved.", text_color="#86d993")
        except Exception as exc:
            self._status.configure(text=f"Save failed: {exc}", text_color="#ff7b7b")
