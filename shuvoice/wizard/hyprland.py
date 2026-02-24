"""Hyprland keybind helpers for setup wizard."""

from __future__ import annotations

from enum import Enum

from ..wizard_state import (
    KEYBIND_PRESETS,
    auto_add_hyprland_keybind,
    format_hyprland_bind,
    format_hyprland_bind_for_keybind,
)


class KeybindSetupStatus(str, Enum):
    ADDED = "added"
    ALREADY_CONFIGURED = "already_configured"
    CONFLICT = "conflict"
    MISSING_CONFIG = "missing_config"
    SKIPPED_CUSTOM = "skipped_custom"
    NOT_ATTEMPTED = "not_attempted"
    ERROR = "error"


def setup_keybind(keybind_id: str) -> tuple[str, str]:
    return auto_add_hyprland_keybind(keybind_id)


__all__ = [
    "KEYBIND_PRESETS",
    "KeybindSetupStatus",
    "format_hyprland_bind",
    "format_hyprland_bind_for_keybind",
    "setup_keybind",
]
