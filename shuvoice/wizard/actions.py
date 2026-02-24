"""Wizard finish actions (config + marker + optional keybind setup)."""

from __future__ import annotations

from ..wizard_state import needs_wizard, write_config, write_marker
from .hyprland import setup_keybind


def finish_setup(
    asr_backend: str,
    keybind_id: str,
    *,
    auto_add_keybind: bool,
    overwrite_existing: bool,
) -> tuple[str, str]:
    """Persist wizard selections and optionally configure Hyprland keybind."""
    write_config(asr_backend, overwrite_existing=overwrite_existing)

    keybind_status = "not_attempted"
    keybind_message = "automatic keybind setup disabled"
    if auto_add_keybind:
        keybind_status, keybind_message = setup_keybind(keybind_id)

    write_marker()
    return keybind_status, keybind_message


__all__ = [
    "finish_setup",
    "needs_wizard",
    "write_config",
    "write_marker",
]
