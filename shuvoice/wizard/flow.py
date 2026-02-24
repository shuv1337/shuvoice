"""Wizard flow helpers."""

from __future__ import annotations

from ..wizard_state import format_summary


def summary_text(asr_backend: str, keybind_id: str, *, auto_add_keybind: bool) -> str:
    return format_summary(asr_backend, keybind_id, auto_add_keybind=auto_add_keybind)
