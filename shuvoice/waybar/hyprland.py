"""Hyprland integration helpers for Waybar module."""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any

_CACHE_TTL_SEC = 2.0
_cached_value: dict[str, str | None] | None = None
_cached_at_monotonic: float = 0.0
_COMMAND_PATTERNS: dict[str, tuple[str, ...]] = {
    "start": ("--control start", " control start"),
    "tts_speak": ("--control tts_speak", " control tts_speak"),
}


def _format_bind(bind: dict[str, Any]) -> str | None:
    key = str(bind.get("key", "")).strip()
    if not key:
        return None

    modmask = int(bind.get("modmask", 0) or 0)
    mod_names: list[str] = []
    if modmask & 64:
        mod_names.append("Super")
    if modmask & 4:
        mod_names.append("Ctrl")
    if modmask & 8:
        mod_names.append("Alt")
    if modmask & 1:
        mod_names.append("Shift")
    if mod_names:
        return " + ".join(mod_names + [key])
    return key


def _matches_shuvoice_command(arg: str, command: str) -> bool:
    arg_lc = str(arg).lower()
    if "shuvoice" not in arg_lc:
        return False
    return any(pattern in arg_lc for pattern in _COMMAND_PATTERNS.get(command, ()))


def _detect_keybinds_uncached() -> dict[str, str | None]:
    detected: dict[str, str | None] = {command: None for command in _COMMAND_PATTERNS}

    try:
        result = subprocess.run(
            ["hyprctl", "binds", "-j"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        if result.returncode != 0:
            return detected
        binds = json.loads(result.stdout)
        for bind in binds:
            arg = bind.get("arg", "")
            for command in detected:
                if detected[command] is not None:
                    continue
                if not _matches_shuvoice_command(arg, command):
                    continue
                detected[command] = _format_bind(bind)
            if all(value is not None for value in detected.values()):
                break
    except Exception:
        return {command: None for command in _COMMAND_PATTERNS}
    return detected


def detect_keybinds(*, ttl_sec: float = _CACHE_TTL_SEC) -> dict[str, str | None]:
    """Detect active ShuVoice Hyprland keybinds with short-lived caching."""
    global _cached_at_monotonic, _cached_value

    now = time.monotonic()
    if _cached_value is not None and now - _cached_at_monotonic <= ttl_sec:
        return dict(_cached_value)

    _cached_value = _detect_keybinds_uncached()
    _cached_at_monotonic = now
    return dict(_cached_value)


def detect_keybind(command: str = "start", *, ttl_sec: float = _CACHE_TTL_SEC) -> str | None:
    """Detect a specific ShuVoice keybind.

    Supported commands currently include ``start`` (push-to-talk/STT) and
    ``tts_speak``.
    """
    return detect_keybinds(ttl_sec=ttl_sec).get(command)


def clear_keybind_cache() -> None:
    global _cached_at_monotonic, _cached_value
    _cached_at_monotonic = 0.0
    _cached_value = None
