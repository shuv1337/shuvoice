"""Hyprland integration helpers for Waybar module."""

from __future__ import annotations

import json
import subprocess
import time

_CACHE_TTL_SEC = 2.0
_cached_value: str | None = None
_cached_at_monotonic: float = 0.0


def _detect_keybind_uncached() -> str | None:
    try:
        result = subprocess.run(
            ["hyprctl", "binds", "-j"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        if result.returncode != 0:
            return None
        binds = json.loads(result.stdout)
        for bind in binds:
            arg = bind.get("arg", "")
            if "shuvoice" not in arg or "start" not in arg:
                continue
            key = bind.get("key", "")
            if not key:
                continue
            modmask = bind.get("modmask", 0)
            mod_names: list[str] = []
            if modmask & 1:
                mod_names.append("Shift")
            if modmask & 4:
                mod_names.append("Ctrl")
            if modmask & 8:
                mod_names.append("Alt")
            if modmask & 64:
                mod_names.append("Super")
            if mod_names:
                return " + ".join(mod_names + [key])
            return key
    except Exception:
        return None
    return None


def detect_keybind(*, ttl_sec: float = _CACHE_TTL_SEC) -> str | None:
    """Detect active ShuVoice keybind with short-lived caching."""
    global _cached_at_monotonic, _cached_value

    now = time.monotonic()
    if now - _cached_at_monotonic <= ttl_sec:
        return _cached_value

    _cached_value = _detect_keybind_uncached()
    _cached_at_monotonic = now
    return _cached_value


def clear_keybind_cache() -> None:
    global _cached_at_monotonic, _cached_value
    _cached_at_monotonic = 0.0
    _cached_value = None
