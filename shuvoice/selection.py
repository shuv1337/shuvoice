"""Wayland selection capture helpers for TTS speak-selected-text flow."""

from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)

_SELECTION_TIMEOUT_SEC = 2.0


class SelectionError(RuntimeError):
    """Raised when no usable selection text can be captured."""


def _capture_wl_paste(*args: str) -> str | None:
    cmd = ["wl-paste", "--no-newline", *args]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=_SELECTION_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        log.warning("Selection capture command timed out: %s", cmd[0])
        return None
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("Selection capture command failed: %s", type(exc).__name__)
        return None

    text = result.stdout.strip()
    if text:
        log.debug("Selection capture succeeded (len=%d)", len(text))
    return text or None


def capture_selection() -> str:
    """Capture selected text using primary selection, then clipboard fallback.

    Order:
    1. ``wl-paste --primary --no-newline``
    2. ``wl-paste --no-newline``

    Returns:
        Trimmed selection text.

    Raises:
        SelectionError: if both capture attempts are empty/unavailable.
    """

    primary = _capture_wl_paste("--primary")
    if primary:
        return primary

    clipboard = _capture_wl_paste()
    if clipboard:
        return clipboard

    raise SelectionError("No selected text found (primary selection and clipboard were empty)")
