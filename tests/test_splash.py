"""Tests for the splash screen overlay module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.gui

# splash.py requires gi (GTK4) at import time.
gi = pytest.importorskip("gi")
from shuvoice.splash import SplashOverlay, _find_logo


def test_find_logo_returns_none_when_missing(tmp_path):
    """_find_logo returns None when no branding files exist at expected paths."""
    with patch("shuvoice.splash._LOGO_CANDIDATES", [tmp_path / "nonexistent.png"]):
        assert _find_logo() is None


def test_find_logo_returns_first_existing(tmp_path):
    """_find_logo returns the first existing candidate path."""
    logo = tmp_path / "logo.png"
    logo.write_bytes(b"\x89PNG")

    with patch("shuvoice.splash._LOGO_CANDIDATES", [logo, tmp_path / "other.png"]):
        assert _find_logo() == logo


def test_splash_overlay_dismiss_clears_window():
    """dismiss() nulls out window and status references."""
    splash = object.__new__(SplashOverlay)
    splash._window = MagicMock()
    splash._status = MagicMock()

    splash._do_dismiss()

    assert splash._window is None
    assert splash._status is None


def test_splash_overlay_set_status_updates_label():
    """_do_set_status updates the label text."""
    splash = object.__new__(SplashOverlay)
    splash._status = MagicMock()

    splash._do_set_status("Downloading model\u2026")

    splash._status.set_text.assert_called_once_with("Downloading model\u2026")


def test_splash_overlay_set_status_no_crash_when_dismissed():
    """_do_set_status tolerates None status label (already dismissed)."""
    splash = object.__new__(SplashOverlay)
    splash._status = None

    splash._do_set_status("anything")  # Should not raise


def test_splash_overlay_dismiss_is_idempotent():
    """Calling _do_dismiss twice does not raise."""
    splash = object.__new__(SplashOverlay)
    splash._window = MagicMock()
    splash._status = MagicMock()

    splash._do_dismiss()
    splash._do_dismiss()  # Should not raise
