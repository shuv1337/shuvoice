from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock sounddevice/gi to prevent import errors in CI/headless environments
if "sounddevice" not in sys.modules:
    mock_sd = MagicMock()
    mock_sd.PortAudioError = OSError
    sys.modules["sounddevice"] = mock_sd

if "evdev" not in sys.modules:
    try:
        import evdev  # noqa: F401
    except ImportError:
        sys.modules["evdev"] = MagicMock()

# Check if Gtk is available; if not, mock gi
try:
    import gi

    gi.require_version("Gtk", "4.0")
    # Also ensure LayerShell is available, as the wizard depends on it.
    # If Gtk is present but LayerShell is missing (e.g. CI), we must still mock.
    gi.require_version("Gtk4LayerShell", "1.0")
except (ImportError, ValueError):
    # Mock gi structure
    mock_gi = MagicMock()
    sys.modules["gi"] = mock_gi

    # Mock gi.repository
    mock_repo = MagicMock()
    sys.modules["gi.repository"] = mock_repo

    # Setup Gtk/GLib/Gdk/LayerShell mocks
    mock_gtk = MagicMock()
    mock_glib = MagicMock()
    # Define constants used in return values
    mock_glib.SOURCE_REMOVE = 0
    mock_glib.SOURCE_CONTINUE = 1
    mock_gdk = MagicMock()
    mock_layershell = MagicMock()

    # Mock properties to avoid AttributeErrors
    mock_gtk.Application = MagicMock
    mock_gtk.Window = MagicMock
    mock_gtk.Label = MagicMock
    mock_gtk.Box = MagicMock
    mock_gtk.Image = MagicMock
    mock_gtk.CssProvider = MagicMock
    mock_gtk.StyleContext = MagicMock
    mock_gtk.AccessibleProperty.LABEL = "label"

    # Make them importable from gi.repository
    mock_repo.Gtk = mock_gtk
    mock_repo.GLib = mock_glib
    mock_repo.Gdk = mock_gdk
    mock_repo.Gtk4LayerShell = mock_layershell

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(items: list[pytest.Item]):
    for item in items:
        if "gui" not in item.keywords:
            item.add_marker(pytest.mark.unit)
