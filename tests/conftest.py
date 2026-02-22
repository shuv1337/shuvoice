from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock sounddevice/evdev/gi to prevent import errors in CI/headless environments
if "sounddevice" not in sys.modules:
    mock_sd = MagicMock()
    mock_sd.PortAudioError = OSError
    sys.modules["sounddevice"] = mock_sd

if "gi" not in sys.modules:
    mock_gi = MagicMock()
    # Mock require_version to do nothing
    mock_gi.require_version = MagicMock()
    sys.modules["gi"] = mock_gi

    # Create mock repository
    mock_repo = MagicMock()
    sys.modules["gi.repository"] = mock_repo

    # Mock Gtk and GLib
    mock_gtk = MagicMock()
    mock_gtk.Application = MagicMock
    mock_glib = MagicMock()

    # Assign to repository
    mock_repo.Gtk = mock_gtk
    mock_repo.GLib = mock_glib

    # Also inject directly into sys.modules for 'from gi.repository import ...'
    sys.modules["gi.repository.Gtk"] = mock_gtk
    sys.modules["gi.repository.GLib"] = mock_glib

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(items: list[pytest.Item]):
    for item in items:
        if "gui" not in item.keywords:
            item.add_marker(pytest.mark.unit)
