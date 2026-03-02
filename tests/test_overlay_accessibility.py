"""Tests for the overlay accessibility enhancements."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.gui


def test_caption_overlay_do_set_text_updates_accessible_property():
    """Verify that _do_set_text updates the accessible label property."""
    # overlay.py requires gi (GTK4) + Gtk4LayerShell at import time.
    gi = pytest.importorskip("gi")
    try:
        gi.require_version("Gtk4LayerShell", "1.0")
    except ValueError:
        pytest.skip("Gtk4LayerShell not available")

    from gi.repository import Gtk

    from shuvoice.overlay import CaptionOverlay

    overlay = object.__new__(CaptionOverlay)
    overlay._label = MagicMock()
    overlay._visible = True
    overlay._window = MagicMock()

    text = "Hello world"
    overlay._do_set_text(text)

    overlay._label.set_text.assert_called_with(text)
    overlay._label.update_property.assert_called_with([Gtk.AccessibleProperty.LABEL], [text])
