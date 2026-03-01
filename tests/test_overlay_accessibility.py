from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    from shuvoice.overlay import CaptionOverlay
except ValueError:
    pytest.skip("Gtk4LayerShell not available", allow_module_level=True)


@patch("shuvoice.overlay.Gtk")
@patch("shuvoice.overlay.LayerShell")
def test_caption_overlay_sets_accessible_role(mock_layer_shell, mock_gtk):
    mock_layer_shell.is_supported.return_value = True
    app = mock_gtk.Application()
    config = MagicMock()
    config.bottom_margin = 0
    config.border_radius = 0
    config.bg_opacity = 0
    config.font_size = 0

    mock_label = MagicMock()
    mock_gtk.Label.return_value = mock_label

    CaptionOverlay(app, config)

    mock_label.set_accessible_role.assert_called_once_with(mock_gtk.AccessibleRole.STATUS)


@patch("shuvoice.overlay.Gtk")
@patch("shuvoice.overlay.LayerShell")
@patch("shuvoice.overlay.GLib")
def test_caption_overlay_updates_accessible_property_on_text_change(
    mock_glib, mock_layer_shell, mock_gtk
):
    mock_layer_shell.is_supported.return_value = True
    app = mock_gtk.Application()
    config = MagicMock()
    config.bottom_margin = 0
    config.border_radius = 0
    config.bg_opacity = 0
    config.font_size = 0

    mock_label = MagicMock()
    mock_gtk.Label.return_value = mock_label

    overlay = CaptionOverlay(app, config)

    overlay._do_set_text("Hello World")

    mock_label.set_text.assert_called_with("Hello World")
    mock_label.update_property.assert_called_once_with(
        [mock_gtk.AccessibleProperty.LABEL], ["Hello World"]
    )
