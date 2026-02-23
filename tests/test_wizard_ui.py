from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.gui

gi = pytest.importorskip("gi")
try:
    gi.require_version("Gtk4LayerShell", "1.0")
except ValueError:
    pytest.skip("Gtk4LayerShell not available", allow_module_level=True)


def test_release_input_and_destroy_window_releases_keyboard_mode():
    from shuvoice.wizard import LayerShell, WelcomeWizard

    wizard = WelcomeWizard.__new__(WelcomeWizard)
    win = MagicMock()
    wizard._win = win

    with (
        patch.object(LayerShell, "is_supported", return_value=True),
        patch.object(LayerShell, "set_keyboard_mode") as set_keyboard_mode,
    ):
        WelcomeWizard._release_input_and_destroy_window(wizard)

    set_keyboard_mode.assert_called_once_with(win, LayerShell.KeyboardMode.NONE)
    win.set_visible.assert_called_once_with(False)
    win.destroy.assert_called_once()
    assert wizard._win is None


def test_release_input_and_destroy_window_is_idempotent():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard.__new__(WelcomeWizard)
    wizard._win = None

    WelcomeWizard._release_input_and_destroy_window(wizard)


def test_finish_status_text_maps_known_states():
    from shuvoice.wizard import WelcomeWizard

    assert "Added push-to-talk" in WelcomeWizard._finish_status_text("added")
    assert "already configured" in WelcomeWizard._finish_status_text("already_configured")
    assert "already bound" in WelcomeWizard._finish_status_text("conflict")


def test_on_finish_writes_config_releases_window_and_quits():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard.__new__(WelcomeWizard)
    wizard._asr_backend = "moonshine"
    wizard._keybind = "f9"
    wizard.completed = False
    wizard._release_input_and_destroy_window = MagicMock()
    wizard.quit = MagicMock()

    with (
        patch("shuvoice.wizard.write_config") as write_config,
        patch("shuvoice.wizard.write_marker") as write_marker,
    ):
        WelcomeWizard._on_finish(wizard, None)

    write_config.assert_called_once_with("moonshine", overwrite_existing=False)
    write_marker.assert_called_once()
    wizard._release_input_and_destroy_window.assert_called_once()
    wizard.quit.assert_called_once()
    assert wizard.completed is True
