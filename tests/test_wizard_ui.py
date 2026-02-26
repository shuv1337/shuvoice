from __future__ import annotations

import threading
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


def test_model_status_text_maps_cancelled_state():
    from shuvoice.wizard import WelcomeWizard

    assert "cancelled" in WelcomeWizard._model_download_status_text("cancelled").lower()


def test_cancel_download_sets_event_and_updates_button():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard.__new__(WelcomeWizard)
    wizard._download_cancel_event = threading.Event()
    wizard._cancel_download_button = MagicMock()
    wizard._apply_download_progress = MagicMock()

    WelcomeWizard._on_cancel_download_clicked(wizard, None)

    assert wizard._download_cancel_event.is_set() is True
    wizard._cancel_download_button.set_sensitive.assert_called_once_with(False)
    wizard._cancel_download_button.set_label.assert_called_once_with("Canceling…")
    wizard._apply_download_progress.assert_called_once()


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
        patch(
            "shuvoice.wizard.maybe_download_model", return_value=("skipped", "noop")
        ) as maybe_download,
        patch("shuvoice.wizard.write_marker") as write_marker,
    ):
        WelcomeWizard._on_finish(wizard, None)

    write_config.assert_called_once_with(
        "moonshine",
        overwrite_existing=False,
        sherpa_model_name="sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
    )
    maybe_download.assert_called_once_with(
        "moonshine",
        sherpa_model_name="sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
        progress_callback=None,
    )
    write_marker.assert_called_once()
    wizard._release_input_and_destroy_window.assert_called_once()
    wizard.quit.assert_called_once()
    assert wizard.completed is True


def test_on_finish_passes_parakeet_streaming_profile_to_write_config():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard.__new__(WelcomeWizard)
    wizard._asr_backend = "sherpa"
    wizard._sherpa_model_name = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    wizard._sherpa_enable_parakeet_streaming = True
    wizard._keybind = "f9"
    wizard.completed = False
    wizard._release_input_and_destroy_window = MagicMock()
    wizard.quit = MagicMock()

    with (
        patch("shuvoice.wizard.write_config") as write_config,
        patch(
            "shuvoice.wizard.maybe_download_model", return_value=("skipped", "noop")
        ),
        patch("shuvoice.wizard.write_marker"),
    ):
        WelcomeWizard._on_finish(wizard, None)

    write_config.assert_called_once_with(
        "sherpa",
        overwrite_existing=False,
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        sherpa_enable_parakeet_streaming=True,
    )


def test_complete_finish_shows_launch_button_and_waits_for_click():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard.__new__(WelcomeWizard)
    wizard._asr_backend = "sherpa"
    wizard._keybind = "insert"
    wizard._download_pulse_source_id = None
    wizard._finish_in_progress = True
    wizard.completed = False
    wizard._launch_button = MagicMock()
    wizard._set_cancel_download_visible = MagicMock()
    wizard._set_download_note_visible = MagicMock()
    wizard._apply_download_progress = MagicMock()
    wizard._show_finish_status = MagicMock()
    wizard._set_launch_button_visible = MagicMock()
    wizard._finalize_and_quit = MagicMock()

    with patch("shuvoice.wizard.write_marker") as write_marker:
        WelcomeWizard._complete_finish(
            wizard,
            keybind_status="added",
            sherpa_model_name="sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
            model_status="downloaded",
            model_message="done",
        )

    write_marker.assert_called_once()
    wizard._set_launch_button_visible.assert_called_once_with(True)
    wizard._finalize_and_quit.assert_not_called()
    assert wizard.completed is False
    assert wizard._finish_in_progress is False
