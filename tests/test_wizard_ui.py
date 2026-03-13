from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.gui

gi = pytest.importorskip("gi")
try:
    gi.require_version("Gtk4LayerShell", "1.0")
    gi.require_version("Gtk", "4.0")
except ValueError:
    pytest.skip("Gtk4LayerShell/Gtk not available", allow_module_level=True)

from gi.repository import Gtk  # noqa: E402


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


def test_starting_model_setup_status_text_includes_cuda_hint_for_sherpa():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard.__new__(WelcomeWizard)
    wizard._asr_backend = "sherpa"
    wizard._sherpa_provider = "cuda"

    status = WelcomeWizard._starting_model_setup_status_text(wizard, "not_attempted").lower()
    assert "cuda" in status
    assert "before model download" in status


def test_wizard_defaults_to_parakeet_instant_profile():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard()
    assert wizard._sherpa_model_name == "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    assert wizard._sherpa_enable_parakeet_streaming is False
    assert wizard._sherpa_provider == "cpu"
    assert wizard._tts_backend == "elevenlabs"
    assert wizard._tts_voice_id == "zNsotODqUhvbJ5wMG7Ei"


def test_make_dropdown_section_updates_state_and_description():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard()
    page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    seen: list[str] = []

    _title, dropdown, desc = wizard._make_dropdown_section(
        page,
        "Example",
        [
            ("a", "Option A", "Description A"),
            ("b", "Option B", "Description B"),
        ],
        "a",
        seen.append,
    )

    dropdown.set_selected(1)

    assert desc.get_text() == "Description B"
    assert seen[-1] == "b"


def test_asr_page_uses_dropdowns_for_sherpa_controls():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard()
    wizard._build_asr_page()

    assert hasattr(wizard, "_sherpa_profile_dropdown")
    assert isinstance(wizard._sherpa_profile_dropdown, Gtk.DropDown)
    assert hasattr(wizard, "_sherpa_provider_dropdown")
    assert isinstance(wizard._sherpa_provider_dropdown, Gtk.DropDown)
    assert not hasattr(wizard, "_sherpa_streaming_radio")
    assert not hasattr(wizard, "_sherpa_provider_cpu_radio")


def test_sherpa_dropdowns_only_visible_when_sherpa_selected():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard()
    wizard._build_asr_page()

    assert wizard._sherpa_profile_dropdown.get_visible() is True
    wizard._asr_radios["nemo"].set_active(True)

    assert wizard._sherpa_profile_dropdown.get_visible() is False
    assert wizard._sherpa_provider_dropdown.get_visible() is False


def test_sherpa_profile_dropdown_updates_state_on_selection_change():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard()
    wizard._build_asr_page()
    wizard._sherpa_profile_dropdown.set_selected(0)

    assert wizard._sherpa_model_name == "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
    assert wizard._sherpa_enable_parakeet_streaming is False


def test_keybind_page_uses_dropdowns_and_hides_preview():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard()
    wizard._build_keybind_page()

    assert hasattr(wizard, "_keybind_dropdown")
    assert isinstance(wizard._keybind_dropdown, Gtk.DropDown)
    assert hasattr(wizard, "_final_injection_dropdown")
    assert isinstance(wizard._final_injection_dropdown, Gtk.DropDown)
    assert not hasattr(wizard, "_keybind_preview")
    assert not hasattr(wizard, "_tts_provider_dropdown")


def test_keybind_dropdown_updates_auto_add_state_for_custom():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard()
    wizard._build_keybind_page()
    wizard._keybind_dropdown.set_selected(4)

    assert wizard._keybind == "custom"
    assert wizard._auto_add_keybind.get_sensitive() is False
    assert wizard._auto_add_keybind.get_active() is False


def test_final_injection_dropdown_updates_state():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard()
    wizard._build_keybind_page()
    wizard._final_injection_dropdown.set_selected(2)

    assert wizard._typing_final_injection_mode == "direct"


def test_text_case_dropdown_updates_state():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard()
    wizard._build_keybind_page()
    wizard._typing_text_case_dropdown.set_selected(1)

    assert wizard._typing_text_case == "lowercase"


def test_tts_page_includes_provider_dropdown_and_voice_entry():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard()
    wizard._build_tts_page()

    assert hasattr(wizard, "_tts_provider_dropdown")
    assert isinstance(wizard._tts_provider_dropdown, Gtk.DropDown)
    assert hasattr(wizard, "_tts_voice_entry")
    assert wizard._tts_voice_entry.get_text() == "zNsotODqUhvbJ5wMG7Ei"


def test_tts_provider_dropdown_updates_voice_entry_for_openai():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard()
    wizard._build_tts_page()
    wizard._tts_provider_dropdown.set_selected(1)

    assert wizard._tts_backend == "openai"
    assert wizard._tts_voice_entry.get_text() == "onyx"


def test_local_piper_controls_only_appear_on_tts_page_when_local_selected():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard()
    wizard._build_keybind_page()
    assert not hasattr(wizard, "_tts_local_setup_mode_dropdown")

    wizard._build_tts_page()
    assert wizard._tts_local_setup_mode_dropdown.get_visible() is False

    wizard._tts_provider_dropdown.set_selected(2)

    assert wizard._tts_backend == "local"
    assert wizard._tts_local_setup_mode_dropdown.get_visible() is True
    assert wizard._tts_local_auto_voice_dropdown.get_visible() is True
    assert wizard._tts_local_model_path_entry.get_visible() is False


def test_local_manual_mode_shows_path_entry():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard()
    wizard._build_tts_page()
    wizard._tts_provider_dropdown.set_selected(2)
    wizard._tts_local_setup_mode_dropdown.set_selected(1)

    assert wizard._tts_local_setup_mode == "manual"
    assert wizard._tts_local_model_path_entry.get_visible() is True
    assert wizard._tts_voice_entry.get_visible() is True
    assert wizard._tts_local_auto_voice_dropdown.get_visible() is False


def test_do_activate_registers_new_page_sequence():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard()
    with patch("shuvoice.wizard.LayerShell.is_supported", return_value=False):
        wizard.do_activate()

    assert wizard._stack.get_child_by_name("welcome") is not None
    assert wizard._stack.get_child_by_name("asr") is not None
    assert wizard._stack.get_child_by_name("keybind") is not None
    assert wizard._stack.get_child_by_name("tts") is not None
    assert wizard._stack.get_child_by_name("done") is not None

    wizard._release_input_and_destroy_window()


def test_model_status_text_maps_cancelled_state():
    from shuvoice.wizard import WelcomeWizard

    assert "cancelled" in WelcomeWizard._model_download_status_text("cancelled").lower()


def test_model_status_text_maps_incompatible_streaming_state():
    from shuvoice.wizard import WelcomeWizard

    status = WelcomeWizard._model_download_status_text("incompatible_streaming").lower()
    assert "parakeet streaming" in status
    assert "zipformer" in status


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
        typing_final_injection_mode="auto",
        typing_text_case="default",
        tts_backend="elevenlabs",
        tts_default_voice_id="zNsotODqUhvbJ5wMG7Ei",
        tts_local_model_path=None,
        tts_local_voice=None,
    )
    maybe_download.assert_called_once_with(
        "moonshine",
        sherpa_model_name="sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
        progress_callback=None,
        cancel_requested=None,
        auto_install_missing=True,
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
    wizard._sherpa_provider = "cuda"
    wizard._keybind = "f9"
    wizard._tts_backend = "openai"
    wizard._tts_voice_id = "onyx"
    wizard.completed = False
    wizard._release_input_and_destroy_window = MagicMock()
    wizard.quit = MagicMock()

    with (
        patch("shuvoice.wizard.write_config") as write_config,
        patch("shuvoice.wizard.maybe_download_model", return_value=("skipped", "noop")),
        patch("shuvoice.wizard.write_marker"),
    ):
        WelcomeWizard._on_finish(wizard, None)

    write_config.assert_called_once_with(
        "sherpa",
        overwrite_existing=False,
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        sherpa_enable_parakeet_streaming=True,
        sherpa_provider="cuda",
        typing_final_injection_mode="auto",
        typing_text_case="default",
        tts_backend="openai",
        tts_default_voice_id="onyx",
        tts_local_model_path=None,
        tts_local_voice=None,
    )


def test_on_finish_passes_local_tts_settings_to_write_config():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard.__new__(WelcomeWizard)
    wizard._asr_backend = "moonshine"
    wizard._keybind = "f9"
    wizard._tts_backend = "local"
    wizard._tts_local_setup_mode = "manual"
    wizard._tts_voice_id = "default"
    wizard._tts_local_model_path = "/tmp/piper-models"
    wizard.completed = False
    wizard._release_input_and_destroy_window = MagicMock()
    wizard.quit = MagicMock()

    with (
        patch.object(WelcomeWizard, "_validate_tts_selection_for_finish", return_value=True),
        patch.object(WelcomeWizard, "_local_tts_resolved_voice", return_value="amy"),
        patch("shuvoice.wizard.write_config") as write_config,
        patch("shuvoice.wizard.maybe_download_model", return_value=("skipped", "noop")),
        patch("shuvoice.wizard.write_marker"),
    ):
        WelcomeWizard._on_finish(wizard, None)

    write_config.assert_called_once_with(
        "moonshine",
        overwrite_existing=False,
        sherpa_model_name="sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
        typing_final_injection_mode="auto",
        typing_text_case="default",
        tts_backend="local",
        tts_default_voice_id="amy",
        tts_local_model_path="/tmp/piper-models",
        tts_local_voice="amy",
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


def test_complete_finish_applies_zipformer_fallback_for_incompatible_parakeet_streaming():
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard.__new__(WelcomeWizard)
    wizard._asr_backend = "sherpa"
    wizard._sherpa_model_name = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    wizard._sherpa_enable_parakeet_streaming = True
    wizard._sherpa_provider = "cuda"
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

    with (
        patch("shuvoice.wizard.write_config") as write_config,
        patch("shuvoice.wizard.write_marker") as write_marker,
    ):
        WelcomeWizard._complete_finish(
            wizard,
            keybind_status="added",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            model_status="incompatible_streaming",
            model_message="not compatible",
        )

    write_config.assert_called_once_with(
        "sherpa",
        overwrite_existing=True,
        sherpa_model_name="sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
        sherpa_enable_parakeet_streaming=False,
        sherpa_provider="cuda",
        typing_final_injection_mode="auto",
        typing_text_case="default",
        tts_backend="elevenlabs",
        tts_default_voice_id="zNsotODqUhvbJ5wMG7Ei",
        tts_local_model_path=None,
        tts_local_voice=None,
    )
    write_marker.assert_called_once()
    assert wizard._sherpa_model_name == "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
    assert wizard._sherpa_enable_parakeet_streaming is False


# -- MeloTTS wizard UI (VAL-WIZ-001, VAL-WIZ-002, VAL-WIZ-003) ---------------


def test_melotts_controls_visible_when_melotts_selected():
    """When melotts is selected in TTS dropdown, melotts-specific controls appear."""
    from shuvoice.wizard import WelcomeWizard
    from shuvoice.wizard_state import TTS_BACKENDS

    wizard = WelcomeWizard()
    wizard._build_tts_page()

    # Find melotts index in TTS_BACKENDS
    melotts_idx = next(i for i, (bid, _, _) in enumerate(TTS_BACKENDS) if bid == "melotts")
    wizard._tts_provider_dropdown.set_selected(melotts_idx)

    assert wizard._tts_backend == "melotts"
    # MeloTTS device dropdown should be visible
    assert wizard._tts_melotts_device_dropdown.get_visible() is True
    # Local Piper controls should be hidden
    assert wizard._tts_local_setup_mode_dropdown.get_visible() is False
    # Voice entry should be visible (for voice selection)
    assert wizard._tts_voice_entry.get_visible() is True


def test_melotts_controls_hidden_when_other_backend_selected():
    """MeloTTS device dropdown is hidden when a non-melotts backend is selected."""
    from shuvoice.wizard import WelcomeWizard
    from shuvoice.wizard_state import TTS_BACKENDS

    wizard = WelcomeWizard()
    wizard._build_tts_page()

    # Select melotts first
    melotts_idx = next(i for i, (bid, _, _) in enumerate(TTS_BACKENDS) if bid == "melotts")
    wizard._tts_provider_dropdown.set_selected(melotts_idx)
    assert wizard._tts_melotts_device_dropdown.get_visible() is True

    # Switch to elevenlabs (index 0)
    wizard._tts_provider_dropdown.set_selected(0)
    assert wizard._tts_backend == "elevenlabs"
    assert wizard._tts_melotts_device_dropdown.get_visible() is False


def test_melotts_device_dropdown_updates_state():
    """MeloTTS device dropdown updates internal state correctly."""
    from shuvoice.wizard import WelcomeWizard
    from shuvoice.wizard_state import TTS_BACKENDS

    wizard = WelcomeWizard()
    wizard._build_tts_page()

    melotts_idx = next(i for i, (bid, _, _) in enumerate(TTS_BACKENDS) if bid == "melotts")
    wizard._tts_provider_dropdown.set_selected(melotts_idx)

    # Default should be "auto"
    assert wizard._tts_melotts_device == "auto"

    # Switch to CPU (index 1)
    wizard._tts_melotts_device_dropdown.set_selected(1)
    assert wizard._tts_melotts_device == "cpu"

    # Switch to CUDA (index 2)
    wizard._tts_melotts_device_dropdown.set_selected(2)
    assert wizard._tts_melotts_device == "cuda"


def test_melotts_voice_entry_shows_default_voice():
    """When melotts is selected, voice entry shows the default MeloTTS voice."""
    from shuvoice.wizard import WelcomeWizard
    from shuvoice.wizard_state import TTS_BACKENDS

    wizard = WelcomeWizard()
    wizard._build_tts_page()

    melotts_idx = next(i for i, (bid, _, _) in enumerate(TTS_BACKENDS) if bid == "melotts")
    wizard._tts_provider_dropdown.set_selected(melotts_idx)

    assert wizard._tts_voice_entry.get_text() == "EN-US"


def test_on_finish_passes_melotts_settings_to_write_config():
    """_on_finish correctly passes melotts-specific kwargs to write_config."""
    from shuvoice.wizard import WelcomeWizard

    wizard = WelcomeWizard.__new__(WelcomeWizard)
    wizard._asr_backend = "moonshine"
    wizard._keybind = "f9"
    wizard._tts_backend = "melotts"
    wizard._tts_voice_id = "EN-BR"
    wizard._tts_melotts_device = "cuda"
    wizard.completed = False
    wizard._release_input_and_destroy_window = MagicMock()
    wizard.quit = MagicMock()

    with (
        patch("shuvoice.wizard.write_config") as mock_write_config,
        patch("shuvoice.wizard.maybe_download_model", return_value=("skipped", "noop")),
        patch("shuvoice.wizard.write_marker"),
    ):
        WelcomeWizard._on_finish(wizard, None)

    mock_write_config.assert_called_once()
    call_kwargs = mock_write_config.call_args
    assert call_kwargs[1]["tts_backend"] == "melotts"
    assert call_kwargs[1]["tts_default_voice_id"] == "EN-BR"
    assert call_kwargs[1]["tts_melotts_device"] == "cuda"
