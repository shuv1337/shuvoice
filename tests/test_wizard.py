"""Tests for the welcome wizard state module (headless-safe)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from shuvoice.wizard_state import (
    ASR_BACKENDS,
    DEFAULT_FINAL_INJECTION_MODE,
    DEFAULT_KEYBIND_ID,
    DEFAULT_SHERPA_MODEL_NAME,
    KEYBIND_PRESETS,
    PARAKEET_TDT_V3_INT8_MODEL_NAME,
    TTS_BACKENDS,
    auto_add_hyprland_keybind,
    default_tts_voice_for_backend,
    format_hyprland_bind,
    format_hyprland_bind_for_keybind,
    format_summary,
    needs_wizard,
    tts_voice_label,
    write_config,
    write_marker,
)


def test_needs_wizard_true_on_fresh_install(tmp_path):
    """needs_wizard returns True when no marker file and no config.toml exist."""
    with patch("shuvoice.wizard_state.Config") as mock_config:
        mock_config.data_dir.return_value = tmp_path
        mock_config.config_dir.return_value = tmp_path
        assert needs_wizard() is True


def test_needs_wizard_false_after_completion(tmp_path):
    """needs_wizard returns False after the marker file is created."""
    (tmp_path / ".wizard-done").write_text("done\n")
    with patch("shuvoice.wizard_state.Config") as mock_config:
        mock_config.data_dir.return_value = tmp_path
        mock_config.config_dir.return_value = tmp_path
        assert needs_wizard() is False


def test_needs_wizard_false_with_existing_config(tmp_path):
    """needs_wizard returns False when config.toml already exists (upgrade path).

    An existing installation that upgrades but lacks the .wizard-done marker
    must not be forced through the wizard.
    """
    (tmp_path / "config.toml").write_text("# existing config\n")
    with patch("shuvoice.wizard_state.Config") as mock_config:
        mock_config.data_dir.return_value = tmp_path
        mock_config.config_dir.return_value = tmp_path
        assert needs_wizard() is False


def test_write_marker_creates_file(tmp_path):
    """write_marker creates the .wizard-done marker file."""
    with patch("shuvoice.wizard_state.Config") as mock_config:
        mock_config.data_dir.return_value = tmp_path
        write_marker()
        assert (tmp_path / ".wizard-done").exists()
        assert (tmp_path / ".wizard-done").read_text() == "done\n"


def test_write_config_creates_toml_with_cuda(tmp_path):
    """write_config writes config.toml with CUDA provider when Sherpa CUDA is available."""
    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_sherpa_cuda_provider", return_value=True),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config("sherpa")

        config_file = tmp_path / "config.toml"
        assert config_file.exists()

        content = config_file.read_text()
        assert 'asr_backend = "sherpa"' in content
        assert f'sherpa_model_name = "{DEFAULT_SHERPA_MODEL_NAME}"' in content
        assert 'sherpa_provider = "cuda"' in content
        assert 'sherpa_decode_mode = "auto"' in content
        assert "instant_mode = false" in content
        assert "sherpa_enable_parakeet_streaming = false" in content
        assert 'output_mode = "final_only"' in content
        assert f'typing_final_injection_mode = "{DEFAULT_FINAL_INJECTION_MODE}"' in content
        assert 'typing_text_case = "default"' in content
        assert 'tts_backend = "elevenlabs"' in content
        assert 'tts_default_voice_id = "zNsotODqUhvbJ5wMG7Ei"' in content
        assert "use_clipboard_for_final = true" in content


def test_write_config_creates_toml_without_cuda(tmp_path):
    """write_config writes config.toml with CPU provider when Sherpa CUDA is unavailable."""
    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_sherpa_cuda_provider", return_value=False),
        patch("shuvoice.wizard_state._detect_cuda", return_value=True),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config("sherpa")

        config_file = tmp_path / "config.toml"
        content = config_file.read_text()
        assert 'asr_backend = "sherpa"' in content
        assert f'sherpa_model_name = "{DEFAULT_SHERPA_MODEL_NAME}"' in content
        assert 'sherpa_provider = "cpu"' in content
        assert 'sherpa_decode_mode = "auto"' in content
        assert "instant_mode = false" in content
        assert "sherpa_enable_parakeet_streaming = false" in content
        assert 'output_mode = "final_only"' in content
        assert f'typing_final_injection_mode = "{DEFAULT_FINAL_INJECTION_MODE}"' in content
        assert 'typing_text_case = "default"' in content
        assert 'tts_backend = "elevenlabs"' in content
        assert 'tts_default_voice_id = "zNsotODqUhvbJ5wMG7Ei"' in content
        assert "use_clipboard_for_final = true" in content


def test_write_config_sherpa_explicit_provider_overrides_detection(tmp_path):
    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_sherpa_cuda_provider", return_value=True),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config("sherpa", sherpa_provider="cpu")

    content = (tmp_path / "config.toml").read_text()
    assert 'sherpa_provider = "cpu"' in content


def test_write_config_sherpa_explicit_cuda_persists_even_before_runtime_ready(tmp_path):
    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_sherpa_cuda_provider", return_value=False),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config("sherpa", sherpa_provider="cuda")

    content = (tmp_path / "config.toml").read_text()
    assert 'sherpa_provider = "cuda"' in content


def test_write_config_sherpa_custom_model_name(tmp_path):
    """write_config persists selected Sherpa model name."""
    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_sherpa_cuda_provider", return_value=False),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config("sherpa", sherpa_model_name=PARAKEET_TDT_V3_INT8_MODEL_NAME)

        content = (tmp_path / "config.toml").read_text()
        assert f'sherpa_model_name = "{PARAKEET_TDT_V3_INT8_MODEL_NAME}"' in content
        assert "instant_mode = true" in content
        assert 'sherpa_decode_mode = "offline_instant"' in content
        assert "sherpa_enable_parakeet_streaming = false" in content
        assert 'output_mode = "final_only"' in content
        assert f'typing_final_injection_mode = "{DEFAULT_FINAL_INJECTION_MODE}"' in content
        assert 'typing_text_case = "default"' in content
        assert "use_clipboard_for_final = true" in content


def test_write_config_sherpa_parakeet_streaming_override(tmp_path):
    """write_config can persist Parakeet streaming override profile."""
    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_sherpa_cuda_provider", return_value=False),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config(
            "sherpa",
            sherpa_model_name=PARAKEET_TDT_V3_INT8_MODEL_NAME,
            sherpa_enable_parakeet_streaming=True,
        )

        content = (tmp_path / "config.toml").read_text()
        assert f'sherpa_model_name = "{PARAKEET_TDT_V3_INT8_MODEL_NAME}"' in content
        assert 'sherpa_decode_mode = "streaming"' in content
        assert "instant_mode = false" in content
        assert "sherpa_enable_parakeet_streaming = true" in content
        assert 'output_mode = "final_only"' in content
        assert f'typing_final_injection_mode = "{DEFAULT_FINAL_INJECTION_MODE}"' in content
        assert 'typing_text_case = "default"' in content
        assert "use_clipboard_for_final = true" in content


def test_write_config_nemo_sets_device(tmp_path):
    """write_config sets device key for NeMo backend."""
    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_cuda", return_value=True),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config("nemo")

        content = (tmp_path / "config.toml").read_text()
        assert 'asr_backend = "nemo"' in content
        assert 'device = "cuda"' in content
        assert f'typing_final_injection_mode = "{DEFAULT_FINAL_INJECTION_MODE}"' in content


def test_write_config_typing_mode_direct_updates_legacy_flag(tmp_path):
    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_cuda", return_value=False),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config("moonshine", typing_final_injection_mode="direct")

    content = (tmp_path / "config.toml").read_text()
    assert 'typing_final_injection_mode = "direct"' in content
    assert 'typing_text_case = "default"' in content
    assert "use_clipboard_for_final = false" in content


def test_write_config_rejects_invalid_typing_mode(tmp_path):
    with patch("shuvoice.wizard_state.Config") as mock_config:
        mock_config.config_dir.return_value = tmp_path
        with patch("shuvoice.wizard_state._detect_cuda", return_value=False):
            with pytest.raises(ValueError, match="typing_final_injection_mode"):
                write_config("nemo", typing_final_injection_mode="invalid")


def test_write_config_rejects_invalid_typing_text_case(tmp_path):
    with patch("shuvoice.wizard_state.Config") as mock_config:
        mock_config.config_dir.return_value = tmp_path
        with patch("shuvoice.wizard_state._detect_cuda", return_value=False):
            with pytest.raises(ValueError, match="typing_text_case"):
                write_config("nemo", typing_text_case="titlecase")


def test_write_config_rejects_invalid_sherpa_provider(tmp_path):
    with patch("shuvoice.wizard_state.Config") as mock_config:
        mock_config.config_dir.return_value = tmp_path
        with pytest.raises(ValueError, match="sherpa_provider"):
            write_config("sherpa", sherpa_provider="rocm")


def test_write_config_persists_tts_provider_and_voice(tmp_path):
    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_cuda", return_value=False),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config("nemo", tts_backend="openai", tts_default_voice_id="nova")

    content = (tmp_path / "config.toml").read_text()
    assert 'tts_backend = "openai"' in content
    assert 'tts_default_voice_id = "nova"' in content


def test_write_config_persists_local_tts_settings(tmp_path):
    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_cuda", return_value=False),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config(
            "nemo",
            tts_backend="local",
            tts_default_voice_id="amy",
            tts_local_model_path="/models/piper",
            tts_local_voice="amy",
        )

    content = (tmp_path / "config.toml").read_text()
    assert 'tts_backend = "local"' in content
    assert 'tts_default_voice_id = "amy"' in content
    assert 'tts_local_model_path = "/models/piper"' in content
    assert 'tts_local_voice = "amy"' in content


def test_write_config_does_not_overwrite_existing(tmp_path):
    """write_config preserves an existing config.toml."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("# existing config\n")

    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_cuda", return_value=False),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config("nemo")

        assert config_file.read_text() == "# existing config\n"


def test_write_config_overwrite_updates_existing_asr_backend(tmp_path):
    """Forced wizard reruns should update [asr].asr_backend in-place."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """# existing config
[asr]
asr_backend = \"sherpa\"
sherpa_provider = \"cpu\"
model_name = \"nvidia/nemotron-speech-streaming-en-0.6b\"

[typing]
output_mode = \"final_only\"
"""
    )

    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_cuda", return_value=True),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config("moonshine", overwrite_existing=True)

    content = config_file.read_text()
    assert 'asr_backend = "moonshine"' in content
    assert 'moonshine_provider = "cuda"' in content
    assert 'model_name = "nvidia/nemotron-speech-streaming-en-0.6b"' in content
    assert '[typing]\noutput_mode = "final_only"' in content


def test_write_config_overwrite_adds_asr_section_if_missing(tmp_path):
    """Forced wizard reruns should add [asr] if an old config lacks it."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """# existing config
[typing]
output_mode = \"final_only\"
"""
    )

    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_cuda", return_value=False),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config("nemo", overwrite_existing=True)

    content = config_file.read_text()
    assert '[typing]\noutput_mode = "final_only"' in content
    assert 'asr_backend = "nemo"' in content
    assert 'device = "cpu"' in content


# -- format_summary -----------------------------------------------------------


def test_format_summary_contains_backend_and_keybind():
    """format_summary includes backend name and default keybind label."""
    assert DEFAULT_KEYBIND_ID == "right_ctrl"
    result = format_summary("sherpa")
    assert "Sherpa-ONNX" in result
    assert "Final injection:  Auto (recommended)" in result
    assert "TTS provider:     ElevenLabs" in result
    assert "TTS voice:        Default (zNsotODqUhvbJ5wMG7Ei)" in result
    assert "Right Control" in result
    assert "hyprland.conf" in result


def test_format_summary_shows_selected_final_injection_mode():
    result = format_summary("nemo", typing_final_injection_mode="direct")
    assert "Final injection:  Direct typing (keystroke simulation)" in result


def test_format_summary_shows_selected_typing_text_case():
    result = format_summary("nemo", typing_text_case="lowercase")
    assert "Text case:        Lowercase" in result


def test_format_summary_shows_selected_tts_provider_and_voice():
    result = format_summary("nemo", tts_backend="openai", tts_default_voice_id="nova")
    assert "TTS provider:     OpenAI" in result
    assert "TTS voice:        Nova" in result


def test_format_summary_shows_local_tts_path_and_voice():
    result = format_summary(
        "nemo",
        tts_backend="local",
        tts_default_voice_id="amy",
        tts_local_model_path="/models/piper",
    )
    assert "TTS provider:     Local Piper" in result
    assert "TTS model path:   /models/piper" in result
    assert "TTS voice:        amy" in result


def test_format_summary_sherpa_default_model_shows_streaming_mode():
    result = format_summary("sherpa", sherpa_model_name=DEFAULT_SHERPA_MODEL_NAME)
    assert "Sherpa profile: Streaming" in result
    assert "Sherpa device:  CPU" in result
    assert "Sherpa model:   Zipformer Kroko (default)" in result
    assert "Sherpa decode:  Streaming (auto)" in result
    assert "Output mode:    final_only" in result


def test_format_summary_sherpa_cuda_provider_label():
    result = format_summary(
        "sherpa",
        sherpa_model_name=DEFAULT_SHERPA_MODEL_NAME,
        sherpa_provider="cuda",
    )
    assert "Sherpa device:  GPU (CUDA)" in result


def test_format_summary_sherpa_parakeet_model_label():
    result = format_summary("sherpa", sherpa_model_name=PARAKEET_TDT_V3_INT8_MODEL_NAME)
    assert "Sherpa profile: Instant (Parakeet)" in result
    assert "Sherpa model:   Parakeet TDT v3 (int8)" in result
    assert "Sherpa decode:  Offline instant (auto-enabled)" in result
    assert "Output mode:    final_only" in result


def test_format_summary_sherpa_parakeet_streaming_profile_label():
    result = format_summary(
        "sherpa",
        sherpa_model_name=PARAKEET_TDT_V3_INT8_MODEL_NAME,
        sherpa_enable_parakeet_streaming=True,
    )
    assert "Sherpa profile: Streaming (Parakeet)" in result
    assert "Sherpa model:   Parakeet TDT v3 (int8)" in result
    assert "Sherpa decode:  Streaming (explicit override)" in result
    assert "Output mode:    final_only" in result


def test_format_summary_includes_hyprland_bind_lines_for_preset():
    """Preset keybinds include copy-pasteable Hyprland bind/bindr lines."""
    result = format_summary("nemo", "super_v")
    assert "NeMo" in result
    assert "Super + V" in result
    assert "bind = SUPER, V, exec, shuvoice control start --control-wait-sec 0" in result
    assert "bindr = SUPER, V, exec, shuvoice control stop --control-wait-sec 0" in result
    assert "bind = SUPER CTRL, S, exec, shuvoice control tts_speak --control-wait-sec 0" in result


def test_format_summary_manual_mode_shows_manual_copy_hint():
    result = format_summary("sherpa", "insert", auto_add_keybind=False)
    assert "Add to ~/.config/hypr/hyprland.conf" in result


def test_format_summary_right_ctrl_includes_extra_release_line():
    result = format_summary("sherpa", "right_ctrl")
    assert "bindr = CTRL, Control_R, exec, shuvoice control stop --control-wait-sec 0" in result


def test_format_summary_custom_keybind_shows_readme_hint():
    """Custom keybind selection points user to README for examples."""
    result = format_summary("sherpa", "custom")
    assert "Custom" in result
    assert "README.md" in result
    assert "bind =" not in result


def test_format_summary_with_moonshine():
    """format_summary handles non-default backend selection."""
    result = format_summary("moonshine")
    assert "Moonshine-ONNX" in result


# -- format_hyprland_bind -----------------------------------------------------


def test_format_hyprland_bind_no_modifier():
    result = format_hyprland_bind(", F9")
    assert result == (
        "bind = , F9, exec, shuvoice control start --control-wait-sec 0\n"
        "bindr = , F9, exec, shuvoice control stop --control-wait-sec 0"
    )


def test_format_hyprland_bind_with_modifier():
    result = format_hyprland_bind("SUPER, V")
    assert result == (
        "bind = SUPER, V, exec, shuvoice control start --control-wait-sec 0\n"
        "bindr = SUPER, V, exec, shuvoice control stop --control-wait-sec 0"
    )


def test_format_hyprland_bind_for_right_ctrl_includes_extra_release_line():
    result = format_hyprland_bind_for_keybind("right_ctrl", ", Control_R")
    assert "bind = , Control_R, exec, shuvoice control start --control-wait-sec 0" in result
    assert "bindr = , Control_R, exec, shuvoice control stop --control-wait-sec 0" in result
    assert "bindr = CTRL, Control_R, exec, shuvoice control stop --control-wait-sec 0" in result


# -- auto_add_hyprland_keybind ------------------------------------------------


def test_auto_add_hyprland_keybind_adds_lines_when_key_unused(tmp_path):
    hypr_dir = tmp_path / "hypr"
    hypr_dir.mkdir(parents=True)
    hypr_conf = hypr_dir / "hyprland.conf"
    hypr_conf.write_text("# user config\n")

    with patch("shuvoice.wizard_state.Config") as mock_config:
        mock_config.config_dir.return_value = tmp_path / "shuvoice"
        status, message = auto_add_hyprland_keybind("insert")

    assert status == "added"
    assert "Added ShuVoice keybind" in message
    content = hypr_conf.read_text()
    assert "bind = , Insert, exec," in content
    assert "bindr = , Insert, exec," in content
    assert "control start --control-wait-sec 0" in content
    assert "control stop --control-wait-sec 0" in content
    assert "control tts_speak --control-wait-sec 0" in content


def test_auto_add_hyprland_keybind_uses_resolved_binary(tmp_path):
    hypr_dir = tmp_path / "hypr"
    hypr_dir.mkdir(parents=True)
    hypr_conf = hypr_dir / "hyprland.conf"
    hypr_conf.write_text("# user config\n")

    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch(
            "shuvoice.wizard_state._resolve_shuvoice_command",
            return_value="/opt/shuvoice/bin/shuvoice",
        ),
    ):
        mock_config.config_dir.return_value = tmp_path / "shuvoice"
        status, _message = auto_add_hyprland_keybind("insert")

    assert status == "added"
    content = hypr_conf.read_text()
    assert "/opt/shuvoice/bin/shuvoice control start --control-wait-sec 0" in content
    assert "/opt/shuvoice/bin/shuvoice control stop --control-wait-sec 0" in content
    assert "/opt/shuvoice/bin/shuvoice control tts_speak --control-wait-sec 0" in content


def test_auto_add_hyprland_keybind_reports_conflict(tmp_path):
    hypr_dir = tmp_path / "hypr"
    hypr_dir.mkdir(parents=True)
    hypr_conf = hypr_dir / "hyprland.conf"
    hypr_conf.write_text("bind = , Insert, exec, grimblast save area\n")

    with patch("shuvoice.wizard_state.Config") as mock_config:
        mock_config.config_dir.return_value = tmp_path / "shuvoice"
        status, message = auto_add_hyprland_keybind("insert")

    assert status == "conflict"
    assert "not adding ShuVoice binds" in message
    content = hypr_conf.read_text()
    assert "shuvoice control" not in content


def test_auto_add_hyprland_keybind_detects_existing_bind(tmp_path):
    hypr_dir = tmp_path / "hypr"
    hypr_dir.mkdir(parents=True)
    hypr_conf = hypr_dir / "hyprland.conf"
    hypr_conf.write_text(
        "bind = , Insert, exec, shuvoice control start --control-wait-sec 0\n"
        "bindr = , Insert, exec, shuvoice control stop --control-wait-sec 0\n"
        "bind = SUPER CTRL, S, exec, shuvoice control tts_speak --control-wait-sec 0\n"
    )

    with patch("shuvoice.wizard_state.Config") as mock_config:
        mock_config.config_dir.return_value = tmp_path / "shuvoice"
        status, _message = auto_add_hyprland_keybind("insert")

    assert status == "already_configured"


def test_auto_add_hyprland_keybind_skips_custom():
    status, _message = auto_add_hyprland_keybind("custom")
    assert status == "skipped_custom"


def test_auto_add_hyprland_keybind_updates_existing_shuvoice_bindings_conf(tmp_path):
    hypr_dir = tmp_path / "hypr"
    hypr_dir.mkdir(parents=True)
    bindings_conf = hypr_dir / "bindings.conf"
    hyprland_conf = hypr_dir / "hyprland.conf"

    bindings_conf.write_text(
        "bind = , Insert, exec, /venv/bin/shuvoice control start --control-wait-sec 0\n"
        "bindr = , Insert, exec, /venv/bin/shuvoice control stop --control-wait-sec 0\n"
        "bind = SUPER CTRL, S, exec, /venv/bin/shuvoice control tts_speak --control-wait-sec 0\n"
    )
    hyprland_conf.write_text("source = ~/.config/hypr/bindings.conf\n")

    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._resolve_shuvoice_command", return_value="/venv/bin/shuvoice"),
    ):
        mock_config.config_dir.return_value = tmp_path / "shuvoice"
        status, _message = auto_add_hyprland_keybind("right_ctrl")

    assert status == "added"

    bindings_text = bindings_conf.read_text()
    assert "Insert" not in bindings_text
    assert "Control_R" in bindings_text
    assert "/venv/bin/shuvoice control start --control-wait-sec 0" in bindings_text
    assert "/venv/bin/shuvoice control stop --control-wait-sec 0" in bindings_text
    assert "/venv/bin/shuvoice control tts_speak --control-wait-sec 0" in bindings_text
    assert (
        "bindr = CTRL, Control_R, exec, /venv/bin/shuvoice control stop --control-wait-sec 0"
        in bindings_text
    )

    hyprland_text = hyprland_conf.read_text()
    assert "control start --control-wait-sec 0" not in hyprland_text
    assert "control stop --control-wait-sec 0" not in hyprland_text
    assert "control tts_speak --control-wait-sec 0" not in hyprland_text


# -- KEYBIND_PRESETS ----------------------------------------------------------


def test_keybind_presets_structure():
    """KEYBIND_PRESETS has useful presets plus a custom option."""
    assert len(KEYBIND_PRESETS) >= 4
    ids = [kid for kid, _, _, _ in KEYBIND_PRESETS]
    assert "insert" in ids
    assert "right_ctrl" in ids
    assert "custom" in ids
    # custom must have None for the hyprland key spec
    custom = next(p for p in KEYBIND_PRESETS if p[0] == "custom")
    assert custom[2] is None


def test_asr_backends_has_three_entries():
    """ASR_BACKENDS should list exactly three backend options."""
    assert len(ASR_BACKENDS) == 3
    ids = [bid for bid, _, _ in ASR_BACKENDS]
    assert "nemo" in ids
    assert "sherpa" in ids
    assert "moonshine" in ids


# -- MeloTTS wizard state (VAL-WIZ-001, VAL-WIZ-002, VAL-WIZ-003) -----------


def test_tts_backends_includes_melotts():
    """TTS_BACKENDS includes a melotts entry with name and description."""
    ids = [bid for bid, _, _ in TTS_BACKENDS]
    assert "melotts" in ids

    melotts = next(entry for entry in TTS_BACKENDS if entry[0] == "melotts")
    assert melotts[1] == "MeloTTS"
    assert "MeloTTS" in melotts[2]


def test_default_tts_voice_for_backend_melotts():
    """default_tts_voice_for_backend('melotts') returns 'EN-US'."""
    assert default_tts_voice_for_backend("melotts") == "EN-US"


def test_tts_voice_label_melotts_voices():
    """tts_voice_label handles melotts voice IDs with human-readable labels."""
    assert tts_voice_label("melotts", "EN-US") == "American English"
    assert tts_voice_label("melotts", "EN-BR") == "British English"
    assert tts_voice_label("melotts", "EN-INDIA") == "Indian English"
    assert tts_voice_label("melotts", "EN-AU") == "Australian English"
    assert tts_voice_label("melotts", "EN-Newest") == "Newest English"


def test_tts_voice_label_melotts_unknown_returns_raw():
    """tts_voice_label for melotts with unknown voice ID returns the raw value."""
    assert tts_voice_label("melotts", "custom-voice") == "custom-voice"


def test_write_config_persists_melotts_settings(tmp_path):
    """write_config with melotts settings produces correct config.toml."""
    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_cuda", return_value=False),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config(
            "nemo",
            tts_backend="melotts",
            tts_default_voice_id="EN-BR",
            tts_melotts_device="cuda",
        )

    content = (tmp_path / "config.toml").read_text()
    assert 'tts_backend = "melotts"' in content
    assert 'tts_default_voice_id = "EN-BR"' in content
    assert 'tts_melotts_device = "cuda"' in content


def test_write_config_persists_melotts_default_voice(tmp_path):
    """write_config defaults melotts voice to EN-US when not specified."""
    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_cuda", return_value=False),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config("nemo", tts_backend="melotts")

    content = (tmp_path / "config.toml").read_text()
    assert 'tts_backend = "melotts"' in content
    assert 'tts_default_voice_id = "EN-US"' in content


def test_write_config_melotts_round_trip(tmp_path):
    """Wizard write → config load round-trip for melotts (VAL-CROSS-002)."""
    from shuvoice.config import Config as RealConfig

    config_dir = tmp_path / "shuvoice"
    config_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_cuda", return_value=False),
    ):
        mock_config.config_dir.return_value = config_dir
        write_config(
            "nemo",
            tts_backend="melotts",
            tts_default_voice_id="EN-AU",
            tts_melotts_device="cpu",
        )

    config_file = config_dir / "config.toml"
    assert config_file.exists()

    with patch.object(RealConfig, "config_path", return_value=config_file):
        cfg = RealConfig.load()

    assert cfg.tts_backend == "melotts"
    assert cfg.tts_default_voice_id == "EN-AU"
    assert cfg.tts_melotts_device == "cpu"


def test_format_summary_shows_melotts_provider_and_voice():
    """format_summary displays MeloTTS as TTS provider with correct voice label."""
    result = format_summary("nemo", tts_backend="melotts", tts_default_voice_id="EN-BR")
    assert "TTS provider:     MeloTTS" in result
    assert "TTS voice:        British English" in result


def test_write_config_rejects_invalid_tts_backend_but_accepts_melotts(tmp_path):
    """write_config accepts 'melotts' but rejects unknown tts backends."""
    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_cuda", return_value=False),
    ):
        mock_config.config_dir.return_value = tmp_path
        with pytest.raises(ValueError, match="tts_backend"):
            write_config("nemo", tts_backend="nonexistent")

    # melotts should not raise
    with (
        patch("shuvoice.wizard_state.Config") as mock_config,
        patch("shuvoice.wizard_state._detect_cuda", return_value=False),
    ):
        mock_config.config_dir.return_value = tmp_path
        write_config("nemo", tts_backend="melotts")
