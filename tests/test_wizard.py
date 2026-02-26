"""Tests for the welcome wizard state module (headless-safe)."""

from __future__ import annotations

from unittest.mock import patch

from shuvoice.wizard_state import (
    ASR_BACKENDS,
    DEFAULT_SHERPA_MODEL_NAME,
    KEYBIND_PRESETS,
    PARAKEET_TDT_V3_INT8_MODEL_NAME,
    auto_add_hyprland_keybind,
    format_hyprland_bind,
    format_hyprland_bind_for_keybind,
    format_summary,
    needs_wizard,
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
    """format_summary includes backend name and keybind label."""
    result = format_summary("sherpa")
    assert "Sherpa-ONNX" in result
    assert "Insert" in result
    assert "hyprland.conf" in result


def test_format_summary_sherpa_default_model_shows_streaming_mode():
    result = format_summary("sherpa", sherpa_model_name=DEFAULT_SHERPA_MODEL_NAME)
    assert "Sherpa profile: Streaming" in result
    assert "Sherpa model:   Zipformer Kroko (default)" in result
    assert "Sherpa decode:  Streaming (auto)" in result


def test_format_summary_sherpa_parakeet_model_label():
    result = format_summary("sherpa", sherpa_model_name=PARAKEET_TDT_V3_INT8_MODEL_NAME)
    assert "Sherpa profile: Instant (Parakeet)" in result
    assert "Sherpa model:   Parakeet TDT v3 (int8)" in result
    assert "Sherpa decode:  Offline instant (auto-enabled)" in result


def test_format_summary_includes_hyprland_bind_lines_for_preset():
    """Preset keybinds include copy-pasteable Hyprland bind/bindr lines."""
    result = format_summary("nemo", "super_v")
    assert "NeMo" in result
    assert "Super + V" in result
    assert "bind = SUPER, V, exec, shuvoice --control start" in result
    assert "bindr = SUPER, V, exec, shuvoice --control stop" in result


def test_format_summary_manual_mode_shows_manual_copy_hint():
    result = format_summary("sherpa", "insert", auto_add_keybind=False)
    assert "Add to ~/.config/hypr/hyprland.conf" in result


def test_format_summary_right_ctrl_includes_extra_release_line():
    result = format_summary("sherpa", "right_ctrl")
    assert "bindr = CTRL, Control_R, exec, shuvoice --control stop" in result


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
        "bind = , F9, exec, shuvoice --control start\nbindr = , F9, exec, shuvoice --control stop"
    )


def test_format_hyprland_bind_with_modifier():
    result = format_hyprland_bind("SUPER, V")
    assert result == (
        "bind = SUPER, V, exec, shuvoice --control start\n"
        "bindr = SUPER, V, exec, shuvoice --control stop"
    )


def test_format_hyprland_bind_for_right_ctrl_includes_extra_release_line():
    result = format_hyprland_bind_for_keybind("right_ctrl", ", Control_R")
    assert "bind = , Control_R, exec, shuvoice --control start" in result
    assert "bindr = , Control_R, exec, shuvoice --control stop" in result
    assert "bindr = CTRL, Control_R, exec, shuvoice --control stop" in result


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
    assert "--control start" in content
    assert "--control stop" in content


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
    assert "/opt/shuvoice/bin/shuvoice --control start" in content
    assert "/opt/shuvoice/bin/shuvoice --control stop" in content


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
    assert "shuvoice --control" not in content


def test_auto_add_hyprland_keybind_detects_existing_bind(tmp_path):
    hypr_dir = tmp_path / "hypr"
    hypr_dir.mkdir(parents=True)
    hypr_conf = hypr_dir / "hyprland.conf"
    hypr_conf.write_text(
        "bind = , Insert, exec, shuvoice --control start\n"
        "bindr = , Insert, exec, shuvoice --control stop\n"
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
        "bind = , Insert, exec, /venv/bin/shuvoice --control start\n"
        "bindr = , Insert, exec, /venv/bin/shuvoice --control stop\n"
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
    assert "/venv/bin/shuvoice --control start" in bindings_text
    assert "/venv/bin/shuvoice --control stop" in bindings_text
    assert "bindr = CTRL, Control_R, exec, /venv/bin/shuvoice --control stop" in bindings_text

    hyprland_text = hyprland_conf.read_text()
    assert "--control start" not in hyprland_text
    assert "--control stop" not in hyprland_text


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
