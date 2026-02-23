"""Tests for the welcome wizard state module (headless-safe)."""

from __future__ import annotations

from unittest.mock import patch

from shuvoice.wizard_state import (
    ASR_BACKENDS,
    KEYBIND_PRESETS,
    format_hyprland_bind,
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


def test_write_config_creates_toml(tmp_path):
    """write_config writes a valid config.toml with selected settings."""
    with patch("shuvoice.wizard_state.Config") as mock_config:
        mock_config.config_dir.return_value = tmp_path
        write_config("sherpa")

        config_file = tmp_path / "config.toml"
        assert config_file.exists()

        content = config_file.read_text()
        assert 'asr_backend = "sherpa"' in content
        assert "hotkey_backend" not in content


def test_write_config_does_not_overwrite_existing(tmp_path):
    """write_config preserves an existing config.toml."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("# existing config\n")

    with patch("shuvoice.wizard_state.Config") as mock_config:
        mock_config.config_dir.return_value = tmp_path
        write_config("nemo")

        assert config_file.read_text() == "# existing config\n"


# -- format_summary -----------------------------------------------------------


def test_format_summary_contains_backend_and_keybind():
    """format_summary includes backend name and keybind label."""
    result = format_summary("sherpa")
    assert "Sherpa-ONNX" in result
    assert "F9" in result
    assert "hyprland.conf" in result


def test_format_summary_includes_hyprland_bind_lines_for_preset():
    """Preset keybinds include copy-pasteable Hyprland bind/bindr lines."""
    result = format_summary("nemo", "super_v")
    assert "NeMo" in result
    assert "Super + V" in result
    assert "bind = SUPER, V, exec, shuvoice --control start" in result
    assert "bindr = SUPER, V, exec, shuvoice --control stop" in result


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
        "bind = , F9, exec, shuvoice --control start\n"
        "bindr = , F9, exec, shuvoice --control stop"
    )


def test_format_hyprland_bind_with_modifier():
    result = format_hyprland_bind("SUPER, V")
    assert result == (
        "bind = SUPER, V, exec, shuvoice --control start\n"
        "bindr = SUPER, V, exec, shuvoice --control stop"
    )


# -- KEYBIND_PRESETS ----------------------------------------------------------


def test_keybind_presets_structure():
    """KEYBIND_PRESETS has at least 3 presets plus a custom option."""
    assert len(KEYBIND_PRESETS) >= 4
    ids = [kid for kid, _, _, _ in KEYBIND_PRESETS]
    assert "f9" in ids
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
