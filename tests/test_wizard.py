"""Tests for the welcome wizard state module (headless-safe)."""

from __future__ import annotations

from unittest.mock import patch

from shuvoice.wizard_state import (
    ASR_BACKENDS,
    HOTKEY_BACKENDS,
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
        write_config("sherpa", "ipc")

        config_file = tmp_path / "config.toml"
        assert config_file.exists()

        content = config_file.read_text()
        assert 'asr_backend = "sherpa"' in content
        assert 'hotkey_backend = "ipc"' in content


def test_write_config_does_not_overwrite_existing(tmp_path):
    """write_config preserves an existing config.toml."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("# existing config\n")

    with patch("shuvoice.wizard_state.Config") as mock_config:
        mock_config.config_dir.return_value = tmp_path
        write_config("nemo", "evdev")

        assert config_file.read_text() == "# existing config\n"


def test_format_summary_contains_backend_names():
    """format_summary includes human-readable backend names."""
    result = format_summary("sherpa", "ipc")
    assert "Sherpa-ONNX" in result
    assert "IPC socket" in result
    assert "KEY_RIGHTCTRL" in result


def test_format_summary_with_nemo():
    """format_summary handles the default nemo/evdev selection."""
    result = format_summary("nemo", "evdev")
    assert "NeMo (NVIDIA)" in result
    assert "evdev" in result


def test_asr_backends_has_three_entries():
    """ASR_BACKENDS should list exactly three backend options."""
    assert len(ASR_BACKENDS) == 3
    ids = [bid for bid, _, _ in ASR_BACKENDS]
    assert "nemo" in ids
    assert "sherpa" in ids
    assert "moonshine" in ids


def test_hotkey_backends_has_two_entries():
    """HOTKEY_BACKENDS should list exactly two backend options."""
    assert len(HOTKEY_BACKENDS) == 2
    ids = [bid for bid, _, _ in HOTKEY_BACKENDS]
    assert "evdev" in ids
    assert "ipc" in ids
