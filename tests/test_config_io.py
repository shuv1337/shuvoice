from __future__ import annotations

from pathlib import Path

import pytest

from shuvoice.config import CURRENT_CONFIG_VERSION
from shuvoice.config_io import load_raw, write_atomic


def test_load_raw_missing_file_returns_current_schema(tmp_path: Path):
    raw = load_raw(tmp_path / "config.toml")
    assert raw["config_version"] == CURRENT_CONFIG_VERSION


def test_write_atomic_creates_backup_for_existing_file(tmp_path: Path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("config_version = 1\n")

    backup = write_atomic(
        config_file, {"config_version": CURRENT_CONFIG_VERSION, "asr": {"asr_backend": "sherpa"}}
    )

    assert backup is not None
    assert backup.exists()
    content = config_file.read_text()
    assert "[asr]" in content
    assert 'asr_backend = "sherpa"' in content


def test_write_atomic_preserves_existing_file_when_replace_fails(tmp_path: Path, monkeypatch):
    config_file = tmp_path / "config.toml"
    original = 'config_version = 1\n[asr]\nasr_backend = "nemo"\n'
    config_file.write_text(original)

    def fail_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr("shuvoice.config_io.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_atomic(
            config_file,
            {"config_version": CURRENT_CONFIG_VERSION, "asr": {"asr_backend": "sherpa"}},
        )

    assert config_file.read_text() == original


def test_round_trip_load_write_reload(tmp_path: Path):
    config_file = tmp_path / "config.toml"

    write_atomic(
        config_file,
        {
            "config_version": CURRENT_CONFIG_VERSION,
            "audio": {"sample_rate": 16000, "chunk_ms": 100},
            "asr": {"asr_backend": "moonshine", "moonshine_model_name": "moonshine/tiny"},
        },
    )

    raw = load_raw(config_file)

    assert raw["config_version"] == CURRENT_CONFIG_VERSION
    assert raw["audio"]["sample_rate"] == 16000
    assert raw["asr"]["asr_backend"] == "moonshine"


def test_write_atomic_quotes_non_bare_keys(tmp_path: Path):
    config_file = tmp_path / "config.toml"

    write_atomic(
        config_file,
        {
            "config_version": CURRENT_CONFIG_VERSION,
            "typing": {
                "text_replacements": {
                    "shove voice": "ShuVoice",
                    "high-per-land": "Hyprland",
                }
            },
        },
    )

    content = config_file.read_text()
    assert '"shove voice" = "ShuVoice"' in content
    assert 'high-per-land = "Hyprland"' in content or '"high-per-land" = "Hyprland"' in content

    raw = load_raw(config_file)
    assert raw["typing"]["text_replacements"]["shove voice"] == "ShuVoice"
