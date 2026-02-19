from __future__ import annotations

from pathlib import Path

from shuvoice.config import Config


def test_load_defaults_when_config_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))

    cfg = Config.load()

    assert cfg.sample_rate == 16000
    assert cfg.hotkey == "KEY_RIGHTCTRL"
    assert cfg.hotkey_backend == "evdev"
    assert cfg.output_mode == "final_only"


def test_load_flattens_sections_and_ignores_unknown(monkeypatch, tmp_path: Path):
    cfg_home = tmp_path / "cfg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))

    cfg_dir = cfg_home / "shuvoice"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    config_file = cfg_dir / "config.toml"

    config_file.write_text(
        """
[audio]
chunk_ms = 80
sample_rate = 16000
unknown_audio_key = 999

[hotkey]
hotkey_backend = "ipc"
hotkey = "KEY_F9"
hold_threshold_ms = 250
hotkey_device = "/dev/input/event9"

[typing]
output_mode = "streaming_partial"
use_clipboard_for_final = true
preserve_clipboard = true
typing_retry_attempts = 3
typing_retry_delay_ms = 20

[extra]
foo = "bar"
""".strip()
    )

    cfg = Config.load()

    assert cfg.chunk_ms == 80
    assert cfg.hotkey_backend == "ipc"
    assert cfg.hotkey == "KEY_F9"
    assert cfg.hold_threshold_ms == 250
    assert cfg.hotkey_device == "/dev/input/event9"
    assert cfg.output_mode == "streaming_partial"
    assert cfg.use_clipboard_for_final is True
    assert cfg.preserve_clipboard is True
    assert cfg.typing_retry_attempts == 3
    assert cfg.typing_retry_delay_ms == 20


def test_config_helpers_create_dirs(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    cdir = Config.config_dir()
    ddir = Config.data_dir()

    assert cdir.exists() and cdir.is_dir()
    assert ddir.exists() and ddir.is_dir()
