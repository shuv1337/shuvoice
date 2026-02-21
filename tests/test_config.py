from __future__ import annotations

from pathlib import Path

import pytest

from shuvoice.config import Config


def test_load_defaults_when_config_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))

    cfg = Config.load()

    assert cfg.sample_rate == 16000
    assert cfg.hotkey == "KEY_RIGHTCTRL"
    assert cfg.hotkey_backend == "evdev"
    assert cfg.output_mode == "final_only"
    assert cfg.audio_queue_max_size == 200
    assert cfg.audio_feedback is True
    assert cfg.auto_capitalize is True
    assert cfg.streaming_stall_guard is True
    assert cfg.streaming_stall_chunks == 4


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
audio_queue_max_size = 55
silence_rms_threshold = 0.007
silence_rms_multiplier = 2.2
min_speech_ms = 90
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
auto_capitalize = false

[streaming]
streaming_stall_guard = false
streaming_stall_chunks = 6
streaming_stall_rms_ratio = 0.9
streaming_stall_flush_chunks = 2

[feedback]
audio_feedback = false
feedback_start_freq = 500
feedback_stop_freq = 400
feedback_duration_ms = 50
feedback_volume = 0.2

[extra]
foo = "bar"
""".strip()
    )

    cfg = Config.load()

    assert cfg.chunk_ms == 80
    assert cfg.audio_queue_max_size == 55
    assert cfg.silence_rms_threshold == 0.007
    assert cfg.silence_rms_multiplier == 2.2
    assert cfg.min_speech_ms == 90
    assert cfg.hotkey_backend == "ipc"
    assert cfg.hotkey == "KEY_F9"
    assert cfg.hold_threshold_ms == 250
    assert cfg.hotkey_device == "/dev/input/event9"
    assert cfg.output_mode == "streaming_partial"
    assert cfg.use_clipboard_for_final is True
    assert cfg.preserve_clipboard is True
    assert cfg.typing_retry_attempts == 3
    assert cfg.typing_retry_delay_ms == 20
    assert cfg.auto_capitalize is False
    assert cfg.streaming_stall_guard is False
    assert cfg.streaming_stall_chunks == 6
    assert cfg.streaming_stall_rms_ratio == 0.9
    assert cfg.streaming_stall_flush_chunks == 2
    assert cfg.audio_feedback is False
    assert cfg.feedback_start_freq == 500
    assert cfg.feedback_stop_freq == 400
    assert cfg.feedback_duration_ms == 50
    assert cfg.feedback_volume == 0.2


def test_config_helpers_create_dirs(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    cdir = Config.config_dir()
    ddir = Config.data_dir()

    assert cdir.exists() and cdir.is_dir()
    assert ddir.exists() and ddir.is_dir()


def test_native_chunk_samples_scaling():
    c = Config(right_context=0)
    assert c.native_chunk_samples == 1280

    c = Config(right_context=1)
    assert c.native_chunk_samples == 2560

    c = Config(right_context=6)
    assert c.native_chunk_samples == 8960

    c = Config(right_context=13)
    assert c.native_chunk_samples == 17920

    c = Config(right_context=999)
    assert c.native_chunk_samples == 17920


def test_audio_queue_max_size_validation():
    with pytest.raises(ValueError):
        Config(audio_queue_max_size=0)


def test_streaming_stall_validation():
    with pytest.raises(ValueError):
        Config(streaming_stall_chunks=0)

    with pytest.raises(ValueError):
        Config(streaming_stall_flush_chunks=0)

    with pytest.raises(ValueError):
        Config(streaming_stall_rms_ratio=0)
