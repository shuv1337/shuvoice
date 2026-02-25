from __future__ import annotations

from pathlib import Path

import pytest

from shuvoice.config import DEFAULT_TEXT_REPLACEMENTS, Config


def test_load_defaults_when_config_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))

    cfg = Config.load()

    assert cfg.sample_rate == 16000
    assert cfg.output_mode == "final_only"
    assert cfg.typing_final_injection_mode == "auto"
    assert cfg.typing_clipboard_settle_delay_ms == 40
    assert cfg.audio_queue_max_size == 200
    assert cfg.auto_gain_target_peak == 0.15
    assert cfg.auto_gain_max == 10.0
    assert cfg.auto_gain_settle_chunks == 2
    assert cfg.audio_feedback is True
    assert cfg.auto_capitalize is True
    assert cfg.text_replacements == DEFAULT_TEXT_REPLACEMENTS
    assert cfg.font_family is None
    assert cfg.streaming_stall_guard is True
    assert cfg.streaming_stall_chunks == 4
    assert cfg.asr_backend == "sherpa"
    assert cfg.instant_mode is False
    assert cfg.sherpa_model_name == "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
    assert cfg.sherpa_provider == "cpu"
    assert cfg.sherpa_num_threads == 2
    assert cfg.sherpa_chunk_ms == 100
    assert cfg.moonshine_model_name == "moonshine/tiny"
    assert cfg.moonshine_model_precision == "float"
    assert cfg.moonshine_chunk_ms == 100
    assert cfg.moonshine_max_window_sec == 5.0
    assert cfg.moonshine_max_tokens == 64
    assert cfg.moonshine_provider == "cpu"
    assert cfg.moonshine_onnx_threads == 0


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
auto_gain_target_peak = 0.11
auto_gain_max = 7.5
auto_gain_settle_chunks = 3
unknown_audio_key = 999

[asr]
asr_backend = "sherpa"
sherpa_model_name = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
model_name = "nvidia/nemotron-speech-streaming-en-0.6b"
right_context = 13
device = "cuda"
sherpa_model_dir = "/tmp/sherpa-model"
sherpa_provider = "cuda"
sherpa_num_threads = 4
sherpa_chunk_ms = 120
moonshine_model_name = "moonshine/tiny"
moonshine_model_dir = "/tmp/moonshine-model"
moonshine_model_precision = "float"
moonshine_chunk_ms = 110
moonshine_max_window_sec = 20.0
moonshine_max_tokens = 160

[overlay]
font_size = 28
font_family = "JetBrains Mono"

[typing]
output_mode = "streaming_partial"
use_clipboard_for_final = true
preserve_clipboard = true
typing_retry_attempts = 3
typing_retry_delay_ms = 20
auto_capitalize = false

[typing.text_replacements]
"shove voice" = "ShuVoice"
"hyper land" = "Hyprland"
"um" = ""

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
    assert cfg.auto_gain_target_peak == 0.11
    assert cfg.auto_gain_max == 7.5
    assert cfg.auto_gain_settle_chunks == 3
    assert cfg.asr_backend == "sherpa"
    assert cfg.sherpa_model_name == "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    assert cfg.sherpa_model_dir == "/tmp/sherpa-model"
    assert cfg.sherpa_provider == "cuda"
    assert cfg.sherpa_num_threads == 4
    assert cfg.sherpa_chunk_ms == 120
    assert cfg.moonshine_model_name == "moonshine/tiny"
    assert cfg.moonshine_model_dir == "/tmp/moonshine-model"
    assert cfg.moonshine_model_precision == "float"
    assert cfg.moonshine_chunk_ms == 110
    assert cfg.moonshine_max_window_sec == 20.0
    assert cfg.moonshine_max_tokens == 160
    assert cfg.font_size == 28
    assert cfg.font_family == "JetBrains Mono"
    assert cfg.output_mode == "streaming_partial"
    assert cfg.typing_final_injection_mode == "clipboard"
    assert cfg.use_clipboard_for_final is True
    assert cfg.preserve_clipboard is True
    assert cfg.typing_retry_attempts == 3
    assert cfg.typing_retry_delay_ms == 20
    assert cfg.auto_capitalize is False
    assert cfg.text_replacements["shove voice"] == "ShuVoice"
    assert cfg.text_replacements["hyper land"] == "Hyprland"
    assert cfg.text_replacements["um"] == ""
    assert cfg.text_replacements["shu voice"] == "ShuVoice"
    assert cfg.text_replacements["hyperland"] == "Hyprland"
    assert cfg.streaming_stall_guard is False
    assert cfg.streaming_stall_chunks == 6
    assert cfg.streaming_stall_rms_ratio == 0.9
    assert cfg.streaming_stall_flush_chunks == 2
    assert cfg.audio_feedback is False
    assert cfg.feedback_start_freq == 500
    assert cfg.feedback_stop_freq == 400
    assert cfg.feedback_duration_ms == 50
    assert cfg.feedback_volume == 0.2


def test_load_legacy_clipboard_flag_false_maps_to_direct_mode(monkeypatch, tmp_path: Path):
    cfg_home = tmp_path / "cfg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))

    cfg_dir = cfg_home / "shuvoice"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.toml").write_text(
        """
[typing]
use_clipboard_for_final = false
""".strip()
    )

    cfg = Config.load()

    assert cfg.use_clipboard_for_final is False
    assert cfg.typing_final_injection_mode == "direct"


def test_load_explicit_typing_mode_wins_over_legacy_flag(monkeypatch, tmp_path: Path):
    cfg_home = tmp_path / "cfg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))

    cfg_dir = cfg_home / "shuvoice"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.toml").write_text(
        """
[typing]
typing_final_injection_mode = "auto"
use_clipboard_for_final = false
""".strip()
    )

    cfg = Config.load()

    assert cfg.use_clipboard_for_final is False
    assert cfg.typing_final_injection_mode == "auto"


def test_config_helpers_create_dirs(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    cdir = Config.config_dir()
    ddir = Config.data_dir()

    assert cdir.exists() and cdir.is_dir()
    assert ddir.exists() and ddir.is_dir()


def test_asr_backend_validation():
    with pytest.raises(ValueError, match="asr_backend"):
        Config(asr_backend="bad-backend")


def test_instant_mode_validation():
    with pytest.raises(ValueError, match="instant_mode"):
        Config(instant_mode="yes")


def test_sherpa_provider_validation():
    with pytest.raises(ValueError, match="sherpa_provider"):
        Config(sherpa_provider="rocm")


def test_sherpa_model_name_validation():
    with pytest.raises(ValueError, match="sherpa_model_name"):
        Config(sherpa_model_name="   ")


def test_sherpa_chunk_ms_validation():
    with pytest.raises(ValueError, match="sherpa_chunk_ms"):
        Config(sherpa_chunk_ms=0)


def test_sherpa_num_threads_validation():
    with pytest.raises(ValueError, match="sherpa_num_threads"):
        Config(sherpa_num_threads=0)


def test_moonshine_chunk_ms_validation():
    with pytest.raises(ValueError, match="moonshine_chunk_ms"):
        Config(moonshine_chunk_ms=0)


def test_moonshine_max_window_sec_validation():
    with pytest.raises(ValueError, match="moonshine_max_window_sec"):
        Config(moonshine_max_window_sec=0)


def test_moonshine_max_tokens_validation():
    with pytest.raises(ValueError, match="moonshine_max_tokens"):
        Config(moonshine_max_tokens=0)


def test_moonshine_model_name_validation():
    with pytest.raises(ValueError, match="moonshine_model_name"):
        Config(moonshine_model_name="   ")


def test_moonshine_model_precision_validation():
    with pytest.raises(ValueError, match="moonshine_model_precision"):
        Config(moonshine_model_precision="   ")


def test_instant_mode_nemo_profile_sets_lowest_right_context():
    cfg = Config(asr_backend="nemo", right_context=13, instant_mode=True)
    assert cfg.right_context == 0


def test_instant_mode_sherpa_profile_caps_chunk_ms():
    cfg = Config(asr_backend="sherpa", sherpa_chunk_ms=120, instant_mode=True)
    assert cfg.sherpa_chunk_ms == 80


def test_instant_mode_moonshine_profile_forces_tiny_and_caps_window():
    cfg = Config(
        asr_backend="moonshine",
        instant_mode=True,
        moonshine_model_name="moonshine/base",
        moonshine_max_window_sec=5.0,
        moonshine_max_tokens=64,
    )

    assert cfg.moonshine_model_name == "moonshine/tiny"
    assert cfg.moonshine_max_window_sec == 3.0
    assert cfg.moonshine_max_tokens == 48


def test_audio_queue_max_size_validation():
    with pytest.raises(ValueError):
        Config(audio_queue_max_size=0)


def test_auto_gain_validation():
    with pytest.raises(ValueError, match="auto_gain_target_peak"):
        Config(auto_gain_target_peak=0)

    with pytest.raises(ValueError, match="auto_gain_max"):
        Config(auto_gain_max=0.9)

    with pytest.raises(ValueError, match="auto_gain_settle_chunks"):
        Config(auto_gain_settle_chunks=0)


def test_streaming_stall_validation():
    with pytest.raises(ValueError):
        Config(streaming_stall_chunks=0)

    with pytest.raises(ValueError):
        Config(streaming_stall_flush_chunks=0)

    with pytest.raises(ValueError):
        Config(streaming_stall_rms_ratio=0)


def test_text_replacements_validation():
    # Bad type
    with pytest.raises(ValueError, match="text_replacements"):
        Config(text_replacements="nope")

    # Non-string key/value types
    with pytest.raises(ValueError, match="keys must be strings"):
        Config(text_replacements={1: "value"})

    with pytest.raises(ValueError, match="values must be strings"):
        Config(text_replacements={"um": 1})

    # Empty/whitespace key
    with pytest.raises(ValueError, match="keys must not be empty"):
        Config(text_replacements={"   ": "value"})

    # Empty values are allowed (deletion)
    cfg = Config(text_replacements={"um": ""})
    assert cfg.text_replacements["um"] == ""
    assert cfg.text_replacements["shove voice"] == "ShuVoice"


def test_text_replacements_are_normalized():
    cfg = Config(text_replacements={" shove voice ": " ShuVoice "})
    assert cfg.text_replacements["shove voice"] == "ShuVoice"
    assert cfg.text_replacements["hyper land"] == "Hyprland"


def test_default_text_replacements_include_common_brand_variants():
    cfg = Config()

    assert cfg.text_replacements["shu voice"] == "ShuVoice"
    assert cfg.text_replacements["show voice"] == "ShuVoice"
    assert cfg.text_replacements["hyperland"] == "Hyprland"
    assert cfg.text_replacements["high per land"] == "Hyprland"
