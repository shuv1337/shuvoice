from __future__ import annotations

from shuvoice.config import Config
from shuvoice.waybar.format import build_waybar_payload, config_info_lines


def test_waybar_format_build_payload_idle():
    payload = build_waybar_payload("idle")
    assert payload["alt"] == "idle"
    assert "ShuVoice: Ready" in payload["tooltip"]


def test_waybar_format_config_info_lines():
    cfg = Config(asr_backend="moonshine", moonshine_model_name="moonshine/tiny")
    lines = config_info_lines(cfg)
    assert any("Moonshine" in line for line in lines)
    assert any("TTS:" in line and "ElevenLabs" in line for line in lines)
    assert any("Voice:" in line for line in lines)
    assert "Speed:    1.0× (default synth)" in lines


def test_waybar_format_instant_profile_line():
    cfg = Config(asr_backend="moonshine", instant_mode=True)
    lines = config_info_lines(cfg)
    assert "Profile:  Instant" in lines


def test_waybar_format_sherpa_model_name_is_shown_when_auto_downloading():
    cfg = Config(
        asr_backend="sherpa", sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    )
    lines = config_info_lines(cfg)
    assert any("parakeet-tdt-0.6b-v3-int8" in line for line in lines)


def test_waybar_format_shows_openai_tts_voice_name():
    cfg = Config(tts_backend="openai")
    lines = config_info_lines(cfg)
    assert "TTS:      OpenAI" in lines
    assert "Voice:    Onyx" in lines


def test_waybar_format_shows_local_auto_voice_label():
    cfg = Config(tts_backend="local")
    lines = config_info_lines(cfg)
    assert "TTS:      Local Piper" in lines
    assert "Voice:    Auto" in lines


def test_waybar_format_shows_tts_disabled_state():
    cfg = Config(tts_enabled=False)
    lines = config_info_lines(cfg)
    assert "TTS:      Disabled" in lines
