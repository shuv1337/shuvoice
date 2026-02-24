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
