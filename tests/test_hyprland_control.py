from __future__ import annotations

from shuvoice.hyprland_control import (
    matches_control_command_token,
    matches_shuvoice_command,
    matches_shuvoice_description,
)


def test_matches_control_command_token_distinguishes_clipboard_variant():
    clipboard = "shuvoice control tts_speak_clipboard --control-wait-sec 0"
    selection = "shuvoice control tts_speak --control-wait-sec 0"

    assert matches_control_command_token(clipboard, "tts_speak_clipboard")
    assert not matches_control_command_token(clipboard, "tts_speak")
    assert matches_control_command_token(selection, "tts_speak")
    assert not matches_control_command_token(selection, "tts_speak_clipboard")


def test_matches_shuvoice_description_distinguishes_clipboard_variant():
    clipboard = "ShuVoice TTS speak clipboard"
    selection = "ShuVoice TTS speak"

    assert matches_shuvoice_description(clipboard, "tts_speak_clipboard")
    assert not matches_shuvoice_description(clipboard, "tts_speak")
    assert matches_shuvoice_description(selection, "tts_speak")
    assert not matches_shuvoice_description(selection, "tts_speak_clipboard")


def test_matches_shuvoice_command_supports_legacy_control_flag_style():
    arg = "shuvoice --control tts_speak --control-wait-sec 0"
    assert matches_shuvoice_command(arg, "tts_speak")
    assert not matches_shuvoice_command(arg, "tts_speak_clipboard")