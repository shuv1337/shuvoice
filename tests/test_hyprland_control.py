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


def test_matches_shuvoice_command_checks_all_occurrences():
    # The first "tts_speak" substring belongs to "tts_speak_clipboard" and has
    # no boundary; the second occurrence is the real command and should match.
    arg = "shuvoice control tts_speak_clipboard --control tts_speak"
    assert matches_shuvoice_command(arg, "tts_speak")
    assert matches_shuvoice_command(arg, "tts_speak_clipboard")


def test_matches_shuvoice_description_allows_trailing_text():
    assert matches_shuvoice_description("ShuVoice start (push to talk)", "start")
    assert matches_shuvoice_description("ShuVoice TTS speak now", "tts_speak")
    # But a longer command phrase still wins over the shorter prefix.
    assert not matches_shuvoice_description("ShuVoice TTS speak now", "tts_speak_clipboard")


def test_matches_shuvoice_description_prefers_longer_command():
    clipboard = "ShuVoice TTS speak clipboard (read aloud)"
    assert matches_shuvoice_description(clipboard, "tts_speak_clipboard")
    assert not matches_shuvoice_description(clipboard, "tts_speak")
