from __future__ import annotations

import pytest

from shuvoice.tts_speed import (
    TTS_PLAYBACK_SPEED_MAX,
    TTS_PLAYBACK_SPEED_MIN,
    format_tts_playback_speed,
    normalize_tts_playback_speed,
    step_tts_playback_speed,
    validate_tts_playback_speed,
)


def test_validate_tts_playback_speed_accepts_in_range_values():
    assert validate_tts_playback_speed(1.0) == 1.0
    assert validate_tts_playback_speed("1.25") == 1.25


@pytest.mark.parametrize("value", [0.49, 2.01, "fast", float("inf")])
def test_validate_tts_playback_speed_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="tts_playback_speed"):
        validate_tts_playback_speed(value)


def test_normalize_tts_playback_speed_clamps_to_supported_range():
    assert normalize_tts_playback_speed(0.1) == TTS_PLAYBACK_SPEED_MIN
    assert normalize_tts_playback_speed(9.9) == TTS_PLAYBACK_SPEED_MAX


def test_step_tts_playback_speed_moves_by_fixed_increment():
    assert step_tts_playback_speed(1.0, 1) == 1.1
    assert step_tts_playback_speed(1.0, -1) == 0.9


def test_format_tts_playback_speed_uses_readable_multiplier_text():
    assert format_tts_playback_speed(1) == "1.0×"
    assert format_tts_playback_speed(1.25) == "1.25×"
