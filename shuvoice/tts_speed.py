"""Shared TTS playback speed helpers."""

from __future__ import annotations

import math

TTS_PLAYBACK_SPEED_MIN = 0.5
TTS_PLAYBACK_SPEED_MAX = 2.0
TTS_PLAYBACK_SPEED_STEP = 0.1
TTS_PLAYBACK_SPEED_DEFAULT = 1.0


def _parse_tts_playback_speed(speed: float | int | str) -> float:
    try:
        value = float(speed)
    except (TypeError, ValueError) as exc:
        raise ValueError("tts_playback_speed must be a number") from exc

    if not math.isfinite(value):
        raise ValueError("tts_playback_speed must be a finite number")

    return value


def validate_tts_playback_speed(speed: float | int | str) -> float:
    value = _parse_tts_playback_speed(speed)
    if not (TTS_PLAYBACK_SPEED_MIN <= value <= TTS_PLAYBACK_SPEED_MAX):
        raise ValueError(
            "tts_playback_speed must be between "
            f"{TTS_PLAYBACK_SPEED_MIN:.1f} and {TTS_PLAYBACK_SPEED_MAX:.1f}"
        )
    return round(value, 2)


def normalize_tts_playback_speed(speed: float | int | str) -> float:
    value = _parse_tts_playback_speed(speed)
    value = min(TTS_PLAYBACK_SPEED_MAX, max(TTS_PLAYBACK_SPEED_MIN, value))
    return round(value, 2)


def step_tts_playback_speed(speed: float | int | str, steps: int) -> float:
    current = normalize_tts_playback_speed(speed)
    return normalize_tts_playback_speed(current + (int(steps) * TTS_PLAYBACK_SPEED_STEP))


def format_tts_playback_speed(speed: float | int | str) -> str:
    value = normalize_tts_playback_speed(speed)
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return f"{text}×"


__all__ = [
    "TTS_PLAYBACK_SPEED_DEFAULT",
    "TTS_PLAYBACK_SPEED_MAX",
    "TTS_PLAYBACK_SPEED_MIN",
    "TTS_PLAYBACK_SPEED_STEP",
    "format_tts_playback_speed",
    "normalize_tts_playback_speed",
    "step_tts_playback_speed",
    "validate_tts_playback_speed",
]
