"""Audio feedback tone helpers."""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


def generate_tone(
    freq: float,
    duration_ms: int,
    volume: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Generate a mono sine-wave tone."""
    duration_ms = max(1, int(duration_ms))
    sample_rate = max(1, int(sample_rate))
    volume = max(0.0, float(volume))

    sample_count = max(1, int(sample_rate * duration_ms / 1000.0))
    t = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
    tone = np.sin(2.0 * np.pi * float(freq) * t).astype(np.float32)

    # Tiny fade to reduce clicks.
    fade = min(32, sample_count // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]

    tone *= volume
    return np.clip(tone, -1.0, 1.0).astype(np.float32)


def play_tone(
    freq: float,
    duration_ms: int,
    volume: float,
    sample_rate: int = 16000,
):
    """Play a short tone without blocking the caller."""
    try:
        tone = generate_tone(freq, duration_ms, volume, sample_rate=sample_rate)
        sd.play(tone, samplerate=sample_rate, blocking=False)
    except Exception:
        log.debug("Audio feedback tone failed", exc_info=True)
