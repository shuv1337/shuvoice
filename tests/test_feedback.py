from __future__ import annotations

import numpy as np

from shuvoice.feedback import generate_tone, play_tone


def test_generate_tone_length_and_amplitude():
    tone = generate_tone(freq=440, duration_ms=100, volume=0.25, sample_rate=1000)

    assert len(tone) == 100
    assert tone.dtype == np.float32
    assert np.max(np.abs(tone)) <= 0.25 + 1e-6


def test_play_tone_calls_sounddevice_play(monkeypatch):
    calls: list[tuple[np.ndarray, int, bool]] = []

    def fake_play(data, samplerate, blocking=False):
        calls.append((data, samplerate, blocking))

    monkeypatch.setattr("shuvoice.feedback.sd.play", fake_play)

    play_tone(freq=880, duration_ms=50, volume=0.1, sample_rate=8000)

    assert len(calls) == 1
    data, samplerate, blocking = calls[0]
    assert len(data) == 400
    assert samplerate == 8000
    assert blocking is False
