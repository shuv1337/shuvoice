from __future__ import annotations

import numpy as np

from shuvoice.asr_moonshine import MoonshineBackend
from shuvoice.config import Config


def test_guard_truncates_hyphenated_token_repetition():
    text = "The six-six-hake-hake-hake-hake-hake"

    assert MoonshineBackend._guard_repetition(text, audio_seconds=2.0) == "The six-six-hake"


def test_guard_truncates_digit_token_repetition():
    text = "Invoice 4827 totals 127127127127 and fees"

    assert (
        MoonshineBackend._guard_repetition(text, audio_seconds=2.0)
        == "Invoice 4827 totals 127 and fees"
    )


def test_guard_truncates_clause_level_repetition():
    clause = "we still have issues with recording cutting out on long sentences"
    text = " ".join([clause, clause, clause])

    assert MoonshineBackend._guard_repetition(text, audio_seconds=7.0) == clause


def test_guard_character_cap_truncates_long_single_token():
    text = "abcdefghijklmnopqrstuvwxyz" * 12  # 312 chars, no 1-10 char 4x loop

    out = MoonshineBackend._guard_repetition(text, audio_seconds=1.0)

    assert len(out) == 100
    assert out == text[:100]


def test_guard_keeps_normal_text_unchanged():
    text = "This is a normal sentence with ordinary wording."

    assert MoonshineBackend._guard_repetition(text, audio_seconds=3.0) == text


def test_guard_keeps_short_non_pathological_text_unchanged():
    text = "all systems nominal now"

    assert MoonshineBackend._guard_repetition(text, audio_seconds=2.0) == text


def test_commit_pending_audio_merges_once_and_trims_window():
    cfg = Config(asr_backend="moonshine", moonshine_max_window_sec=0.5)
    backend = MoonshineBackend(cfg)

    existing = np.arange(6000, dtype=np.float32)
    p1 = np.full(3000, 1.0, dtype=np.float32)
    p2 = np.full(3000, 2.0, dtype=np.float32)

    backend._audio_buffer = existing
    backend._pending_chunks = [p1, p2]
    backend._pending_samples = p1.size + p2.size

    backend._commit_pending_audio()

    expected = np.concatenate([existing, p1, p2])[-backend._max_window_samples :]
    assert np.array_equal(backend._audio_buffer, expected)
    assert backend._pending_chunks == []
    assert backend._pending_samples == 0
