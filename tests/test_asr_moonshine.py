from __future__ import annotations

from shuvoice.asr_moonshine import MoonshineBackend


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
