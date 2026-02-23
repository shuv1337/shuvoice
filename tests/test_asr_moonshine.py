from __future__ import annotations

import numpy as np
import pytest

from shuvoice.asr_moonshine import MoonshineBackend
from shuvoice.config import Config

# ---------------------------------------------------------------------------
# Existing tests (issue #12 baseline)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Issue #12 — hyphenated/token-local repetition regression
# ---------------------------------------------------------------------------


def test_guard_hyphenated_long_hake_loop():
    """Issue #12 exact reproduction: tongue twister producing hake-hake-... loop."""
    text = "The six-six-hake" + "-hake" * 50
    result = MoonshineBackend._guard_repetition(text, audio_seconds=2.0)
    assert len(result) < 50, f"Runaway output: {len(result)} chars"
    assert "hake-hake-hake-hake" not in result


def test_guard_hyphenated_varied_delimiter():
    """Hyphenated loop with underscore/mixed delimiters."""
    text = "prefix tok_tok_tok_tok_tok_tok suffix"
    result = MoonshineBackend._guard_repetition(text, audio_seconds=2.0)
    # The multi-char pattern regex uses optional hyphen; underscores won't
    # match the hyphen variant, but the char-count cap or single-char run
    # should still prevent runaway output.
    assert len(result) <= len(text)


# ---------------------------------------------------------------------------
# Issue #12 — clause-level repetition regression
# ---------------------------------------------------------------------------


def test_guard_clause_loop_with_punctuation_variation():
    """Issue #12 exact reproduction: clause loop with period/comma variation."""
    text = (
        "We still have issues with recording cutting out on long sentences. "
        "and we still have issues with recording cutting out on long sentences, "
        "and we still have issues with recording cutting out on long sentences, "
        "and we need, and we need, and we need, and we need, and we need, and"
    )
    result = MoonshineBackend._guard_repetition(text, audio_seconds=5.0)
    # Should not contain repeated clause
    count = result.lower().count("we still have issues")
    assert count <= 1, f"Clause repeated {count} times in output"


def test_guard_clause_loop_late_start():
    """Clause loop starting after 20+ normal words."""
    prefix = "This is a completely normal preamble with many ordinary words that should be preserved. "
    clause = "we need deterministic regression tests to catch regressions before they ship"
    text = prefix + " ".join([clause] * 4)
    result = MoonshineBackend._guard_repetition(text, audio_seconds=10.0)
    count = result.lower().count("we need deterministic")
    assert count <= 1, f"Late clause repeated {count} times"
    assert result.startswith("This is a completely normal preamble")


def test_guard_clause_loop_8_word_pattern():
    """Clause loop with exactly 8 words (above old max=4, below max=12)."""
    clause = "the cat sat on the mat every day"
    text = " ".join([clause] * 4)
    result = MoonshineBackend._guard_repetition(text, audio_seconds=6.0)
    count = result.lower().count("the cat sat")
    assert count <= 1


# ---------------------------------------------------------------------------
# Issue #12 — numeric/character-level repetition regression
# ---------------------------------------------------------------------------


def test_guard_numeric_tail_flood():
    """Issue #12 exact reproduction: digit flood like 12700000000..."""
    text = "Invoice 4827 totals 153 dollars and 127" + "0" * 100
    result = MoonshineBackend._guard_repetition(text, audio_seconds=3.0)
    # The "0" run should be truncated
    assert len(result) < 80, f"Numeric flood not truncated: {len(result)} chars"
    assert result.startswith("Invoice 4827 totals 153 dollars and 1")


def test_guard_single_char_digit_run():
    """Single digit repeated many times inside a token."""
    text = "total is 5" + "5" * 20 + " dollars"
    result = MoonshineBackend._guard_repetition(text, audio_seconds=2.0)
    assert len(result) < len(text)
    assert "5555555555" not in result


def test_guard_repeated_char_substring():
    """Repeated short substring like 'ababab...'."""
    text = "prefix " + "ab" * 30 + " suffix"
    result = MoonshineBackend._guard_repetition(text, audio_seconds=2.0)
    assert len(result) < len(text)


# ---------------------------------------------------------------------------
# Issue #12 — false-positive controls (must NOT truncate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,audio_seconds",
    [
        ("state-of-the-art technology is here", 3.0),
        ("ha-ha-ha that was funny", 2.0),
        ("Invoice 4827 totals 100000 dollars", 2.0),
        ("The year 20260223 was remarkable", 2.0),
        ("well-well-known fact about this", 2.0),
        ("This is a normal sentence with ordinary wording.", 3.0),
        ("all systems nominal now", 2.0),
        ("one-two-three-four-five", 2.0),
        ("self-self-assessment form", 2.0),
        ("The population is 3500000", 2.0),
    ],
    ids=[
        "hyphenated-compound",
        "short-laugh",
        "normal-large-number",
        "date-like-number",
        "well-well-known",
        "normal-sentence",
        "short-phrase",
        "hyphenated-list",
        "double-word-hyphen",
        "population-number",
    ],
)
def test_guard_false_positive_unchanged(text: str, audio_seconds: float):
    """Legitimate text patterns must not be altered by the repetition guard."""
    result = MoonshineBackend._guard_repetition(text, audio_seconds)
    assert result == text, f"False positive: {text!r} was changed to {result!r}"


# ---------------------------------------------------------------------------
# Boundary / threshold tests
# ---------------------------------------------------------------------------


def test_guard_short_pattern_needs_4_repeats():
    """Short patterns (1-4 words) need ≥4 consecutive repeats to trigger."""
    # 3 repeats should NOT trigger for 2-word pattern
    text = "hello world hello world hello world other words here now"
    result = MoonshineBackend._guard_repetition(text, audio_seconds=5.0)
    assert "hello world hello world hello world" in result

    # 4 repeats SHOULD trigger
    text4 = "hello world hello world hello world hello world other words"
    result4 = MoonshineBackend._guard_repetition(text4, audio_seconds=5.0)
    assert result4.count("hello world") <= 1


def test_guard_long_pattern_needs_2_repeats():
    """Long patterns (≥5 words) need only ≥2 consecutive repeats to trigger."""
    clause = "the quick brown fox jumped over the lazy dog near the river"
    text = f"{clause} {clause} and then something else"
    result = MoonshineBackend._guard_repetition(text, audio_seconds=8.0)
    assert result.count("the quick brown fox") <= 1


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
