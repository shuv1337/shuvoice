from __future__ import annotations

from shuvoice.transcript import MIN_OVERLAP_CHARS, prefer_transcript


def test_normal_growth_prefers_candidate():
    assert prefer_transcript("hello", "hello world") == "hello world"


def test_regression_rejection_prefers_previous():
    assert prefer_transcript("hello world", "hello") == "hello world"


def test_empty_whitespace_handling():
    assert prefer_transcript("", "hello") == "hello"
    assert prefer_transcript("hello", "   ") == "hello"


def test_overlap_stitching_behavior():
    previous = "the quick brown fox"
    candidate = "brown fox jumps high"
    assert prefer_transcript(previous, candidate) == "the quick brown fox jumps high"


def test_false_positive_short_overlap_does_not_stitch():
    previous = "alpha123beta"
    candidate = "123beta and gamma"
    # overlap is 7 chars ("123beta"), below required threshold
    assert MIN_OVERLAP_CHARS == 8
    assert prefer_transcript(previous, candidate) == candidate


def test_rewrite_acceptance_for_longer_contextual_candidate():
    previous = "quick brown"
    candidate = "the quick brown dog jumped"
    assert prefer_transcript(previous, candidate) == candidate


def test_equal_length_divergent_determinism_keeps_previous():
    previous = "hello world"
    candidate = "jello world"
    assert len(previous) == len(candidate)
    assert prefer_transcript(previous, candidate) == previous


def test_overlap_threshold_at_eight_chars_stitches():
    previous = "alpha1234beta"
    candidate = "1234beta and gamma"
    # overlap is exactly 8 chars ("1234beta")
    assert prefer_transcript(previous, candidate) == "alpha1234beta and gamma"
