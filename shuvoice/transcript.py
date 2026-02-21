"""Transcript merge helpers for streaming ASR hypotheses."""

from __future__ import annotations

MIN_OVERLAP_CHARS = 8
MIN_OVERLAP_WORDS = 2


def _normalize_word(word: str) -> str:
    return word.strip(".,!?;:\"'()[]{}").lower()


def _stitch_by_word_overlap(previous: str, candidate: str) -> str | None:
    prev_words = previous.split()
    new_words = candidate.split()
    if not prev_words or not new_words:
        return None

    min_words = max(1, MIN_OVERLAP_WORDS)
    max_words = min(len(prev_words), len(new_words))
    for overlap in range(max_words, min_words - 1, -1):
        prev_tail = [_normalize_word(word) for word in prev_words[-overlap:]]
        new_head = [_normalize_word(word) for word in new_words[:overlap]]
        if prev_tail != new_head:
            continue

        suffix_words = new_words[overlap:]
        if not suffix_words:
            return None

        glue = "" if previous.endswith((" ", "\n", "\t")) else " "
        return previous + glue + " ".join(suffix_words)

    return None


def prefer_transcript(previous: str, candidate: str) -> str:
    """Prefer stable cumulative transcript growth over regressions.

    Rules:
    - Empty candidate never replaces non-empty previous text.
    - Candidate that extends previous text is accepted.
    - Candidate that is a shorter prefix is rejected.
    - Candidate can be stitched when overlap is strong (>= MIN_OVERLAP_CHARS).
    - Longer contextual rewrites are accepted.
    - Equal-length divergent rewrites keep previous text for deterministic stability.
    """
    previous_raw = previous or ""
    candidate_raw = candidate or ""

    prev = previous_raw.strip()
    new = candidate_raw.strip()

    if not new:
        return previous_raw
    if not prev:
        return candidate_raw

    if new.startswith(prev):
        return candidate_raw

    if prev.startswith(new):
        return previous_raw

    min_len = min(len(prev), len(new))
    for overlap in range(min_len, MIN_OVERLAP_CHARS - 1, -1):
        if prev.endswith(new[:overlap]):
            return prev + new[overlap:]

    stitched = _stitch_by_word_overlap(previous_raw, candidate_raw)
    if stitched:
        return stitched

    if len(new) > len(prev):
        return candidate_raw

    return previous_raw
