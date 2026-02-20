"""Transcript merge helpers for streaming ASR hypotheses."""

from __future__ import annotations

MIN_OVERLAP_CHARS = 8


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

    if len(new) > len(prev):
        return candidate_raw

    return previous_raw
