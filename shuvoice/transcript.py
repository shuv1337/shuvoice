"""Transcript merge helpers for streaming ASR hypotheses."""

from __future__ import annotations

MIN_OVERLAP_CHARS = 8
MIN_OVERLAP_WORDS = 2

# Guard against pathological repetition bursts emitted by some backends
# under noisy/unstable conditions (e.g. "just just just ..." hundreds of times).
REPETITION_MIN_WORDS = 20
REPETITION_MAX_RUN = 8
REPETITION_MAX_UNIQUE_RATIO = 0.2


def _normalize_word(word: str) -> str:
    return word.strip(".,!?;:\"'()[]{}").lower()


def _max_consecutive_run(words: list[str]) -> int:
    if not words:
        return 0

    best = 1
    current = 1
    last = words[0]
    for word in words[1:]:
        if word == last:
            current += 1
            best = max(best, current)
        else:
            current = 1
            last = word

    return best


def _is_pathological_repetition(text: str) -> bool:
    words = [_normalize_word(word) for word in text.split()]
    words = [word for word in words if word]
    if len(words) < REPETITION_MIN_WORDS:
        return False

    max_run = _max_consecutive_run(words)
    unique_ratio = len(set(words)) / max(1, len(words))

    if max_run >= REPETITION_MAX_RUN:
        return True

    return unique_ratio <= REPETITION_MAX_UNIQUE_RATIO


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
    - Pathological repetition candidates are rejected.
    - If previous text is pathological repetition and candidate is sane, candidate replaces it.
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

    if _is_pathological_repetition(new):
        return previous_raw

    if not prev:
        return candidate_raw

    if _is_pathological_repetition(prev):
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
