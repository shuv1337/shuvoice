"""Post-processing helpers for final committed text."""

from __future__ import annotations

import re


def capitalize_first(text: str) -> str:
    """Capitalize the first alphabetic character, preserving leading spacing."""
    if not text:
        return text

    chars = list(text)
    for idx, char in enumerate(chars):
        if char.isalpha():
            chars[idx] = char.upper()
            break
    return "".join(chars)


def apply_text_replacements(text: str, replacements: dict[str, str]) -> str:
    """Apply custom whole-word/phrase replacements case-insensitively.

    Longer source phrases are matched first to prevent partial overlaps.
    Empty replacement values delete the matched word/phrase, and any
    resulting multi-space runs are collapsed to a single space.
    """
    if not text or not replacements:
        return text

    result = text
    for source in sorted(replacements, key=len, reverse=True):
        if not source:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", re.IGNORECASE)
        result = pattern.sub(replacements[source], result)

    # Collapse double-spaces left by deletions (empty replacements).
    result = re.sub(r" {2,}", " ", result).strip()
    return result
