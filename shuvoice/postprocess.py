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
    """Apply custom whole-word/phrase replacements case-insensitively."""
    if not text or not replacements:
        return text

    result = text
    keys = sorted(replacements.keys(), key=len, reverse=True)
    for source in keys:
        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", re.IGNORECASE)
        result = pattern.sub(replacements[source], result)
    return result
