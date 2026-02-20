"""Post-processing helpers for final committed text."""

from __future__ import annotations


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
