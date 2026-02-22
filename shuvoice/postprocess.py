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


def apply_text_mappings(text: str, mappings: dict[str, str] | None) -> str:
    """Apply case-insensitive whole-word/phrase mappings to text."""
    if not text or not mappings:
        return text

    mapped = text
    for source, target in sorted(mappings.items(), key=lambda item: len(item[0]), reverse=True):
        if not source or not target:
            continue

        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", flags=re.IGNORECASE)
        mapped = pattern.sub(target, mapped)

    return mapped
