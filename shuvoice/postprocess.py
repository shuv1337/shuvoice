"""Post-processing helpers for final committed text."""

from __future__ import annotations

import re
from collections.abc import Mapping

CompiledTextReplacements = tuple[tuple[re.Pattern[str], str], ...]


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


def compile_text_replacements(replacements: Mapping[str, str] | None) -> CompiledTextReplacements:
    """Compile whole-word replacement regex patterns once for hot-path reuse."""
    if not replacements:
        return tuple()

    compiled: list[tuple[re.Pattern[str], str]] = []
    for source in sorted(replacements, key=len, reverse=True):
        if not source:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", re.IGNORECASE)
        compiled.append((pattern, replacements[source]))
    return tuple(compiled)


def apply_text_replacements(
    text: str,
    replacements: Mapping[str, str] | None,
    *,
    compiled_replacements: CompiledTextReplacements | None = None,
) -> str:
    """Apply custom whole-word/phrase replacements case-insensitively.

    Longer source phrases are matched first to prevent partial overlaps.
    Empty replacement values delete the matched word/phrase, and any
    resulting multi-space runs are collapsed to a single space.

    Replacement values are treated literally (not as regex backreferences).
    """
    if not text:
        return text

    compiled = compiled_replacements
    if compiled is None:
        compiled = compile_text_replacements(replacements)

    if not compiled:
        return text

    result = text
    for pattern, replacement in compiled:
        result = pattern.sub(lambda _m, replacement=replacement: replacement, result)

    # Collapse double-spaces left by deletions (empty replacements).
    result = re.sub(r" {2,}", " ", result).strip()
    return result
