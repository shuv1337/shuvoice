"""Shared Hyprland ShuVoice control-command matching helpers."""

from __future__ import annotations

CONTROL_COMMAND_PATTERNS: dict[str, tuple[str, ...]] = {
    "start": ("--control start", " control start"),
    "tts_speak": ("--control tts_speak", " control tts_speak"),
    "tts_speak_clipboard": (
        "--control tts_speak_clipboard",
        " control tts_speak_clipboard",
    ),
}

DESCRIPTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "start": ("shuvoice start",),
    "tts_speak": ("shuvoice tts speak", "shuvoice tts_speak"),
    "tts_speak_clipboard": (
        "shuvoice tts speak clipboard",
        "shuvoice tts_speak_clipboard",
    ),
}


def _pattern_has_command_boundary(text_lc: str, pattern: str) -> bool:
    start = 0
    while True:
        idx = text_lc.find(pattern, start)
        if idx == -1:
            return False
        end = idx + len(pattern)
        if end >= len(text_lc):
            return True
        next_char = text_lc[end]
        if next_char in (" ", "\t") or text_lc[end:].startswith("--"):
            return True
        start = end


def _pattern_has_description_boundary(text_lc: str, pattern: str) -> bool:
    idx = text_lc.find(pattern)
    if idx == -1:
        return False
    end = idx + len(pattern)
    if end >= len(text_lc):
        return True
    next_char = text_lc[end]
    return not (next_char.isalnum() or next_char == "_")


def matches_control_command_token(command_lc: str, token: str) -> bool:
    """Return True when ``token`` appears as a complete ShuVoice control subcommand."""
    for prefix in ("--control ", " control "):
        pattern = f"{prefix}{token}"
        if _pattern_has_command_boundary(command_lc, pattern):
            return True
    return False


def matches_shuvoice_command(arg: str, command: str) -> bool:
    arg_lc = str(arg).lower()
    if "shuvoice" not in arg_lc:
        return False
    return any(
        _pattern_has_command_boundary(arg_lc, pattern)
        for pattern in CONTROL_COMMAND_PATTERNS.get(command, ())
    )


def matches_shuvoice_description(description: str, command: str) -> bool:
    desc_lc = str(description).lower()
    if "shuvoice" not in desc_lc:
        return False

    own_patterns = DESCRIPTION_PATTERNS.get(command, ())
    best_own = ""
    for pattern in own_patterns:
        if _pattern_has_description_boundary(desc_lc, pattern) and len(pattern) > len(best_own):
            best_own = pattern

    if not best_own:
        return False

    for other_command, other_patterns in DESCRIPTION_PATTERNS.items():
        if other_command == command:
            continue
        for other_pattern in other_patterns:
            if len(other_pattern) <= len(best_own):
                continue
            if _pattern_has_description_boundary(desc_lc, other_pattern):
                return False

    return True
