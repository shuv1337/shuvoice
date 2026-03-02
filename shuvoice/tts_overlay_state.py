"""Pure TTS overlay state helpers (headless-safe)."""

from __future__ import annotations

TTS_OVERLAY_IDLE = "idle"
TTS_OVERLAY_SYNTHESIZING = "synthesizing"
TTS_OVERLAY_PLAYING = "playing"
TTS_OVERLAY_PAUSED = "paused"
TTS_OVERLAY_ERROR = "error"


def summarize_preview(text: str, *, max_chars: int = 50) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    if len(value) <= max_chars:
        return value
    return value[: max(1, max_chars - 1)].rstrip() + "…"


def status_label_for_state(state: str, *, error_message: str | None = None) -> str:
    if state == TTS_OVERLAY_SYNTHESIZING:
        return "🔊 Synthesizing…"
    if state == TTS_OVERLAY_PLAYING:
        return "🔊 Speaking…"
    if state == TTS_OVERLAY_PAUSED:
        return "⏸ Paused"
    if state == TTS_OVERLAY_ERROR:
        detail = (error_message or "TTS error").strip()
        return f"⚠ {detail}"
    return "🔈 Idle"
