from __future__ import annotations

from shuvoice.tts_overlay_state import status_label_for_state, summarize_preview


def test_summarize_preview_truncates_long_text():
    text = "a" * 80
    rendered = summarize_preview(text, max_chars=20)
    assert rendered.endswith("…")
    assert len(rendered) == 20


def test_summarize_preview_keeps_short_text():
    assert summarize_preview("hello", max_chars=20) == "hello"


def test_status_label_for_error_state_includes_message():
    label = status_label_for_state("error", error_message="Network timeout")
    assert "Network timeout" in label


def test_status_label_for_playing_state():
    assert status_label_for_state("playing") == "🔊 Speaking…"
