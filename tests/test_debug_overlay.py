from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from shuvoice.app import ShuVoiceApp
from shuvoice.debug_log import RecentLogBuffer
from shuvoice.metrics import MetricsCollector
from shuvoice.utterance_state import _UtteranceState


@pytest.fixture
def recent_logs() -> RecentLogBuffer:
    handler = RecentLogBuffer(max_entries=20)
    logger = logging.getLogger("shuvoice.test.debug_overlay")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    try:
        logger.info("debug overlay smoke log")
        logger.warning("audio queue high")
        yield handler
    finally:
        logger.removeHandler(handler)


def test_recent_log_buffer_returns_tail(recent_logs: RecentLogBuffer):
    tail = recent_logs.tail(max_lines=1)
    assert len(tail) == 1
    assert "audio queue high" in tail[0]


def test_build_debug_status_includes_logs_and_runtime(recent_logs: RecentLogBuffer):
    metrics = MetricsCollector()
    metrics.increment("chunks_processed", 3)

    app = SimpleNamespace(
        config=SimpleNamespace(
            asr_backend="sherpa",
            tts_backend="elevenlabs",
            overlay_debug_mode=True,
            overlay_debug_max_lines=2,
        ),
        control=SimpleNamespace(socket_path="/tmp/shuvoice/control.sock"),
        audio=SimpleNamespace(queue=SimpleNamespace(qsize=lambda: 4, maxsize=200)),
        asr=SimpleNamespace(debug_step_num=7, native_chunk_samples=1600, wants_raw_audio=False),
        metrics=metrics,
        _recent_logs=recent_logs,
        _recording=threading.Event(),
        _processing=threading.Event(),
        _asr_disabled_event=threading.Event(),
        _asr_thread_alive=True,
        _model_load_failed=False,
        _noise_floor_rms=0.0042,
        _speech_rms_threshold=0.008,
        _speech_rms_multiplier=1.8,
        _consecutive_asr_failures=0,
        _debug_current_transcript="partial text",
        _debug_last_final_transcript="final text",
    )
    app._recording.set()

    payload = ShuVoiceApp._build_debug_status(app)

    assert payload["app"]["overlay_debug_mode"] is True
    assert payload["app"]["recording"] is True
    assert payload["audio"]["queue_depth"] == 4
    assert payload["asr"]["debug_step_num"] == 7
    assert payload["asr"]["current_transcript"] == "partial text"
    assert payload["asr"]["last_final_transcript"] == "final text"
    assert payload["metrics"]["counters"]["chunks_processed"] == 3
    assert len(payload["logs"]) == 2
    assert any("audio queue high" in line for line in payload["logs"])


def test_debug_status_serializes_to_json(recent_logs: RecentLogBuffer):
    app = SimpleNamespace(
        _build_debug_status=lambda: {"logs": recent_logs.tail(max_lines=2), "ok": True}
    )

    rendered = ShuVoiceApp._debug_status(app)
    payload = json.loads(rendered)

    assert payload["ok"] is True
    assert len(payload["logs"]) == 2


def test_update_debug_overlay_pushes_runtime_lines_and_logs(recent_logs: RecentLogBuffer):
    state = _UtteranceState(total=3200, speech_samples=1600, peak_rms=0.021, utterance_gain=1.4)
    state.unchanged_steps = 2

    overlay = SimpleNamespace(set_debug_text=Mock())
    metrics = MetricsCollector()
    metrics.increment("chunks_processed", 5)
    metrics.increment("final_commits", 1)

    app = SimpleNamespace(
        overlay=overlay,
        config=SimpleNamespace(
            overlay_debug_mode=True, overlay_debug_max_lines=12, asr_backend="sherpa"
        ),
        metrics=metrics,
        audio=SimpleNamespace(queue=SimpleNamespace(qsize=lambda: 3, maxsize=200)),
        asr=SimpleNamespace(debug_step_num=9, native_chunk_samples=1600, wants_raw_audio=False),
        _recording=threading.Event(),
        _processing=threading.Event(),
        _asr_disabled_event=threading.Event(),
        _asr_thread_alive=True,
        _noise_floor_rms=0.004,
        _speech_rms_threshold=0.008,
        _recent_logs=recent_logs,
        _debug_current_transcript="working partial",
        _debug_last_final_transcript="finished final",
    )

    ShuVoiceApp._update_debug_overlay(app, state)

    debug_text = overlay.set_debug_text.call_args.args[0]
    assert "state rec=0 proc=0 asr_disabled=0 thread_alive=1" in debug_text
    assert "audio q=3/200" in debug_text
    assert "asr backend=sherpa step=9 chunk=1600 raw=0" in debug_text
    assert "utt buf=3200 speech_samples=1600 peak=0.0210 gain=1.40 unchanged=2" in debug_text
    assert "partial: working partial" in debug_text
    assert "final: finished final" in debug_text
    assert "logs:" in debug_text


def test_update_debug_overlay_noops_when_disabled():
    overlay = SimpleNamespace(set_debug_text=Mock())
    app = SimpleNamespace(
        overlay=overlay,
        config=SimpleNamespace(overlay_debug_mode=False, overlay_debug_max_lines=12),
    )

    ShuVoiceApp._update_debug_overlay(app)

    overlay.set_debug_text.assert_not_called()
