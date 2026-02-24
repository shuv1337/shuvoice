from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

from shuvoice.runtime.state_machine import on_recording_start, on_recording_stop, recording_status


def test_on_recording_start_sets_state_and_overlay():
    app = SimpleNamespace(
        _recording=threading.Event(),
        _processing=threading.Event(),
        _asr_thread_alive=True,
        _show_overlay_error=Mock(),
        _asr_lock=threading.Lock(),
        _asr_disabled=False,
        _consecutive_asr_failures=0,
        _disable_asr=Mock(),
        _ASR_MAX_FAILURES=10,
        audio=SimpleNamespace(clear=Mock()),
        asr=SimpleNamespace(reset=Mock()),
        overlay=SimpleNamespace(show=Mock(), set_state=Mock(), set_text=Mock()),
        _play_feedback_tone=Mock(),
    )

    on_recording_start(app)

    assert app._recording.is_set()
    assert not app._processing.is_set()
    app.overlay.show.assert_called_once()


def test_recording_status_matrix():
    app = SimpleNamespace(
        _asr_disabled=False,
        _asr_thread_alive=True,
        _recording=threading.Event(),
        _processing=threading.Event(),
    )

    assert recording_status(app) == "idle"
    app._processing.set()
    assert recording_status(app) == "processing"
    app._recording.set()
    assert recording_status(app) == "recording"


def test_on_recording_stop_sets_processing():
    app = SimpleNamespace(
        _recording=threading.Event(),
        _processing=threading.Event(),
        overlay=SimpleNamespace(set_state=Mock()),
        _play_feedback_tone=Mock(),
    )
    app._recording.set()

    on_recording_stop(app)

    assert not app._recording.is_set()
    assert app._processing.is_set()
    app.overlay.set_state.assert_called_once_with("processing")
