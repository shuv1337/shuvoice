from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from shuvoice.utterance_state import _UtteranceState

pytest.importorskip("gi")
from shuvoice.app import ShuVoiceApp


def test_recording_start_stop_transitions():
    tones: list[bool] = []

    app = SimpleNamespace(
        _recording=threading.Event(),
        _asr_thread_alive=True,
        _show_overlay_error=Mock(),
        _asr_lock=threading.Lock(),
        _asr_disabled=False,
        _consecutive_asr_failures=0,
        _disable_asr=Mock(),
        audio=SimpleNamespace(clear=Mock()),
        asr=SimpleNamespace(reset=Mock()),
        overlay=SimpleNamespace(show=Mock(), set_state=Mock(), set_text=Mock()),
        _play_feedback_tone=lambda is_start: tones.append(is_start),
    )

    ShuVoiceApp._on_recording_start(app)

    assert app._recording.is_set()
    assert app.audio.clear.call_count == 2
    app.asr.reset.assert_called_once()
    app.overlay.show.assert_called_once()
    app.overlay.set_state.assert_any_call("listening")
    app.overlay.set_text.assert_called_with("Listening…")
    assert tones == [True]

    ShuVoiceApp._on_recording_stop(app)

    assert not app._recording.is_set()
    app.overlay.set_state.assert_any_call("processing")
    assert tones == [True, False]


def test_begin_utterance_resets_asr_before_threshold_setup():
    reset_calls = 0

    def reset():
        nonlocal reset_calls
        reset_calls += 1

    state = _UtteranceState(last_text="stale", speech_samples=123, utterance_rms_threshold=0.0)
    app = SimpleNamespace(
        _asr_lock=threading.Lock(),
        _asr_disabled=False,
        asr=SimpleNamespace(reset=reset),
        _recover_asr_after_failure=Mock(),
        _speech_rms_threshold=0.008,
        _noise_floor_rms=0.010,
        _speech_rms_multiplier=1.8,
    )

    ShuVoiceApp._begin_utterance(app, state)

    assert reset_calls == 1
    assert state.last_text == ""
    assert state.speech_samples == 0
    assert state.utterance_rms_threshold == pytest.approx(0.018)
    app._recover_asr_after_failure.assert_not_called()


def test_flush_tail_silence_aborts_when_new_recording_already_started():
    process_calls = 0

    def process_chunk(_audio):
        nonlocal process_calls
        process_calls += 1
        return ""

    recording = threading.Event()
    recording.set()

    state = _UtteranceState(last_text="existing transcript")
    app = SimpleNamespace(
        _asr_disabled=False,
        _recording=recording,
        asr=SimpleNamespace(native_chunk_samples=1600, wants_raw_audio=True),
        _make_flush_noise=lambda _n, escalation=1.0: None,
        _process_chunk_safe=process_chunk,
        _recover_asr_after_failure=Mock(),
        overlay=None,
        config=SimpleNamespace(output_mode="final_only"),
        _FLUSH_NOISE_ESCALATION=ShuVoiceApp._FLUSH_NOISE_ESCALATION,
        _FLUSH_NOISE_MAX_RMS=ShuVoiceApp._FLUSH_NOISE_MAX_RMS,
    )

    ShuVoiceApp._flush_tail_silence(app, state)

    assert process_calls == 0


def test_flush_tail_silence_aborts_if_recording_restarts_mid_flush():
    process_calls = 0
    recording = threading.Event()

    def process_chunk(_audio):
        nonlocal process_calls
        process_calls += 1
        recording.set()
        return ""

    state = _UtteranceState(last_text="")
    app = SimpleNamespace(
        _asr_disabled=False,
        _recording=recording,
        asr=SimpleNamespace(native_chunk_samples=1600, wants_raw_audio=True),
        _make_flush_noise=lambda n, escalation=1.0: [0.0] * n,
        _process_chunk_safe=process_chunk,
        _recover_asr_after_failure=Mock(),
        overlay=None,
        config=SimpleNamespace(output_mode="final_only"),
        _FLUSH_NOISE_ESCALATION=ShuVoiceApp._FLUSH_NOISE_ESCALATION,
        _FLUSH_NOISE_MAX_RMS=ShuVoiceApp._FLUSH_NOISE_MAX_RMS,
    )

    ShuVoiceApp._flush_tail_silence(app, state)

    assert process_calls == 1


def test_handle_recording_stop_ignores_silent_utterances_without_commit():
    commit_calls = 0

    def commit(_state):
        nonlocal commit_calls
        commit_calls += 1

    state = _UtteranceState(
        speech_samples=20,
        peak_rms=0.012,
        utterance_rms_threshold=0.008,
    )

    app = SimpleNamespace(
        overlay=SimpleNamespace(set_state=Mock(), hide=Mock()),
        _drain_and_buffer=lambda _state: None,
        _min_speech_samples=100,
        _speech_rms_threshold=0.008,
        typer=SimpleNamespace(reset=Mock()),
        _asr_disabled=False,
        _commit_utterance=commit,
    )

    ShuVoiceApp._handle_recording_stop(app, state)

    assert commit_calls == 0
    app.overlay.hide.assert_called_once()
    app.typer.reset.assert_called_once()


def test_handle_recording_stop_commits_when_speech_threshold_met():
    commit_calls = 0

    def commit(_state):
        nonlocal commit_calls
        commit_calls += 1

    state = _UtteranceState(
        speech_samples=400,
        peak_rms=0.022,
        utterance_rms_threshold=0.008,
        last_text="hello world",
    )

    app = SimpleNamespace(
        overlay=SimpleNamespace(set_state=Mock(), hide=Mock()),
        _drain_and_buffer=lambda _state: None,
        _min_speech_samples=100,
        _asr_disabled=False,
        asr=SimpleNamespace(native_chunk_samples=1600, wants_raw_audio=True, debug_step_num=0),
        _flush_tail_silence=lambda _state: None,
        _commit_utterance=commit,
        _speech_rms_threshold=0.008,
        typer=SimpleNamespace(reset=Mock()),
    )

    ShuVoiceApp._handle_recording_stop(app, state)

    assert commit_calls == 1
    app.overlay.hide.assert_called_once()
    app.typer.reset.assert_called_once()
