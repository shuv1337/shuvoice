from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from shuvoice.tts_base import TTSCapabilities
from shuvoice.utterance_state import _UtteranceState

pytest.importorskip("gi")
from shuvoice.app import ShuVoiceApp


def test_recording_start_stop_transitions():
    tones: list[bool] = []

    app = SimpleNamespace(
        _recording=threading.Event(),
        _processing=threading.Event(),
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
    assert not app._processing.is_set()
    assert app.audio.clear.call_count == 2
    app.asr.reset.assert_called_once()
    app.overlay.show.assert_called_once()
    app.overlay.set_state.assert_any_call("listening")
    app.overlay.set_text.assert_called_with("Listening…")
    assert tones == [True]

    ShuVoiceApp._on_recording_stop(app)

    assert not app._recording.is_set()
    assert app._processing.is_set()
    app.overlay.set_state.assert_any_call("processing")
    assert tones == [True, False]


def test_recording_status_reports_processing_between_stop_and_commit():
    app = SimpleNamespace(
        _asr_disabled=False,
        _asr_thread_alive=True,
        _recording=threading.Event(),
        _processing=threading.Event(),
    )

    assert ShuVoiceApp._recording_status(app) == "idle"

    app._processing.set()
    assert ShuVoiceApp._recording_status(app) == "processing"

    app._recording.set()
    assert ShuVoiceApp._recording_status(app) == "recording"


def test_recording_start_ignores_spurious_restart_during_processing_rearm(monkeypatch):
    monkeypatch.setattr("shuvoice.runtime.state_machine.time.monotonic", lambda: 10.2)

    app = SimpleNamespace(
        _recording=threading.Event(),
        _processing=threading.Event(),
        _asr_thread_alive=True,
        _last_stop_monotonic=10.0,
        _show_overlay_error=Mock(),
    )
    app._processing.set()

    ShuVoiceApp._on_recording_start(app)

    assert app._recording.is_set() is False


def test_recording_start_allows_restart_after_processing_rearm_window(monkeypatch):
    monkeypatch.setattr("shuvoice.runtime.state_machine.time.monotonic", lambda: 11.0)

    tones: list[bool] = []
    app = SimpleNamespace(
        _recording=threading.Event(),
        _processing=threading.Event(),
        _asr_thread_alive=True,
        _show_overlay_error=Mock(),
        _asr_lock=threading.Lock(),
        _asr_disabled=False,
        _consecutive_asr_failures=0,
        _disable_asr=Mock(),
        _last_stop_monotonic=10.0,
        _ASR_MAX_FAILURES=10,
        audio=SimpleNamespace(clear=Mock()),
        asr=SimpleNamespace(reset=Mock()),
        overlay=None,
        _play_feedback_tone=lambda is_start: tones.append(is_start),
    )
    app._processing.set()

    ShuVoiceApp._on_recording_start(app)

    assert app._recording.is_set() is True
    assert app._processing.is_set() is False
    assert tones == [True]


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
        _is_offline_instant_mode=False,
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
        _is_offline_instant_mode=False,
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


def test_handle_recording_stop_offline_mode_decodes_once_and_commits():
    state = _UtteranceState(
        speech_samples=400,
        peak_rms=0.022,
        utterance_rms_threshold=0.008,
    )

    def decode(_state: _UtteranceState):
        _state.last_text = "decoded once"

    commit = Mock()

    app = SimpleNamespace(
        overlay=SimpleNamespace(set_state=Mock(), hide=Mock()),
        _drain_and_buffer=lambda _state: None,
        _min_speech_samples=100,
        _asr_disabled=False,
        _is_offline_instant_mode=True,
        _decode_offline_utterance=decode,
        _commit_utterance=commit,
        _speech_rms_threshold=0.008,
        typer=SimpleNamespace(reset=Mock()),
    )

    ShuVoiceApp._handle_recording_stop(app, state)

    commit.assert_called_once_with(state)
    app.overlay.hide.assert_called_once()
    app.typer.reset.assert_called_once()


def test_decode_offline_utterance_applies_gain_for_non_raw_backend():
    raw_audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    gained_audio = np.array([0.2, -0.4, 0.6], dtype=np.float32)
    state = _UtteranceState(
        buffer=[raw_audio],
        total=3,
        utterance_gain=2.0,
    )

    app = SimpleNamespace(
        _asr_disabled=False,
        asr=SimpleNamespace(wants_raw_audio=False),
        _apply_utterance_gain=Mock(return_value=gained_audio),
        _process_utterance_safe=Mock(return_value="offline text"),
        _recover_asr_after_failure=Mock(),
    )

    ShuVoiceApp._decode_offline_utterance(app, state)

    app._apply_utterance_gain.assert_called_once_with(raw_audio, 2.0)
    app._process_utterance_safe.assert_called_once_with(gained_audio)
    assert state.last_text == "offline text"


def test_process_recording_chunks_is_noop_in_offline_mode(monkeypatch):
    called = {"value": False}

    def fake_process_recording_chunks(_app, _state):
        called["value"] = True

    monkeypatch.setattr("shuvoice.app.process_recording_chunks", fake_process_recording_chunks)

    app = SimpleNamespace(_is_offline_instant_mode=True)
    state = _UtteranceState()

    result = ShuVoiceApp._process_recording_chunks(app, state)

    assert result is None
    assert called["value"] is False


def test_on_transcript_update_skips_partials_in_offline_mode():
    metrics = SimpleNamespace(observe_partial_update=Mock())
    app = SimpleNamespace(
        _render_transcript_text=lambda text: f"rendered:{text}",
        overlay=SimpleNamespace(set_text=Mock()),
        config=SimpleNamespace(output_mode="streaming_partial"),
        _is_offline_instant_mode=True,
        typer=SimpleNamespace(update_partial=Mock()),
        metrics=metrics,
    )

    ShuVoiceApp._on_transcript_update(app, "hello")

    app.overlay.set_text.assert_called_once_with("rendered:hello")
    app.typer.update_partial.assert_not_called()
    metrics.observe_partial_update.assert_not_called()


def test_on_transcript_update_keeps_partials_in_streaming_mode():
    metrics = SimpleNamespace(observe_partial_update=Mock())
    app = SimpleNamespace(
        _render_transcript_text=lambda text: f"rendered:{text}",
        overlay=SimpleNamespace(set_text=Mock()),
        config=SimpleNamespace(output_mode="streaming_partial"),
        _is_offline_instant_mode=False,
        typer=SimpleNamespace(update_partial=Mock()),
        metrics=metrics,
    )

    ShuVoiceApp._on_transcript_update(app, "hello")

    app.overlay.set_text.assert_called_once_with("rendered:hello")
    app.typer.update_partial.assert_called_once_with("rendered:hello")
    metrics.observe_partial_update.assert_called_once()


def test_render_transcript_text_applies_replacements_and_capitalize():
    app = SimpleNamespace(
        config=SimpleNamespace(
            text_replacements={"shove voice": "ShuVoice", "um": ""},
            auto_capitalize=True,
        )
    )

    rendered = ShuVoiceApp._render_transcript_text(app, "shove voice um")

    assert rendered == "ShuVoice"


def test_commit_utterance_uses_rendered_text_for_overlay_and_typing():
    overlay = SimpleNamespace(set_text=Mock())
    typer = SimpleNamespace(commit_final=Mock(), update_partial=Mock(), reset=Mock())
    app = SimpleNamespace(
        _render_transcript_text=lambda _text: "Rendered final",
        overlay=overlay,
        typer=typer,
        config=SimpleNamespace(use_clipboard_for_final=True),
    )
    state = _UtteranceState(last_text="raw transcript")

    ShuVoiceApp._commit_utterance(app, state)

    overlay.set_text.assert_called_once_with("Rendered final")
    typer.commit_final.assert_called_once_with("Rendered final")
    typer.update_partial.assert_not_called()


def test_commit_utterance_does_not_branch_on_legacy_clipboard_flag():
    overlay = SimpleNamespace(set_text=Mock())
    typer = SimpleNamespace(commit_final=Mock(), update_partial=Mock(), reset=Mock())
    app = SimpleNamespace(
        _render_transcript_text=lambda _text: "Rendered final",
        overlay=overlay,
        typer=typer,
        config=SimpleNamespace(use_clipboard_for_final=False),
    )
    state = _UtteranceState(last_text="raw transcript")

    ShuVoiceApp._commit_utterance(app, state)

    overlay.set_text.assert_called_once_with("Rendered final")
    typer.commit_final.assert_called_once_with("Rendered final")
    typer.update_partial.assert_not_called()


def test_commit_utterance_skips_when_rendered_text_is_empty():
    overlay = SimpleNamespace(set_text=Mock())
    typer = SimpleNamespace(commit_final=Mock(), update_partial=Mock(), reset=Mock())
    app = SimpleNamespace(
        _render_transcript_text=lambda _text: "",
        overlay=overlay,
        typer=typer,
        config=SimpleNamespace(use_clipboard_for_final=True),
    )
    state = _UtteranceState(last_text="um")

    ShuVoiceApp._commit_utterance(app, state)

    overlay.set_text.assert_not_called()
    typer.commit_final.assert_not_called()
    typer.update_partial.assert_not_called()


def test_remaining_splash_ms_is_zero_without_splash_timestamp():
    assert ShuVoiceApp._remaining_splash_ms(None, 2.0, now_monotonic=10.0) == 0


def test_remaining_splash_ms_counts_down_to_zero():
    remaining = ShuVoiceApp._remaining_splash_ms(10.0, 2.0, now_monotonic=10.3)
    assert 1690 <= remaining <= 1700

    assert ShuVoiceApp._remaining_splash_ms(10.0, 2.0, now_monotonic=12.5) == 0


def test_on_model_loaded_defers_activation_when_splash_is_too_fast(monkeypatch):
    timeout_add = Mock()
    monkeypatch.setattr("shuvoice.app.GLib.timeout_add", timeout_add)
    monkeypatch.setattr("shuvoice.app.time.monotonic", lambda: 10.2)

    app = SimpleNamespace(
        _model_loaded=False,
        _splash_started_monotonic=10.0,
        _MIN_SPLASH_VISIBLE_SEC=2.0,
        _complete_model_loaded_startup=Mock(),
    )

    result = ShuVoiceApp._on_model_loaded(app)

    assert app._model_loaded is True
    assert result == 0
    timeout_add.assert_called_once()
    delay_ms, callback = timeout_add.call_args.args
    assert delay_ms == 2000
    assert callback is app._complete_model_loaded_startup
    app._complete_model_loaded_startup.assert_not_called()


def test_complete_model_loaded_startup_dismisses_splash_and_finishes():
    splash = SimpleNamespace(dismiss=Mock())
    app = SimpleNamespace(
        _splash=splash,
        _splash_started_monotonic=10.0,
        _finish_activation=Mock(),
    )

    result = ShuVoiceApp._complete_model_loaded_startup(app)

    splash.dismiss.assert_called_once()
    assert app._splash is None
    assert app._splash_started_monotonic is None
    app._finish_activation.assert_called_once()
    assert result == 0


def test_on_model_loaded_prefers_realized_splash_timestamp(monkeypatch):
    timeout_add = Mock()
    monkeypatch.setattr("shuvoice.app.GLib.timeout_add", timeout_add)
    monkeypatch.setattr("shuvoice.app.time.monotonic", lambda: 11.0)

    splash = SimpleNamespace(shown_monotonic=10.8)
    app = SimpleNamespace(
        _model_loaded=False,
        _splash=splash,
        _splash_started_monotonic=10.0,
        _MIN_SPLASH_VISIBLE_SEC=2.0,
        _complete_model_loaded_startup=Mock(),
    )

    ShuVoiceApp._on_model_loaded(app)

    timeout_add.assert_called_once()
    delay_ms, callback = timeout_add.call_args.args
    assert delay_ms == 2000
    assert callback is app._complete_model_loaded_startup


def test_apply_utterance_gain_uses_float32_and_does_not_mutate_input():
    audio = np.array([0.1, -0.2, 0.95], dtype=np.float32)
    audio_before = audio.copy()

    result = ShuVoiceApp._apply_utterance_gain(SimpleNamespace(), audio, 2.0)

    np.testing.assert_array_equal(audio, audio_before)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.array([0.2, -0.4, 1.0], dtype=np.float32))


def test_apply_utterance_gain_returns_input_when_gain_near_unity():
    audio = np.array([0.1, -0.2], dtype=np.float32)

    result = ShuVoiceApp._apply_utterance_gain(SimpleNamespace(), audio, 1.01)

    assert result is audio


def test_recording_start_stops_active_tts_first(monkeypatch):
    called = {"started": False}

    monkeypatch.setattr(
        "shuvoice.app.on_recording_start",
        lambda _app: called.__setitem__("started", True),
    )

    tts_player = SimpleNamespace(is_active=lambda: True, stop=Mock())
    app = SimpleNamespace(tts_player=tts_player, tts_overlay=SimpleNamespace(hide=Mock()))

    ShuVoiceApp._on_recording_start(app)

    tts_player.stop.assert_called_once()
    app.tts_overlay.hide.assert_called_once()
    assert called["started"] is True


def test_tts_speak_selection_stops_recording_and_starts_player(monkeypatch):
    monkeypatch.setattr("shuvoice.app.capture_selection", lambda: "selected text")

    recording = threading.Event()
    recording.set()

    app = SimpleNamespace(
        _tts_runtime_ready=lambda: True,
        _recording=recording,
        _processing=threading.Event(),
        _on_recording_stop=Mock(side_effect=lambda: recording.clear()),
        _wait_for_stt_processing_clear=lambda timeout_sec=5.0: True,
        config=SimpleNamespace(tts_max_chars=5000, tts_model_id="model", tts_backend="elevenlabs"),
        _tts_voice_id="voice",
        tts_player=SimpleNamespace(speak=Mock(return_value=False)),
        metrics=SimpleNamespace(observe_tts_interrupt=Mock(), observe_tts_speak=Mock()),
        tts_overlay=SimpleNamespace(set_state=Mock()),
        _tts_last_preview_text="",
    )

    ShuVoiceApp._tts_speak_selection(app)

    app._on_recording_stop.assert_called_once()
    app.tts_player.speak.assert_called_once_with("selected text", "voice", "model")
    app.metrics.observe_tts_speak.assert_called_once()
    app.metrics.observe_tts_interrupt.assert_not_called()


def test_tts_set_playback_speed_updates_player_overlay_and_metrics():
    app = SimpleNamespace(
        _tts_playback_speed=1.0,
        config=SimpleNamespace(tts_backend="openai"),
        _tts_last_preview_text="preview",
        tts_backend=SimpleNamespace(
            capabilities=TTSCapabilities(supports_speed_control=True, speed_min=0.5, speed_max=2.0)
        ),
        tts_player=SimpleNamespace(
            set_playback_speed=Mock(return_value=1.4),
            is_active=lambda: False,
            restart=Mock(return_value=False),
        ),
        tts_overlay=SimpleNamespace(set_speed=Mock(), set_state=Mock()),
        metrics=SimpleNamespace(
            observe_tts_speed_change=Mock(),
            observe_tts_speed_restart=Mock(),
            observe_tts_speed_unsupported=Mock(),
        ),
        _tts_speed_capabilities=lambda: TTSCapabilities(
            supports_speed_control=True, speed_min=0.5, speed_max=2.0
        ),
    )

    result = ShuVoiceApp._tts_set_playback_speed(app, 1.4)

    assert result == 1.4
    assert app._tts_playback_speed == 1.4
    app.tts_player.set_playback_speed.assert_called_once_with(1.4)
    app.tts_overlay.set_speed.assert_called_once_with(1.4)
    app.metrics.observe_tts_speed_change.assert_called_once()
    app.metrics.observe_tts_speed_restart.assert_not_called()
    app.metrics.observe_tts_speed_unsupported.assert_not_called()


def test_tts_set_playback_speed_restarts_active_utterance():
    app = SimpleNamespace(
        _tts_playback_speed=1.0,
        config=SimpleNamespace(tts_backend="openai"),
        _tts_last_preview_text="preview",
        tts_backend=SimpleNamespace(
            capabilities=TTSCapabilities(supports_speed_control=True, speed_min=0.5, speed_max=2.0)
        ),
        tts_player=SimpleNamespace(
            set_playback_speed=Mock(return_value=1.3),
            is_active=lambda: True,
            restart=Mock(return_value=True),
        ),
        tts_overlay=SimpleNamespace(set_speed=Mock(), set_state=Mock()),
        metrics=SimpleNamespace(
            observe_tts_speed_change=Mock(),
            observe_tts_speed_restart=Mock(),
            observe_tts_speed_unsupported=Mock(),
        ),
        _tts_speed_capabilities=lambda: TTSCapabilities(
            supports_speed_control=True, speed_min=0.5, speed_max=2.0
        ),
    )

    result = ShuVoiceApp._tts_set_playback_speed(app, 1.3)

    assert result == 1.3
    app.tts_player.restart.assert_called_once_with()
    app.tts_overlay.set_state.assert_called_once_with("synthesizing", preview_text="preview")
    app.metrics.observe_tts_speed_restart.assert_called_once()


def test_tts_set_playback_speed_ignores_unsupported_backend():
    app = SimpleNamespace(
        _tts_playback_speed=1.0,
        config=SimpleNamespace(tts_backend="mock"),
        tts_backend=SimpleNamespace(capabilities=TTSCapabilities(supports_speed_control=False)),
        tts_player=SimpleNamespace(
            set_playback_speed=Mock(return_value=1.4),
            is_active=lambda: False,
            restart=Mock(return_value=False),
        ),
        tts_overlay=SimpleNamespace(set_speed=Mock()),
        metrics=SimpleNamespace(
            observe_tts_speed_change=Mock(),
            observe_tts_speed_restart=Mock(),
            observe_tts_speed_unsupported=Mock(),
        ),
        _tts_speed_capabilities=lambda: TTSCapabilities(supports_speed_control=False),
    )

    result = ShuVoiceApp._tts_set_playback_speed(app, 1.4)

    assert result == 1.0
    app.tts_player.set_playback_speed.assert_not_called()
    app.metrics.observe_tts_speed_change.assert_not_called()
    app.metrics.observe_tts_speed_restart.assert_not_called()
    app.metrics.observe_tts_speed_unsupported.assert_called_once()


def test_handle_tts_command_returns_disabled_error_when_config_disabled():
    app = SimpleNamespace(config=SimpleNamespace(tts_enabled=False))

    assert ShuVoiceApp._handle_tts_command(app, "tts_status") == "ERROR tts disabled"


def test_handle_tts_status_command_reports_player_state():
    app = SimpleNamespace(
        config=SimpleNamespace(tts_enabled=True),
        _tts_runtime_ready=lambda: True,
        tts_player=SimpleNamespace(state="playing"),
    )

    assert ShuVoiceApp._handle_tts_command(app, "tts_status") == "OK playing"
