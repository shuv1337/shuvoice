from __future__ import annotations

import base64
import builtins
import json

import numpy as np

from shuvoice.asr import create_backend, get_backend_class
from shuvoice.asr_openai_realtime import OPENAI_REALTIME_SAMPLE_RATE, OpenAIRealtimeBackend
from shuvoice.config import Config


class _FakeSocket:
    def __init__(self, on_send=None):
        self.sent: list[dict] = []
        self.closed = False
        self._on_send = on_send

    def send(self, payload: str) -> None:
        decoded = json.loads(payload)
        self.sent.append(decoded)
        if self._on_send is not None:
            self._on_send(decoded)

    def close(self) -> None:
        self.closed = True


class _FakeTimeout(Exception):
    pass


def test_config_and_registry_accept_openai_realtime():
    cfg = Config(asr_backend="openai_realtime")

    backend_cls = get_backend_class("openai_realtime")
    backend = create_backend("openai_realtime", cfg)

    assert backend_cls is OpenAIRealtimeBackend
    assert backend.capabilities.finalization_mode == "remote_manual_commit"
    assert backend.capabilities.preferred_sample_rate == OPENAI_REALTIME_SAMPLE_RATE
    assert backend.native_chunk_samples == 2400


def test_dependency_errors_when_websocket_client_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "websocket":
            raise ImportError("missing websocket")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert "websocket-client" in OpenAIRealtimeBackend.dependency_errors()[0]


def test_startup_errors_validate_api_key_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    errors = OpenAIRealtimeBackend.startup_errors(Config(asr_backend="openai_realtime"))

    assert errors == ["Missing OpenAI API key environment variable: OPENAI_API_KEY"]

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert OpenAIRealtimeBackend.startup_errors(Config(asr_backend="openai_realtime")) == []


def test_startup_errors_reject_non_manual_turn_detection(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    errors = OpenAIRealtimeBackend.startup_errors(
        Config(asr_backend="openai_realtime", openai_realtime_turn_detection="server_vad")
    )

    assert errors == [
        "OpenAI Realtime ASR currently supports only openai_realtime_turn_detection='manual'"
    ]


def test_encode_audio_clamps_float32_to_pcm16_base64():
    encoded = OpenAIRealtimeBackend._encode_audio(
        np.array([-2.0, -1.0, 0.0, 0.5, 2.0], dtype=np.float32)
    )

    pcm = np.frombuffer(base64.b64decode(encoded), dtype="<i2")

    assert pcm.tolist() == [-32767, -32767, 0, 16383, 32767]


def test_session_update_uses_current_transcription_session_shape():
    backend = OpenAIRealtimeBackend(Config(asr_backend="openai_realtime"))
    backend._ws = _FakeSocket()

    backend._send_session_update()

    assert backend._ws.sent == [
        {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "gpt-4o-transcribe",
                    "language": "en",
                },
                "turn_detection": None,
                "input_audio_noise_reduction": {"type": "near_field"},
            },
        }
    ]


def test_receive_loop_keeps_running_on_idle_websocket_timeout(caplog):
    class TimeoutSocket:
        def __init__(self):
            self.calls = 0

        def recv(self):
            self.calls += 1
            if self.calls == 1:
                raise _FakeTimeout("idle")
            return json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": "item-a",
                    "delta": "hello",
                }
            )

    _FakeTimeout.__name__ = "WebSocketTimeoutException"
    backend = OpenAIRealtimeBackend(Config(asr_backend="openai_realtime"))
    backend._ws = TimeoutSocket()

    original_handle_event = backend._handle_event

    def handle_event_and_stop(event: dict):
        original_handle_event(event)
        backend._stop_event.set()

    backend._handle_event = handle_event_and_stop

    backend._receive_loop()

    assert backend._latest_partial == "hello"
    assert "receiver stopped" not in caplog.text


def test_delta_and_completed_events_track_current_item_only():
    backend = OpenAIRealtimeBackend(Config(asr_backend="openai_realtime"))

    backend._handle_event(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "item-a",
            "delta": "hello ",
        }
    )
    backend._handle_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item-b",
            "transcript": "wrong turn",
        }
    )
    backend._handle_event(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "item-a",
            "delta": "world",
        }
    )
    backend._handle_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item-a",
            "transcript": "hello world",
        }
    )

    assert backend._latest_partial == "hello world"
    assert backend._latest_final == "hello world"


def test_late_completion_before_current_item_is_ignored():
    backend = OpenAIRealtimeBackend(Config(asr_backend="openai_realtime"))

    backend._handle_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "old-item",
            "transcript": "stale text",
        }
    )
    backend._handle_event(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "new-item",
            "delta": "fresh",
        }
    )

    assert backend._current_item_id == "new-item"
    assert backend._latest_partial == "fresh"
    assert backend._latest_final == ""


def test_committed_event_promotes_matching_stored_completion():
    backend = OpenAIRealtimeBackend(Config(asr_backend="openai_realtime"))

    backend._handle_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item-a",
            "transcript": "done",
        }
    )
    backend._handle_event({"type": "input_audio_buffer.committed", "item_id": "item-a"})

    assert backend._current_item_id == "item-a"
    assert backend._latest_final == "done"


def test_finish_utterance_sends_commit_and_returns_final():
    backend = OpenAIRealtimeBackend(
        Config(asr_backend="openai_realtime", openai_realtime_commit_timeout_sec=0.1)
    )

    def on_send(payload: dict) -> None:
        if payload["type"] != "input_audio_buffer.commit":
            return
        backend._handle_event({"type": "input_audio_buffer.committed", "item_id": "item-a"})
        backend._handle_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-a",
                "transcript": "done",
            }
        )

    backend._ws = _FakeSocket(on_send=on_send)

    assert backend.finish_utterance() == "done"
    assert backend._ws.sent[-1]["type"] == "input_audio_buffer.commit"


def test_finish_utterance_timeout_returns_best_partial():
    backend = OpenAIRealtimeBackend(
        Config(asr_backend="openai_realtime", openai_realtime_commit_timeout_sec=0.01)
    )
    backend._ws = _FakeSocket()
    backend._latest_partial = "partial"

    assert backend.finish_utterance() == "partial"


def test_reset_clears_transcript_state_and_close_closes_socket():
    backend = OpenAIRealtimeBackend(Config(asr_backend="openai_realtime"))
    backend._ws = _FakeSocket()
    backend._latest_partial = "stale"
    backend._latest_final = "stale"
    backend._current_item_id = "item-a"

    backend.reset()
    backend.close()

    assert backend._latest_partial == ""
    assert backend._latest_final == ""
    assert backend._current_item_id is None
    assert backend._ws is None
