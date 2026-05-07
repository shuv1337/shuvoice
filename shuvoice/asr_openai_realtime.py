"""OpenAI Realtime transcription ASR backend."""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from typing import Any

import numpy as np

from .asr_base import ASRBackend, ASRCapabilities

log = logging.getLogger(__name__)

OPENAI_REALTIME_SAMPLE_RATE = 24000
OPENAI_REALTIME_WS_URL = "wss://api.openai.com/v1/realtime?intent=transcription"


class OpenAIRealtimeBackend(ASRBackend):
    """Cloud ASR backend using OpenAI Realtime transcription sessions."""

    capabilities = ASRCapabilities(
        supports_gpu=False,
        supports_model_download=False,
        wants_raw_audio=True,
        expected_chunking="streaming",
        finalization_mode="remote_manual_commit",
        preferred_sample_rate=OPENAI_REALTIME_SAMPLE_RATE,
    )

    def __init__(self, config):
        self.config = config
        self._ws = None
        self._receiver_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._completion_event = threading.Event()
        self._partial_by_item: dict[str, str] = {}
        self._completed_by_item: dict[str, str] = {}
        self._current_item_id: str | None = None
        self._latest_partial = ""
        self._latest_final = ""

    @staticmethod
    def dependency_errors() -> list[str]:
        try:
            import websocket  # noqa: F401
        except Exception as exc:
            return [f"Missing websocket-client dependency: {exc}"]
        return []

    @classmethod
    def startup_errors(cls, config) -> list[str]:
        errors = []
        env_name = str(config.openai_realtime_api_key_env).strip()
        if not env_name:
            errors.append("openai_realtime_api_key_env must not be empty")
        elif env_name.startswith(("sk_", "sk-")):
            errors.append(
                "openai_realtime_api_key_env looks like a raw API key value; "
                "set it to an environment variable name"
            )
        elif not os.environ.get(env_name, "").strip():
            errors.append(f"Missing OpenAI API key environment variable: {env_name}")
        turn_detection = str(config.openai_realtime_turn_detection).strip().lower()
        if turn_detection != "manual":
            errors.append(
                "OpenAI Realtime ASR currently supports only "
                "openai_realtime_turn_detection='manual'"
            )
        return errors

    @property
    def native_chunk_samples(self) -> int:
        return OPENAI_REALTIME_SAMPLE_RATE * int(self.config.chunk_ms) // 1000

    def load(self) -> None:
        import websocket

        api_key = os.environ[str(self.config.openai_realtime_api_key_env).strip()]
        timeout = float(self.config.openai_realtime_request_timeout_sec)
        self._ws = websocket.create_connection(
            OPENAI_REALTIME_WS_URL,
            timeout=timeout,
            header=[f"Authorization: Bearer {api_key}", "OpenAI-Beta: realtime=v1"],
        )
        self._stop_event.clear()
        self._send_session_update()
        self._receiver_thread = threading.Thread(
            target=self._receive_loop,
            name="openai-realtime-asr",
            daemon=True,
        )
        self._receiver_thread.start()

    def reset(self) -> None:
        with self._lock:
            self._partial_by_item.clear()
            self._completed_by_item.clear()
            self._current_item_id = None
            self._latest_partial = ""
            self._latest_final = ""
            self._completion_event.clear()

    def process_chunk(self, audio_chunk: np.ndarray) -> str:
        payload = {
            "type": "input_audio_buffer.append",
            "audio": self._encode_audio(audio_chunk),
        }
        self._send(payload)
        with self._lock:
            return self._latest_partial

    def finish_utterance(self, timeout_sec: float | None = None) -> str:
        timeout = (
            float(timeout_sec)
            if timeout_sec is not None
            else float(self.config.openai_realtime_commit_timeout_sec)
        )
        with self._lock:
            self._completion_event.clear()
            self._latest_final = ""
        self._send({"type": "input_audio_buffer.commit"})
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if self._completion_event.wait(min(0.05, remaining)):
                break
        with self._lock:
            if self._latest_final:
                return self._latest_final
            if self._latest_partial:
                log.warning("OpenAI Realtime commit timed out; using best partial transcript")
            return self._latest_partial

    def close(self) -> None:
        self._stop_event.set()
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                log.debug("OpenAI Realtime socket close failed", exc_info=True)
        if self._receiver_thread is not None and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=2.0)

    def _send_session_update(self) -> None:
        transcription: dict[str, Any] = {"model": str(self.config.openai_realtime_model).strip()}
        language = str(self.config.openai_realtime_language).strip()
        if language:
            transcription["language"] = language

        self._send(
            {
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": transcription,
                    "turn_detection": self._turn_detection_payload(),
                    "input_audio_noise_reduction": {"type": "near_field"},
                },
            }
        )

    def _turn_detection_payload(self):
        return None

    def _send(self, payload: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            raise RuntimeError("OpenAI Realtime WebSocket is not connected")
        ws.send(json.dumps(payload, separators=(",", ":")))

    def _receive_loop(self) -> None:
        while not self._stop_event.is_set():
            ws = self._ws
            if ws is None:
                return
            try:
                raw = ws.recv()
            except Exception as exc:
                if exc.__class__.__name__ == "WebSocketTimeoutException":
                    continue
                if not self._stop_event.is_set():
                    log.exception("OpenAI Realtime receiver stopped")
                return
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                log.debug("Ignoring non-JSON OpenAI Realtime event")
                continue
            self._handle_event(event)

    def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "conversation.item.input_audio_transcription.delta":
            self._handle_delta(event)
        elif event_type == "conversation.item.input_audio_transcription.completed":
            self._handle_completed(event)
        elif event_type == "input_audio_buffer.committed":
            self._handle_committed(event)
        elif event_type == "error":
            log.error("OpenAI Realtime error event: %s", event)

    def _handle_delta(self, event: dict[str, Any]) -> None:
        item_id = self._event_item_id(event)
        delta = str(event.get("delta") or event.get("transcript") or "")
        if not item_id or not delta:
            return
        with self._lock:
            if self._current_item_id is None:
                self._current_item_id = item_id
            self._partial_by_item[item_id] = self._partial_by_item.get(item_id, "") + delta
            if item_id == self._current_item_id:
                self._latest_partial = self._partial_by_item[item_id]

    def _handle_completed(self, event: dict[str, Any]) -> None:
        item_id = self._event_item_id(event)
        transcript = str(event.get("transcript") or "")
        if not item_id:
            return
        with self._lock:
            self._completed_by_item[item_id] = transcript
            if item_id == self._current_item_id:
                self._latest_final = transcript or self._partial_by_item.get(item_id, "")
                self._completion_event.set()

    def _handle_committed(self, event: dict[str, Any]) -> None:
        item_id = self._event_item_id(event)
        if not item_id:
            return
        with self._lock:
            self._current_item_id = item_id
            completed = self._completed_by_item.get(item_id)
            if completed is not None:
                self._latest_final = completed or self._partial_by_item.get(item_id, "")
                self._completion_event.set()

    @staticmethod
    def _event_item_id(event: dict[str, Any]) -> str:
        return str(event.get("item_id") or event.get("item", {}).get("id") or "")

    @staticmethod
    def _encode_audio(audio_chunk: np.ndarray) -> str:
        audio = np.asarray(audio_chunk, dtype=np.float32)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        clamped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clamped * 32767.0).astype("<i2")
        return base64.b64encode(pcm16.tobytes()).decode("ascii")
