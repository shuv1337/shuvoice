"""Moonshine ASR worker handler (protocol v1)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from shuvoice_worker_proto.constants import PROTOCOL_VERSION
from shuvoice_worker_proto.errors import ProtocolError
from shuvoice_worker_proto.messages import msg_final_transcript, msg_partial_transcript, require_request_id
from shuvoice_worker_proto.server import WorkerServer

from .engine import EXPECTED_SAMPLE_RATE, MoonshineLoadConfig, create_engine, dependency_errors

log = logging.getLogger(__name__)


class MoonshineAsrHandler:
    backend_id = "moonshine"

    def __init__(self, *, fake: bool = False) -> None:
        self._engine = create_engine(fake=fake)
        self._fake = fake

    def manifest(self) -> dict[str, Any]:
        model = self._engine.model_name if self._engine.loaded else None
        # Honest caps: windowed re-decode, not true frame-streaming ASR.
        return {
            "backend_id": self.backend_id,
            "kind": "asr",
            "runtime_version": f"shuvoice-workers/{PROTOCOL_VERSION}",
            "model": model,
            "asr": {
                "supports_model_download": False,
                "wants_raw_audio": True,
                # process_chunk is supported as windowed offline re-decode.
                "supports_streaming": True,
                "supports_offline_utterance": True,
                "supports_cancel": True,
                "native_sample_rate_hz": EXPECTED_SAMPLE_RATE,
            },
        }

    def on_cancel(self, request_id: uuid.UUID, server: WorkerServer) -> None:
        if self._engine.loaded:
            try:
                self._engine.reset()
            except Exception:
                log.exception("moonshine reset on cancel failed")

    def handle(self, msg_type: str, msg: dict[str, Any], server: WorkerServer) -> None:
        if msg_type == "load":
            rid = require_request_id(msg)
            if not self._fake:
                errors = dependency_errors()
                if errors:
                    raise ProtocolError("dependency_missing", errors[0][:300])
            cfg = MoonshineLoadConfig.from_mapping(
                msg.get("config") if isinstance(msg.get("config"), dict) else {}
            )
            try:
                self._engine.load(cfg)
            except RuntimeError as exc:
                text = str(exc)
                if text.startswith("deps:"):
                    raise ProtocolError("dependency_missing", text[5:][:300]) from exc
                raise ProtocolError("load_failed", "model load failed") from exc
            except Exception as exc:
                raise ProtocolError("load_failed", "model load failed") from exc
            server.send_ack(
                rid,
                result={
                    "model_name": cfg.model_name,
                    "max_window_sec": cfg.max_window_sec,
                    "max_tokens": cfg.max_tokens,
                    "provider": cfg.provider,
                },
            )
            return
        if msg_type == "reset":
            rid = require_request_id(msg)
            if not self._engine.loaded:
                raise ProtocolError("not_loaded", "ASR model is not loaded")
            self._engine.reset()
            server.send_ack(rid)
            return
        if msg_type in {"process_chunk", "process_utterance"}:
            rid = require_request_id(msg)
            if not self._engine.loaded:
                raise ProtocolError("not_loaded", "ASR model is not loaded")
            encoding, data = server.read_pcm_for_request(rid)
            if server.is_cancelled(rid):
                server.clear_cancel(rid)
                raise ProtocolError("cancelled", "request cancelled")
            server.validate_audio_meta(
                msg, expected_sample_rate_hz=EXPECTED_SAMPLE_RATE, frame_encoding=encoding
            )
            if encoding == "f32_le":
                samples = list(data)  # type: ignore[arg-type]
            elif encoding == "i16_le":
                samples = [float(s) / 32768.0 for s in data]  # type: ignore[arg-type]
            else:
                raise ProtocolError("unsupported_encoding", "expected pcm f32_le or i16_le")
            try:
                text = self._engine.process_chunk(samples)
                if msg_type == "process_utterance":
                    text = self._engine.finish() or text
            except Exception as exc:
                raise ProtocolError("decode_failed", "ASR decode failed") from exc
            if server.is_cancelled(rid):
                server.clear_cancel(rid)
                raise ProtocolError("cancelled", "request cancelled")
            if msg_type == "process_utterance":
                server.send(msg_final_transcript(rid, text or ""))
            else:
                server.send(msg_partial_transcript(rid, text or ""))
            return
        if msg_type == "finish":
            rid = require_request_id(msg)
            if not self._engine.loaded:
                raise ProtocolError("not_loaded", "ASR model is not loaded")
            if server.is_cancelled(rid):
                server.clear_cancel(rid)
                raise ProtocolError("cancelled", "request cancelled")
            text = self._engine.finish(timeout_ms=msg.get("timeout_ms"))
            server.send(msg_final_transcript(rid, text or ""))
            return
        raise ProtocolError("unsupported_type", f"unsupported message type: {msg_type}")
