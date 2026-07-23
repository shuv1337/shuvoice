"""NeMo ASR worker handler (protocol v1)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from shuvoice_worker_proto.constants import PROTOCOL_VERSION
from shuvoice_worker_proto.errors import ProtocolError
from shuvoice_worker_proto.messages import (
    msg_final_transcript,
    msg_partial_transcript,
    require_request_id,
)
from shuvoice_worker_proto.server import WorkerServer

from .engine import (
    NemoLoadConfig,
    create_engine,
    dependency_errors,
    native_chunk_samples,
)

log = logging.getLogger(__name__)

_NATIVE_SR = 16000


class NemoAsrHandler:
    backend_id = "nemo"

    def __init__(self, *, fake: bool = False) -> None:
        self._engine = create_engine(fake=fake)
        self._fake = fake

    def manifest(self) -> dict[str, Any]:
        rc = int(getattr(self._engine, "right_context", 13) or 13)
        model = getattr(self._engine, "model_name", None) if self._engine.loaded else None
        provider = getattr(self._engine, "device", None) if self._engine.loaded else None
        return {
            "backend_id": self.backend_id,
            "kind": "asr",
            "runtime_version": f"shuvoice-workers/{PROTOCOL_VERSION}",
            "model": model,
            "provider": provider,
            "asr": {
                "supports_model_download": True,
                "wants_raw_audio": True,
                "supports_streaming": True,
                "supports_offline_utterance": True,
                "supports_cancel": True,
                "native_sample_rate_hz": _NATIVE_SR,
                "native_chunk_samples": native_chunk_samples(rc),
            },
        }

    def on_cancel(self, request_id: uuid.UUID, server: WorkerServer) -> None:
        if self._engine.loaded:
            try:
                self._engine.reset()
            except Exception:
                log.exception("reset during cancel failed")

    def handle(self, msg_type: str, msg: dict[str, Any], server: WorkerServer) -> None:
        if msg_type == "load":
            self._load(msg, server)
            return
        if msg_type == "reset":
            self._reset(msg, server)
            return
        if msg_type == "process_chunk":
            self._process_chunk(msg, server, final=False)
            return
        if msg_type == "process_utterance":
            self._process_chunk(msg, server, final=True)
            return
        if msg_type == "finish":
            self._finish(msg, server)
            return
        raise ProtocolError("unsupported_type", f"unsupported message type: {msg_type}")

    def _load(self, msg: dict[str, Any], server: WorkerServer) -> None:
        rid = require_request_id(msg)
        if not self._fake:
            errors = dependency_errors()
            if errors:
                # Actionable, no secrets.
                raise ProtocolError("dependency_missing", errors[0][:300])
        cfg_raw = msg.get("config") if isinstance(msg.get("config"), dict) else {}
        cfg = NemoLoadConfig.from_mapping(cfg_raw)
        try:
            self._engine.load(cfg)
        except RuntimeError as exc:
            text = str(exc)
            if text.startswith("deps:"):
                raise ProtocolError("dependency_missing", text[5:].split("|", 1)[0][:300]) from exc
            raise ProtocolError("load_failed", "model load failed") from exc
        except Exception as exc:
            raise ProtocolError("load_failed", "model load failed") from exc
        server.send_ack(
            rid,
            result={
                "native_chunk_samples": native_chunk_samples(cfg.right_context),
                "right_context": cfg.right_context,
                "model_name": cfg.model_name,
                "device": cfg.device,
            },
        )

    def _reset(self, msg: dict[str, Any], server: WorkerServer) -> None:
        rid = require_request_id(msg)
        if not self._engine.loaded:
            raise ProtocolError("not_loaded", "ASR model is not loaded")
        try:
            self._engine.reset()
        except Exception as exc:
            raise ProtocolError("reset_failed", "reset failed") from exc
        server.send_ack(rid)

    def _process_chunk(self, msg: dict[str, Any], server: WorkerServer, *, final: bool) -> None:
        rid = require_request_id(msg)
        if not self._engine.loaded:
            raise ProtocolError("not_loaded", "ASR model is not loaded")
        # Demux cancel/close while waiting for PCM (same-request cancel is cooperative).
        encoding, data = server.read_pcm_for_request(rid)
        if server.is_cancelled(rid):
            server.clear_cancel(rid)
            raise ProtocolError("cancelled", "request cancelled")
        server.validate_audio_meta(
            msg, expected_sample_rate_hz=_NATIVE_SR, frame_encoding=encoding
        )
        if encoding == "f32_le":
            samples = list(data)  # type: ignore[arg-type]
        elif encoding == "i16_le":
            samples = [float(s) / 32768.0 for s in data]  # type: ignore[arg-type]
        else:
            raise ProtocolError("unsupported_encoding", "expected pcm f32_le or i16_le")

        # Residual: third-party NeMo inference is not preemptible mid-call; cancel
        # observed after the engine returns is honored below. Hosts may kill the
        # worker process to bound runaway inference.
        try:
            text = self._engine.process_chunk(samples)
            if final:
                if server.is_cancelled(rid):
                    server.clear_cancel(rid)
                    raise ProtocolError("cancelled", "request cancelled")
                text = self._engine.finish() or text
        except ProtocolError:
            raise
        except Exception as exc:
            raise ProtocolError("decode_failed", "ASR decode failed") from exc

        if server.is_cancelled(rid):
            server.clear_cancel(rid)
            raise ProtocolError("cancelled", "request cancelled")

        if final:
            server.send(msg_final_transcript(rid, text or ""))
        else:
            server.send(msg_partial_transcript(rid, text or ""))

    def _finish(self, msg: dict[str, Any], server: WorkerServer) -> None:
        rid = require_request_id(msg)
        if not self._engine.loaded:
            raise ProtocolError("not_loaded", "ASR model is not loaded")
        if server.is_cancelled(rid):
            server.clear_cancel(rid)
            raise ProtocolError("cancelled", "request cancelled")
        # Optional host-provided deadline (ms) — best-effort cooperative bound.
        timeout_ms = msg.get("timeout_ms")
        try:
            text = self._engine.finish(timeout_ms=timeout_ms)
        except Exception as exc:
            raise ProtocolError("finish_failed", "finish failed") from exc
        if server.is_cancelled(rid):
            server.clear_cancel(rid)
            raise ProtocolError("cancelled", "request cancelled")
        server.send(msg_final_transcript(rid, text or ""))
