"""MeloTTS worker handler (protocol v1)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from shuvoice_worker_proto.constants import FRAME_KIND_PCM_I16_LE, PROTOCOL_VERSION
from shuvoice_worker_proto.errors import ProtocolError
from shuvoice_worker_proto.framing import Frame
from shuvoice_worker_proto.messages import msg_audio_end, msg_audio_start, msg_voices, require_request_id
from shuvoice_worker_proto.server import WorkerServer

from .engine import SAMPLE_RATE_HZ, MeloSynthRequest, create_engine, dependency_errors

log = logging.getLogger(__name__)

# Chunk PCM into ~4 KiB frames for streaming-friendly transport.
_PCM_CHUNK_BYTES = 4096


class MeloTtsHandler:
    backend_id = "melotts"

    def __init__(self, *, fake: bool = False, device: str = "auto") -> None:
        self._engine = create_engine(fake=fake, device=device)
        self._fake = fake
        self._device = device
        self._active: uuid.UUID | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "kind": "tts",
            "runtime_version": f"shuvoice-workers/{PROTOCOL_VERSION}",
            "provider": self._device,
            "tts": {
                "requires_api_key": False,
                "supports_native_speed": True,
                # Full utterance is synthesized before any PCM is emitted.
                "supports_streaming_audio": False,
                "supports_list_voices": True,
                "supports_cancel": True,
                "default_sample_rate_hz": SAMPLE_RATE_HZ,
                "max_chars": 5000,
            },
        }

    def on_cancel(self, request_id: uuid.UUID, server: WorkerServer) -> None:
        if self._active == request_id:
            self._active = None

    def handle(self, msg_type: str, msg: dict[str, Any], server: WorkerServer) -> None:
        if msg_type == "load":
            rid = require_request_id(msg)
            if not self._fake:
                errors = dependency_errors()
                if errors:
                    raise ProtocolError("dependency_missing", errors[0][:300])
            server.send_ack(rid, result={"device": self._device})
            return
        if msg_type == "list_voices":
            rid = require_request_id(msg)
            server.send(msg_voices(rid, self._engine.list_voices()))
            return
        if msg_type == "synthesize":
            self._synthesize(msg, server)
            return
        if msg_type in {"reset", "process_chunk", "process_utterance", "finish"}:
            raise ProtocolError(
                "unsupported_type", f"ASR message not supported by TTS worker: {msg_type}"
            )
        raise ProtocolError("unsupported_type", f"unsupported message type: {msg_type}")

    def _synthesize(self, msg: dict[str, Any], server: WorkerServer) -> None:
        rid = require_request_id(msg)
        text = str(msg.get("text", ""))
        # Do not put text into errors.
        if not text.strip():
            raise ProtocolError("empty_text", "synthesis text is empty")
        if len(text) > 5000:
            raise ProtocolError("text_too_long", "synthesis text exceeds max_chars")

        voice_id = str(msg.get("voice_id") or "EN-US")
        speed = float(msg.get("speed") or 1.0)
        speed = max(0.5, min(2.0, speed))
        encoding = str(msg.get("output_encoding") or "f32_le")
        if encoding not in {"i16_le", "f32_le"}:
            encoding = "f32_le"

        self._active = rid
        try:
            # Residual: MeloTTS inference is not preemptible mid-call. Cancel is
            # checked before/after synthesis and between PCM chunks; hosts may
            # kill the worker process to bound runaway inference.
            pcm_i16 = self._engine.synthesize_i16(
                MeloSynthRequest(text=text, voice_id=voice_id, speed=speed)
            )
        except ValueError as exc:
            if str(exc) == "empty_text":
                raise ProtocolError("empty_text", "synthesis text is empty") from exc
            raise ProtocolError("synth_failed", "synthesis failed") from exc
        except RuntimeError as exc:
            if str(exc).startswith("import_failed"):
                raise ProtocolError(
                    "dependency_missing",
                    "MeloTTS import failed; run inside the MeloTTS venv",
                ) from exc
            raise ProtocolError("synth_failed", "synthesis failed") from exc
        except Exception as exc:
            raise ProtocolError("synth_failed", "synthesis failed") from exc

        if server.is_cancelled(rid) or self._active != rid:
            server.clear_cancel(rid)
            raise ProtocolError("cancelled", "request cancelled")

        server.send(
            msg_audio_start(
                rid,
                sample_rate_hz=SAMPLE_RATE_HZ,
                channels=1,
                encoding=encoding,
            )
        )

        if encoding == "f32_le":
            # Convert i16 LE bytes to f32 LE for clients that prefer float frames.
            import array
            import struct

            samples_i16 = array.array("h")
            samples_i16.frombytes(pcm_i16)
            body = b"".join(struct.pack("<f", float(s) / 32768.0) for s in samples_i16)
            kind = 2  # FRAME_KIND_PCM_F32_LE
            payload_body = body
        else:
            kind = FRAME_KIND_PCM_I16_LE
            payload_body = pcm_i16

        if not payload_body:
            raise ProtocolError("empty_audio", "synthesis produced no PCM")

        offset = 0
        while offset < len(payload_body):
            if server.is_cancelled(rid) or self._active != rid:
                server.clear_cancel(rid)
                raise ProtocolError("cancelled", "request cancelled")
            chunk = payload_body[offset : offset + _PCM_CHUNK_BYTES]
            offset += len(chunk)
            server.send_frame(Frame.binary(kind, rid, chunk))

        server.send(
            msg_audio_end(
                rid,
                sample_rate_hz=SAMPLE_RATE_HZ,
                channels=1,
                encoding=encoding,
            )
        )
        self._active = None
