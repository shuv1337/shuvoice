"""Minimal client for tests and manual probing (not required by the Rust shell)."""

from __future__ import annotations

import time
import uuid
from typing import Any, BinaryIO

from .constants import (
    DEFAULT_LOAD_TIMEOUT_SEC,
    DEFAULT_MAX_IGNORED_MESSAGES,
    DEFAULT_RPC_TIMEOUT_SEC,
    PROTOCOL_VERSION,
)
from .errors import ProtocolError
from .framing import Frame, read_frame, write_frame
from .messages import parse_control_message


class WorkerClient:
    def __init__(
        self,
        reader: BinaryIO,
        writer: BinaryIO,
        *,
        rpc_timeout_sec: float = DEFAULT_RPC_TIMEOUT_SEC,
        load_timeout_sec: float = DEFAULT_LOAD_TIMEOUT_SEC,
        max_ignored_messages: int = DEFAULT_MAX_IGNORED_MESSAGES,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.manifest: dict[str, Any] | None = None
        self.rpc_timeout_sec = float(rpc_timeout_sec)
        self.load_timeout_sec = float(load_timeout_sec)
        self.max_ignored_messages = int(max_ignored_messages)

    def _send(self, msg: dict[str, Any]) -> None:
        write_frame(self.writer, Frame.json_bytes(msg))

    def _recv(self) -> Frame:
        return read_frame(self.reader)

    def _recv_msg(self) -> dict[str, Any]:
        frame = self._recv()
        if frame.kind != 1:
            raise ProtocolError("unexpected_binary", "expected JSON control frame")
        return parse_control_message(frame.payload)

    def handshake(self, client_name: str = "test") -> dict[str, Any]:
        self._send(
            {
                "type": "hello",
                "protocol_version": PROTOCOL_VERSION,
                "client_name": client_name,
            }
        )
        reply = self._recv_msg()
        if reply.get("type") == "hello_err":
            code = str(reply.get("code") or "handshake")
            if code == "unsupported_version":
                raise ProtocolError("unsupported_version", str(reply.get("message")))
            raise ProtocolError(code, str(reply.get("message")))
        if reply.get("type") != "hello_ok":
            raise ProtocolError("unexpected_message", f"expected hello_ok, got {reply.get('type')}")
        if int(reply.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise ProtocolError("unsupported_version", "version mismatch")
        self.manifest = reply.get("manifest") if isinstance(reply.get("manifest"), dict) else {}
        return self.manifest or {}

    def load(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        rid = uuid.uuid4()
        self._send({"type": "load", "request_id": str(rid), "config": config or {}})
        return self._wait_for(rid, {"ack", "error"}, timeout_sec=self.load_timeout_sec)

    def reset(self) -> dict[str, Any]:
        rid = uuid.uuid4()
        self._send({"type": "reset", "request_id": str(rid)})
        return self._wait_for(rid, {"ack", "error"})

    def process_chunk_f32(self, samples: list[float], sample_rate_hz: int = 16000) -> dict[str, Any]:
        rid = uuid.uuid4()
        self._send(
            {
                "type": "process_chunk",
                "request_id": str(rid),
                "sample_rate_hz": sample_rate_hz,
                "channels": 1,
                "encoding": "f32_le",
                "end": True,
            }
        )
        write_frame(self.writer, Frame.pcm_f32le(rid, samples))
        return self._wait_for(rid, {"partial_transcript", "final_transcript", "error"})

    def process_utterance_f32(
        self, samples: list[float], sample_rate_hz: int = 16000
    ) -> dict[str, Any]:
        rid = uuid.uuid4()
        self._send(
            {
                "type": "process_utterance",
                "request_id": str(rid),
                "sample_rate_hz": sample_rate_hz,
                "channels": 1,
                "encoding": "f32_le",
                "end": True,
            }
        )
        write_frame(self.writer, Frame.pcm_f32le(rid, samples))
        return self._wait_for(rid, {"final_transcript", "error"})

    def finish(self) -> dict[str, Any]:
        rid = uuid.uuid4()
        self._send({"type": "finish", "request_id": str(rid)})
        return self._wait_for(rid, {"final_transcript", "error"})

    def cancel(self, request_id: uuid.UUID) -> dict[str, Any]:
        self._send({"type": "cancel", "request_id": str(request_id)})
        return self._wait_for(request_id, {"ack", "error"})

    def list_voices(self) -> dict[str, Any]:
        rid = uuid.uuid4()
        self._send({"type": "list_voices", "request_id": str(rid)})
        return self._wait_for(rid, {"voices", "error"})

    def synthesize(
        self, text: str, voice_id: str = "EN-US", speed: float = 1.0
    ) -> tuple[dict, bytes]:
        rid = uuid.uuid4()
        # Align with Rust WorkerClient default output_encoding (f32_le).
        self._send(
            {
                "type": "synthesize",
                "request_id": str(rid),
                "text": text,
                "voice_id": voice_id,
                "speed": speed,
                "output_encoding": "f32_le",
            }
        )
        pcm = bytearray()
        meta: dict[str, Any] = {}
        ignored = 0
        deadline = time.monotonic() + self.rpc_timeout_sec
        while True:
            if time.monotonic() > deadline:
                raise ProtocolError("rpc_timeout", "synthesize timed out")
            frame = self._recv()
            if frame.kind == 1:
                msg = parse_control_message(frame.payload)
                if msg.get("request_id") not in (None, str(rid)):
                    ignored += 1
                    if ignored > self.max_ignored_messages:
                        raise ProtocolError("too_many_ignored", "ignored message limit exceeded")
                    continue
                if msg.get("type") == "error":
                    raise ProtocolError(str(msg.get("code")), str(msg.get("message")))
                if msg.get("type") == "audio_start":
                    meta = msg
                    continue
                if msg.get("type") == "audio_end":
                    if not pcm:
                        raise ProtocolError("empty_audio", "TTS audio stream produced no PCM")
                    return meta, bytes(pcm)
                ignored += 1
                if ignored > self.max_ignored_messages:
                    raise ProtocolError("too_many_ignored", "ignored message limit exceeded")
                continue
            frid, body = frame.split_binary_payload()
            if frid != rid:
                ignored += 1
                if ignored > self.max_ignored_messages:
                    raise ProtocolError("too_many_ignored", "ignored message limit exceeded")
                continue
            pcm.extend(body)

    def close(self) -> None:
        rid = uuid.uuid4()
        self._send({"type": "close", "request_id": str(rid)})
        try:
            self._wait_for(rid, {"ack", "error"})
        except ProtocolError as exc:
            if exc.code == "unexpected_eof":
                return
            raise

    def _wait_for(
        self,
        rid: uuid.UUID,
        types: set[str],
        *,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + float(
            self.rpc_timeout_sec if timeout_sec is None else timeout_sec
        )
        ignored = 0
        while True:
            if time.monotonic() > deadline:
                raise ProtocolError("rpc_timeout", "RPC timed out")
            try:
                msg = self._recv_msg()
            except ProtocolError:
                raise
            if msg.get("type") == "error" and msg.get("request_id") in (None, str(rid)):
                raise ProtocolError(str(msg.get("code")), str(msg.get("message")))
            if msg.get("type") in types and msg.get("request_id") == str(rid):
                return msg
            ignored += 1
            if ignored > self.max_ignored_messages:
                raise ProtocolError("too_many_ignored", "ignored message limit exceeded")
