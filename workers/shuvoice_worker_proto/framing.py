"""Length-prefixed binary frames (u32 BE length + kind + payload)."""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass
from typing import BinaryIO

from .constants import (
    BINARY_KINDS,
    BINARY_REQUEST_ID_LEN,
    FRAME_KIND_BYTES,
    FRAME_KIND_JSON,
    FRAME_KIND_PCM_F32_LE,
    FRAME_KIND_PCM_I16_LE,
    MAX_FRAME_LEN,
    MAX_JSON_PAYLOAD_LEN,
    MIN_BINARY_PAYLOAD_LEN,
    MIN_FRAME_LEN,
)
from .errors import ProtocolError

_U32_BE = struct.Struct(">I")
_F32_LE = struct.Struct("<f")
_I16_LE = struct.Struct("<h")


@dataclass(frozen=True, slots=True)
class Frame:
    kind: int
    payload: bytes

    @property
    def is_binary(self) -> bool:
        return self.kind in BINARY_KINDS

    def encode(self) -> bytes:
        length = 1 + len(self.payload)
        if length < MIN_FRAME_LEN:
            raise ProtocolError("frame_too_small", f"length {length} < {MIN_FRAME_LEN}")
        if length > MAX_FRAME_LEN:
            raise ProtocolError("frame_too_large", f"length {length} > {MAX_FRAME_LEN}")
        return _U32_BE.pack(length) + bytes([self.kind]) + self.payload

    @classmethod
    def json_bytes(cls, payload: bytes | str | dict) -> Frame:
        if isinstance(payload, dict):
            import json

            data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        elif isinstance(payload, str):
            data = payload.encode("utf-8")
        else:
            data = payload
        _check_json_payload_len(len(data))
        _check_payload_len(len(data))
        return cls(kind=FRAME_KIND_JSON, payload=data)

    @classmethod
    def binary(cls, kind: int, request_id: uuid.UUID, body: bytes = b"") -> Frame:
        if kind not in BINARY_KINDS:
            raise ProtocolError("invalid_binary", "frame kind is not binary")
        total = BINARY_REQUEST_ID_LEN + len(body)
        _check_payload_len(total)
        return cls(kind=kind, payload=request_id.bytes + body)

    @classmethod
    def pcm_f32le(cls, request_id: uuid.UUID, samples: list[float] | tuple[float, ...]) -> Frame:
        body = b"".join(_F32_LE.pack(float(s)) for s in samples)
        return cls.binary(FRAME_KIND_PCM_F32_LE, request_id, body)

    @classmethod
    def pcm_i16le(cls, request_id: uuid.UUID, samples: list[int] | tuple[int, ...]) -> Frame:
        body = b"".join(_I16_LE.pack(int(s)) for s in samples)
        return cls.binary(FRAME_KIND_PCM_I16_LE, request_id, body)

    @classmethod
    def bytes_frame(cls, request_id: uuid.UUID, body: bytes) -> Frame:
        return cls.binary(FRAME_KIND_BYTES, request_id, body)

    def split_binary_payload(self) -> tuple[uuid.UUID, bytes]:
        if not self.is_binary:
            raise ProtocolError("invalid_binary", "not a binary frame kind")
        if len(self.payload) < MIN_BINARY_PAYLOAD_LEN:
            raise ProtocolError("invalid_binary", "binary payload shorter than request id")
        req = uuid.UUID(bytes=self.payload[:BINARY_REQUEST_ID_LEN])
        return req, self.payload[BINARY_REQUEST_ID_LEN:]



def _check_json_payload_len(payload_len: int) -> None:
    if payload_len > MAX_JSON_PAYLOAD_LEN:
        raise ProtocolError(
            "json_too_large",
            f"JSON payload length {payload_len} > {MAX_JSON_PAYLOAD_LEN}",
        )


def _validate_kind_payload(kind: int, payload: bytes) -> None:
    if kind == FRAME_KIND_JSON:
        _check_json_payload_len(len(payload))
    if kind in BINARY_KINDS and len(payload) < MIN_BINARY_PAYLOAD_LEN:
        raise ProtocolError("invalid_binary", "binary payload shorter than request id")

def _check_payload_len(payload_len: int) -> None:
    length = payload_len + 1
    if length < MIN_FRAME_LEN:
        raise ProtocolError("frame_too_small", f"length {length} < {MIN_FRAME_LEN}")
    if length > MAX_FRAME_LEN:
        raise ProtocolError("frame_too_large", f"length {length} > {MAX_FRAME_LEN}")


def _validate_length_field(length: int) -> None:
    if length < MIN_FRAME_LEN:
        raise ProtocolError("frame_too_small", f"length {length} < {MIN_FRAME_LEN}")
    if length > MAX_FRAME_LEN:
        raise ProtocolError("frame_too_large", f"length {length} > {MAX_FRAME_LEN}")


def decode_frame(buf: bytes) -> tuple[Frame, int]:
    """Decode one frame from *buf*; return ``(frame, bytes_consumed)``."""
    if len(buf) < 4:
        raise ProtocolError("truncated_frame", f"need 4 length bytes, got {len(buf)}")
    (length,) = _U32_BE.unpack(buf[:4])
    _validate_length_field(length)
    total = 4 + length
    if len(buf) < total:
        raise ProtocolError("truncated_frame", f"declared {total} bytes, got {len(buf)}")
    kind = buf[4]
    if kind not in (
        FRAME_KIND_JSON,
        FRAME_KIND_PCM_F32_LE,
        FRAME_KIND_PCM_I16_LE,
        FRAME_KIND_BYTES,
    ):
        raise ProtocolError("unsupported_frame_kind", f"unsupported frame kind: {kind}")
    payload = buf[5:total]
    _validate_kind_payload(kind, payload)
    return Frame(kind=kind, payload=payload), total


def read_frame(stream: BinaryIO) -> Frame:
    """Read exactly one frame; allocate only after length validation."""
    len_buf = _read_exact(stream, 4, context="frame length")
    (length,) = _U32_BE.unpack(len_buf)
    _validate_length_field(length)
    body = _read_exact(stream, length, context="frame body")
    kind = body[0]
    if kind not in (
        FRAME_KIND_JSON,
        FRAME_KIND_PCM_F32_LE,
        FRAME_KIND_PCM_I16_LE,
        FRAME_KIND_BYTES,
    ):
        raise ProtocolError("unsupported_frame_kind", f"unsupported frame kind: {kind}")
    payload = body[1:]
    _validate_kind_payload(kind, payload)
    return Frame(kind=kind, payload=payload)


def write_frame(stream: BinaryIO, frame: Frame) -> None:
    stream.write(frame.encode())
    stream.flush()


def _read_exact(stream: BinaryIO, n: int, *, context: str) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if chunk is None:  # pragma: no cover
            continue
        if chunk == b"":
            if not buf and context == "frame length":
                raise ProtocolError("unexpected_eof", "unexpected end of stream while reading frame length")
            raise ProtocolError(
                "truncated_frame",
                f"truncated {context}: declared {n} bytes, got {len(buf)}",
            )
        buf.extend(chunk)
    return bytes(buf)


def decode_f32le_samples(body: bytes) -> list[float]:
    if len(body) % 4 != 0:
        raise ProtocolError("invalid_binary", "f32le PCM length not multiple of 4")
    return [_F32_LE.unpack_from(body, i)[0] for i in range(0, len(body), 4)]


def decode_i16le_samples(body: bytes) -> list[int]:
    if len(body) % 2 != 0:
        raise ProtocolError("invalid_binary", "i16le PCM length not multiple of 2")
    return [_I16_LE.unpack_from(body, i)[0] for i in range(0, len(body), 2)]
