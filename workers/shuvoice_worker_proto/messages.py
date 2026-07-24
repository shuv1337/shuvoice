"""JSON control-plane helpers (schema mirrors Rust ControlMessage)."""

from __future__ import annotations

import json
import uuid
from typing import Any


def parse_control_message(payload: bytes | str) -> dict[str, Any]:
    if isinstance(payload, bytes):
        text = payload.decode("utf-8")
    else:
        text = payload
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("control message must be a JSON object")
    if "type" not in data:
        raise ValueError("control message missing type")
    return data


def dumps_control(msg: dict[str, Any]) -> bytes:
    """Serialize a control message with stable separators (no spaces)."""
    return json.dumps(msg, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def parse_request_id(value: Any) -> uuid.UUID | None:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def require_request_id(msg: dict[str, Any]) -> uuid.UUID:
    rid = parse_request_id(msg.get("request_id"))
    if rid is None:
        raise ValueError("missing request_id")
    return rid


# ── constructors (keep field order stable for golden tests) ─────────────


def msg_hello_ok(*, protocol_version: int, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "hello_ok",
        "protocol_version": protocol_version,
        "manifest": manifest,
    }


def msg_hello_err(
    *,
    message: str,
    code: str | None = None,
    protocol_version: int | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "hello_err", "message": message}
    if code is not None:
        out["code"] = code
    if protocol_version is not None:
        out["protocol_version"] = int(protocol_version)
    return out


def msg_ack(request_id: uuid.UUID, result: Any | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "ack", "request_id": str(request_id)}
    if result is not None:
        out["result"] = result
    return out


def msg_error(
    *,
    code: str,
    message: str,
    request_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    # Intentionally never include user text / secrets.
    out: dict[str, Any] = {"type": "error", "code": code, "message": message}
    if request_id is not None:
        out["request_id"] = str(request_id)
    return out


def msg_partial_transcript(request_id: uuid.UUID, text: str) -> dict[str, Any]:
    return {
        "type": "partial_transcript",
        "request_id": str(request_id),
        "text": text,
    }


def msg_final_transcript(request_id: uuid.UUID, text: str) -> dict[str, Any]:
    return {
        "type": "final_transcript",
        "request_id": str(request_id),
        "text": text,
    }


def msg_progress(
    request_id: uuid.UUID,
    *,
    fraction: float | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "progress", "request_id": str(request_id)}
    if fraction is not None:
        out["fraction"] = fraction
    if message is not None:
        out["message"] = message
    return out


def msg_voices(request_id: uuid.UUID, voices: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "voices",
        "request_id": str(request_id),
        "voices": voices,
    }


def msg_audio_start(
    request_id: uuid.UUID,
    *,
    sample_rate_hz: int,
    channels: int = 1,
    encoding: str = "f32_le",
) -> dict[str, Any]:
    return {
        "type": "audio_start",
        "request_id": str(request_id),
        "sample_rate_hz": sample_rate_hz,
        "channels": channels,
        "encoding": encoding,
    }


def msg_audio_end(
    request_id: uuid.UUID,
    *,
    sample_rate_hz: int | None = None,
    channels: int | None = None,
    encoding: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "audio_end", "request_id": str(request_id)}
    if sample_rate_hz is not None:
        out["sample_rate_hz"] = sample_rate_hz
    if channels is not None:
        out["channels"] = channels
    if encoding is not None:
        out["encoding"] = encoding
    return out


def msg_event(name: str, *, request_id: uuid.UUID | None = None, message: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "event", "name": name}
    if request_id is not None:
        out["request_id"] = str(request_id)
    if message is not None:
        out["message"] = message
    return out
