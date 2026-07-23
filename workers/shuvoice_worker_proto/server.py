"""Generic worker server loop over framed stdio."""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Callable
from typing import Any, BinaryIO, Protocol

from .constants import (
    FRAME_KIND_JSON,
    FRAME_KIND_PCM_F32_LE,
    FRAME_KIND_PCM_I16_LE,
    PROTOCOL_VERSION,
)
from .errors import ProtocolError
from .framing import Frame, decode_f32le_samples, decode_i16le_samples
from .io import FramedStdio
from .messages import (
    msg_ack,
    msg_error,
    msg_hello_err,
    msg_hello_ok,
    parse_control_message,
    parse_request_id,
    require_request_id,
)

log = logging.getLogger(__name__)


class WorkerHandler(Protocol):
    """Backend-specific hooks. Heavy imports stay inside the handler."""

    def manifest(self) -> dict[str, Any]:
        """Return handshake manifest (without wrapping hello_ok)."""

    def handle(self, msg_type: str, msg: dict[str, Any], server: WorkerServer) -> None:
        """Handle one control message after handshake."""


class WorkerServer:
    """Deterministic request loop: hello → commands until close/EOF.

    Pending PCM reads demux cancel/close for the same request id so a cancel
    between process_* meta and the binary frame does not desync the stream.
    """

    def __init__(
        self,
        handler: WorkerHandler,
        *,
        reader: BinaryIO | None = None,
        writer: BinaryIO | None = None,
    ) -> None:
        self.handler = handler
        self.io = FramedStdio(reader=reader, writer=writer)
        self._closed = False
        self._handshaken = False
        # Pending audio for process_chunk / process_utterance (request_id → meta)
        self._expect_audio: dict[uuid.UUID, dict[str, Any]] = {}
        self._cancel_flags: set[uuid.UUID] = set()

    # ── public emit helpers ────────────────────────────────────────────

    def send(self, msg: dict[str, Any]) -> None:
        self.io.write_json(msg)

    def send_frame(self, frame: Frame) -> None:
        self.io.write_frame(frame)

    def send_ack(self, request_id: uuid.UUID, result: Any | None = None) -> None:
        self.send(msg_ack(request_id, result=result))

    def send_error(
        self,
        *,
        code: str,
        message: str,
        request_id: uuid.UUID | None = None,
    ) -> None:
        # Never log or echo secrets/transcripts here.
        self.send(msg_error(code=code, message=message, request_id=request_id))

    def is_cancelled(self, request_id: uuid.UUID) -> bool:
        return request_id in self._cancel_flags

    def clear_cancel(self, request_id: uuid.UUID) -> None:
        self._cancel_flags.discard(request_id)

    def expect_audio(self, request_id: uuid.UUID, meta: dict[str, Any]) -> None:
        self._expect_audio[request_id] = meta

    def take_expected_audio(self, request_id: uuid.UUID) -> dict[str, Any] | None:
        return self._expect_audio.pop(request_id, None)

    def read_pcm_for_request(
        self, request_id: uuid.UUID
    ) -> tuple[str, list[float] | list[int] | bytes]:
        """Read the next binary frame for *request_id*, demuxing cancel/close.

        Returns ``(encoding, samples_or_bytes)`` where encoding is
        ``f32_le``, ``i16_le``, or ``bytes``.

        Same-request ``cancel`` raises ``ProtocolError(cancelled)`` after acking.
        ``close`` is honored (ack + mark closed) and raises ``cancelled``.
        Unrelated cancel messages are applied without consuming the PCM wait.
        """
        while not self._closed:
            frame = self.io.read_frame()
            if frame.kind == FRAME_KIND_JSON:
                try:
                    msg = parse_control_message(frame.payload)
                except Exception as exc:  # noqa: BLE001
                    raise ProtocolError("invalid_json", "invalid JSON control payload") from exc
                msg_type = str(msg.get("type", ""))
                if msg_type == "cancel":
                    try:
                        rid = require_request_id(msg)
                    except ValueError as exc:
                        self.send_error(
                            code="missing_request_id", message="cancel requires request_id"
                        )
                        continue
                    self._apply_cancel(rid)
                    if rid == request_id:
                        raise ProtocolError("cancelled", "request cancelled")
                    # Unrelated cancel — keep waiting for PCM.
                    continue
                if msg_type == "close":
                    self._handle_close(msg)
                    raise ProtocolError("cancelled", "worker closing")
                raise ProtocolError(
                    "unexpected_message",
                    "expected PCM frame after process_* meta",
                )

            rid, body = frame.split_binary_payload()
            if rid != request_id:
                raise ProtocolError("request_id_mismatch", "binary frame request_id mismatch")
            if frame.kind == FRAME_KIND_PCM_F32_LE:
                return "f32_le", decode_f32le_samples(body)
            if frame.kind == FRAME_KIND_PCM_I16_LE:
                return "i16_le", decode_i16le_samples(body)
            return "bytes", body

        raise ProtocolError("cancelled", "worker closed")

    def validate_audio_meta(
        self,
        msg: dict[str, Any],
        *,
        expected_sample_rate_hz: int | None = None,
        frame_encoding: str,
    ) -> None:
        """Enforce channels/encoding/sample_rate from process_* meta vs frame."""
        channels = int(msg.get("channels") or 1)
        if channels != 1:
            raise ProtocolError("unsupported_channels", "only mono (channels=1) audio is supported")
        meta_enc = str(msg.get("encoding") or "f32_le")
        if meta_enc not in {"f32_le", "i16_le"}:
            raise ProtocolError("unsupported_encoding", "expected pcm f32_le or i16_le")
        if meta_enc != frame_encoding:
            raise ProtocolError(
                "encoding_mismatch",
                "process_* encoding does not match binary frame kind",
            )
        if expected_sample_rate_hz is not None:
            sr = int(msg.get("sample_rate_hz") or 0)
            if sr != int(expected_sample_rate_hz):
                raise ProtocolError(
                    "sample_rate_mismatch",
                    f"expected sample_rate_hz={expected_sample_rate_hz}",
                )

    # ── main loop ──────────────────────────────────────────────────────

    def run(self) -> int:
        """Run until close or clean EOF. Returns process exit code."""
        try:
            while not self._closed:
                try:
                    frame = self.io.read_frame()
                except ProtocolError as exc:
                    if exc.code == "unexpected_eof":
                        return 0
                    log.error("framing error: %s", exc.code)
                    # Cannot reliably write if stream is broken mid-frame.
                    return 2

                if not self._handshaken:
                    self._handle_handshake_frame(frame)
                    continue

                if frame.kind != FRAME_KIND_JSON:
                    # Unexpected binary without prior meta — reject without echoing payload.
                    self.send_error(
                        code="unexpected_binary", message="binary frame without pending request"
                    )
                    continue

                try:
                    msg = parse_control_message(frame.payload)
                except Exception:
                    self.send_error(code="invalid_json", message="invalid JSON control payload")
                    continue

                msg_type = str(msg.get("type", ""))
                if msg_type == "close":
                    self._handle_close(msg)
                    break
                if msg_type == "cancel":
                    self._handle_cancel(msg)
                    continue
                if msg_type == "hello":
                    # Already handshaken — ignore or error.
                    self.send_error(code="already_handshaken", message="hello after handshake")
                    continue

                try:
                    self.handler.handle(msg_type, msg, self)
                except ProtocolError as exc:
                    rid = parse_request_id(msg.get("request_id"))
                    self.send_error(code=exc.code, message=exc.message, request_id=rid)
                except Exception as exc:  # noqa: BLE001 — boundary
                    rid = parse_request_id(msg.get("request_id"))
                    log.exception("handler failure type=%s", msg_type)
                    # Do not include exception text if it might hold user content;
                    # use a generic message. Engine code should raise ProtocolError.
                    self.send_error(
                        code="engine_error",
                        message=f"handler failed for {msg_type}",
                        request_id=rid,
                    )
                    _ = exc
        finally:
            self._closed = True
        return 0

    def _handle_handshake_frame(self, frame: Frame) -> None:
        if frame.kind != FRAME_KIND_JSON:
            self.io.write_json(
                msg_hello_err(message="expected hello JSON frame", code="invalid_handshake")
            )
            self._closed = True
            return
        try:
            msg = parse_control_message(frame.payload)
        except Exception:
            self.io.write_json(
                msg_hello_err(message="invalid hello JSON", code="invalid_handshake")
            )
            self._closed = True
            return
        if msg.get("type") != "hello":
            self.io.write_json(
                msg_hello_err(message="expected type=hello", code="invalid_handshake")
            )
            self._closed = True
            return
        try:
            remote_version = int(msg.get("protocol_version", -1))
        except (TypeError, ValueError):
            remote_version = -1
        if remote_version != PROTOCOL_VERSION:
            self.io.write_json(
                msg_hello_err(
                    message=(
                        f"unsupported protocol version {remote_version} "
                        f"(server supports {PROTOCOL_VERSION})"
                    ),
                    code="unsupported_version",
                    protocol_version=remote_version if remote_version >= 0 else None,
                )
            )
            self._closed = True
            return
        try:
            manifest = self.handler.manifest()
        except Exception:
            log.exception("manifest() failed")
            self.io.write_json(
                msg_hello_err(message="failed to build worker manifest", code="manifest_error")
            )
            self._closed = True
            return
        self.io.write_json(msg_hello_ok(protocol_version=PROTOCOL_VERSION, manifest=manifest))
        self._handshaken = True

    def _handle_close(self, msg: dict[str, Any]) -> None:
        rid = parse_request_id(msg.get("request_id"))
        if rid is not None:
            self.send_ack(rid)
        self._closed = True

    def _apply_cancel(self, rid: uuid.UUID) -> None:
        self._cancel_flags.add(rid)
        self._expect_audio.pop(rid, None)
        try:
            cancel = getattr(self.handler, "on_cancel", None)
            if callable(cancel):
                cancel(rid, self)
        except Exception:
            log.exception("on_cancel failed")
        self.send_ack(rid)

    def _handle_cancel(self, msg: dict[str, Any]) -> None:
        try:
            rid = require_request_id(msg)
        except ValueError:
            self.send_error(code="missing_request_id", message="cancel requires request_id")
            return
        self._apply_cancel(rid)


def configure_logging(*, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def run_worker(handler_factory: Callable[[], WorkerHandler], *, argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    verbose = "-v" in argv or "--verbose" in argv
    configure_logging(verbose=verbose)
    handler = handler_factory()
    return WorkerServer(handler).run()
