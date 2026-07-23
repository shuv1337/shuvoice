"""Framing unit tests + golden byte compatibility."""

from __future__ import annotations

import io
import struct
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shuvoice_worker_proto.constants import (  # noqa: E402
    FRAME_KIND_JSON,
    FRAME_KIND_PCM_F32_LE,
    MAX_FRAME_LEN,
    PROTOCOL_VERSION,
)
from shuvoice_worker_proto.errors import ProtocolError  # noqa: E402
from shuvoice_worker_proto.framing import (  # noqa: E402
    Frame,
    decode_f32le_samples,
    decode_frame,
    read_frame,
)


GOLDEN = Path(__file__).resolve().parent / "golden"


class FramingTests(unittest.TestCase):
    def test_json_roundtrip(self) -> None:
        frame = Frame.json_bytes({"type": "hello", "protocol_version": PROTOCOL_VERSION})
        encoded = frame.encode()
        decoded, n = decode_frame(encoded)
        self.assertEqual(n, len(encoded))
        self.assertEqual(decoded.kind, FRAME_KIND_JSON)
        self.assertEqual(decoded.payload, frame.payload)

    def test_pcm_f32_roundtrip(self) -> None:
        rid = uuid.UUID(int=1)
        samples = [0.0, 1.0, -1.0]
        frame = Frame.pcm_f32le(rid, samples)
        decoded, _ = decode_frame(frame.encode())
        got_id, body = decoded.split_binary_payload()
        self.assertEqual(got_id, rid)
        self.assertEqual(decode_f32le_samples(body), samples)

    def test_rejects_oversize_before_body(self) -> None:
        buf = struct.pack(">I", MAX_FRAME_LEN + 1) + b"\x01"
        with self.assertRaises(ProtocolError) as ctx:
            decode_frame(buf)
        self.assertEqual(ctx.exception.code, "frame_too_large")

    def test_rejects_zero_length(self) -> None:
        with self.assertRaises(ProtocolError) as ctx:
            decode_frame(struct.pack(">I", 0))
        self.assertEqual(ctx.exception.code, "frame_too_small")

    def test_rejects_unknown_kind(self) -> None:
        buf = struct.pack(">I", 1) + bytes([0xFF])
        with self.assertRaises(ProtocolError) as ctx:
            decode_frame(buf)
        self.assertEqual(ctx.exception.code, "unsupported_frame_kind")

    def test_truncated_body(self) -> None:
        buf = struct.pack(">I", 10) + bytes([FRAME_KIND_JSON, ord("{")])
        with self.assertRaises(ProtocolError) as ctx:
            decode_frame(buf)
        self.assertEqual(ctx.exception.code, "truncated_frame")

    def test_read_frame_eof(self) -> None:
        with self.assertRaises(ProtocolError) as ctx:
            read_frame(io.BytesIO(b""))
        self.assertEqual(ctx.exception.code, "unexpected_eof")

    def test_binary_requires_request_id(self) -> None:
        buf = struct.pack(">I", 2) + bytes([FRAME_KIND_PCM_F32_LE, 0])
        with self.assertRaises(ProtocolError) as ctx:
            decode_frame(buf)
        self.assertEqual(ctx.exception.code, "invalid_binary")

    def test_golden_hello_v1(self) -> None:
        data = (GOLDEN / "hello_v1.bin").read_bytes()
        frame, n = decode_frame(data)
        self.assertEqual(n, len(data))
        self.assertEqual(frame.kind, FRAME_KIND_JSON)
        # Exact bytes must match recomputation (cross-lang lock).
        expected = Frame.json_bytes(
            {"type": "hello", "protocol_version": 1, "client_name": "golden"}
        ).encode()
        self.assertEqual(data, expected)

    def test_golden_pcm_f32(self) -> None:
        data = (GOLDEN / "pcm_f32_three_samples.bin").read_bytes()
        frame, n = decode_frame(data)
        self.assertEqual(n, len(data))
        rid, body = frame.split_binary_payload()
        self.assertEqual(rid, uuid.UUID(int=1))
        self.assertEqual(decode_f32le_samples(body), [0.0, 1.0, -1.0])
        expected = Frame.pcm_f32le(uuid.UUID(int=1), [0.0, 1.0, -1.0]).encode()
        self.assertEqual(data, expected)

    def test_json_payload_cap(self) -> None:
        from shuvoice_worker_proto.constants import MAX_JSON_PAYLOAD_LEN

        with self.assertRaises(ProtocolError) as ctx:
            Frame.json_bytes(b"y" * (MAX_JSON_PAYLOAD_LEN + 3))
        self.assertEqual(ctx.exception.code, "json_too_large")

    def test_golden_hello_ok_nemo_decodes(self) -> None:
        data = (GOLDEN / "hello_ok_nemo_v1.bin").read_bytes()
        frame, _ = decode_frame(data)
        import json

        msg = json.loads(frame.payload.decode("utf-8"))
        self.assertEqual(msg["type"], "hello_ok")
        self.assertEqual(msg["manifest"]["backend_id"], "nemo")
        self.assertTrue(msg["manifest"]["asr"]["wants_raw_audio"])


if __name__ == "__main__":
    unittest.main()
