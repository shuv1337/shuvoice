"""End-to-end worker tests using fake engines (no NeMo/Melo packages)."""

from __future__ import annotations

import os
import select
import struct
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shuvoice_worker_proto.client import WorkerClient  # noqa: E402
from shuvoice_worker_proto.constants import MAX_JSON_PAYLOAD_LEN  # noqa: E402
from shuvoice_worker_proto.errors import ProtocolError  # noqa: E402
from shuvoice_worker_proto.framing import Frame, read_frame, write_frame  # noqa: E402
from shuvoice_worker_proto.messages import parse_control_message  # noqa: E402


def _run_worker(module: str, extra_args: list[str] | None = None) -> subprocess.Popen:
    args = [sys.executable, "-m", module, "--fake"]
    if extra_args:
        args.extend(extra_args)
    return subprocess.Popen(
        args,
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"},
    )


def _cleanup(proc: subprocess.Popen) -> None:
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None and not getattr(stream, "closed", False):
            try:
                stream.close()
            except Exception:
                pass
    if proc.poll() is None:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass


class NemoWorkerFakeTests(unittest.TestCase):
    def test_load_chunk_finish_close(self) -> None:
        proc = _run_worker("nemo_asr")
        assert proc.stdin and proc.stdout
        try:
            client = WorkerClient(proc.stdout, proc.stdin)
            manifest = client.handshake("unit")
            self.assertEqual(manifest.get("backend_id"), "nemo")
            self.assertEqual(manifest.get("kind"), "asr")
            self.assertTrue(manifest["asr"]["wants_raw_audio"])

            ack = client.load({"right_context": 0, "device": "cpu", "model_name": "fake"})
            self.assertEqual(ack["type"], "ack")
            self.assertEqual(ack["result"]["native_chunk_samples"], 1280)

            partial = client.process_chunk_f32([0.01] * 1280)
            self.assertEqual(partial["type"], "partial_transcript")
            self.assertEqual(partial["text"], "step-1")

            final = client.finish()
            self.assertEqual(final["type"], "final_transcript")
            self.assertEqual(final["text"], "step-1")

            client.close()
            self.assertEqual(proc.wait(timeout=5), 0)
        finally:
            _cleanup(proc)

    def test_cancel_between_meta_and_pcm(self) -> None:
        proc = _run_worker("nemo_asr")
        assert proc.stdin and proc.stdout
        try:
            client = WorkerClient(proc.stdout, proc.stdin)
            client.handshake("unit")
            client.load({"right_context": 0, "device": "cpu"})
            rid = uuid.uuid4()
            write_frame(
                proc.stdin,
                Frame.json_bytes(
                    {
                        "type": "process_chunk",
                        "request_id": str(rid),
                        "sample_rate_hz": 16000,
                        "channels": 1,
                        "encoding": "f32_le",
                        "end": True,
                    }
                ),
            )
            write_frame(
                proc.stdin,
                Frame.json_bytes({"type": "cancel", "request_id": str(rid)}),
            )
            # Expect cancel ack then cancelled error (order: cancel path acks then raises).
            msgs = []
            for _ in range(3):
                ready, _, _ = select.select([proc.stdout], [], [], 2.0)
                self.assertTrue(ready, "worker did not respond to cancel interleave")
                msg = parse_control_message(read_frame(proc.stdout).payload)
                msgs.append(msg)
                if msg.get("type") == "error":
                    break
            types = [m.get("type") for m in msgs]
            self.assertIn("ack", types)
            err = next(m for m in msgs if m.get("type") == "error")
            self.assertEqual(err.get("code"), "cancelled")
            # Stream still usable for a subsequent request.
            partial = client.process_chunk_f32([0.02] * 1280)
            self.assertEqual(partial["type"], "partial_transcript")
            client.close()
            self.assertEqual(proc.wait(timeout=5), 0)
        finally:
            _cleanup(proc)

    def test_encoding_mismatch_rejected(self) -> None:
        proc = _run_worker("nemo_asr")
        assert proc.stdin and proc.stdout
        try:
            client = WorkerClient(proc.stdout, proc.stdin)
            client.handshake("unit")
            client.load({"right_context": 0})
            rid = uuid.uuid4()
            write_frame(
                proc.stdin,
                Frame.json_bytes(
                    {
                        "type": "process_chunk",
                        "request_id": str(rid),
                        "sample_rate_hz": 16000,
                        "channels": 1,
                        "encoding": "f32_le",
                        "end": True,
                    }
                ),
            )
            write_frame(proc.stdin, Frame.pcm_i16le(rid, [1000, 2000, 3000]))
            ready, _, _ = select.select([proc.stdout], [], [], 2.0)
            self.assertTrue(ready)
            msg = parse_control_message(read_frame(proc.stdout).payload)
            self.assertEqual(msg.get("type"), "error")
            self.assertEqual(msg.get("code"), "encoding_mismatch")
            client.close()
        finally:
            _cleanup(proc)


class MeloWorkerFakeTests(unittest.TestCase):
    def test_list_voices_and_synthesize(self) -> None:
        proc = _run_worker("melotts")
        assert proc.stdin and proc.stdout
        try:
            client = WorkerClient(proc.stdout, proc.stdin)
            manifest = client.handshake("unit")
            self.assertEqual(manifest.get("backend_id"), "melotts")
            self.assertEqual(manifest.get("kind"), "tts")
            self.assertFalse(manifest["tts"].get("supports_streaming_audio"))

            voices = client.list_voices()
            self.assertEqual(voices["type"], "voices")
            ids = {v["id"] for v in voices["voices"]}
            self.assertIn("EN-US", ids)

            meta, pcm = client.synthesize("hi", voice_id="EN-US", speed=1.0)
            self.assertEqual(meta.get("sample_rate_hz"), 44100)
            self.assertGreater(len(pcm), 0)
            # Default client encoding is f32_le.
            self.assertEqual(len(pcm) % 4, 0)
            self.assertEqual(meta.get("encoding"), "f32_le")

            client.close()
            self.assertEqual(proc.wait(timeout=5), 0)
        finally:
            _cleanup(proc)


class MeloLoadDepsTests(unittest.TestCase):
    def test_real_load_fails_when_venv_missing(self) -> None:
        # No --fake: dependency probe should fail load with dependency_missing.
        proc = subprocess.Popen(
            [sys.executable, "-m", "melotts"],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "PYTHONUNBUFFERED": "1",
                "SHUVOICE_MELOTTS_VENV": "/tmp/shuvoice-melotts-venv-does-not-exist",
            },
        )
        assert proc.stdin and proc.stdout
        try:
            client = WorkerClient(proc.stdout, proc.stdin)
            client.handshake("deps")
            with self.assertRaises(ProtocolError) as ctx:
                client.load({})
            self.assertEqual(ctx.exception.code, "dependency_missing")
            client.close()
        finally:
            _cleanup(proc)


class MoonshineWorkerFakeTests(unittest.TestCase):
    def test_utterance(self) -> None:
        proc = _run_worker("moonshine_asr")
        assert proc.stdin and proc.stdout
        try:
            client = WorkerClient(proc.stdout, proc.stdin)
            manifest = client.handshake("unit")
            self.assertEqual(manifest.get("backend_id"), "moonshine")
            ack = client.load({"model_name": "moonshine/tiny", "max_window_sec": 0.1})
            self.assertEqual(ack["type"], "ack")
            self.assertAlmostEqual(float(ack["result"]["max_window_sec"]), 0.1)
            final = client.process_utterance_f32([0.0] * 1600)
            self.assertEqual(final["type"], "final_transcript")
            self.assertTrue(str(final["text"]).startswith("moon-samples-"))
            # Window 0.1s @ 16k = 1600 samples max.
            self.assertEqual(final["text"], "moon-samples-1600")
            client.close()
            self.assertEqual(proc.wait(timeout=5), 0)
        finally:
            _cleanup(proc)


class HandshakeVersionTests(unittest.TestCase):
    def test_bad_version_rejected(self) -> None:
        proc = _run_worker("nemo_asr")
        assert proc.stdin and proc.stdout
        try:
            write_frame(
                proc.stdin,
                Frame.json_bytes({"type": "hello", "protocol_version": 0, "client_name": "old"}),
            )
            reply = parse_control_message(read_frame(proc.stdout).payload)
            self.assertEqual(reply["type"], "hello_err")
            self.assertEqual(reply.get("code"), "unsupported_version")
            self.assertEqual(reply.get("protocol_version"), 0)
            proc.wait(timeout=5)
        finally:
            _cleanup(proc)


class FramingCapTests(unittest.TestCase):
    def test_json_over_max_rejected(self) -> None:
        with self.assertRaises(ProtocolError) as ctx:
            Frame.json_bytes(b"x" * (MAX_JSON_PAYLOAD_LEN + 1))
        self.assertEqual(ctx.exception.code, "json_too_large")


if __name__ == "__main__":
    unittest.main()
