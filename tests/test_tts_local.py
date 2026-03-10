from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

from shuvoice.config import Config
from shuvoice.tts_base import TTSSynthesisRequest
from shuvoice.tts_local import LocalTTSBackend


class _FakePopen:
    def __init__(self, command, *, stdout_chunks=None, returncode=0, stderr=b""):
        self.command = list(command)
        self.stdin = io.BytesIO()
        self.stdout = _ChunkReader(stdout_chunks or [b"aa", b"bb", b""])
        self._returncode = returncode
        self.returncode = returncode
        self._stderr = stderr
        self.killed = False

    def communicate(self, timeout=None):
        return (b"", self._stderr)

    def kill(self):
        self.killed = True


class _ChunkReader:
    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)
        self._index = 0

    def read(self, _size: int = -1) -> bytes:
        if self._index >= len(self._chunks):
            return b""
        value = self._chunks[self._index]
        self._index += 1
        return value


def test_local_backend_shapes_piper_command_with_length_scale(monkeypatch, tmp_path: Path):
    model_file = tmp_path / "amy.onnx"
    model_file.write_bytes(b"model")

    cfg = Config(tts_backend="local", tts_local_model_path=str(model_file), tts_local_voice="amy")
    backend = LocalTTSBackend(cfg)

    seen: dict[str, object] = {}

    def fake_popen(command, stdin=None, stdout=None, stderr=None):
        proc = _FakePopen(command)
        seen["command"] = proc.command
        seen["proc"] = proc
        return proc

    monkeypatch.setattr("shuvoice.tts_local.subprocess.Popen", fake_popen)

    chunks = list(
        backend.synthesize_stream(
            TTSSynthesisRequest(
                text="hello",
                voice_id="amy",
                model_id="ignored",
                playback_speed=1.25,
            )
        )
    )

    assert chunks == [b"aa", b"bb"]
    assert seen["command"] == [
        "piper",
        "--model",
        str(model_file),
        "--output_raw",
        "--length-scale",
        "0.8000",
    ]


def test_local_backend_length_scale_mapping_is_inverse():
    assert LocalTTSBackend._length_scale_for_speed(1.0) == 1.0
    assert LocalTTSBackend._length_scale_for_speed(2.0) == 0.5
    assert LocalTTSBackend._length_scale_for_speed(0.5) == 2.0


def test_local_backend_sample_rate_uses_sidecar_metadata(tmp_path: Path):
    model_file = tmp_path / "amy.onnx"
    model_file.write_bytes(b"model")
    (tmp_path / "amy.onnx.json").write_text('{"audio": {"sample_rate": 24000}}')

    cfg = Config(tts_backend="local", tts_local_model_path=str(model_file))
    backend = LocalTTSBackend(cfg)

    assert backend.sample_rate_hz() == 24000


def test_local_backend_default_voice_resolves_first_model_in_directory(tmp_path: Path):
    (tmp_path / "alpha.onnx").write_bytes(b"model")
    (tmp_path / "beta.onnx").write_bytes(b"model")
    (tmp_path / "alpha.onnx.json").write_text('{"audio": {"sample_rate": 22050}}')

    cfg = Config(tts_backend="local", tts_local_model_path=str(tmp_path))
    backend = LocalTTSBackend(cfg)

    assert backend.sample_rate_hz() == 22050
    assert backend.list_voices()[0].id == "alpha"


def test_local_backend_requires_model_path():
    cfg = Config(tts_backend="local")

    with pytest.raises(RuntimeError, match="tts_local_model_path"):
        LocalTTSBackend(cfg)


def test_local_backend_reports_stderr_failures(monkeypatch, tmp_path: Path):
    model_file = tmp_path / "amy.onnx"
    model_file.write_bytes(b"model")

    cfg = Config(tts_backend="local", tts_local_model_path=str(model_file))
    backend = LocalTTSBackend(cfg)

    def fake_popen(command, stdin=None, stdout=None, stderr=None):
        return _FakePopen(command, stdout_chunks=[b""], returncode=1, stderr=b"boom")

    monkeypatch.setattr("shuvoice.tts_local.subprocess.Popen", fake_popen)

    with pytest.raises(RuntimeError, match="boom"):
        list(
            backend.synthesize_stream(
                TTSSynthesisRequest(
                    text="hello",
                    voice_id="amy",
                    model_id="ignored",
                    playback_speed=1.0,
                )
            )
        )


def test_local_backend_timeout_kills_process(monkeypatch, tmp_path: Path):
    model_file = tmp_path / "amy.onnx"
    model_file.write_bytes(b"model")

    cfg = Config(tts_backend="local", tts_local_model_path=str(model_file))
    backend = LocalTTSBackend(cfg)

    proc = _FakePopen(["piper"])

    def fake_communicate(timeout=None):
        raise subprocess.TimeoutExpired(cmd=proc.command, timeout=timeout or 0)

    proc.communicate = fake_communicate

    def fake_popen(command, stdin=None, stdout=None, stderr=None):
        proc.command = list(command)
        return proc

    monkeypatch.setattr("shuvoice.tts_local.subprocess.Popen", fake_popen)

    with pytest.raises(RuntimeError, match="timed out"):
        list(
            backend.synthesize_stream(
                TTSSynthesisRequest(
                    text="hello",
                    voice_id="amy",
                    model_id="ignored",
                    playback_speed=1.0,
                )
            )
        )

    assert proc.killed is True
