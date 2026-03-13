"""Tests for the MeloTTS backend module and helper protocol."""

from __future__ import annotations

import io
import json
import struct
import subprocess
from pathlib import Path

import pytest

from shuvoice.tts_base import TTSSynthesisRequest

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _pcm_frame(pcm_bytes: bytes) -> bytes:
    """Build a framing header (4-byte LE uint32 length) + raw PCM payload."""
    return struct.pack("<I", len(pcm_bytes)) + pcm_bytes


class _CapturingBytesIO(io.BytesIO):
    """BytesIO that preserves written data even after close."""

    def __init__(self):
        super().__init__()
        self._captured: bytes = b""

    def close(self):
        self._captured = self.getvalue()
        super().close()

    def getvalue(self) -> bytes:
        if self.closed:
            return self._captured
        return super().getvalue()


class _FakePopen:
    """Minimal subprocess.Popen stand-in for MeloTTS helper tests."""

    def __init__(
        self,
        command,
        *,
        stdout_data: bytes = b"",
        returncode: int = 0,
        stderr: bytes = b"",
        hang: bool = False,
    ):
        self.command = list(command)
        self.stdin = _CapturingBytesIO()
        self.stdout = io.BytesIO(stdout_data)
        self._returncode = returncode
        self.returncode = returncode
        self.stderr = io.BytesIO(stderr)
        self.killed = False
        self._hang = hang

    def wait(self, timeout=None):
        if self._hang:
            raise subprocess.TimeoutExpired(cmd=self.command, timeout=timeout or 0)
        self.returncode = self._returncode
        return self._returncode

    def kill(self):
        self.killed = True


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_capabilities_supports_streaming_false():
    from shuvoice.tts_melotts import MeloTTSBackend

    caps = MeloTTSBackend.capabilities
    assert caps.supports_streaming is False


def test_capabilities_supports_voice_list():
    from shuvoice.tts_melotts import MeloTTSBackend

    caps = MeloTTSBackend.capabilities
    assert caps.supports_voice_list is True


def test_capabilities_requires_no_api_key():
    from shuvoice.tts_melotts import MeloTTSBackend

    caps = MeloTTSBackend.capabilities
    assert caps.requires_api_key is False


def test_capabilities_speed_control_enabled():
    from shuvoice.tts_melotts import MeloTTSBackend

    caps = MeloTTSBackend.capabilities
    assert caps.supports_speed_control is True
    assert caps.speed_min == 0.5
    assert caps.speed_max == 2.0


# ---------------------------------------------------------------------------
# Sample rate
# ---------------------------------------------------------------------------


def test_sample_rate_hz_returns_44100(tmp_path: Path):
    from shuvoice.config import Config
    from shuvoice.tts_melotts import MeloTTSBackend

    venv_dir = tmp_path / "melotts-venv"
    venv_dir.mkdir()
    (venv_dir / "bin").mkdir()
    python_bin = venv_dir / "bin" / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)

    cfg = Config(tts_backend="melotts", tts_melotts_venv_path=str(venv_dir))
    backend = MeloTTSBackend(cfg)
    assert backend.sample_rate_hz() == 44100


# ---------------------------------------------------------------------------
# list_voices
# ---------------------------------------------------------------------------


def test_list_voices_returns_five_entries(tmp_path: Path):
    from shuvoice.config import Config
    from shuvoice.tts_melotts import MeloTTSBackend

    venv_dir = tmp_path / "melotts-venv"
    venv_dir.mkdir()
    (venv_dir / "bin").mkdir()
    python_bin = venv_dir / "bin" / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)

    cfg = Config(tts_backend="melotts", tts_melotts_venv_path=str(venv_dir))
    backend = MeloTTSBackend(cfg)
    voices = backend.list_voices()

    assert len(voices) == 5
    voice_ids = {v.id for v in voices}
    assert voice_ids == {"EN-US", "EN-BR", "EN-INDIA", "EN-AU", "EN-Newest"}


def test_list_voices_have_human_readable_names(tmp_path: Path):
    from shuvoice.config import Config
    from shuvoice.tts_melotts import MeloTTSBackend

    venv_dir = tmp_path / "melotts-venv"
    venv_dir.mkdir()
    (venv_dir / "bin").mkdir()
    python_bin = venv_dir / "bin" / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)

    cfg = Config(tts_backend="melotts", tts_melotts_venv_path=str(venv_dir))
    backend = MeloTTSBackend(cfg)
    voices = backend.list_voices()

    name_map = {v.id: v.name for v in voices}
    assert "American" in name_map["EN-US"]
    assert "British" in name_map["EN-BR"]
    assert "Indian" in name_map["EN-INDIA"]
    assert "Australian" in name_map["EN-AU"]
    assert "Newest" in name_map["EN-Newest"]


# ---------------------------------------------------------------------------
# dependency_errors
# ---------------------------------------------------------------------------


def test_dependency_errors_missing_venv(tmp_path: Path):
    from shuvoice.tts_melotts import MeloTTSBackend

    errors = MeloTTSBackend.dependency_errors(venv_path=str(tmp_path / "nonexistent"))
    assert len(errors) > 0
    assert any("venv" in e.lower() for e in errors)


def test_dependency_errors_missing_python_binary(tmp_path: Path):
    from shuvoice.tts_melotts import MeloTTSBackend

    venv_dir = tmp_path / "melotts-venv"
    venv_dir.mkdir()
    # No bin/python
    errors = MeloTTSBackend.dependency_errors(venv_path=str(venv_dir))
    assert len(errors) > 0
    assert any("python" in e.lower() for e in errors)


def test_dependency_errors_python_not_executable(tmp_path: Path):
    from shuvoice.tts_melotts import MeloTTSBackend

    venv_dir = tmp_path / "melotts-venv"
    venv_dir.mkdir()
    (venv_dir / "bin").mkdir()
    python_bin = venv_dir / "bin" / "python"
    python_bin.write_text("not executable")
    python_bin.chmod(0o644)

    errors = MeloTTSBackend.dependency_errors(venv_path=str(venv_dir))
    assert len(errors) > 0
    assert any("executable" in e.lower() for e in errors)


def test_dependency_errors_all_ok(tmp_path: Path):
    from shuvoice.tts_melotts import MeloTTSBackend

    venv_dir = tmp_path / "melotts-venv"
    venv_dir.mkdir()
    (venv_dir / "bin").mkdir()
    python_bin = venv_dir / "bin" / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)

    errors = MeloTTSBackend.dependency_errors(venv_path=str(venv_dir))
    assert errors == []


# ---------------------------------------------------------------------------
# synthesize_stream – happy path
# ---------------------------------------------------------------------------


def test_synthesize_stream_yields_pcm_chunks(monkeypatch, tmp_path: Path):
    from shuvoice.config import Config
    from shuvoice.tts_melotts import MeloTTSBackend

    venv_dir = tmp_path / "melotts-venv"
    venv_dir.mkdir()
    (venv_dir / "bin").mkdir()
    python_bin = venv_dir / "bin" / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)

    pcm_data = b"\x00\x01" * 100  # 200 bytes of PCM
    stdout_payload = _pcm_frame(pcm_data)

    seen: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        proc = _FakePopen(command, stdout_data=stdout_payload)
        seen["command"] = proc.command
        seen["proc"] = proc
        return proc

    monkeypatch.setattr("shuvoice.tts_melotts.subprocess.Popen", fake_popen)

    cfg = Config(tts_backend="melotts", tts_melotts_venv_path=str(venv_dir))
    backend = MeloTTSBackend(cfg)

    request = TTSSynthesisRequest(
        text="Hello world",
        voice_id="EN-US",
        model_id="melotts",
        playback_speed=1.0,
    )
    chunks = list(backend.synthesize_stream(request))

    # All chunks concatenated should equal our PCM data
    assert b"".join(chunks) == pcm_data


def test_synthesize_stream_sends_json_request(monkeypatch, tmp_path: Path):
    from shuvoice.config import Config
    from shuvoice.tts_melotts import MeloTTSBackend

    venv_dir = tmp_path / "melotts-venv"
    venv_dir.mkdir()
    (venv_dir / "bin").mkdir()
    python_bin = venv_dir / "bin" / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)

    pcm_data = b"\x00\x01" * 10
    stdout_payload = _pcm_frame(pcm_data)

    seen: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        proc = _FakePopen(command, stdout_data=stdout_payload)
        seen["proc"] = proc
        return proc

    monkeypatch.setattr("shuvoice.tts_melotts.subprocess.Popen", fake_popen)

    cfg = Config(tts_backend="melotts", tts_melotts_venv_path=str(venv_dir))
    backend = MeloTTSBackend(cfg)

    request = TTSSynthesisRequest(
        text="Hello world",
        voice_id="EN-BR",
        model_id="melotts",
        playback_speed=1.5,
    )
    list(backend.synthesize_stream(request))

    proc = seen["proc"]
    stdin_data = proc.stdin.getvalue()
    parsed = json.loads(stdin_data.decode("utf-8").strip())
    assert parsed["text"] == "Hello world"
    assert parsed["voice_id"] == "EN-BR"
    assert parsed["speed"] == 1.5


# ---------------------------------------------------------------------------
# synthesize_stream – speed forwarding
# ---------------------------------------------------------------------------


def test_speed_forwarded_to_helper(monkeypatch, tmp_path: Path):
    from shuvoice.config import Config
    from shuvoice.tts_melotts import MeloTTSBackend

    venv_dir = tmp_path / "melotts-venv"
    venv_dir.mkdir()
    (venv_dir / "bin").mkdir()
    python_bin = venv_dir / "bin" / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)

    pcm_data = b"\x00\x01" * 10
    stdout_payload = _pcm_frame(pcm_data)

    seen: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        proc = _FakePopen(command, stdout_data=stdout_payload)
        seen["proc"] = proc
        return proc

    monkeypatch.setattr("shuvoice.tts_melotts.subprocess.Popen", fake_popen)

    cfg = Config(tts_backend="melotts", tts_melotts_venv_path=str(venv_dir))
    backend = MeloTTSBackend(cfg)

    for speed in [0.5, 1.0, 1.5, 2.0]:
        request = TTSSynthesisRequest(
            text="Test",
            voice_id="EN-US",
            model_id="melotts",
            playback_speed=speed,
        )
        list(backend.synthesize_stream(request))

        proc = seen["proc"]
        parsed = json.loads(proc.stdin.getvalue().decode("utf-8").strip())
        assert parsed["speed"] == speed


# ---------------------------------------------------------------------------
# synthesize_stream – error handling
# ---------------------------------------------------------------------------


def test_subprocess_crash_raises_error(monkeypatch, tmp_path: Path):
    from shuvoice.config import Config
    from shuvoice.tts_melotts import MeloTTSBackend

    venv_dir = tmp_path / "melotts-venv"
    venv_dir.mkdir()
    (venv_dir / "bin").mkdir()
    python_bin = venv_dir / "bin" / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)

    def fake_popen(command, **kwargs):
        return _FakePopen(command, stdout_data=b"", returncode=1, stderr=b"crash boom")

    monkeypatch.setattr("shuvoice.tts_melotts.subprocess.Popen", fake_popen)

    cfg = Config(tts_backend="melotts", tts_melotts_venv_path=str(venv_dir))
    backend = MeloTTSBackend(cfg)

    request = TTSSynthesisRequest(
        text="Hello",
        voice_id="EN-US",
        model_id="melotts",
        playback_speed=1.0,
    )
    with pytest.raises(RuntimeError, match="crash boom"):
        list(backend.synthesize_stream(request))


def test_subprocess_timeout_kills_process(monkeypatch, tmp_path: Path):
    from shuvoice.config import Config
    from shuvoice.tts_melotts import MeloTTSBackend

    venv_dir = tmp_path / "melotts-venv"
    venv_dir.mkdir()
    (venv_dir / "bin").mkdir()
    python_bin = venv_dir / "bin" / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)

    seen: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        proc = _FakePopen(command, stdout_data=b"", hang=True)
        seen["proc"] = proc
        return proc

    monkeypatch.setattr("shuvoice.tts_melotts.subprocess.Popen", fake_popen)

    cfg = Config(tts_backend="melotts", tts_melotts_venv_path=str(venv_dir))
    backend = MeloTTSBackend(cfg)

    request = TTSSynthesisRequest(
        text="Hello",
        voice_id="EN-US",
        model_id="melotts",
        playback_speed=1.0,
    )
    with pytest.raises(RuntimeError, match="timed out"):
        list(backend.synthesize_stream(request))

    assert seen["proc"].killed is True


def test_subprocess_popen_failure_raises(monkeypatch, tmp_path: Path):
    from shuvoice.config import Config
    from shuvoice.tts_melotts import MeloTTSBackend

    venv_dir = tmp_path / "melotts-venv"
    venv_dir.mkdir()
    (venv_dir / "bin").mkdir()
    python_bin = venv_dir / "bin" / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)

    def fake_popen(command, **kwargs):
        raise OSError("No such file")

    monkeypatch.setattr("shuvoice.tts_melotts.subprocess.Popen", fake_popen)

    cfg = Config(tts_backend="melotts", tts_melotts_venv_path=str(venv_dir))
    backend = MeloTTSBackend(cfg)

    request = TTSSynthesisRequest(
        text="Hello",
        voice_id="EN-US",
        model_id="melotts",
        playback_speed=1.0,
    )
    with pytest.raises(RuntimeError, match="Failed to start"):
        list(backend.synthesize_stream(request))


def test_synthesize_stream_empty_text_raises(tmp_path: Path):
    from shuvoice.config import Config
    from shuvoice.tts_melotts import MeloTTSBackend

    venv_dir = tmp_path / "melotts-venv"
    venv_dir.mkdir()
    (venv_dir / "bin").mkdir()
    python_bin = venv_dir / "bin" / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)

    cfg = Config(tts_backend="melotts", tts_melotts_venv_path=str(venv_dir))
    backend = MeloTTSBackend(cfg)

    request = TTSSynthesisRequest(
        text="  ",
        voice_id="EN-US",
        model_id="melotts",
        playback_speed=1.0,
    )
    with pytest.raises(ValueError, match="empty"):
        list(backend.synthesize_stream(request))


def test_synthesize_stream_text_too_long_raises(tmp_path: Path):
    from shuvoice.config import Config
    from shuvoice.tts_melotts import MeloTTSBackend

    venv_dir = tmp_path / "melotts-venv"
    venv_dir.mkdir()
    (venv_dir / "bin").mkdir()
    python_bin = venv_dir / "bin" / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)

    cfg = Config(
        tts_backend="melotts",
        tts_melotts_venv_path=str(venv_dir),
        tts_max_chars=10,
    )
    backend = MeloTTSBackend(cfg)

    request = TTSSynthesisRequest(
        text="A" * 20,
        voice_id="EN-US",
        model_id="melotts",
        playback_speed=1.0,
    )
    with pytest.raises(ValueError, match="too long"):
        list(backend.synthesize_stream(request))


# ---------------------------------------------------------------------------
# synthesize_stream – truncated / malformed framing
# ---------------------------------------------------------------------------


def test_synthesize_stream_truncated_header_raises(monkeypatch, tmp_path: Path):
    """If stdout only has 2 bytes (incomplete 4-byte header), raise error."""
    from shuvoice.config import Config
    from shuvoice.tts_melotts import MeloTTSBackend

    venv_dir = tmp_path / "melotts-venv"
    venv_dir.mkdir()
    (venv_dir / "bin").mkdir()
    python_bin = venv_dir / "bin" / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)

    def fake_popen(command, **kwargs):
        return _FakePopen(command, stdout_data=b"\x00\x01")

    monkeypatch.setattr("shuvoice.tts_melotts.subprocess.Popen", fake_popen)

    cfg = Config(tts_backend="melotts", tts_melotts_venv_path=str(venv_dir))
    backend = MeloTTSBackend(cfg)

    request = TTSSynthesisRequest(
        text="Hello",
        voice_id="EN-US",
        model_id="melotts",
        playback_speed=1.0,
    )
    with pytest.raises(RuntimeError, match="(?i)incomplete|truncated|unexpected"):
        list(backend.synthesize_stream(request))


# ---------------------------------------------------------------------------
# Helper protocol: voice → model routing
# ---------------------------------------------------------------------------


def test_helper_voice_model_routing():
    """Verify the voice-to-model mapping used by the helper."""
    from shuvoice.melo_helper import _model_for_voice

    assert _model_for_voice("EN-US") == "EN_V2"
    assert _model_for_voice("EN-BR") == "EN_V2"
    assert _model_for_voice("EN-INDIA") == "EN_V2"
    assert _model_for_voice("EN-AU") == "EN_V2"
    assert _model_for_voice("EN-Newest") == "EN_NEWEST"


def test_helper_voice_model_routing_unknown():
    from shuvoice.melo_helper import _model_for_voice

    # Unknown voice should fall back to EN_V2
    assert _model_for_voice("UNKNOWN") == "EN_V2"


# ---------------------------------------------------------------------------
# Helper protocol: build_request_json
# ---------------------------------------------------------------------------


def test_helper_builds_correct_request_json():
    """Verify the JSON request format the backend sends to the helper."""
    from shuvoice.melo_helper import _build_request_json

    result = _build_request_json("Hello world", "EN-BR", 1.5)
    parsed = json.loads(result)
    assert parsed == {"text": "Hello world", "voice_id": "EN-BR", "speed": 1.5}
