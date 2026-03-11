from __future__ import annotations

import io
from pathlib import Path

from shuvoice.piper_setup import (
    ensure_local_piper_ready,
    find_piper_binary,
    get_curated_piper_voice,
    piper_install_commands,
    validate_piper_voice_artifacts,
)


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._stream = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_find_piper_binary_prefers_upstream_name(monkeypatch):
    monkeypatch.setattr(
        "shuvoice.piper_setup.shutil.which",
        lambda name: "/usr/bin/piper" if name == "piper" else "/usr/bin/piper-tts",
    )

    assert find_piper_binary() == "piper"


def test_piper_install_commands_target_piper_tts_package():
    commands = piper_install_commands()
    assert ["yay", "-S", "--needed", "piper-tts"] in commands
    assert ["paru", "-S", "--needed", "piper-tts"] in commands


def test_validate_piper_voice_artifacts_requires_sidecar(tmp_path: Path):
    model_file = tmp_path / "en_US-amy-medium.onnx"
    model_file.write_bytes(b"model")

    valid, detail = validate_piper_voice_artifacts(tmp_path, "en_US-amy-medium")

    assert valid is False
    assert "sidecar" in detail.lower()


def test_validate_piper_voice_artifacts_accepts_valid_voice_dir(tmp_path: Path):
    model_file = tmp_path / "en_US-amy-medium.onnx"
    sidecar = tmp_path / "en_US-amy-medium.onnx.json"
    model_file.write_bytes(b"model")
    sidecar.write_text('{"audio": {"sample_rate": 22050}}')

    valid, detail = validate_piper_voice_artifacts(tmp_path, "en_US-amy-medium")

    assert valid is True
    assert "22050" in detail


def test_ensure_local_piper_ready_returns_missing_deps_when_binary_absent(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("shuvoice.piper_setup.find_piper_binary", lambda: None)

    result = ensure_local_piper_ready(
        get_curated_piper_voice("en_US-amy-medium"),
        model_dir=tmp_path,
        auto_install_missing=False,
    )

    assert result.status == "skipped_missing_deps"
    assert "missing" in result.message.lower()


def test_ensure_local_piper_ready_downloads_model_and_sidecar(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("shuvoice.piper_setup.find_piper_binary", lambda: "piper-tts")

    def fake_urlopen(request, timeout=0):
        url = getattr(request, "full_url", str(request))
        if url.endswith(".onnx?download=true"):
            return _FakeResponse(b"model-bytes")
        return _FakeResponse(b'{"audio": {"sample_rate": 22050}}')

    monkeypatch.setattr("shuvoice.piper_setup.urllib.request.urlopen", fake_urlopen)

    result = ensure_local_piper_ready(
        get_curated_piper_voice("en_US-amy-medium"),
        model_dir=tmp_path,
        auto_install_missing=False,
    )

    assert result.status == "downloaded"
    assert result.binary_name == "piper-tts"
    assert (tmp_path / "en_US-amy-medium.onnx").exists()
    assert (tmp_path / "en_US-amy-medium.onnx.json").exists()
    assert result.sample_rate_hz == 22050
