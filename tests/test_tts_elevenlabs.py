from __future__ import annotations

import io
import json
import urllib.error

import pytest

from shuvoice.config import Config
from shuvoice.tts_elevenlabs import ElevenLabsTTSBackend


class _ChunkResponse:
    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)
        self._index = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, _size: int = -1) -> bytes:
        if self._index >= len(self._chunks):
            return b""
        value = self._chunks[self._index]
        self._index += 1
        return value


def test_synthesize_stream_shapes_request(monkeypatch):
    cfg = Config(tts_enabled=True)
    backend = ElevenLabsTTSBackend(cfg)

    monkeypatch.setenv("ELEVENLABS_API_KEY", "secret")

    seen: dict[str, object] = {}

    def fake_urlopen(request, timeout=0):
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["headers"] = dict(request.header_items())
        seen["body"] = request.data
        seen["timeout"] = timeout
        return _ChunkResponse([b"aa", b"bb", b""])

    monkeypatch.setattr("shuvoice.tts_elevenlabs.urllib.request.urlopen", fake_urlopen)

    chunks = list(
        backend.synthesize_stream(
            "Hello world",
            voice_id="zNsotODqUhvbJ5wMG7Ei",
            model_id="eleven_multilingual_v2",
        )
    )

    assert chunks == [b"aa", b"bb"]
    assert "/text-to-speech/zNsotODqUhvbJ5wMG7Ei/stream" in str(seen["url"])
    assert "output_format=pcm_24000" in str(seen["url"])
    assert seen["method"] == "POST"
    assert "xi-api-key" in {k.lower() for k in seen["headers"]}

    payload = json.loads(seen["body"].decode("utf-8"))
    assert payload["text"] == "Hello world"
    assert payload["model_id"] == "eleven_multilingual_v2"


def test_synthesize_stream_requires_env_var(monkeypatch):
    cfg = Config(tts_api_key_env="MY_CUSTOM_KEY")
    backend = ElevenLabsTTSBackend(cfg)

    monkeypatch.delenv("MY_CUSTOM_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MY_CUSTOM_KEY"):
        list(backend.synthesize_stream("hello", voice_id="v", model_id="m"))


def test_synthesize_stream_http_error_classification(monkeypatch):
    cfg = Config()
    backend = ElevenLabsTTSBackend(cfg)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "secret")

    def fake_urlopen(_request, timeout=0):
        raise urllib.error.HTTPError(
            url="https://api.elevenlabs.io",
            code=429,
            msg="rate limited",
            hdrs=None,
            fp=io.BytesIO(b""),
        )

    monkeypatch.setattr("shuvoice.tts_elevenlabs.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="rate limit"):
        list(backend.synthesize_stream("hello", voice_id="v", model_id="m"))


def test_list_voices_parses_payload_and_caches(monkeypatch):
    cfg = Config()
    backend = ElevenLabsTTSBackend(cfg)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "secret")

    calls = {"count": 0}

    payload = {
        "voices": [
            {"voice_id": "v1", "name": "Voice One", "description": "desc"},
            {"voice_id": "v2", "name": "Voice Two"},
        ]
    }

    def fake_urlopen(_request, timeout=0):
        calls["count"] += 1
        return _ChunkResponse([json.dumps(payload).encode("utf-8")])

    monkeypatch.setattr("shuvoice.tts_elevenlabs.urllib.request.urlopen", fake_urlopen)

    voices_a = backend.list_voices()
    voices_b = backend.list_voices()

    assert calls["count"] == 1
    assert [voice.id for voice in voices_a] == ["v1", "v2"]
    assert [voice.name for voice in voices_b] == ["Voice One", "Voice Two"]


def test_dependency_errors_checks_default_env(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    errors = ElevenLabsTTSBackend.dependency_errors()
    assert errors

    monkeypatch.setenv("ELEVENLABS_API_KEY", "secret")
    assert ElevenLabsTTSBackend.dependency_errors() == []
