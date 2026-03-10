from __future__ import annotations

import io
import json
import urllib.error

import pytest

from shuvoice.config import Config
from shuvoice.tts_base import TTSSynthesisRequest
from shuvoice.tts_openai import OpenAITTSBackend


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


def test_sample_rate_hz_uses_pcm_output_format():
    cfg = Config(tts_backend="openai", tts_output_format="pcm_24000")
    backend = OpenAITTSBackend(cfg)

    assert backend.sample_rate_hz() == 24000


def test_synthesize_stream_shapes_request(monkeypatch):
    cfg = Config(tts_backend="openai")
    backend = OpenAITTSBackend(cfg)

    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    seen: dict[str, object] = {}

    def fake_urlopen(request, timeout=0):
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["headers"] = dict(request.header_items())
        seen["body"] = request.data
        seen["timeout"] = timeout
        return _ChunkResponse([b"aa", b"bb", b""])

    monkeypatch.setattr("shuvoice.tts_openai.urllib.request.urlopen", fake_urlopen)

    chunks = list(
        backend.synthesize_stream(
            TTSSynthesisRequest(
                text="Hello world",
                voice_id="onyx",
                model_id="gpt-4o-mini-tts",
                playback_speed=1.3,
            )
        )
    )

    assert chunks == [b"aa", b"bb"]
    assert str(seen["url"]).endswith("/audio/speech")
    assert seen["method"] == "POST"

    headers = {k.lower(): v for k, v in seen["headers"].items()}
    assert headers["authorization"] == "Bearer secret"

    payload = json.loads(seen["body"].decode("utf-8"))
    assert payload["input"] == "Hello world"
    assert payload["voice"] == "onyx"
    assert payload["model"] == "gpt-4o-mini-tts"
    assert payload["response_format"] == "pcm"
    assert payload["speed"] == 1.3


def test_synthesize_stream_clamps_provider_native_speed(monkeypatch):
    cfg = Config(tts_backend="openai")
    backend = OpenAITTSBackend(cfg)

    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    seen: dict[str, object] = {}

    def fake_urlopen(request, timeout=0):
        seen["body"] = request.data
        return _ChunkResponse([b""])

    monkeypatch.setattr("shuvoice.tts_openai.urllib.request.urlopen", fake_urlopen)

    list(
        backend.synthesize_stream(
            TTSSynthesisRequest(
                text="Hello world",
                voice_id="onyx",
                model_id="gpt-4o-mini-tts",
                playback_speed=9.0,
            )
        )
    )

    payload = json.loads(seen["body"].decode("utf-8"))
    assert payload["speed"] == 4.0


def test_synthesize_stream_requires_env_var(monkeypatch):
    cfg = Config(tts_backend="openai", tts_api_key_env="MY_CUSTOM_KEY")
    backend = OpenAITTSBackend(cfg)

    monkeypatch.delenv("MY_CUSTOM_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MY_CUSTOM_KEY"):
        list(
            backend.synthesize_stream(
                TTSSynthesisRequest(
                    text="hello",
                    voice_id="alloy",
                    model_id="gpt-4o-mini-tts",
                    playback_speed=1.0,
                )
            )
        )


def test_synthesize_stream_http_error_classification(monkeypatch):
    cfg = Config(tts_backend="openai")
    backend = OpenAITTSBackend(cfg)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    def fake_urlopen(_request, timeout=0):
        raise urllib.error.HTTPError(
            url="https://api.openai.com/v1/audio/speech",
            code=429,
            msg="rate limited",
            hdrs=None,
            fp=io.BytesIO(b""),
        )

    monkeypatch.setattr("shuvoice.tts_openai.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="rate limit"):
        list(
            backend.synthesize_stream(
                TTSSynthesisRequest(
                    text="hello",
                    voice_id="alloy",
                    model_id="gpt-4o-mini-tts",
                    playback_speed=1.0,
                )
            )
        )


def test_list_voices_returns_known_openai_voices():
    cfg = Config(tts_backend="openai")
    backend = OpenAITTSBackend(cfg)

    voices = backend.list_voices()

    assert "alloy" in [voice.id for voice in voices]
    assert "nova" in [voice.id for voice in voices]
    assert "shimmer" in [voice.id for voice in voices]


def test_dependency_errors_checks_default_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    errors = OpenAITTSBackend.dependency_errors()
    assert errors

    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    assert OpenAITTSBackend.dependency_errors() == []


def test_openai_backend_rejects_non_pcm_output_format(monkeypatch):
    cfg = Config(tts_backend="openai", tts_output_format="mp3")
    backend = OpenAITTSBackend(cfg)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    with pytest.raises(ValueError, match="raw PCM output"):
        list(
            backend.synthesize_stream(
                TTSSynthesisRequest(
                    text="hello",
                    voice_id="alloy",
                    model_id="gpt-4o-mini-tts",
                    playback_speed=1.0,
                )
            )
        )
