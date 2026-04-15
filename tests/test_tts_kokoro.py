"""Tests for the Kokoro TTS backend."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from shuvoice.config import Config
from shuvoice.tts_base import TTSSynthesisRequest
from shuvoice.tts_kokoro import KokoroTTSBackend


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


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_kokoro_backend_instantiates():
    cfg = Config(tts_backend="kokoro")
    backend = KokoroTTSBackend(cfg)
    assert backend is not None


# ---------------------------------------------------------------------------
# dependency_errors
# ---------------------------------------------------------------------------


def test_dependency_errors_returns_empty_list():
    errors = KokoroTTSBackend.dependency_errors()
    assert errors == []


# ---------------------------------------------------------------------------
# sample_rate_hz
# ---------------------------------------------------------------------------


def test_sample_rate_hz_pcm_24000():
    cfg = Config(tts_backend="kokoro", tts_output_format="pcm_24000")
    backend = KokoroTTSBackend(cfg)
    assert backend.sample_rate_hz() == 24000


def test_sample_rate_hz_pcm_defaults_to_24000():
    cfg = Config(tts_backend="kokoro", tts_output_format="pcm")
    backend = KokoroTTSBackend(cfg)
    assert backend.sample_rate_hz() == 24000


def test_sample_rate_hz_pcm_custom():
    cfg = Config(tts_backend="kokoro", tts_output_format="pcm_44100")
    backend = KokoroTTSBackend(cfg)
    assert backend.sample_rate_hz() == 44100


def test_sample_rate_hz_mp3_returns_default():
    cfg = Config(tts_backend="kokoro", tts_output_format="mp3")
    backend = KokoroTTSBackend(cfg)
    assert backend.sample_rate_hz() == 24000


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_capabilities_no_api_key_required():
    caps = KokoroTTSBackend.capabilities
    assert caps.requires_api_key is False


def test_capabilities_supports_streaming():
    caps = KokoroTTSBackend.capabilities
    assert caps.supports_streaming is True


def test_capabilities_supports_voice_list():
    caps = KokoroTTSBackend.capabilities
    assert caps.supports_voice_list is True


def test_capabilities_speed_control():
    caps = KokoroTTSBackend.capabilities
    assert caps.supports_speed_control is True
    assert caps.speed_min == 0.5
    assert caps.speed_max == 2.0


# ---------------------------------------------------------------------------
# list_voices
# ---------------------------------------------------------------------------


def test_list_voices_parses_response(monkeypatch):
    cfg = Config(tts_backend="kokoro")
    backend = KokoroTTSBackend(cfg)

    voice_data = {
        "voices": [
            {"id": "am_onyx", "name": "Onyx"},
            {"id": "af_heart", "name": "Heart", "description": "Warm voice"},
        ]
    }

    def fake_urlopen(request, timeout=0):
        return _ChunkResponse([json.dumps(voice_data).encode("utf-8")])

    monkeypatch.setattr("shuvoice.tts_kokoro.urllib.request.urlopen", fake_urlopen)

    voices = backend.list_voices()
    assert len(voices) == 2
    assert voices[0].id == "am_onyx"
    assert voices[0].name == "Onyx"
    assert voices[1].id == "af_heart"
    assert voices[1].name == "Heart"
    assert voices[1].description == "Warm voice"


def test_list_voices_caching(monkeypatch):
    cfg = Config(tts_backend="kokoro")
    backend = KokoroTTSBackend(cfg)

    voice_data = {"voices": [{"id": "af_heart", "name": "Heart"}]}
    call_count = 0

    def fake_urlopen(request, timeout=0):
        nonlocal call_count
        call_count += 1
        return _ChunkResponse([json.dumps(voice_data).encode("utf-8")])

    monkeypatch.setattr("shuvoice.tts_kokoro.urllib.request.urlopen", fake_urlopen)

    # First call fetches
    voices1 = backend.list_voices()
    assert call_count == 1

    # Second call uses cache
    voices2 = backend.list_voices()
    assert call_count == 1
    assert len(voices1) == len(voices2)


def test_list_voices_cache_expires(monkeypatch):
    cfg = Config(tts_backend="kokoro")
    backend = KokoroTTSBackend(cfg)

    voice_data = {"voices": [{"id": "af_heart", "name": "Heart"}]}
    call_count = 0

    def fake_urlopen(request, timeout=0):
        nonlocal call_count
        call_count += 1
        return _ChunkResponse([json.dumps(voice_data).encode("utf-8")])

    monkeypatch.setattr("shuvoice.tts_kokoro.urllib.request.urlopen", fake_urlopen)

    # First call
    backend.list_voices()
    assert call_count == 1

    # Expire the cache
    backend._voice_cache_expires_at = 0.0

    # Should fetch again
    backend.list_voices()
    assert call_count == 2


def test_list_voices_skips_empty_ids(monkeypatch):
    cfg = Config(tts_backend="kokoro")
    backend = KokoroTTSBackend(cfg)

    voice_data = {
        "voices": [
            {"id": "", "name": "Empty"},
            {"id": "af_heart", "name": "Heart"},
        ]
    }

    def fake_urlopen(request, timeout=0):
        return _ChunkResponse([json.dumps(voice_data).encode("utf-8")])

    monkeypatch.setattr("shuvoice.tts_kokoro.urllib.request.urlopen", fake_urlopen)

    voices = backend.list_voices()
    assert len(voices) == 1
    assert voices[0].id == "af_heart"


# ---------------------------------------------------------------------------
# synthesize_stream
# ---------------------------------------------------------------------------


def test_synthesize_stream_shapes_request(monkeypatch):
    cfg = Config(tts_backend="kokoro")
    backend = KokoroTTSBackend(cfg)

    seen: dict[str, object] = {}

    def fake_urlopen(request, timeout=0):
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["headers"] = dict(request.header_items())
        seen["body"] = request.data
        seen["timeout"] = timeout
        return _ChunkResponse([b"aa", b"bb", b""])

    monkeypatch.setattr("shuvoice.tts_kokoro.urllib.request.urlopen", fake_urlopen)

    chunks = list(
        backend.synthesize_stream(
            TTSSynthesisRequest(
                text="Hello world",
                voice_id="af_heart",
                model_id="kokoro",
                playback_speed=1.3,
            )
        )
    )

    assert chunks == [b"aa", b"bb"]
    assert str(seen["url"]).endswith("/audio/speech")
    assert seen["method"] == "POST"

    headers = {k.lower(): v for k, v in seen["headers"].items()}
    assert headers["authorization"] == "Bearer sk-local"

    payload = json.loads(seen["body"].decode("utf-8"))
    assert payload["input"] == "Hello world"
    assert payload["voice"] == "af_heart"
    assert payload["model"] == "kokoro"
    assert payload["response_format"] == "pcm"
    assert payload["speed"] == 1.3


def test_synthesize_stream_clamps_speed(monkeypatch):
    cfg = Config(tts_backend="kokoro")
    backend = KokoroTTSBackend(cfg)

    seen: dict[str, object] = {}

    def fake_urlopen(request, timeout=0):
        seen["body"] = request.data
        return _ChunkResponse([b""])

    monkeypatch.setattr("shuvoice.tts_kokoro.urllib.request.urlopen", fake_urlopen)

    list(
        backend.synthesize_stream(
            TTSSynthesisRequest(
                text="Hello world",
                voice_id="af_heart",
                model_id="kokoro",
                playback_speed=9.0,
            )
        )
    )

    payload = json.loads(seen["body"].decode("utf-8"))
    assert payload["speed"] == 2.0


def test_synthesize_stream_http_error(monkeypatch):
    cfg = Config(tts_backend="kokoro")
    backend = KokoroTTSBackend(cfg)

    def fake_urlopen(_request, timeout=0):
        raise urllib.error.HTTPError(
            url="http://localhost:8880/v1/audio/speech",
            code=500,
            msg="server error",
            hdrs=None,
            fp=io.BytesIO(b""),
        )

    monkeypatch.setattr("shuvoice.tts_kokoro.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="server error"):
        list(
            backend.synthesize_stream(
                TTSSynthesisRequest(
                    text="hello",
                    voice_id="af_heart",
                    model_id="kokoro",
                    playback_speed=1.0,
                )
            )
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_synthesize_stream_empty_text_raises():
    cfg = Config(tts_backend="kokoro")
    backend = KokoroTTSBackend(cfg)

    with pytest.raises(ValueError, match="empty"):
        list(
            backend.synthesize_stream(
                TTSSynthesisRequest(
                    text="   ",
                    voice_id="af_heart",
                    model_id="kokoro",
                    playback_speed=1.0,
                )
            )
        )


def test_synthesize_stream_text_too_long_raises():
    cfg = Config(tts_backend="kokoro", tts_max_chars=10)
    backend = KokoroTTSBackend(cfg)

    with pytest.raises(ValueError, match="too long"):
        list(
            backend.synthesize_stream(
                TTSSynthesisRequest(
                    text="A" * 20,
                    voice_id="af_heart",
                    model_id="kokoro",
                    playback_speed=1.0,
                )
            )
        )


# ---------------------------------------------------------------------------
# response_format
# ---------------------------------------------------------------------------


def test_response_format_rejects_unsupported(monkeypatch):
    cfg = Config(tts_backend="kokoro", tts_output_format="ogg")
    backend = KokoroTTSBackend(cfg)

    def fake_urlopen(request, timeout=0):
        return _ChunkResponse([b""])

    monkeypatch.setattr("shuvoice.tts_kokoro.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="supported output format"):
        list(
            backend.synthesize_stream(
                TTSSynthesisRequest(
                    text="hello",
                    voice_id="af_heart",
                    model_id="kokoro",
                    playback_speed=1.0,
                )
            )
        )


def test_response_format_mp3_accepted(monkeypatch):
    cfg = Config(tts_backend="kokoro", tts_output_format="mp3")
    backend = KokoroTTSBackend(cfg)

    seen: dict[str, object] = {}

    def fake_urlopen(request, timeout=0):
        seen["body"] = request.data
        return _ChunkResponse([b"mp3data"])

    monkeypatch.setattr("shuvoice.tts_kokoro.urllib.request.urlopen", fake_urlopen)

    chunks = list(
        backend.synthesize_stream(
            TTSSynthesisRequest(
                text="hello",
                voice_id="af_heart",
                model_id="kokoro",
                playback_speed=1.0,
            )
        )
    )

    payload = json.loads(seen["body"].decode("utf-8"))
    assert payload["response_format"] == "mp3"
    assert chunks == [b"mp3data"]
