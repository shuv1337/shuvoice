from __future__ import annotations

import pytest

from shuvoice.config import Config
from shuvoice.tts import create_tts_backend, get_tts_backend_class
from shuvoice.tts_elevenlabs import ElevenLabsTTSBackend
from shuvoice.tts_kokoro import KokoroTTSBackend
from shuvoice.tts_local import LocalTTSBackend
from shuvoice.tts_melotts import MeloTTSBackend
from shuvoice.tts_openai import OpenAITTSBackend


def test_get_tts_backend_class_resolves_known_backends():
    assert get_tts_backend_class("elevenlabs") is ElevenLabsTTSBackend
    assert get_tts_backend_class("openai") is OpenAITTSBackend
    assert get_tts_backend_class("local") is LocalTTSBackend
    assert get_tts_backend_class("melotts") is MeloTTSBackend
    assert get_tts_backend_class("kokoro") is KokoroTTSBackend


def test_get_tts_backend_class_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown TTS backend"):
        get_tts_backend_class("unknown")


def test_create_tts_backend_uses_config_backend_name():
    cfg = Config(tts_backend="openai")
    backend = create_tts_backend(cfg)
    assert isinstance(backend, OpenAITTSBackend)


def test_create_tts_backend_melotts():
    cfg = Config(tts_backend="melotts")
    backend = create_tts_backend(cfg)
    assert isinstance(backend, MeloTTSBackend)


def test_create_tts_backend_kokoro():
    cfg = Config(tts_backend="kokoro")
    backend = create_tts_backend(cfg)
    assert isinstance(backend, KokoroTTSBackend)
