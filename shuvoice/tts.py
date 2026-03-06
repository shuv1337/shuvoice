"""TTS backend registry and factory."""

from __future__ import annotations

from collections.abc import Callable

from .tts_base import TTSBackend

BackendResolver = Callable[[], type[TTSBackend]]


def _resolve_elevenlabs() -> type[TTSBackend]:
    from .tts_elevenlabs import ElevenLabsTTSBackend

    return ElevenLabsTTSBackend


def _resolve_openai() -> type[TTSBackend]:
    from .tts_openai import OpenAITTSBackend

    return OpenAITTSBackend


def _resolve_local() -> type[TTSBackend]:
    from .tts_local import LocalTTSBackend

    return LocalTTSBackend


_TTS_BACKEND_REGISTRY: dict[str, BackendResolver] = {
    "elevenlabs": _resolve_elevenlabs,
    "openai": _resolve_openai,
    "local": _resolve_local,
}


def get_tts_backend_class(name: str) -> type[TTSBackend]:
    key = (name or "").strip().lower()
    resolver = _TTS_BACKEND_REGISTRY.get(key)
    if resolver is None:
        supported = ", ".join(sorted(_TTS_BACKEND_REGISTRY))
        raise ValueError(f"Unknown TTS backend '{name}'. Supported backends: {supported}")
    return resolver()


def create_tts_backend(config) -> TTSBackend:
    backend_cls = get_tts_backend_class(config.tts_backend)
    return backend_cls(config)


__all__ = [
    "TTSBackend",
    "create_tts_backend",
    "get_tts_backend_class",
]
