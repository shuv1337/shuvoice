"""ASR backend registry, factory, and compatibility exports."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

from .asr_base import ASRBackend

if TYPE_CHECKING:
    from .config import Config

BackendResolver = Callable[[], type[ASRBackend]]


def _resolve_nemo_backend() -> type[ASRBackend]:
    from .asr_nemo import NemoBackend

    return NemoBackend


def _resolve_sherpa_backend() -> type[ASRBackend]:
    from .asr_sherpa import SherpaBackend

    return SherpaBackend


def _resolve_moonshine_backend() -> type[ASRBackend]:
    from .asr_moonshine import MoonshineBackend

    return MoonshineBackend


_BACKEND_REGISTRY: dict[str, BackendResolver] = {
    "nemo": _resolve_nemo_backend,
    "sherpa": _resolve_sherpa_backend,
    "moonshine": _resolve_moonshine_backend,
}


def get_backend_class(name: str) -> type[ASRBackend]:
    """Resolve an ASR backend class by registry name."""
    key = (name or "").strip().lower()
    resolver = _BACKEND_REGISTRY.get(key)
    if resolver is None:
        supported = ", ".join(sorted(_BACKEND_REGISTRY))
        raise ValueError(f"Unknown ASR backend '{name}'. Supported backends: {supported}")
    return resolver()


def create_backend(backend_name: str, config: Config) -> ASRBackend:
    """Create a configured ASR backend instance."""
    backend_cls = get_backend_class(backend_name)
    return backend_cls(config)


def __getattr__(name: str):
    if name != "ASREngine":
        raise AttributeError(name)

    warnings.warn(
        "`shuvoice.asr.ASREngine` is deprecated; use `get_backend_class('nemo')` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_backend_class("nemo")


__all__ = [
    "ASRBackend",
    "create_backend",
    "get_backend_class",
]
