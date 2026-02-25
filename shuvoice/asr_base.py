"""Shared ASR backend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config

import numpy as np


@dataclass(frozen=True)
class ASRCapabilities:
    supports_gpu: bool = False
    supports_model_download: bool = False
    wants_raw_audio: bool = False
    expected_chunking: str = "streaming"  # "streaming" | "windowed"


class ASRBackend(ABC):
    """Common runtime surface used by the ShuVoice app loop."""

    capabilities = ASRCapabilities()

    @abstractmethod
    def load(self) -> None:
        """Load backend model/runtime resources."""

    @abstractmethod
    def reset(self) -> None:
        """Reset backend state for a new utterance."""

    @abstractmethod
    def process_chunk(self, audio_chunk: np.ndarray) -> str:
        """Process one native chunk and return cumulative transcript text."""

    @property
    @abstractmethod
    def native_chunk_samples(self) -> int:
        """Backend-preferred chunk size in PCM samples."""

    @property
    def wants_raw_audio(self) -> bool:
        """Backward-compatible passthrough to ``capabilities.wants_raw_audio``."""
        return bool(self.capabilities.wants_raw_audio)

    @property
    def debug_step_num(self) -> int | None:
        """Optional backend step counter for diagnostics/logging."""
        return None

    @staticmethod
    @abstractmethod
    def dependency_errors() -> list[str]:
        """Return import/runtime dependency errors for this backend."""

    @classmethod
    def dependency_diagnostics(cls) -> dict[str, Any]:
        errors = cls.dependency_errors()
        return {
            "backend": cls.__name__,
            "ok": not errors,
            "errors": errors,
            "capabilities": asdict(cls.capabilities),
        }

    @classmethod
    def startup_errors(cls, _config: Config) -> list[str]:
        """Return startup-blocking compatibility errors for a config.

        This runs before backend load() and should avoid heavy operations.
        """
        return []

    @classmethod
    def startup_warnings(cls, _config: Config, *, apply_fixes: bool = False) -> list[str]:
        """Return non-blocking startup warnings for a config.

        If ``apply_fixes`` is true, implementations may apply safe runtime-only
        fallbacks (for example, downgrading a provider selection).
        """
        return []

    @classmethod
    def download_model(cls, **kwargs) -> None:
        """Optional model pre-download hook."""
        if cls.capabilities.supports_model_download:
            raise NotImplementedError(
                f"{cls.__name__} advertises model download support but does not implement it"
            )
        raise NotImplementedError(f"{cls.__name__} does not support model download")
