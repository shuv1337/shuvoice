"""Shared ASR backend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ASRBackend(ABC):
    """Common runtime surface used by the ShuVoice app loop."""

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
        """If True the app must **not** apply per-chunk utterance gain.

        This is for backends that already normalize internally (for example,
        Moonshine's buffer-level normalization and NeMo's preprocessor
        feature normalization). Applying app-side per-chunk gain on top can
        introduce level discontinuities and hurt recognition.

        Backends that rely on app-side gain tuning (for example Sherpa)
        should keep the default ``False``.
        """
        return False

    @property
    def debug_step_num(self) -> int | None:
        """Optional backend step counter for diagnostics/logging."""
        return None

    @staticmethod
    @abstractmethod
    def dependency_errors() -> list[str]:
        """Return import/runtime dependency errors for this backend."""

    @classmethod
    def download_model(cls, **kwargs) -> None:
        """Optional model pre-download hook."""
        raise NotImplementedError(f"{cls.__name__} does not support model download")
