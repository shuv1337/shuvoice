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

        Batch encoder-decoder backends (e.g. Moonshine) re-encode the
        full accumulated buffer on every inference call.  Per-chunk gain
        creates wildly inconsistent levels (40× on early noise, 5× on
        speech) which degrades quality.  These backends receive raw
        audio and apply their own uniform normalization at inference time.

        Streaming backends (NeMo, Sherpa) process chunks incrementally
        and benefit from the app's per-chunk gain, so they return False.
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
