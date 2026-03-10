"""Shared TTS backend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from .tts_speed import TTS_PLAYBACK_SPEED_MAX, TTS_PLAYBACK_SPEED_MIN

if TYPE_CHECKING:
    from .config import Config


DEFAULT_LOCAL_TTS_VOICE_ID = "default"
DEFAULT_LOCAL_TTS_MODEL_ID = "piper"
LOCAL_TTS_AUTO_VOICE_IDS = frozenset({"default", "auto"})


@dataclass(frozen=True)
class VoiceInfo:
    id: str
    name: str
    description: str = ""


@dataclass(frozen=True)
class TTSSynthesisRequest:
    """Immutable per-utterance synthesis request.

    Speed is captured here so the backend receives the exact speed chosen when
    the utterance started. Playback must not rewrite PCM timing after synthesis.
    """

    text: str
    voice_id: str
    model_id: str
    playback_speed: float


@dataclass(frozen=True)
class TTSCapabilities:
    supports_streaming: bool = True
    supports_voice_list: bool = True
    requires_api_key: bool = False
    supports_speed_control: bool = False
    speed_min: float | None = None
    speed_max: float | None = None

    def speed_bounds(self) -> tuple[float, float] | None:
        if not self.supports_speed_control:
            return None

        minimum = (
            TTS_PLAYBACK_SPEED_MIN
            if self.speed_min is None
            else max(TTS_PLAYBACK_SPEED_MIN, self.speed_min)
        )
        maximum = (
            TTS_PLAYBACK_SPEED_MAX
            if self.speed_max is None
            else min(TTS_PLAYBACK_SPEED_MAX, self.speed_max)
        )
        if minimum > maximum:
            minimum, maximum = maximum, minimum
        return (round(minimum, 2), round(maximum, 2))


class TTSSpeedApplyError(RuntimeError):
    """Raised when a backend cannot apply a requested synthesis speed."""


class TTSBackend(ABC):
    """Common runtime surface used by the ShuVoice TTS player."""

    capabilities = TTSCapabilities()

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def sample_rate_hz(self) -> int:
        """Return the PCM sample rate emitted by :meth:`synthesize_stream`."""

    @abstractmethod
    def synthesize_stream(self, request: TTSSynthesisRequest) -> Iterator[bytes]:
        """Yield PCM audio chunks as they become available."""

    @abstractmethod
    def list_voices(self) -> list[VoiceInfo]:
        """Return available voices for UI selectors."""

    @staticmethod
    @abstractmethod
    def dependency_errors() -> list[str]:
        """Return missing dependency/runtime errors for this backend."""
