"""Shared TTS backend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .config import Config


@dataclass(frozen=True)
class VoiceInfo:
    id: str
    name: str
    description: str = ""


@dataclass(frozen=True)
class TTSCapabilities:
    supports_streaming: bool = True
    supports_voice_list: bool = True
    requires_api_key: bool = False


class TTSBackend(ABC):
    """Common runtime surface used by the ShuVoice TTS player."""

    capabilities = TTSCapabilities()

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def synthesize_stream(self, text: str, voice_id: str, model_id: str) -> Iterator[bytes]:
        """Yield PCM audio chunks as they become available."""

    @abstractmethod
    def list_voices(self) -> list[VoiceInfo]:
        """Return available voices for UI selectors."""

    @staticmethod
    @abstractmethod
    def dependency_errors() -> list[str]:
        """Return missing dependency/runtime errors for this backend."""
