"""MeloTTS engine boundary — real Melo is lazy; fakes used in tests."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

SAMPLE_RATE_HZ = 44100

VOICE_TO_MODEL: dict[str, str] = {
    "EN-US": "EN_V2",
    "EN-BR": "EN_V2",
    "EN-INDIA": "EN_V2",
    "EN-AU": "EN_V2",
    "EN-Newest": "EN_NEWEST",
}

VOICES: list[dict[str, str]] = [
    {"id": "EN-US", "name": "American English", "locale": "en-US"},
    {"id": "EN-BR", "name": "British English", "locale": "en-GB"},
    {"id": "EN-INDIA", "name": "Indian English", "locale": "en-IN"},
    {"id": "EN-AU", "name": "Australian English", "locale": "en-AU"},
    {"id": "EN-Newest", "name": "Newest English", "locale": "en"},
]

DEFAULT_VENV = "~/.local/share/shuvoice/melotts-venv"


def model_for_voice(voice_id: str) -> str:
    return VOICE_TO_MODEL.get(voice_id, "EN_V2")


def dependency_errors(venv_path: str | None = None) -> list[str]:
    errors: list[str] = []
    venv_dir = Path(venv_path or os.environ.get("SHUVOICE_MELOTTS_VENV") or DEFAULT_VENV).expanduser()
    if not venv_dir.is_dir():
        errors.append(
            f"MeloTTS venv directory does not exist: {venv_dir}. "
            "Create an isolated venv and pip install melotts."
        )
        return errors
    python_bin = venv_dir / "bin" / "python"
    if not python_bin.exists():
        errors.append(f"MeloTTS venv python binary not found: {python_bin}")
    return errors


@dataclass
class MeloSynthRequest:
    text: str
    voice_id: str = "EN-US"
    speed: float = 1.0


class TtsEngine(Protocol):
    def list_voices(self) -> list[dict[str, str]]: ...
    def synthesize_i16(self, request: MeloSynthRequest) -> bytes: ...


class FakeMeloEngine:
    """Deterministic PCM generator for tests."""

    def list_voices(self) -> list[dict[str, str]]:
        return list(VOICES)

    def synthesize_i16(self, request: MeloSynthRequest) -> bytes:
        if not request.text.strip():
            raise ValueError("empty_text")
        # 100 samples of silence + length derived from text length (not content).
        n = 100 + min(500, len(request.text.strip()) * 3)
        # int16 little-endian zeros
        return b"\x00\x00" * n


class RealMeloEngine:
    """In-process MeloTTS when running inside the Melo venv (lazy import)."""

    def __init__(self, *, device: str = "auto") -> None:
        self._device = device
        self._models: dict[str, Any] = {}

    def list_voices(self) -> list[dict[str, str]]:
        return list(VOICES)

    def synthesize_i16(self, request: MeloSynthRequest) -> bytes:
        text = request.text.strip()
        if not text:
            raise ValueError("empty_text")
        try:
            import numpy as np
            from melo.api import TTS
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"import_failed:{type(exc).__name__}") from exc

        model_key = model_for_voice(request.voice_id)
        if model_key not in self._models:
            self._models[model_key] = TTS(language=model_key, device=self._device)
        model = self._models[model_key]
        speaker_ids = model.hps.data.spk2id
        speaker_id = speaker_ids.get(request.voice_id, 0)
        audio = model.tts_to_file(
            text,
            speaker_id,
            output_path=None,
            speed=float(request.speed),
            quiet=True,
        )
        pcm = (np.asarray(audio) * 32768.0).clip(-32768, 32767).astype("<i2")
        return pcm.tobytes()


def create_engine(*, fake: bool = False, device: str = "auto") -> TtsEngine:
    if fake:
        return FakeMeloEngine()
    return RealMeloEngine(device=device)
