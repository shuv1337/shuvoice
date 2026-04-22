"""Kokoro TTS backend (local self-hosted, OpenAI-compatible API)."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

from .tts_base import (
    TTSBackend,
    TTSCapabilities,
    TTSSpeedApplyError,
    TTSSynthesisRequest,
    VoiceInfo,
)
from .tts_speed import TTS_PLAYBACK_SPEED_MAX, TTS_PLAYBACK_SPEED_MIN

log = logging.getLogger(__name__)


class KokoroTTSBackend(TTSBackend):
    """Kokoro text-to-speech backend using stdlib urllib.

    Kokoro exposes an OpenAI-compatible ``/v1/audio/speech`` endpoint locally.
    No API key is required (the ``Authorization`` header is accepted but ignored).
    """

    capabilities = TTSCapabilities(
        supports_streaming=True,
        supports_voice_list=True,
        requires_api_key=False,
        supports_speed_control=True,
        speed_min=TTS_PLAYBACK_SPEED_MIN,
        speed_max=TTS_PLAYBACK_SPEED_MAX,
    )

    _OUTPUT_FORMAT_ALIASES = {
        "pcm": "pcm",
        "pcm_24000": "pcm",
        "mp3": "mp3",
    }
    _PROVIDER_SPEED_MIN = 0.5
    _PROVIDER_SPEED_MAX = 2.0
    _DEFAULT_VOICE_CACHE_TTL_SEC = 300.0

    def __init__(self, config):
        super().__init__(config)
        self._base_url = str(getattr(config, "tts_kokoro_base_url", "http://localhost:8880/v1")).rstrip("/")
        self._voice_cache: list[VoiceInfo] = []
        self._voice_cache_expires_at = 0.0
        self._cache_lock = threading.Lock()

    @staticmethod
    def dependency_errors() -> list[str]:
        return []

    def sample_rate_hz(self) -> int:
        output_format = str(self.config.tts_output_format).strip().lower()
        if output_format.startswith("pcm_"):
            maybe_rate = output_format.split("_", 1)[1]
            if maybe_rate.isdigit() and int(maybe_rate) > 0:
                return int(maybe_rate)
        return 24000

    def _response_format(self) -> str:
        output_format = str(self.config.tts_output_format).strip().lower()
        response_format = self._OUTPUT_FORMAT_ALIASES.get(output_format)
        if response_format is None:
            raise ValueError(
                "Kokoro TTS requires a supported output format; set [tts].tts_output_format to "
                '"pcm_24000" (or "pcm" or "mp3")'
            )
        return response_format

    @staticmethod
    def _classify_http_error(exc: urllib.error.HTTPError) -> str:
        if exc.code == 401:
            return "Kokoro authentication failed (401)"
        if exc.code == 429:
            return "Kokoro rate limit exceeded (429)"
        if 500 <= exc.code <= 599:
            return f"Kokoro server error ({exc.code})"
        return f"Kokoro request failed ({exc.code})"

    def _native_speed_for_request(self, request: TTSSynthesisRequest) -> float:
        speed = float(request.playback_speed)
        if not math.isfinite(speed) or speed <= 0:
            raise TTSSpeedApplyError("Kokoro speed must be a positive finite number")

        native_speed = min(self._PROVIDER_SPEED_MAX, max(self._PROVIDER_SPEED_MIN, speed))
        native_speed = round(native_speed, 2)
        if abs(native_speed - speed) >= 1e-6:
            log.info(
                "Kokoro TTS speed clamped: requested=%sx native=%sx",
                round(speed, 2),
                native_speed,
            )
        return native_speed

    def synthesize_stream(self, request: TTSSynthesisRequest) -> Iterator[bytes]:
        text_value = str(request.text).strip()
        if not text_value:
            raise ValueError("TTS text must not be empty")
        if len(text_value) > int(self.config.tts_max_chars):
            raise ValueError(
                f"Selected text is too long ({len(text_value)} chars, max {self.config.tts_max_chars})"
            )

        voice = str(request.voice_id or self.config.tts_default_voice_id).strip()
        model = str(request.model_id or self.config.tts_model_id).strip()
        response_format = self._response_format()
        native_speed = self._native_speed_for_request(request)

        log.info(
            "Kokoro TTS request: voice=%s model=%s speed=%sx native_speed=%sx",
            voice,
            model,
            round(float(request.playback_speed), 2),
            native_speed,
        )

        payload = json.dumps(
            {
                "model": model,
                "voice": voice,
                "input": text_value,
                "response_format": response_format,
                "speed": native_speed,
            }
        ).encode("utf-8")
        http_request = urllib.request.Request(
            url=f"{self._base_url}/audio/speech",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/octet-stream",
                "Authorization": "Bearer sk-local",
            },
            method="POST",
        )

        timeout = float(self.config.tts_request_timeout_sec)

        try:
            with urllib.request.urlopen(http_request, timeout=timeout) as response:
                while True:
                    chunk = response.read(4096)
                    if not chunk:
                        break
                    yield bytes(chunk)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(self._classify_http_error(exc)) from exc
        except TimeoutError as exc:
            raise RuntimeError("Kokoro request timed out") from exc
        except OSError as exc:
            raise RuntimeError(f"Kokoro request failed: {type(exc).__name__}") from exc

    def list_voices(self) -> list[VoiceInfo]:
        now = time.monotonic()
        with self._cache_lock:
            if self._voice_cache and now < self._voice_cache_expires_at:
                return list(self._voice_cache)

        request = urllib.request.Request(
            url=f"{self._base_url}/audio/voices",
            headers={
                "Accept": "application/json",
                "Authorization": "Bearer sk-local",
            },
            method="GET",
        )

        timeout = float(self.config.tts_request_timeout_sec)

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(self._classify_http_error(exc)) from exc
        except TimeoutError as exc:
            raise RuntimeError("Kokoro voice list request timed out") from exc
        except OSError as exc:
            raise RuntimeError(
                f"Kokoro voice list request failed: {type(exc).__name__}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid Kokoro voice list response") from exc

        raw_voices = payload.get("voices", [])
        if not isinstance(raw_voices, list):
            raw_voices = []

        voices: list[VoiceInfo] = []
        for raw in raw_voices:
            # Kokoro-FastAPI returns voices as plain strings (e.g. "af_heart").
            # Some forks/proxies may return dicts with id/name/description fields.
            # Support both shapes defensively.
            if isinstance(raw, str):
                voice_identifier = raw.strip()
                if not voice_identifier:
                    continue
                voices.append(
                    VoiceInfo(id=voice_identifier, name=voice_identifier, description="")
                )
                continue

            if not isinstance(raw, dict):
                continue

            voice_identifier = str(raw.get("id", "")).strip()
            if not voice_identifier:
                continue
            name = str(raw.get("name", "")).strip() or voice_identifier
            description = str(raw.get("description", "")).strip()
            voices.append(VoiceInfo(id=voice_identifier, name=name, description=description))

        with self._cache_lock:
            self._voice_cache = list(voices)
            self._voice_cache_expires_at = now + self._DEFAULT_VOICE_CACHE_TTL_SEC

        return voices
