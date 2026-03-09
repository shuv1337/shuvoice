"""OpenAI TTS backend."""

from __future__ import annotations

import json
import logging
import math
import os
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

_OPENAI_VOICES: tuple[VoiceInfo, ...] = (
    VoiceInfo(id="alloy", name="Alloy", description="Balanced neutral voice"),
    VoiceInfo(id="ash", name="Ash", description="Clear and steady voice"),
    VoiceInfo(id="coral", name="Coral", description="Warm expressive voice"),
    VoiceInfo(id="echo", name="Echo", description="Bright conversational voice"),
    VoiceInfo(id="fable", name="Fable", description="Narrative story-like voice"),
    VoiceInfo(id="onyx", name="Onyx", description="Deep resonant voice"),
    VoiceInfo(id="nova", name="Nova", description="Upbeat modern voice"),
    VoiceInfo(id="sage", name="Sage", description="Calm measured voice"),
    VoiceInfo(id="shimmer", name="Shimmer", description="Light energetic voice"),
)


class OpenAITTSBackend(TTSBackend):
    """OpenAI text-to-speech backend using stdlib urllib."""

    # Verified from the generated openai-python client / OpenAPI types:
    # `speed` is a native request field with a documented provider range of
    # 0.25–4.0. ShuVoice intentionally keeps a narrower shared 0.5–2.0 UI range,
    # so current requests pass through directly without extra mapping.
    capabilities = TTSCapabilities(
        supports_streaming=True,
        supports_voice_list=True,
        requires_api_key=True,
        supports_speed_control=True,
        speed_min=TTS_PLAYBACK_SPEED_MIN,
        speed_max=TTS_PLAYBACK_SPEED_MAX,
    )

    _API_BASE = "https://api.openai.com/v1"
    _OUTPUT_FORMAT_ALIASES = {
        "pcm": "pcm",
        "pcm_24000": "pcm",
    }
    _PROVIDER_SPEED_MIN = 0.25
    _PROVIDER_SPEED_MAX = 4.0

    @staticmethod
    def dependency_errors() -> list[str]:
        if os.environ.get("OPENAI_API_KEY"):
            return []
        return [
            "Missing OPENAI_API_KEY environment variable "
            "(or configure [tts].tts_api_key_env and set that variable)"
        ]

    def _api_key(self) -> str:
        env_name = str(self.config.tts_api_key_env).strip()
        key = os.environ.get(env_name, "").strip()
        if not key:
            raise RuntimeError(f"Missing OpenAI API key environment variable: {env_name}")
        return key

    def _response_format(self) -> str:
        output_format = str(self.config.tts_output_format).strip().lower()
        response_format = self._OUTPUT_FORMAT_ALIASES.get(output_format)
        if response_format is None:
            raise ValueError(
                "OpenAI TTS requires raw PCM output; set [tts].tts_output_format to "
                '"pcm_24000" (or "pcm")'
            )
        return response_format

    @staticmethod
    def _classify_http_error(exc: urllib.error.HTTPError) -> str:
        if exc.code == 401:
            return "OpenAI authentication failed (401)"
        if exc.code == 429:
            return "OpenAI rate limit exceeded (429)"
        if 500 <= exc.code <= 599:
            return f"OpenAI server error ({exc.code})"
        return f"OpenAI request failed ({exc.code})"

    def _native_speed_for_request(self, request: TTSSynthesisRequest) -> float:
        speed = float(request.playback_speed)
        if not math.isfinite(speed) or speed <= 0:
            raise TTSSpeedApplyError("OpenAI speed must be a positive finite number")

        native_speed = min(self._PROVIDER_SPEED_MAX, max(self._PROVIDER_SPEED_MIN, speed))
        native_speed = round(native_speed, 2)
        if abs(native_speed - speed) >= 1e-6:
            log.info(
                "OpenAI TTS speed clamped: requested=%sx native=%sx",
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
        api_key = self._api_key()
        response_format = self._response_format()
        native_speed = self._native_speed_for_request(request)

        log.info(
            "OpenAI TTS request: voice=%s model=%s speed=%sx native_speed=%sx",
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
            url=f"{self._API_BASE}/audio/speech",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/octet-stream",
                "Authorization": f"Bearer {api_key}",
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
            raise RuntimeError("OpenAI request timed out") from exc
        except OSError as exc:
            raise RuntimeError(f"OpenAI request failed: {type(exc).__name__}") from exc

    def list_voices(self) -> list[VoiceInfo]:
        return list(_OPENAI_VOICES)
