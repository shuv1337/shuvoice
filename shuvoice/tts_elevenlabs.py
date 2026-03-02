"""ElevenLabs streaming TTS backend."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator

from .tts_base import TTSBackend, TTSCapabilities, VoiceInfo


class ElevenLabsTTSBackend(TTSBackend):
    """Streaming ElevenLabs backend using stdlib urllib."""

    capabilities = TTSCapabilities(
        supports_streaming=True,
        supports_voice_list=True,
        requires_api_key=True,
    )

    _API_BASE = "https://api.elevenlabs.io/v1"
    _DEFAULT_VOICE_CACHE_TTL_SEC = 300.0

    def __init__(self, config):
        super().__init__(config)
        self._voice_cache: list[VoiceInfo] = []
        self._voice_cache_expires_at = 0.0
        self._cache_lock = threading.Lock()

    @staticmethod
    def dependency_errors() -> list[str]:
        if os.environ.get("ELEVENLABS_API_KEY"):
            return []
        return [
            "Missing ELEVENLABS_API_KEY environment variable "
            "(or configure [tts].tts_api_key_env and set that variable)"
        ]

    def _api_key(self) -> str:
        env_name = str(self.config.tts_api_key_env).strip()
        key = os.environ.get(env_name, "").strip()
        if not key:
            raise RuntimeError(f"Missing ElevenLabs API key environment variable: {env_name}")
        return key

    @staticmethod
    def _classify_http_error(exc: urllib.error.HTTPError) -> str:
        if exc.code == 401:
            return "ElevenLabs authentication failed (401)"
        if exc.code == 429:
            return "ElevenLabs rate limit exceeded (429)"
        if 500 <= exc.code <= 599:
            return f"ElevenLabs server error ({exc.code})"
        return f"ElevenLabs request failed ({exc.code})"

    def synthesize_stream(self, text: str, voice_id: str, model_id: str) -> Iterator[bytes]:
        text_value = str(text).strip()
        if not text_value:
            raise ValueError("TTS text must not be empty")
        if len(text_value) > int(self.config.tts_max_chars):
            raise ValueError(
                f"Selected text is too long ({len(text_value)} chars, max {self.config.tts_max_chars})"
            )

        voice = str(voice_id or self.config.tts_default_voice_id).strip()
        model = str(model_id or self.config.tts_model_id).strip()
        api_key = self._api_key()

        encoded_voice = urllib.parse.quote(voice, safe="")
        query = urllib.parse.urlencode({"output_format": self.config.tts_output_format})
        url = f"{self._API_BASE}/text-to-speech/{encoded_voice}/stream?{query}"

        payload = json.dumps({"text": text_value, "model_id": model}).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/octet-stream",
                "xi-api-key": api_key,
            },
            method="POST",
        )

        timeout = float(self.config.tts_request_timeout_sec)

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                while True:
                    chunk = response.read(4096)
                    if not chunk:
                        break
                    yield bytes(chunk)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(self._classify_http_error(exc)) from exc
        except TimeoutError as exc:
            raise RuntimeError("ElevenLabs request timed out") from exc
        except OSError as exc:
            raise RuntimeError(f"ElevenLabs request failed: {type(exc).__name__}") from exc

    def list_voices(self) -> list[VoiceInfo]:
        now = time.monotonic()
        with self._cache_lock:
            if self._voice_cache and now < self._voice_cache_expires_at:
                return list(self._voice_cache)

        api_key = self._api_key()
        request = urllib.request.Request(
            url=f"{self._API_BASE}/voices",
            headers={
                "Accept": "application/json",
                "xi-api-key": api_key,
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
            raise RuntimeError("ElevenLabs voice list request timed out") from exc
        except OSError as exc:
            raise RuntimeError(f"ElevenLabs voice list request failed: {type(exc).__name__}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid ElevenLabs voice list response") from exc

        voices: list[VoiceInfo] = []
        for raw in payload.get("voices", []):
            voice_identifier = str(raw.get("voice_id", "")).strip()
            if not voice_identifier:
                continue
            name = str(raw.get("name", "")).strip() or voice_identifier
            description = str(raw.get("description", "")).strip()
            voices.append(VoiceInfo(id=voice_identifier, name=name, description=description))

        with self._cache_lock:
            self._voice_cache = list(voices)
            self._voice_cache_expires_at = now + self._DEFAULT_VOICE_CACHE_TTL_SEC

        return voices
