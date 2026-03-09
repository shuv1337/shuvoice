"""Local TTS backend scaffold (Piper CLI)."""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from .tts_base import (
    TTSBackend,
    TTSCapabilities,
    TTSSpeedApplyError,
    TTSSynthesisRequest,
    VoiceInfo,
)
from .tts_speed import TTS_PLAYBACK_SPEED_MAX, TTS_PLAYBACK_SPEED_MIN

log = logging.getLogger(__name__)


class LocalTTSBackend(TTSBackend):
    """Local TTS backend using Piper CLI when available.

    This keeps the same backend contract used by remote providers so the
    control surface and overlay logic do not change when switching to
    ``tts_backend = "local"``.
    """

    # Verified from the current Piper CLI (`piper --help` / `src/piper/__main__.py`)
    # in OHF-Voice/piper1-gpl: synthesis-time duration control is exposed as
    # `--length-scale`. Lower values speak faster; higher values speak slower.
    # ShuVoice therefore maps `speed` inversely as `length_scale = 1.0 / speed`.
    capabilities = TTSCapabilities(
        supports_streaming=True,
        supports_voice_list=True,
        requires_api_key=False,
        supports_speed_control=True,
        speed_min=TTS_PLAYBACK_SPEED_MIN,
        speed_max=TTS_PLAYBACK_SPEED_MAX,
    )

    def __init__(self, config):
        super().__init__(config)
        self._voice_cache = self._discover_voices()

    @staticmethod
    def dependency_errors() -> list[str]:
        errors: list[str] = []
        if shutil.which("piper") is None:
            errors.append(
                "Missing piper binary for local TTS backend. "
                "Install Piper and set [tts].tts_local_model_path."
            )
        return errors

    def _discover_voices(self) -> list[VoiceInfo]:
        model_path = self.config.tts_local_model_path
        if not model_path:
            return []

        resolved = Path(model_path).expanduser()
        voices: list[VoiceInfo] = []

        if resolved.is_file():
            voices.append(
                VoiceInfo(
                    id=resolved.stem,
                    name=resolved.stem,
                    description=str(resolved),
                )
            )
            return voices

        if resolved.is_dir():
            for candidate in sorted(resolved.glob("*.onnx")):
                voices.append(
                    VoiceInfo(
                        id=candidate.stem,
                        name=candidate.stem,
                        description=str(candidate),
                    )
                )
        return voices

    def list_voices(self) -> list[VoiceInfo]:
        if not self._voice_cache:
            self._voice_cache = self._discover_voices()
        return list(self._voice_cache)

    def _resolve_model_file(self, voice_id: str) -> Path:
        configured = self.config.tts_local_model_path
        if not configured:
            raise RuntimeError(
                "Local TTS requires [tts].tts_local_model_path to point to a Piper model"
            )

        path = Path(configured).expanduser()
        if path.is_file():
            return path

        if not path.is_dir():
            raise RuntimeError(f"Local TTS model path does not exist: {path}")

        requested_voice = str(voice_id or self.config.tts_local_voice or "").strip()
        if requested_voice:
            requested_file = path / f"{requested_voice}.onnx"
            if requested_file.is_file():
                return requested_file

        first = next(iter(sorted(path.glob("*.onnx"))), None)
        if first is None:
            raise RuntimeError(f"No .onnx model files found under local TTS path: {path}")
        return first

    @staticmethod
    def _length_scale_for_speed(speed: float) -> float:
        speed_value = float(speed)
        if not math.isfinite(speed_value) or speed_value <= 0:
            raise TTSSpeedApplyError("Local Piper speed must be a positive finite number")

        # Piper length-scale is inverse duration control:
        #   faster ShuVoice speed  -> smaller length_scale
        #   slower ShuVoice speed  -> larger length_scale
        return round(1.0 / speed_value, 4)

    def synthesize_stream(self, request: TTSSynthesisRequest) -> Iterator[bytes]:
        text_value = str(request.text).strip()
        if not text_value:
            raise ValueError("TTS text must not be empty")
        if len(text_value) > int(self.config.tts_max_chars):
            raise ValueError(
                f"Selected text is too long ({len(text_value)} chars, max {self.config.tts_max_chars})"
            )

        model_file = self._resolve_model_file(request.voice_id)
        length_scale = self._length_scale_for_speed(request.playback_speed)

        command = [
            "piper",
            "--model",
            str(model_file),
            "--output_raw",
            "--length-scale",
            f"{length_scale:.4f}",
        ]
        if self.config.tts_local_device is not None:
            log.debug("Local TTS device hint configured: %s", self.config.tts_local_device)

        log.info(
            "Local Piper TTS request: voice=%s speed=%sx length_scale=%s model=%s",
            request.voice_id or self.config.tts_local_voice or model_file.stem,
            round(float(request.playback_speed), 2),
            f"{length_scale:.4f}",
            model_file.name,
        )

        timeout = max(1.0, float(self.config.tts_request_timeout_sec) * 4.0)

        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to start Piper local TTS process: {exc}") from exc

        assert proc.stdin is not None
        assert proc.stdout is not None

        try:
            proc.stdin.write(text_value.encode("utf-8"))
            proc.stdin.close()

            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                yield chunk

            _stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            raise RuntimeError("Local TTS synthesis timed out") from exc

        if proc.returncode not in (0, None):
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            if stderr_text:
                raise RuntimeError(f"Local TTS synthesis failed: {stderr_text}")
            raise RuntimeError(f"Local TTS synthesis failed with exit code {proc.returncode}")
