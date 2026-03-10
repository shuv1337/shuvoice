"""Local Piper-backed TTS backend."""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from .tts_base import (
    DEFAULT_LOCAL_TTS_VOICE_ID,
    LOCAL_TTS_AUTO_VOICE_IDS,
    TTSBackend,
    TTSCapabilities,
    TTSSpeedApplyError,
    TTSSynthesisRequest,
    VoiceInfo,
)
from .tts_speed import TTS_PLAYBACK_SPEED_MAX, TTS_PLAYBACK_SPEED_MIN

log = logging.getLogger(__name__)

_DEFAULT_PIPER_SAMPLE_RATE_HZ = 22050

# Piper binary names in order of preference.
# The AUR `piper-tts` package installs as `piper-tts`; upstream installs as
# `piper`.  We accept whichever is available.
_PIPER_BINARY_NAMES = ("piper", "piper-tts")


def _find_piper_binary() -> str | None:
    """Return the first available Piper binary name, or ``None``."""
    for name in _PIPER_BINARY_NAMES:
        if shutil.which(name) is not None:
            return name
    return None


class LocalTTSBackend(TTSBackend):
    """Local TTS backend using the Piper CLI."""

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
        self._piper_binary = _find_piper_binary()
        if self._piper_binary is None:
            raise RuntimeError(
                "Missing piper binary for local TTS backend. "
                "Install Piper (piper or piper-tts) and set [tts].tts_local_model_path."
            )
        self._model_path = self._validate_model_path(config.tts_local_model_path)
        self._voice_cache = self._discover_voices()
        if not self._voice_cache:
            raise RuntimeError(
                f"No .onnx model files found under local TTS path: {self._model_path}"
            )

    @staticmethod
    def dependency_errors() -> list[str]:
        errors: list[str] = []
        if _find_piper_binary() is None:
            errors.append(
                "Missing piper binary for local TTS backend. "
                "Install Piper (piper or piper-tts) and set [tts].tts_local_model_path."
            )
        return errors

    @staticmethod
    def _validate_model_path(model_path: str | None) -> Path:
        if not model_path:
            raise RuntimeError(
                "Local TTS requires [tts].tts_local_model_path to point to a Piper model"
            )

        path = Path(model_path).expanduser()
        if path.is_file():
            if path.suffix.lower() != ".onnx":
                raise RuntimeError(f"Local TTS model path must point to a .onnx file: {path}")
            return path

        if not path.exists():
            raise RuntimeError(f"Local TTS model path does not exist: {path}")

        if not path.is_dir():
            raise RuntimeError(f"Local TTS model path must be a file or directory: {path}")

        if not any(candidate.is_file() for candidate in sorted(path.glob("*.onnx"))):
            raise RuntimeError(f"No .onnx model files found under local TTS path: {path}")

        return path

    @staticmethod
    def _normalize_voice_id(value: str | None) -> str | None:
        voice_id = str(value or "").strip()
        if not voice_id:
            return None
        if voice_id.lower() in LOCAL_TTS_AUTO_VOICE_IDS:
            return None
        return voice_id

    def _discover_voices(self) -> list[VoiceInfo]:
        voices: list[VoiceInfo] = []
        if self._model_path.is_file():
            voices.append(
                VoiceInfo(
                    id=self._model_path.stem,
                    name=self._model_path.stem,
                    description=str(self._model_path),
                )
            )
            return voices

        for candidate in sorted(self._model_path.glob("*.onnx")):
            voices.append(
                VoiceInfo(
                    id=candidate.stem,
                    name=candidate.stem,
                    description=str(candidate),
                )
            )
        return voices

    def list_voices(self) -> list[VoiceInfo]:
        return list(self._voice_cache)

    def _resolve_model_file(self, voice_id: str) -> Path:
        if self._model_path.is_file():
            return self._model_path

        requested_voice = self._normalize_voice_id(voice_id)
        if requested_voice is None:
            requested_voice = self._normalize_voice_id(self.config.tts_local_voice)

        if requested_voice:
            requested_file = self._model_path / f"{requested_voice}.onnx"
            if requested_file.is_file():
                return requested_file
            raise RuntimeError(
                f"Requested local TTS voice '{requested_voice}' not found in {self._model_path}"
            )

        first = next(iter(sorted(self._model_path.glob("*.onnx"))), None)
        if first is None:
            raise RuntimeError(f"No .onnx model files found under local TTS path: {self._model_path}")
        return first

    @staticmethod
    def _sample_rate_from_sidecar(model_file: Path) -> int | None:
        sidecar = model_file.with_name(f"{model_file.name}.json")
        if not sidecar.is_file():
            return None

        try:
            payload = json.loads(sidecar.read_text())
        except Exception:  # noqa: BLE001
            log.warning("Failed to read Piper sidecar metadata: %s", sidecar, exc_info=True)
            return None

        candidates = [
            payload.get("audio", {}).get("sample_rate") if isinstance(payload.get("audio"), dict) else None,
            payload.get("sample_rate"),
            payload.get("sampleRate"),
        ]
        for value in candidates:
            try:
                sample_rate = int(value)
            except (TypeError, ValueError):
                continue
            if sample_rate > 0:
                return sample_rate
        return None

    def sample_rate_hz(self) -> int:
        model_file = self._resolve_model_file(self.config.tts_default_voice_id)
        sample_rate = self._sample_rate_from_sidecar(model_file)
        if sample_rate is not None:
            return sample_rate

        log.warning(
            "Local Piper sample rate metadata missing for %s; falling back to %s Hz",
            model_file,
            _DEFAULT_PIPER_SAMPLE_RATE_HZ,
        )
        return _DEFAULT_PIPER_SAMPLE_RATE_HZ

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
            self._piper_binary,
            "--model",
            str(model_file),
            "--output_raw",
            "--length-scale",
            f"{length_scale:.4f}",
        ]
        if self.config.tts_local_device is not None:
            log.debug("Local TTS device hint configured: %s", self.config.tts_local_device)

        effective_voice = self._normalize_voice_id(request.voice_id) or model_file.stem
        if effective_voice == DEFAULT_LOCAL_TTS_VOICE_ID:
            effective_voice = model_file.stem

        sample_rate = self._sample_rate_from_sidecar(model_file) or _DEFAULT_PIPER_SAMPLE_RATE_HZ
        log.info(
            "Local Piper TTS request: voice=%s speed=%sx length_scale=%s model=%s sample_rate=%sHz",
            effective_voice,
            round(float(request.playback_speed), 2),
            f"{length_scale:.4f}",
            model_file.name,
            sample_rate,
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

            # stdin is already closed; use wait() + stderr.read() instead of
            # communicate() which would try to flush the closed stdin handle.
            proc.stdout.close()
            stderr_bytes = proc.stderr.read() if proc.stderr else b""
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            raise RuntimeError("Local TTS synthesis timed out") from exc

        if proc.returncode not in (0, None):
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            if stderr_text:
                raise RuntimeError(f"Local TTS synthesis failed: {stderr_text}")
            raise RuntimeError(f"Local TTS synthesis failed with exit code {proc.returncode}")
