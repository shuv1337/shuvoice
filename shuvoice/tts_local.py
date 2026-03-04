"""Local TTS backend scaffold (Piper CLI)."""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from .tts_base import TTSBackend, TTSCapabilities, VoiceInfo

log = logging.getLogger(__name__)


class LocalTTSBackend(TTSBackend):
    """Local TTS backend using Piper CLI when available.

    This keeps the same backend contract used by remote providers so the
    control surface and overlay logic do not change when switching to
    ``tts_backend = "local"``.
    """

    capabilities = TTSCapabilities(
        supports_streaming=True,
        supports_voice_list=True,
        requires_api_key=False,
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

    def synthesize_stream(self, text: str, voice_id: str, model_id: str) -> Iterator[bytes]:
        del model_id  # Reserved for future local model families.

        text_value = str(text).strip()
        if not text_value:
            raise ValueError("TTS text must not be empty")
        if len(text_value) > int(self.config.tts_max_chars):
            raise ValueError(
                f"Selected text is too long ({len(text_value)} chars, max {self.config.tts_max_chars})"
            )

        model_file = self._resolve_model_file(voice_id)

        command = ["piper", "--model", str(model_file), "--output_raw"]
        if self.config.tts_local_device is not None:
            log.debug("Local TTS device hint configured: %s", self.config.tts_local_device)

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
            raise RuntimeError(f"Local TTS synthesis failed with exit code {proc.returncode}")
