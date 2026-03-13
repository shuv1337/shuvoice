"""MeloTTS backend — subprocess-isolated local TTS using MeloTTS."""

from __future__ import annotations

import json
import logging
import struct
import subprocess
from collections.abc import Iterator
from pathlib import Path

from .tts_base import (
    TTSBackend,
    TTSCapabilities,
    TTSSynthesisRequest,
    VoiceInfo,
)

log = logging.getLogger(__name__)

_MELOTTS_SAMPLE_RATE_HZ = 44100

# Default venv location managed by ``shuvoice setup``.
_DEFAULT_MELOTTS_VENV_DIR = "~/.local/share/shuvoice/melotts-venv"

# Path to the helper module inside the ShuVoice package.
_HELPER_MODULE = "shuvoice.melo_helper"

# Frame header size: 4-byte little-endian uint32.
_FRAME_HEADER_SIZE = 4

# Available MeloTTS English voices with human-readable names.
_MELOTTS_VOICES: list[VoiceInfo] = [
    VoiceInfo(id="EN-US", name="American English", description="MeloTTS EN_V2 — American accent"),
    VoiceInfo(id="EN-BR", name="British English", description="MeloTTS EN_V2 — British accent"),
    VoiceInfo(id="EN-INDIA", name="Indian English", description="MeloTTS EN_V2 — Indian accent"),
    VoiceInfo(
        id="EN-AU", name="Australian English", description="MeloTTS EN_V2 — Australian accent"
    ),
    VoiceInfo(
        id="EN-Newest",
        name="Newest English",
        description="MeloTTS EN_NEWEST — latest improved voice",
    ),
]


class MeloTTSBackend(TTSBackend):
    """MeloTTS backend using a subprocess helper in an isolated venv."""

    capabilities = TTSCapabilities(
        supports_streaming=False,
        supports_voice_list=True,
        requires_api_key=False,
        supports_speed_control=True,
        speed_min=0.5,
        speed_max=2.0,
    )

    def __init__(self, config):
        super().__init__(config)
        venv_path = getattr(config, "tts_melotts_venv_path", None)
        if venv_path:
            self._venv_dir = Path(venv_path).expanduser()
        else:
            self._venv_dir = Path(_DEFAULT_MELOTTS_VENV_DIR).expanduser()

        self._device = getattr(config, "tts_melotts_device", "auto") or "auto"

    # ------------------------------------------------------------------
    # TTSBackend interface
    # ------------------------------------------------------------------

    def sample_rate_hz(self) -> int:
        return _MELOTTS_SAMPLE_RATE_HZ

    def list_voices(self) -> list[VoiceInfo]:
        return list(_MELOTTS_VOICES)

    @staticmethod
    def dependency_errors(venv_path: str | None = None) -> list[str]:
        """Check for missing MeloTTS runtime dependencies.

        Parameters
        ----------
        venv_path:
            Override path to the MeloTTS venv.  When ``None`` the default
            location is used.
        """
        errors: list[str] = []

        venv_dir = Path(venv_path or _DEFAULT_MELOTTS_VENV_DIR).expanduser()

        if not venv_dir.is_dir():
            errors.append(
                f"MeloTTS venv directory does not exist: {venv_dir}. "
                "Run 'shuvoice setup --install-missing' to create it."
            )
            return errors

        python_bin = venv_dir / "bin" / "python"
        if not python_bin.exists():
            errors.append(f"MeloTTS venv python binary not found: {python_bin}")
            return errors

        if not python_bin.stat().st_mode & 0o111:
            errors.append(f"MeloTTS venv python is not executable: {python_bin}")

        # Check helper script existence via importlib
        helper_path = Path(__file__).with_name("melo_helper.py")
        if not helper_path.is_file():
            errors.append(f"MeloTTS helper script not found: {helper_path}")

        return errors

    def synthesize_stream(self, request: TTSSynthesisRequest) -> Iterator[bytes]:
        """Spawn the MeloTTS helper, send a request, and yield PCM chunks."""
        text = str(request.text).strip()
        if not text:
            raise ValueError("TTS text must not be empty")
        if len(text) > int(self.config.tts_max_chars):
            raise ValueError(
                f"Selected text is too long ({len(text)} chars, max {self.config.tts_max_chars})"
            )

        python_bin = str(self._venv_dir / "bin" / "python")
        command = [python_bin, "-m", _HELPER_MODULE, self._device]

        log.info(
            "MeloTTS synthesis: voice=%s speed=%sx text_len=%d",
            request.voice_id,
            round(float(request.playback_speed), 2),
            len(text),
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
            raise RuntimeError(f"Failed to start MeloTTS helper process: {exc}") from exc

        assert proc.stdin is not None
        assert proc.stdout is not None

        try:
            # Send the JSON request followed by a newline, then close stdin
            # so the helper knows there are no more requests.
            req_json = json.dumps(
                {
                    "text": text,
                    "voice_id": request.voice_id,
                    "speed": request.playback_speed,
                }
            )
            proc.stdin.write((req_json + "\n").encode("utf-8"))
            proc.stdin.close()

            # Read framing header: 4-byte LE uint32 payload length
            header = proc.stdout.read(_FRAME_HEADER_SIZE)
            if len(header) == 0:
                # No output at all — check for process error below
                pass
            elif len(header) < _FRAME_HEADER_SIZE:
                raise RuntimeError(
                    f"Incomplete frame header from MeloTTS helper "
                    f"(expected {_FRAME_HEADER_SIZE} bytes, got {len(header)})"
                )
            else:
                (payload_len,) = struct.unpack("<I", header)

                # Read exactly payload_len bytes of PCM data, yielding in chunks
                bytes_remaining = payload_len
                chunk_size = 4096
                while bytes_remaining > 0:
                    to_read = min(chunk_size, bytes_remaining)
                    chunk = proc.stdout.read(to_read)
                    if not chunk:
                        break
                    bytes_remaining -= len(chunk)
                    yield chunk

            proc.stdout.close()
            stderr_bytes = proc.stderr.read() if proc.stderr else b""
            proc.wait(timeout=timeout)

        except subprocess.TimeoutExpired as exc:
            proc.kill()
            raise RuntimeError("MeloTTS synthesis timed out") from exc

        if proc.returncode not in (0, None):
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            if stderr_text:
                raise RuntimeError(f"MeloTTS synthesis failed: {stderr_text}")
            raise RuntimeError(f"MeloTTS synthesis failed with exit code {proc.returncode}")
