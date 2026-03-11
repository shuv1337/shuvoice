"""Shared Local Piper setup and managed voice automation helpers."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)

_PIPER_BINARY_NAMES = ("piper", "piper-tts")
_DOWNLOAD_CHUNK_SIZE = 1024 * 256
_DOWNLOAD_TIMEOUT_SEC = 120


@dataclass(frozen=True)
class PiperVoiceOption:
    id: str
    label: str
    stem: str
    language: str
    quality: str
    description: str
    model_url: str
    sidecar_url: str


@dataclass(frozen=True)
class LocalPiperSetupResult:
    status: str
    message: str
    binary_name: str | None
    model_dir: Path
    voice: PiperVoiceOption
    sample_rate_hz: int | None


_CURATED_PIPER_VOICES: tuple[PiperVoiceOption, ...] = (
    PiperVoiceOption(
        id="en_US-amy-medium",
        label="US English — Amy (medium, recommended)",
        stem="en_US-amy-medium",
        language="en-US",
        quality="medium",
        description="Balanced default voice. Good quality with fast local inference.",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true",
        sidecar_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true",
    ),
    PiperVoiceOption(
        id="en_US-lessac-medium",
        label="US English — Lessac (medium)",
        stem="en_US-lessac-medium",
        language="en-US",
        quality="medium",
        description="Popular clean US voice. Similar size to Amy with slightly different tone.",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true",
        sidecar_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true",
    ),
    PiperVoiceOption(
        id="en_US-ryan-medium",
        label="US English — Ryan (medium)",
        stem="en_US-ryan-medium",
        language="en-US",
        quality="medium",
        description="Medium-quality male US voice. Good alternative to Amy/Lessac.",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx?download=true",
        sidecar_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json?download=true",
    ),
    PiperVoiceOption(
        id="en_US-lessac-high",
        label="US English — Lessac (high)",
        stem="en_US-lessac-high",
        language="en-US",
        quality="high",
        description="Higher-quality Lessac voice. Larger download and slower inference.",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/en_US-lessac-high.onnx?download=true",
        sidecar_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/en_US-lessac-high.onnx.json?download=true",
    ),
    PiperVoiceOption(
        id="en_US-ljspeech-high",
        label="US English — LJSpeech (high)",
        stem="en_US-ljspeech-high",
        language="en-US",
        quality="high",
        description="Higher-quality female US voice trained on LJSpeech.",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx?download=true",
        sidecar_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json?download=true",
    ),
    PiperVoiceOption(
        id="en_US-ryan-high",
        label="US English — Ryan (high)",
        stem="en_US-ryan-high",
        language="en-US",
        quality="high",
        description="Higher-quality male US voice. Largest of the curated set.",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx?download=true",
        sidecar_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json?download=true",
    ),
)


def curated_piper_voices() -> tuple[PiperVoiceOption, ...]:
    return _CURATED_PIPER_VOICES


def recommended_piper_voice() -> PiperVoiceOption:
    return _CURATED_PIPER_VOICES[0]


def get_curated_piper_voice(voice_id: str) -> PiperVoiceOption:
    key = str(voice_id or "").strip().lower()
    for option in _CURATED_PIPER_VOICES:
        if option.id.lower() == key or option.stem.lower() == key:
            return option
    supported = ", ".join(option.id for option in _CURATED_PIPER_VOICES)
    raise ValueError(f"Unknown curated Piper voice '{voice_id}'. Supported voices: {supported}")


def managed_piper_model_dir() -> Path:
    return Config.data_dir() / "models" / "piper"


def find_piper_binary() -> str | None:
    for name in _PIPER_BINARY_NAMES:
        if shutil.which(name) is not None:
            return name
    return None


def piper_install_commands() -> list[list[str]]:
    return [
        ["yay", "-S", "--needed", "piper-tts"],
        ["paru", "-S", "--needed", "piper-tts"],
    ]


def piper_install_hints() -> tuple[str, ...]:
    hints: list[str] = []
    if shutil.which("yay"):
        hints.append("Arch (AUR): yay -S --needed piper-tts")
    if shutil.which("paru"):
        hints.append("Arch (AUR): paru -S --needed piper-tts")
    hints.append("Manual: install upstream Piper CLI and ensure `piper` or `piper-tts` is in PATH")
    return tuple(hints)


def attempt_piper_auto_install() -> bool:
    for command in piper_install_commands():
        executable = command[0]
        if shutil.which(executable) is None:
            continue
        log.info("Attempting Local Piper install: %s", " ".join(command))
        proc = subprocess.run(command, check=False)
        if proc.returncode != 0:
            continue
        if find_piper_binary() is not None:
            return True
    return find_piper_binary() is not None


def _selected_voice_id(voice_id: str | None) -> str | None:
    value = str(voice_id or "").strip()
    return value or None


def _model_file_for_voice(path_or_dir: Path, voice_id: str | None = None) -> Path:
    path = Path(path_or_dir).expanduser()
    if path.is_file():
        return path

    if not path.exists():
        raise RuntimeError(f"Local Piper path does not exist: {path}")
    if not path.is_dir():
        raise RuntimeError(f"Local Piper path must be a file or directory: {path}")

    requested_voice = _selected_voice_id(voice_id)
    if requested_voice is not None:
        requested_file = path / f"{requested_voice}.onnx"
        if requested_file.is_file():
            return requested_file
        raise RuntimeError(f"Local Piper voice '{requested_voice}' not found in {path}")

    first = next(iter(sorted(path.glob("*.onnx"))), None)
    if first is None:
        raise RuntimeError(f"No .onnx model files found under: {path}")
    return first


def installed_piper_voice_stems(path_or_dir: Path | None) -> tuple[str, ...]:
    if path_or_dir is None:
        return tuple()

    path = Path(path_or_dir).expanduser()
    if path.is_file() and path.suffix.lower() == ".onnx":
        return (path.stem,)
    if not path.is_dir():
        return tuple()
    return tuple(sorted(candidate.stem for candidate in path.glob("*.onnx") if candidate.is_file()))


def piper_sample_rate_from_sidecar(model_file: Path) -> int | None:
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


def validate_piper_voice_artifacts(path_or_dir: Path, voice_id: str | None = None) -> tuple[bool, str]:
    path = Path(path_or_dir).expanduser()

    if path.is_file():
        if path.suffix.lower() != ".onnx":
            return False, f"Local Piper model path must point to a .onnx file: {path}"
        model_file = path
    else:
        if not path.exists():
            return False, f"Local Piper path does not exist: {path}"
        if not path.is_dir():
            return False, f"Local Piper path must be a file or directory: {path}"
        model_files = sorted(path.glob("*.onnx"))
        if not model_files:
            return False, f"No .onnx model files found under: {path}"
        try:
            model_file = _model_file_for_voice(path, voice_id)
        except RuntimeError as exc:
            return False, str(exc)

    sidecar = model_file.with_name(f"{model_file.name}.json")
    if not sidecar.is_file():
        return False, f"Missing Piper sidecar metadata: {sidecar}"

    try:
        json.loads(sidecar.read_text())
    except Exception as exc:  # noqa: BLE001
        return False, f"Invalid Piper sidecar metadata {sidecar}: {exc}"

    sample_rate = piper_sample_rate_from_sidecar(model_file)
    if sample_rate is not None:
        return True, f"ready ({model_file.name}; sample_rate={sample_rate}Hz)"
    return True, f"ready ({model_file.name}; sample_rate metadata missing)"


def _emit_progress(
    progress_callback: Callable[[float | None, str], None] | None,
    fraction: float | None,
    message: str,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(fraction, message)
    except Exception:  # noqa: BLE001
        log.debug("Local Piper progress callback failed", exc_info=True)


def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise RuntimeError("Local Piper setup cancelled")


def _download_to_file(
    url: str,
    destination: Path,
    *,
    start_fraction: float,
    end_fraction: float,
    progress_message: str,
    progress_callback: Callable[[float | None, str], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f"{destination.name}.part")

    _check_cancel(cancel_check)
    req = urllib.request.Request(url, headers={"User-Agent": "shuvoice"})
    try:
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_SEC) as response, temp_path.open(
            "wb"
        ) as handle:
            header_total = response.headers.get("Content-Length")
            total = int(header_total) if header_total and header_total.isdigit() else None
            downloaded = 0
            while True:
                _check_cancel(cancel_check)
                chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total and total > 0:
                    frac = min(1.0, downloaded / total)
                    overall = start_fraction + ((end_fraction - start_fraction) * frac)
                    _emit_progress(progress_callback, overall, progress_message)
                elif downloaded == len(chunk):
                    _emit_progress(progress_callback, None, progress_message)
    except (urllib.error.URLError, TimeoutError) as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            log.debug("Failed to clean partial Piper download %s", temp_path, exc_info=True)
        raise RuntimeError(f"Failed to download Local Piper artifact from {url}: {exc}") from exc
    except RuntimeError:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            log.debug("Failed to clean partial Piper download %s", temp_path, exc_info=True)
        raise

    temp_path.replace(destination)


def ensure_piper_voice_downloaded(
    voice: PiperVoiceOption,
    *,
    model_dir: Path | None = None,
    progress_callback: Callable[[float | None, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Path:
    target_dir = Path(model_dir).expanduser() if model_dir is not None else managed_piper_model_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    model_file = target_dir / f"{voice.stem}.onnx"
    sidecar_file = target_dir / f"{voice.stem}.onnx.json"

    valid, detail = validate_piper_voice_artifacts(target_dir, voice.stem)
    if valid:
        log.info("Local Piper voice already available: %s (%s)", voice.stem, target_dir)
        _emit_progress(progress_callback, 1.0, f"Local Piper voice already available: {voice.label}")
        return _model_file_for_voice(target_dir, voice.stem)

    log.info(
        "Downloading Local Piper voice %s to %s (%s)",
        voice.stem,
        target_dir,
        detail,
    )

    try:
        _emit_progress(progress_callback, 0.0, f"Downloading Local Piper voice: {voice.label}")
        _download_to_file(
            voice.model_url,
            model_file,
            start_fraction=0.0,
            end_fraction=0.9,
            progress_message=f"Downloading Local Piper model: {voice.stem}",
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        _download_to_file(
            voice.sidecar_url,
            sidecar_file,
            start_fraction=0.9,
            end_fraction=0.98,
            progress_message=f"Downloading Local Piper metadata: {voice.stem}",
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        _check_cancel(cancel_check)
        valid, detail = validate_piper_voice_artifacts(target_dir, voice.stem)
        if not valid:
            raise RuntimeError(detail)
    except Exception:
        if not validate_piper_voice_artifacts(target_dir, voice.stem)[0]:
            try:
                model_file.unlink(missing_ok=True)
                sidecar_file.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                log.debug("Failed to clean partial Local Piper artifacts", exc_info=True)
        raise

    _emit_progress(progress_callback, 1.0, f"Local Piper voice ready: {voice.label}")
    log.info("Local Piper voice ready: %s (%s)", voice.stem, target_dir)
    return model_file


def ensure_local_piper_ready(
    voice: PiperVoiceOption,
    *,
    model_dir: Path | None = None,
    auto_install_missing: bool = False,
    progress_callback: Callable[[float | None, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> LocalPiperSetupResult:
    target_dir = Path(model_dir).expanduser() if model_dir is not None else managed_piper_model_dir()

    binary_name = find_piper_binary()
    if binary_name is None and auto_install_missing:
        _emit_progress(progress_callback, None, "Installing Local Piper runtime…")
        if attempt_piper_auto_install():
            binary_name = find_piper_binary()

    if binary_name is None:
        hints = "; ".join(piper_install_hints())
        return LocalPiperSetupResult(
            status="skipped_missing_deps",
            message=(
                "Local Piper runtime is missing. Install `piper-tts` (or provide `piper` in PATH). "
                f"Hints: {hints}"
            ),
            binary_name=None,
            model_dir=target_dir,
            voice=voice,
            sample_rate_hz=None,
        )

    try:
        ensure_piper_voice_downloaded(
            voice,
            model_dir=target_dir,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
    except RuntimeError as exc:
        message = str(exc)
        status = "cancelled" if "cancelled" in message.lower() else "error"
        return LocalPiperSetupResult(
            status=status,
            message=message,
            binary_name=binary_name,
            model_dir=target_dir,
            voice=voice,
            sample_rate_hz=None,
        )

    sample_rate = piper_sample_rate_from_sidecar(target_dir / f"{voice.stem}.onnx")
    detail = f"Local Piper ready: {voice.stem}"
    if sample_rate is not None:
        detail += f" ({sample_rate}Hz)"
    return LocalPiperSetupResult(
        status="downloaded",
        message=detail,
        binary_name=binary_name,
        model_dir=target_dir,
        voice=voice,
        sample_rate_hz=sample_rate,
    )


__all__ = [
    "LocalPiperSetupResult",
    "PiperVoiceOption",
    "attempt_piper_auto_install",
    "curated_piper_voices",
    "ensure_local_piper_ready",
    "ensure_piper_voice_downloaded",
    "find_piper_binary",
    "get_curated_piper_voice",
    "installed_piper_voice_stems",
    "managed_piper_model_dir",
    "piper_install_commands",
    "piper_install_hints",
    "piper_sample_rate_from_sidecar",
    "recommended_piper_voice",
    "validate_piper_voice_artifacts",
]
