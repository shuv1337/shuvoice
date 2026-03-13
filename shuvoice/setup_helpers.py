"""Setup and dependency guidance helpers."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .asr import get_backend_class
from .config import Config
from .piper_setup import (
    find_piper_binary,
    installed_piper_voice_stems,
    piper_install_hints,
    piper_sample_rate_from_sidecar,
    validate_piper_voice_artifacts,
)
from .tts_base import LOCAL_TTS_AUTO_VOICE_IDS
from .tts_melotts import _DEFAULT_MELOTTS_VENV_DIR

DEPENDENCY_EXIT_CODE = 78
"""Exit code used for startup-blocking backend/config/runtime errors.

Used by the packaged systemd unit via ``RestartPreventExitStatus`` so
service startup does not loop forever on unrecoverable setup issues.
"""


@dataclass(frozen=True)
class BackendSetupReport:
    backend: str
    missing_dependencies: tuple[str, ...]
    install_hints: tuple[str, ...]
    model_status: str


@dataclass(frozen=True)
class LocalTTSSetupReport:
    binary_present: bool
    binary_name: str | None
    model_dir: Path | None
    installed_voices: tuple[str, ...]
    missing_artifacts: tuple[str, ...]
    model_status: str


@dataclass(frozen=True)
class MeloTTSSetupReport:
    venv_present: bool
    venv_dir: Path
    python_executable: bool
    missing_dependencies: tuple[str, ...]
    model_status: str


def _sherpa_model_default_dir(model_name: str | None = None) -> Path:
    backend_cls = get_backend_class("sherpa")
    default_name = model_name or getattr(
        backend_cls,
        "_DEFAULT_MODEL_NAME",
        "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
    )
    return Config.data_dir() / "models" / "sherpa" / str(default_name).strip()


def _is_complete_sherpa_model_dir(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False

    if not (model_dir / "tokens.txt").is_file():
        return False

    for stem in ("encoder", "decoder", "joiner"):
        if not any(path.is_file() for path in model_dir.glob(f"{stem}*.onnx")):
            return False

    return True


def _configured_local_tts_voice(config: Config) -> str | None:
    requested = str(getattr(config, "tts_local_voice", "") or "").strip()
    if requested and requested.lower() not in LOCAL_TTS_AUTO_VOICE_IDS:
        return requested

    fallback = str(getattr(config, "tts_default_voice_id", "") or "").strip()
    if fallback and fallback.lower() not in LOCAL_TTS_AUTO_VOICE_IDS:
        return fallback
    return None


def _local_tts_model_dir(config: Config) -> Path | None:
    if not config.tts_local_model_path:
        return None
    return Path(config.tts_local_model_path).expanduser()


def model_status_for_backend(config: Config) -> str:
    backend = config.asr_backend

    if backend == "sherpa":
        model_name = str(config.sherpa_model_name).strip()
        decode_mode = config.resolved_sherpa_decode_mode or "streaming"

        parakeet_note = ""
        if "parakeet" in model_name.lower():
            if decode_mode == "offline_instant":
                parakeet_note = "; Parakeet offline instant mode configured"
            elif bool(getattr(config, "sherpa_enable_parakeet_streaming", False)):
                parakeet_note = "; Parakeet streaming mode enabled"
            else:
                parakeet_note = (
                    "; Parakeet selected but decode mode resolves to streaming "
                    "(set instant_mode=true or sherpa_decode_mode='offline_instant', "
                    "or enable sherpa_enable_parakeet_streaming for streaming)"
                )

        model_dir = Path(config.sherpa_model_dir).expanduser() if config.sherpa_model_dir else None
        if model_dir is None:
            model_dir = _sherpa_model_default_dir(config.sherpa_model_name)

        if _is_complete_sherpa_model_dir(model_dir):
            return f"present ({model_dir}; decode_mode={decode_mode}){parakeet_note}"

        return (
            f"missing ({model_dir}); will auto-download model "
            f"'{config.sherpa_model_name}' on first successful startup "
            f"after dependencies are installed (decode_mode={decode_mode}){parakeet_note}"
        )

    if backend == "nemo":
        return f"fetched from Hugging Face cache on first load (model_name={config.model_name})"

    if backend == "moonshine":
        if config.moonshine_model_dir:
            model_dir = Path(config.moonshine_model_dir).expanduser()
            if model_dir.is_dir():
                return f"configured local directory ({model_dir})"
            return f"configured local directory missing ({model_dir})"
        return "fetched lazily from Hugging Face on first load"

    return "unknown"


def local_tts_model_status(config: Config) -> str:
    model_dir = _local_tts_model_dir(config)
    if model_dir is None:
        return "missing (tts_local_model_path is not configured)"

    voice_id = _configured_local_tts_voice(config)
    valid, detail = validate_piper_voice_artifacts(model_dir, voice_id=voice_id)
    if not valid:
        return f"missing ({detail})"

    try:
        if model_dir.is_file():
            model_file = model_dir
        else:
            selected = voice_id or next(iter(installed_piper_voice_stems(model_dir)), None)
            model_file = model_dir / f"{selected}.onnx" if selected else model_dir
        sample_rate = piper_sample_rate_from_sidecar(model_file)
    except Exception:  # noqa: BLE001
        sample_rate = None

    if sample_rate is not None:
        return f"present ({detail}; path={model_dir})"
    return f"present ({detail}; path={model_dir})"


def install_hints_for_backend(backend: str) -> tuple[str, ...]:
    hints: list[str] = []

    if backend == "sherpa":
        if shutil.which("yay"):
            hints.append("Arch (AUR, recommended): yay -S --needed python-sherpa-onnx-bin")
            hints.append("Arch (AUR, alternate provider): yay -S --needed python-sherpa-onnx")
        elif shutil.which("paru"):
            hints.append("Arch (AUR, recommended): paru -S --needed python-sherpa-onnx-bin")
            hints.append("Arch (AUR, alternate provider): paru -S --needed python-sherpa-onnx")

        hints.extend(
            [
                "uv (project venv): uv sync --extra asr-sherpa",
                "pip (venv): python -m pip install sherpa-onnx",
            ]
        )
        return tuple(hints)

    if backend == "nemo":
        hints.extend(
            [
                "uv (project venv): uv sync --extra asr-nemo",
                "Arch GPU base: sudo pacman -S python-pytorch-cuda",
                "pip (venv): python -m pip install 'nemo-toolkit[asr]' torch",
            ]
        )
        return tuple(hints)

    if backend == "moonshine":
        hints.extend(
            [
                "uv (project venv): uv sync --extra asr-moonshine",
                "pip (venv): python -m pip install useful-moonshine-onnx",
            ]
        )
        return tuple(hints)

    return tuple(hints)


def build_backend_setup_report(config: Config) -> BackendSetupReport:
    backend_cls = get_backend_class(config.asr_backend)
    missing = tuple(backend_cls.dependency_errors())
    return BackendSetupReport(
        backend=config.asr_backend,
        missing_dependencies=missing,
        install_hints=install_hints_for_backend(config.asr_backend),
        model_status=model_status_for_backend(config),
    )


def build_local_tts_setup_report(config: Config) -> LocalTTSSetupReport:
    model_dir = _local_tts_model_dir(config)
    voice_id = _configured_local_tts_voice(config)
    binary_name = find_piper_binary()
    missing_artifacts: list[str] = []

    if model_dir is None:
        missing_artifacts.append("tts_local_model_path is not configured")
    else:
        valid, detail = validate_piper_voice_artifacts(model_dir, voice_id=voice_id)
        if not valid:
            missing_artifacts.append(detail)

    return LocalTTSSetupReport(
        binary_present=binary_name is not None,
        binary_name=binary_name,
        model_dir=model_dir,
        installed_voices=installed_piper_voice_stems(model_dir),
        missing_artifacts=tuple(missing_artifacts),
        model_status=local_tts_model_status(config),
    )


def format_missing_dependency_report(report: BackendSetupReport) -> str:
    lines = [
        f"Missing dependencies for backend '{report.backend}':",
    ]

    for error in report.missing_dependencies:
        lines.append(f"  - {error}")

    lines.append(f"Model status: {report.model_status}")

    if report.install_hints:
        lines.append("Install one of:")
        for hint in report.install_hints:
            lines.append(f"  * {hint}")

    lines.append("Then run: shuvoice setup")
    return "\n".join(lines)


def format_local_tts_report(report: LocalTTSSetupReport) -> str:
    lines = ["Local TTS (Piper):"]
    if report.binary_present:
        lines.append(f"  Binary: {report.binary_name}")
    else:
        lines.append("  Binary: missing")
        for hint in piper_install_hints():
            lines.append(f"    * {hint}")

    if report.model_dir is None:
        lines.append("  Model path: not configured")
    else:
        lines.append(f"  Model path: {report.model_dir}")

    lines.append(f"  Model status: {report.model_status}")

    if report.installed_voices:
        lines.append("  Installed voices: " + ", ".join(report.installed_voices))

    for detail in report.missing_artifacts:
        lines.append(f"  Missing: {detail}")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
# MeloTTS helpers                                                     #
# ------------------------------------------------------------------ #


def _melotts_venv_dir(config: Config) -> Path:
    """Return the MeloTTS venv directory from *config* or the default."""
    venv_path = getattr(config, "tts_melotts_venv_path", None)
    if venv_path:
        return Path(venv_path).expanduser()
    return Path(_DEFAULT_MELOTTS_VENV_DIR).expanduser()


def melotts_venv_valid(venv_dir: Path) -> bool:
    """Return ``True`` when *venv_dir* contains an executable Python."""
    python_bin = venv_dir / "bin" / "python"
    if not python_bin.exists():
        return False
    return bool(python_bin.stat().st_mode & 0o111)


def build_melotts_setup_report(config: Config) -> MeloTTSSetupReport:
    """Build a status report for the MeloTTS backend."""
    venv_dir = _melotts_venv_dir(config)
    venv_present = venv_dir.is_dir()
    python_ok = melotts_venv_valid(venv_dir) if venv_present else False

    from .tts_melotts import MeloTTSBackend  # noqa: PLC0415

    missing = tuple(MeloTTSBackend.dependency_errors(venv_path=str(venv_dir)))

    if not venv_present:
        model_status = f"not installed (venv missing: {venv_dir})"
    elif not python_ok:
        model_status = f"broken (venv python not executable: {venv_dir})"
    elif missing:
        model_status = f"incomplete ({'; '.join(missing)})"
    else:
        model_status = f"ready ({venv_dir})"

    return MeloTTSSetupReport(
        venv_present=venv_present,
        venv_dir=venv_dir,
        python_executable=python_ok,
        missing_dependencies=missing,
        model_status=model_status,
    )


def melotts_install_commands(venv_dir: Path | None = None) -> list[list[str]]:
    """Return the ordered list of commands to create and populate a MeloTTS venv.

    The sequence is idempotent: callers should skip individual steps
    when the venv already exists and is valid (see :func:`melotts_venv_valid`).
    """
    target = venv_dir or Path(_DEFAULT_MELOTTS_VENV_DIR).expanduser()
    python_bin = str(target / "bin" / "python")

    commands: list[list[str]] = []

    # Step 1: ensure Python 3.12 is available via uv
    commands.append(["uv", "python", "install", "3.12"])

    # Step 2: create the isolated venv with Python 3.12
    commands.append(["uv", "venv", "--python", "3.12", str(target)])

    # Step 3: install melotts into the venv
    commands.append([python_bin, "-m", "pip", "install", "melotts"])

    # Step 4: download the unidic dictionary required by MeloTTS
    commands.append([python_bin, "-m", "unidic", "download"])

    return commands


def format_melotts_report(report: MeloTTSSetupReport) -> str:
    """Format a human-readable MeloTTS setup status block."""
    lines = ["MeloTTS:"]
    lines.append(f"  Venv: {'present' if report.venv_present else 'missing'} ({report.venv_dir})")
    if report.venv_present:
        lines.append(f"  Python: {'executable' if report.python_executable else 'not executable'}")
    lines.append(f"  Status: {report.model_status}")

    if report.missing_dependencies:
        for dep in report.missing_dependencies:
            lines.append(f"  Missing: {dep}")

    return "\n".join(lines)
