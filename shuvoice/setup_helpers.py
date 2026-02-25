"""Setup and dependency guidance helpers."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .asr import get_backend_class
from .config import Config

DEPENDENCY_EXIT_CODE = 78
"""Exit code used when required backend dependencies are missing.

Used by the packaged systemd unit via ``RestartPreventExitStatus`` so
service startup does not loop forever on missing optional backend stacks.
"""


@dataclass(frozen=True)
class BackendSetupReport:
    backend: str
    missing_dependencies: tuple[str, ...]
    install_hints: tuple[str, ...]
    model_status: str


def _sherpa_model_default_dir() -> Path:
    backend_cls = get_backend_class("sherpa")
    default_name = getattr(
        backend_cls,
        "_DEFAULT_MODEL_NAME",
        "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
    )
    return Config.data_dir() / "models" / "sherpa" / str(default_name)


def _is_complete_sherpa_model_dir(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False

    if not (model_dir / "tokens.txt").is_file():
        return False

    for stem in ("encoder", "decoder", "joiner"):
        if not any(path.is_file() for path in model_dir.glob(f"{stem}*.onnx")):
            return False

    return True


def model_status_for_backend(config: Config) -> str:
    backend = config.asr_backend

    if backend == "sherpa":
        model_dir = Path(config.sherpa_model_dir).expanduser() if config.sherpa_model_dir else None
        if model_dir is None:
            model_dir = _sherpa_model_default_dir()

        if _is_complete_sherpa_model_dir(model_dir):
            return f"present ({model_dir})"

        return (
            f"missing ({model_dir}); will auto-download on first successful startup "
            "after dependencies are installed"
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


def install_hints_for_backend(backend: str) -> tuple[str, ...]:
    hints: list[str] = []

    if backend == "sherpa":
        if shutil.which("yay"):
            hints.append("Arch (AUR, recommended): yay -S --needed python-sherpa-onnx-bin")
            hints.append(
                "Arch (AUR, alternate provider): yay -S --needed python-sherpa-onnx"
            )
        elif shutil.which("paru"):
            hints.append("Arch (AUR, recommended): paru -S --needed python-sherpa-onnx-bin")
            hints.append(
                "Arch (AUR, alternate provider): paru -S --needed python-sherpa-onnx"
            )

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
