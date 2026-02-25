"""One-shot setup workflow for backend dependencies and model readiness."""

from __future__ import annotations

import shutil
import subprocess
import sys

from ...asr import get_backend_class
from ...config import Config
from ...setup_helpers import (
    DEPENDENCY_EXIT_CODE,
    build_backend_setup_report,
    format_missing_dependency_report,
)
from .preflight import run_preflight


def _download_model_for_backend(config: Config) -> None:
    backend_cls = get_backend_class(config.asr_backend)

    if not backend_cls.capabilities.supports_model_download:
        print("Model download: skipped (backend downloads lazily at runtime).")
        return

    kwargs: dict[str, object] = {}
    if config.asr_backend == "nemo":
        kwargs["model_name"] = config.model_name
    elif config.asr_backend == "sherpa":
        kwargs["model_name"] = config.sherpa_model_name
        kwargs["model_dir"] = config.sherpa_model_dir

    backend_cls.download_model(**kwargs)
    print("Model download: complete.")


def _running_in_venv() -> bool:
    return bool(getattr(sys, "base_prefix", sys.prefix) != sys.prefix)


def _auto_install_commands(backend: str) -> list[list[str]]:
    commands: list[list[str]] = []

    if backend == "sherpa":
        commands.extend(
            [
                ["yay", "-S", "--needed", "python-sherpa-onnx-bin"],
                ["yay", "-S", "--needed", "python-sherpa-onnx"],
                ["paru", "-S", "--needed", "python-sherpa-onnx-bin"],
                ["paru", "-S", "--needed", "python-sherpa-onnx"],
            ]
        )
        if _running_in_venv():
            commands.append([sys.executable, "-m", "pip", "install", "sherpa-onnx"])
        return commands

    if backend == "moonshine":
        if _running_in_venv():
            commands.append([sys.executable, "-m", "pip", "install", "useful-moonshine-onnx"])
        return commands

    if backend == "nemo":
        if _running_in_venv():
            commands.append([sys.executable, "-m", "pip", "install", "nemo-toolkit[asr]", "torch"])
        return commands

    return commands


def _attempt_auto_install(backend: str) -> bool:
    for command in _auto_install_commands(backend):
        executable = command[0]
        if executable not in {sys.executable} and not shutil.which(executable):
            continue

        print(f"Attempting install: {' '.join(command)}")
        proc = subprocess.run(command, check=False)
        if proc.returncode == 0:
            return True

    return False


def run_setup(
    config: Config,
    *,
    install_missing: bool = False,
    skip_model_download: bool = False,
    skip_preflight: bool = False,
) -> int:
    print("ShuVoice setup")
    print("=" * 13)
    print(f"Backend: {config.asr_backend}")

    report = build_backend_setup_report(config)
    print(f"Model status: {report.model_status}")

    if report.missing_dependencies:
        print("\n[FAIL] Backend dependencies")
        print(format_missing_dependency_report(report))

        if install_missing:
            print("\nAutomatic install requested.")
            if not _attempt_auto_install(config.asr_backend):
                print("Automatic install failed or no supported installer available.")

            report = build_backend_setup_report(config)

        if report.missing_dependencies:
            print("\nSetup incomplete: missing backend dependencies remain.")
            return DEPENDENCY_EXIT_CODE

    print("\n[PASS] Backend dependencies")

    backend_cls = get_backend_class(config.asr_backend)

    startup_warnings = backend_cls.startup_warnings(config, apply_fixes=False)
    if startup_warnings:
        for warning in startup_warnings:
            print(f"[WARN] {warning}")

    startup_errors = backend_cls.startup_errors(config)
    if startup_errors:
        print("\n[FAIL] Backend runtime compatibility")
        for error in startup_errors:
            print(f"  - {error}")
        return DEPENDENCY_EXIT_CODE

    print("[PASS] Backend runtime compatibility")

    if not skip_model_download:
        try:
            _download_model_for_backend(config)
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            print(f"ERROR: model download failed: {exc}", file=sys.stderr)
            return 1
    else:
        print("Model download: skipped (--skip-model-download).")

    if skip_preflight:
        print("Preflight: skipped (--skip-preflight).")
        print("\nSetup complete.")
        return 0

    print("\nRunning preflight checks...\n")
    ready = run_preflight(config)
    if not ready:
        return 1

    print("\nSetup complete.")
    return 0
