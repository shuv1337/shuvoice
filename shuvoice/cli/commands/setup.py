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


def _detect_cuda_gpu() -> bool:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return True
    except Exception:  # noqa: BLE001
        pass

    return shutil.which("nvidia-smi") is not None


def _venv_install_commands(
    packages: list[str],
    *,
    upgrade: bool = False,
    no_binary: str | None = None,
    env_var: str | None = None,
) -> list[list[str]]:
    commands: list[list[str]] = []

    if shutil.which("uv"):
        uv_cmd = ["uv", "pip", "install"]
        if upgrade:
            uv_cmd.append("--upgrade")
        if no_binary:
            uv_cmd.extend(["--no-binary", no_binary])
        uv_cmd.extend(packages)
        if env_var:
            commands.append(["env", env_var, *uv_cmd])
        else:
            commands.append(uv_cmd)

    pip_cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        pip_cmd.append("--upgrade")
    if no_binary:
        pip_cmd.extend(["--no-binary", no_binary])
    pip_cmd.extend(packages)
    if env_var:
        commands.append(["env", env_var, *pip_cmd])
    else:
        commands.append(pip_cmd)

    return commands


def _auto_install_commands(backend: str, *, prefer_cuda: bool | None = None) -> list[list[str]]:
    commands: list[list[str]] = []

    if backend == "sherpa":
        if prefer_cuda is None:
            prefer_cuda = _detect_cuda_gpu()

        if prefer_cuda:
            commands.extend(
                [
                    # Prefer source provider first on CUDA hosts since it can
                    # be built as a CUDA-capable runtime.
                    ["yay", "-S", "--needed", "python-sherpa-onnx"],
                    ["paru", "-S", "--needed", "python-sherpa-onnx"],
                    ["yay", "-S", "--needed", "python-sherpa-onnx-bin"],
                    ["paru", "-S", "--needed", "python-sherpa-onnx-bin"],
                ]
            )
        else:
            commands.extend(
                [
                    ["yay", "-S", "--needed", "python-sherpa-onnx-bin"],
                    ["yay", "-S", "--needed", "python-sherpa-onnx"],
                    ["paru", "-S", "--needed", "python-sherpa-onnx-bin"],
                    ["paru", "-S", "--needed", "python-sherpa-onnx"],
                ]
            )

        if _running_in_venv():
            if prefer_cuda:
                commands.extend(
                    _venv_install_commands(
                        ["sherpa-onnx"],
                        upgrade=True,
                        no_binary="sherpa-onnx",
                        env_var="SHERPA_ONNX_CMAKE_ARGS=-DSHERPA_ONNX_ENABLE_GPU=ON",
                    )
                )
            commands.extend(_venv_install_commands(["sherpa-onnx"], upgrade=True))
        return commands

    if backend == "moonshine":
        if _running_in_venv():
            commands.extend(_venv_install_commands(["useful-moonshine-onnx"]))
        return commands

    if backend == "nemo":
        if _running_in_venv():
            commands.extend(_venv_install_commands(["nemo-toolkit[asr]", "torch"]))
        return commands

    return commands


def _attempt_auto_install(backend: str) -> bool:
    prefer_cuda = backend == "sherpa" and _detect_cuda_gpu()
    if prefer_cuda:
        print("Detected CUDA GPU; preferring CUDA-capable Sherpa runtime install path.")

    for command in _auto_install_commands(backend, prefer_cuda=prefer_cuda):
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

    # Evaluate startup diagnostics on a copy so we can report effective runtime
    # values (for example provider fallback) without mutating caller config.
    cfg_for_checks = Config(**{name: getattr(config, name) for name in Config.config_field_names()})

    startup_warnings = backend_cls.startup_warnings(cfg_for_checks, apply_fixes=True)
    if startup_warnings:
        for warning in startup_warnings:
            print(f"[WARN] {warning}")

    startup_errors = backend_cls.startup_errors(cfg_for_checks)

    if config.asr_backend == "sherpa":
        decode_mode = cfg_for_checks.resolved_sherpa_decode_mode or "streaming"
        print(f"[INFO] Sherpa decode mode: {decode_mode}")
        print(
            "[INFO] Sherpa provider: "
            f"requested={config.sherpa_provider} effective={cfg_for_checks.sherpa_provider}"
        )

        looks_like_parakeet = False
        detector = getattr(backend_cls, "_looks_like_parakeet_model", None)
        if callable(detector):
            looks_like_parakeet = bool(detector(cfg_for_checks))
        print(f"[INFO] Sherpa Parakeet model: {'yes' if looks_like_parakeet else 'no'}")
        if looks_like_parakeet:
            print(f"[INFO] Sherpa Parakeet runnable: {'yes' if not startup_errors else 'no'}")

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
