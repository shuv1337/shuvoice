"""One-shot setup workflow for backend dependencies and model readiness."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from ...asr import get_backend_class
from ...config import CURRENT_CONFIG_VERSION, Config
from ...piper_setup import (
    attempt_piper_auto_install,
    curated_piper_voices,
    ensure_local_piper_ready,
    get_curated_piper_voice,
    managed_piper_model_dir,
    recommended_piper_voice,
)
from ...setup_helpers import (
    DEPENDENCY_EXIT_CODE,
    build_backend_setup_report,
    build_local_tts_setup_report,
    build_melotts_setup_report,
    format_local_tts_report,
    format_melotts_report,
    format_missing_dependency_report,
    melotts_install_commands,
    melotts_venv_valid,
)
from ...sherpa_cuda import prepare_cuda_runtime
from ...wizard_state import _upsert_tts_key
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


def _detect_cuda_architectures() -> str:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}{minor}"
    except Exception:  # noqa: BLE001
        pass

    return "89"


def _sherpa_cuda_cmake_args() -> str:
    arch = _detect_cuda_architectures()
    return (
        "-DSHERPA_ONNX_ENABLE_GPU=ON "
        f"-DCMAKE_CUDA_ARCHITECTURES={arch} "
        "-DCMAKE_C_FLAGS=-Wno-error=format-security "
        "-DCMAKE_CXX_FLAGS=-Wno-error=format-security"
    )


def _sherpa_cuda_compat_packages() -> list[str]:
    return [
        "nvidia-cublas-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu12",
        "nvidia-curand-cu12",
    ]


def _venv_install_commands(
    packages: list[str],
    *,
    upgrade: bool = False,
    no_binary: str | None = None,
    env_vars: dict[str, str] | None = None,
) -> list[list[str]]:
    commands: list[list[str]] = []

    env_prefix = ["env"]
    if env_vars:
        for key, value in env_vars.items():
            env_prefix.append(f"{key}={value}")

    if shutil.which("uv"):
        uv_cmd = ["uv", "pip", "install"]
        if upgrade:
            uv_cmd.append("--upgrade")
        if no_binary:
            uv_cmd.extend(["--no-binary", no_binary])
        uv_cmd.extend(packages)
        if env_vars:
            commands.append([*env_prefix, *uv_cmd])
        else:
            commands.append(uv_cmd)

    pip_cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        pip_cmd.append("--upgrade")
    if no_binary:
        pip_cmd.extend(["--no-binary", no_binary])
    pip_cmd.extend(packages)
    if env_vars:
        commands.append([*env_prefix, *pip_cmd])
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
                        env_vars={"SHERPA_ONNX_CMAKE_ARGS": _sherpa_cuda_cmake_args()},
                    )
                )
                commands.extend(
                    _venv_install_commands(
                        _sherpa_cuda_compat_packages(),
                        upgrade=True,
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


def _attempt_auto_install(backend: str, *, prefer_cuda: bool | None = None) -> bool:
    if prefer_cuda is None:
        prefer_cuda = backend == "sherpa" and _detect_cuda_gpu()

    if backend == "sherpa" and prefer_cuda:
        print("Detected CUDA GPU; preferring CUDA-capable Sherpa runtime install path.")

    for command in _auto_install_commands(backend, prefer_cuda=prefer_cuda):
        executable = command[0]
        if executable not in {sys.executable} and not shutil.which(executable):
            continue

        print(f"Attempting install: {' '.join(command)}")
        proc = subprocess.run(command, check=False)
        if proc.returncode != 0:
            continue

        if backend == "sherpa" and prefer_cuda:
            repaired, detail = prepare_cuda_runtime()
            print(f"Sherpa CUDA runtime repair: {detail}")
            if not repaired:
                continue

        return True

    return False


def _is_interactive_terminal() -> bool:
    stdin = getattr(sys.stdin, "isatty", lambda: False)()
    stdout = getattr(sys.stdout, "isatty", lambda: False)()
    return bool(stdin and stdout)


def _config_path_string(path: Path) -> str:
    expanded = path.expanduser().resolve()
    home = Path.home().resolve()
    try:
        relative = expanded.relative_to(home)
    except ValueError:
        return str(expanded)
    return f"~/{relative.as_posix()}"


def _ensure_config_file_for_patch() -> Path:
    config_path = Config.config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(f"config_version = {CURRENT_CONFIG_VERSION}\n")
    return config_path


def _persist_local_tts_selection(config: Config, *, model_dir: Path, voice_stem: str) -> None:
    config_path = _ensure_config_file_for_patch()
    model_path_value = _config_path_string(model_dir)

    _upsert_tts_key(config_path, "tts_backend", "local")
    _upsert_tts_key(config_path, "tts_default_voice_id", voice_stem)
    _upsert_tts_key(config_path, "tts_model_id", "piper")
    _upsert_tts_key(config_path, "tts_local_model_path", model_path_value)
    _upsert_tts_key(config_path, "tts_local_voice", voice_stem)

    config.tts_backend = "local"
    config.tts_default_voice_id = voice_stem
    config.tts_model_id = "piper"
    config.tts_local_model_path = model_path_value
    config.tts_local_voice = voice_stem
    config.__post_init__()


def _prompt_for_local_tts_voice(*, default_voice_id: str | None = None):
    options = curated_piper_voices()
    default_option = None
    if default_voice_id:
        try:
            default_option = get_curated_piper_voice(default_voice_id)
        except ValueError:
            default_option = None
    if default_option is None:
        default_option = recommended_piper_voice()

    print("\nChoose a Local Piper voice:")
    for index, option in enumerate(options, start=1):
        marker = " (default)" if option.id == default_option.id else ""
        print(f"  {index}. {option.label}{marker}")
        print(f"     {option.description}")

    while True:
        try:
            answer = input(f"Select voice [1-{len(options)}] (Enter for default): ").strip()
        except EOFError:
            return default_option
        if not answer:
            return default_option
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(options):
                return options[index - 1]
        print("Invalid selection. Please enter one of the listed numbers.")


def _choose_local_tts_voice(
    config: Config,
    *,
    explicit_voice_id: str | None,
    non_interactive: bool,
):
    if explicit_voice_id:
        return get_curated_piper_voice(explicit_voice_id)

    current_voice = str(getattr(config, "tts_local_voice", "") or "").strip()
    if not current_voice:
        current_voice = str(getattr(config, "tts_default_voice_id", "") or "").strip()

    if not non_interactive and _is_interactive_terminal():
        return _prompt_for_local_tts_voice(default_voice_id=current_voice)

    if current_voice:
        try:
            return get_curated_piper_voice(current_voice)
        except ValueError:
            pass
    return recommended_piper_voice()


def _run_local_tts_setup(
    config: Config,
    *,
    install_missing: bool,
    skip_model_download: bool,
    tts_local_voice: str | None,
    tts_local_model_dir: str | None,
    non_interactive: bool,
) -> int:
    print("\nLocal TTS backend: local")
    report = build_local_tts_setup_report(config)
    print(format_local_tts_report(report))

    if not report.binary_present:
        print("\n[FAIL] Local Piper runtime")
        if install_missing:
            print("Automatic install requested for Local Piper.")
            if attempt_piper_auto_install():
                print("Local Piper install: complete.")
            else:
                print("Local Piper install: failed or no supported installer available.")
            report = build_local_tts_setup_report(config)
            print(format_local_tts_report(report))

        if not report.binary_present:
            print("\nSetup incomplete: Local Piper runtime is still missing.")
            return DEPENDENCY_EXIT_CODE

    print("\n[PASS] Local Piper runtime")

    target_dir = (
        Path(tts_local_model_dir).expanduser()
        if tts_local_model_dir
        else Path(config.tts_local_model_path).expanduser()
        if config.tts_local_model_path
        else managed_piper_model_dir()
    )

    needs_voice_download = bool(
        tts_local_voice or tts_local_model_dir or not report.model_dir or report.missing_artifacts
    )

    if skip_model_download:
        print("Local Piper voice download: skipped (--skip-model-download).")
        if report.missing_artifacts:
            print("\nSetup incomplete: Local Piper voice artifacts are missing.")
            return DEPENDENCY_EXIT_CODE
        return 0

    if not needs_voice_download:
        print("Local Piper voice download: skipped (configured voice artifacts already present).")
        return 0

    selected_voice = _choose_local_tts_voice(
        config,
        explicit_voice_id=tts_local_voice,
        non_interactive=non_interactive,
    )
    print(f"Selected Local Piper voice: {selected_voice.label}")
    print(f"Managed voice directory: {target_dir}")

    result = ensure_local_piper_ready(
        selected_voice,
        model_dir=target_dir,
        auto_install_missing=install_missing,
        progress_callback=lambda _fraction, message: print(message),
        cancel_check=None,
    )

    if result.status == "skipped_missing_deps":
        print(f"Local Piper setup: {result.message}")
        print("\nSetup incomplete: Local Piper dependencies remain missing.")
        return DEPENDENCY_EXIT_CODE

    if result.status == "cancelled":
        print(f"Local Piper setup cancelled: {result.message}")
        return 1

    if result.status == "error":
        print(f"Local Piper setup failed: {result.message}")
        return 1

    _persist_local_tts_selection(config, model_dir=result.model_dir, voice_stem=result.voice.stem)
    print(f"Local Piper setup: {result.message}")

    report = build_local_tts_setup_report(config)
    print(format_local_tts_report(report))
    if report.missing_artifacts:
        print("\nSetup incomplete: Local Piper artifacts are still incomplete.")
        return DEPENDENCY_EXIT_CODE

    return 0


def _run_melotts_setup(
    config: Config,
    *,
    install_missing: bool,
) -> int:
    """Set up the MeloTTS isolated venv and report status."""
    print("\nTTS backend: melotts")
    report = build_melotts_setup_report(config)
    print(format_melotts_report(report))

    if report.missing_dependencies and install_missing:
        venv_dir = report.venv_dir
        already_valid = melotts_venv_valid(venv_dir)

        commands = melotts_install_commands(venv_dir)
        for command in commands:
            # Skip venv creation steps when venv already exists and is valid
            if already_valid and command[0] == "uv" and len(command) > 1:
                if command[1] == "python":
                    # uv python install — always safe to run (idempotent)
                    pass
                elif command[1] == "venv":
                    print(f"  Skipping venv creation (already valid): {venv_dir}")
                    continue

            executable = command[0]
            if executable not in {sys.executable} and not shutil.which(executable):
                print(f"  Skipping (executable not found): {executable}")
                continue

            print(f"  Running: {' '.join(command)}")
            proc = subprocess.run(command, check=False)
            if proc.returncode != 0:
                print(f"  Command failed (exit {proc.returncode}): {' '.join(command)}")
                break

        # Re-check status after install
        report = build_melotts_setup_report(config)
        print(format_melotts_report(report))

    if report.missing_dependencies:
        print("\n[FAIL] MeloTTS backend")
        print("Setup incomplete: MeloTTS venv is not ready.")
        print("Install hints:")
        for cmd in melotts_install_commands(report.venv_dir):
            print(f"  $ {' '.join(cmd)}")
        return DEPENDENCY_EXIT_CODE

    print("\n[PASS] MeloTTS backend")
    return 0


def run_setup(
    config: Config,
    *,
    install_missing: bool = False,
    skip_model_download: bool = False,
    skip_preflight: bool = False,
    tts_local_voice: str | None = None,
    tts_local_model_dir: str | None = None,
    non_interactive: bool = False,
) -> int:
    print("ShuVoice setup")
    print("=" * 13)
    print(f"ASR backend: {config.asr_backend}")
    print(f"TTS backend: {config.tts_backend}")

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

    cfg_for_checks = Config(**{name: getattr(config, name) for name in Config.config_field_names()})

    startup_warnings = backend_cls.startup_warnings(cfg_for_checks, apply_fixes=True)
    if (
        install_missing
        and config.asr_backend == "sherpa"
        and config.sherpa_provider == "cuda"
        and startup_warnings
    ):
        print("\n[WARN] Sherpa CUDA runtime is not ready; attempting repair/install.")
        if _attempt_auto_install(config.asr_backend, prefer_cuda=True):
            backend_cls = get_backend_class(config.asr_backend)
            cfg_for_checks = Config(
                **{name: getattr(config, name) for name in Config.config_field_names()}
            )
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

    if config.tts_backend == "local":
        local_setup_code = _run_local_tts_setup(
            config,
            install_missing=install_missing,
            skip_model_download=skip_model_download,
            tts_local_voice=tts_local_voice,
            tts_local_model_dir=tts_local_model_dir,
            non_interactive=non_interactive,
        )
        if local_setup_code != 0:
            return local_setup_code

    if config.tts_backend == "melotts":
        melotts_setup_code = _run_melotts_setup(
            config,
            install_missing=install_missing,
        )
        if melotts_setup_code != 0:
            return melotts_setup_code

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
