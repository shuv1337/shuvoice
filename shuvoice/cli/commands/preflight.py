"""Preflight runtime dependency checks."""

from __future__ import annotations

import importlib
import shutil
import sys
from ctypes import CDLL
from typing import Callable

from ...asr import get_backend_class
from ...config import Config


def run_preflight(config: Config) -> bool:
    """Check runtime prerequisites and print a human-readable report."""
    checks: list[tuple[str, bool, str]] = []

    def add_check(name: str, fn: Callable[[], str]) -> None:
        try:
            detail = fn()
            checks.append((name, True, detail))
        except Exception as exc:  # noqa: BLE001
            checks.append((name, False, str(exc)))

    def check_python() -> str:
        major, minor = sys.version_info[:2]
        if (major, minor) < (3, 10):
            raise RuntimeError(f"Unsupported Python {major}.{minor}; expected >=3.10")
        return f"Python {major}.{minor} is supported"

    def check_import(module: str) -> Callable[[], str]:
        def _inner() -> str:
            importlib.import_module(module)
            return f"import {module}"

        return _inner

    def check_binary(binary: str) -> Callable[[], str]:
        def _inner() -> str:
            path = shutil.which(binary)
            if not path:
                raise RuntimeError(f"{binary} not found in PATH")
            return path

        return _inner

    def check_layer_shell() -> str:
        CDLL("libgtk4-layer-shell.so")
        return "libgtk4-layer-shell.so loaded"

    def check_output_mode() -> str:
        allowed = {"final_only", "streaming_partial"}
        if config.output_mode not in allowed:
            raise RuntimeError(
                f"Invalid output_mode '{config.output_mode}'. Allowed: {sorted(allowed)}"
            )
        return config.output_mode

    def check_audio_device() -> str:
        if config.audio_device is None:
            return "default"

        import sounddevice as sd

        # Raises if the device cannot be resolved as input.
        sd.check_input_settings(device=config.audio_device, samplerate=config.sample_rate)
        return str(config.audio_device)

    def check_asr_stack() -> str:
        backend_cls = get_backend_class(config.asr_backend)
        errors = backend_cls.dependency_errors()
        if errors:
            raise RuntimeError("; ".join(errors))

        startup_errors = backend_cls.startup_errors(config)
        if startup_errors:
            raise RuntimeError("; ".join(startup_errors))

        startup_warnings = backend_cls.startup_warnings(config, apply_fixes=False)

        caps = backend_cls.capabilities
        chunking = caps.expected_chunking
        gpu_support = "yes" if caps.supports_gpu else "no"
        detail = (
            f"{config.asr_backend} deps OK "
            f"(supports_gpu={gpu_support}, expected_chunking={chunking}, "
            f"wants_raw_audio={caps.wants_raw_audio})"
        )
        if startup_warnings:
            detail += " | warnings: " + " | ".join(startup_warnings)
        return detail

    add_check("Python version", check_python)
    add_check("Import numpy", check_import("numpy"))
    add_check("Import sounddevice", check_import("sounddevice"))
    add_check("Import gi", check_import("gi"))
    add_check("Audio input device", check_audio_device)
    add_check("ASR dependencies", check_asr_stack)
    add_check("wtype binary", check_binary("wtype"))
    add_check("wl-copy binary", check_binary("wl-copy"))
    if config.preserve_clipboard:
        add_check("wl-paste binary", check_binary("wl-paste"))
    else:
        checks.append(("wl-paste binary", True, "skipped (preserve_clipboard=false)"))
    add_check("gtk4-layer-shell library", check_layer_shell)
    add_check("Output mode", check_output_mode)

    print("ShuVoice preflight checks")
    print("=" * 24)
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")

    ready = all(result for _, result, _ in checks)
    print("\nResult:", "READY" if ready else "NOT READY")
    return ready
