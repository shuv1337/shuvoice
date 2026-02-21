"""CLI entry point for shuvoice."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import shutil
import sys
from ctypes import CDLL
from typing import Callable


def _run_preflight(config) -> bool:
    """Check runtime prerequisites and print a human-readable report."""
    checks: list[tuple[str, bool, str]] = []

    def add_check(name: str, fn: Callable[[], str]):
        try:
            detail = fn()
            checks.append((name, True, detail))
        except Exception as e:
            checks.append((name, False, str(e)))

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

    def check_hotkey_backend() -> str:
        allowed = {"evdev", "ipc"}
        if config.hotkey_backend not in allowed:
            raise RuntimeError(
                f"Invalid hotkey_backend '{config.hotkey_backend}'. Allowed: {sorted(allowed)}"
            )
        return config.hotkey_backend

    def check_hotkey() -> str:
        from evdev import ecodes

        if not hasattr(ecodes, config.hotkey):
            raise RuntimeError(f"Unknown hotkey: {config.hotkey}")
        return config.hotkey

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
        from .asr import get_backend_class

        backend_cls = get_backend_class(config.asr_backend)
        errors = backend_cls.dependency_errors()
        if errors:
            raise RuntimeError("; ".join(errors))
        return f"{config.asr_backend} backend dependencies are importable"

    add_check("Python version", check_python)
    add_check("Import numpy", check_import("numpy"))
    add_check("Import sounddevice", check_import("sounddevice"))
    add_check("Import evdev", check_import("evdev"))
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
    add_check("Hotkey backend", check_hotkey_backend)

    if config.hotkey_backend == "evdev":
        add_check("Configured hotkey", check_hotkey)
    else:
        checks.append(("Configured hotkey", True, "skipped (backend=ipc)"))

    add_check("Output mode", check_output_mode)

    print("ShuVoice preflight checks")
    print("=" * 24)
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")

    ok = all(result for _, result, _ in checks)
    print("\nResult:", "READY" if ok else "NOT READY")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Streaming speech-to-text overlay for Hyprland",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download model artifacts for selected backend and exit",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check runtime dependencies and exit",
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List audio input devices and exit",
    )
    parser.add_argument(
        "--control",
        choices=["start", "stop", "toggle", "status", "ping"],
        default=None,
        help="Send a control command to a running ShuVoice instance and exit",
    )
    parser.add_argument(
        "--control-socket",
        default=None,
        help="Override control socket path (default: $XDG_RUNTIME_DIR/shuvoice/control.sock)",
    )
    parser.add_argument(
        "--asr-backend",
        choices=["nemo", "sherpa", "moonshine"],
        default=None,
        help="ASR backend selection (default: from config)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device (NeMo backend, default: from config)",
    )
    parser.add_argument(
        "--right-context",
        type=int,
        choices=[0, 1, 6, 13],
        default=None,
        help="NeMo streaming right context (0,1,6,13 => lower to higher latency/accuracy)",
    )
    parser.add_argument(
        "--sherpa-model-dir",
        default=None,
        help="Sherpa model directory containing tokens.txt + encoder/decoder/joiner ONNX files",
    )
    parser.add_argument(
        "--sherpa-provider",
        choices=["cpu", "cuda"],
        default=None,
        help="Sherpa execution provider (default: from config)",
    )
    parser.add_argument(
        "--sherpa-num-threads",
        type=int,
        default=None,
        help="Sherpa decoding threads (default: from config)",
    )
    parser.add_argument(
        "--sherpa-chunk-ms",
        type=int,
        default=None,
        help="Sherpa native chunk duration in milliseconds (default: from config)",
    )
    parser.add_argument(
        "--moonshine-model-name",
        default=None,
        help="Moonshine model name (for example: moonshine/base, moonshine/tiny)",
    )
    parser.add_argument(
        "--moonshine-model-dir",
        default=None,
        help="Local Moonshine model directory (encoder_model.onnx + decoder_model_merged.onnx)",
    )
    parser.add_argument(
        "--moonshine-model-precision",
        default=None,
        help="Moonshine model precision variant (default: from config)",
    )
    parser.add_argument(
        "--moonshine-chunk-ms",
        type=int,
        default=None,
        help="Moonshine native chunk duration in milliseconds (default: from config)",
    )
    parser.add_argument(
        "--moonshine-max-window-sec",
        type=float,
        default=None,
        help="Moonshine cumulative decode window in seconds (default: from config)",
    )
    parser.add_argument(
        "--moonshine-max-tokens",
        type=int,
        default=None,
        help="Moonshine max generated tokens per decode (default: from config)",
    )
    parser.add_argument(
        "--audio-device",
        default=None,
        help="Audio input device name or index (default: from config)",
    )
    parser.add_argument(
        "--input-gain",
        type=float,
        default=None,
        help="Multiply microphone PCM by this factor before ASR (default: from config)",
    )
    parser.add_argument(
        "--hotkey-backend",
        choices=["evdev", "ipc"],
        default=None,
        help="Hotkey backend: evdev or ipc (default: from config)",
    )
    parser.add_argument(
        "--hotkey",
        default=None,
        help="Hotkey name, e.g. KEY_RIGHTCTRL (default: from config)",
    )
    parser.add_argument(
        "--hotkey-device",
        default=None,
        help="Explicit /dev/input/eventX path for hotkey capture",
    )
    parser.add_argument(
        "--hotkey-listen-all-devices",
        action="store_true",
        help="Listen to all matching keyboard devices (can cause duplicate hotkey events)",
    )
    parser.add_argument(
        "--output-mode",
        choices=["final_only", "streaming_partial"],
        default=None,
        help="Text output mode (default: from config)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    journald = bool(os.environ.get("JOURNAL_STREAM"))
    log_format = (
        "%(levelname)s %(name)s: %(message)s"
        if journald
        else "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=log_format,
    )

    from .config import Config

    config = Config.load()

    if args.asr_backend:
        config.asr_backend = args.asr_backend
    if args.device:
        config.device = args.device
    if args.right_context is not None:
        config.right_context = int(args.right_context)
    if args.sherpa_model_dir is not None:
        config.sherpa_model_dir = args.sherpa_model_dir
    if args.sherpa_provider:
        config.sherpa_provider = args.sherpa_provider
    if args.sherpa_num_threads is not None:
        config.sherpa_num_threads = int(args.sherpa_num_threads)
    if args.sherpa_chunk_ms is not None:
        config.sherpa_chunk_ms = int(args.sherpa_chunk_ms)
    if args.moonshine_model_name is not None:
        config.moonshine_model_name = args.moonshine_model_name
    if args.moonshine_model_dir is not None:
        config.moonshine_model_dir = args.moonshine_model_dir
    if args.moonshine_model_precision is not None:
        config.moonshine_model_precision = args.moonshine_model_precision
    if args.moonshine_chunk_ms is not None:
        config.moonshine_chunk_ms = int(args.moonshine_chunk_ms)
    if args.moonshine_max_window_sec is not None:
        config.moonshine_max_window_sec = float(args.moonshine_max_window_sec)
    if args.moonshine_max_tokens is not None:
        config.moonshine_max_tokens = int(args.moonshine_max_tokens)
    if args.audio_device is not None:
        # Accept numeric indexes or raw device names
        config.audio_device = (
            int(args.audio_device) if str(args.audio_device).isdigit() else args.audio_device
        )
    if args.input_gain is not None:
        config.input_gain = float(args.input_gain)
    if args.hotkey_backend:
        config.hotkey_backend = args.hotkey_backend
    if args.hotkey:
        config.hotkey = args.hotkey
    if args.hotkey_device:
        config.hotkey_device = args.hotkey_device
    if args.hotkey_listen_all_devices:
        config.hotkey_listen_all_devices = True
    if args.output_mode:
        config.output_mode = args.output_mode
    if args.control_socket:
        config.control_socket = args.control_socket

    try:
        config.__post_init__()
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.list_audio_devices:
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            print("Audio devices:")
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    print(
                        f"[{idx}] {dev['name']} "
                        f"(in={dev['max_input_channels']}, "
                        f"default_sr={dev['default_samplerate']})"
                    )
        except Exception as e:
            print(f"ERROR: Could not list audio devices: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Socket control command mode (for Hyprland bind/bindr)
    if args.control:
        from .control import send_control_command

        try:
            response = send_control_command(args.control, config.control_socket)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        print(response)
        return

    if args.preflight:
        sys.exit(0 if _run_preflight(config) else 1)

    # --download-model: fetch backend model files (if supported) and exit
    if args.download_model:
        from .asr import get_backend_class

        backend_cls = get_backend_class(config.asr_backend)

        try:
            kwargs = {"model_name": config.model_name} if config.asr_backend == "nemo" else {}
            backend_cls.download_model(**kwargs)
        except (RuntimeError, ValueError, NotImplementedError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        print("Model downloaded successfully.")
        return

    # Load libgtk4-layer-shell BEFORE any gi imports (required by overlay/app)
    try:
        CDLL("libgtk4-layer-shell.so")
    except OSError:
        print(
            "ERROR: libgtk4-layer-shell.so not found.\nInstall it with: pacman -S gtk4-layer-shell",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from .app import ShuVoiceApp
    except ModuleNotFoundError as e:
        if e.name == "gi":
            print(
                "ERROR: Missing PyGObject (module 'gi').\n"
                "Install Python deps with: pip install -e .\n"
                "If that fails, install system packages: pacman -S python-gobject gtk4 gtk4-layer-shell",
                file=sys.stderr,
            )
            sys.exit(1)
        raise

    try:
        app = ShuVoiceApp(config)
        app.load_model()
    except (RuntimeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(app.run(None))


if __name__ == "__main__":
    main()
