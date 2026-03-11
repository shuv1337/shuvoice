"""CLI parser and compatibility mapping."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shuvoice.config import Config

LEGACY_FLAG_WARNING = "This legacy flag is deprecated and will be removed in a future release."

CONTROL_COMMAND_CHOICES = [
    "start",
    "stop",
    "toggle",
    "status",
    "ping",
    "metrics",
    "tts_speak",
    "tts_pause",
    "tts_resume",
    "tts_toggle_pause",
    "tts_restart",
    "tts_stop",
    "tts_status",
]


def _add_runtime_overrides(parser: argparse.ArgumentParser) -> None:
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
        "--sherpa-model-name",
        default=None,
        help=(
            "Sherpa model archive name for auto-download when sherpa_model_dir is unset "
            "(default: from config)"
        ),
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
        help="Moonshine model name (for example: moonshine/tiny, moonshine/base)",
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
        "--moonshine-provider",
        choices=["cpu", "cuda"],
        default=None,
        help="Moonshine execution provider (default: from config)",
    )
    parser.add_argument(
        "--moonshine-onnx-threads",
        type=int,
        default=None,
        help="Moonshine ONNX intra-op threads (0 = auto, default: from config)",
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
        "--output-mode",
        choices=["final_only", "streaming_partial"],
        default=None,
        help="Text output mode (default: from config)",
    )
    parser.add_argument(
        "--control-socket",
        default=None,
        help="Override control socket path (default: $XDG_RUNTIME_DIR/shuvoice/control.sock)",
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Streaming speech-to-text overlay for Hyprland",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    # Legacy top-level compatibility flags (one cycle).
    parser.add_argument(
        "--download-model",
        action="store_true",
        help=f"[legacy] Equivalent to `shuvoice model download`. {LEGACY_FLAG_WARNING}",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=f"[legacy] Equivalent to `shuvoice preflight`. {LEGACY_FLAG_WARNING}",
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help=(f"[legacy] Equivalent to `shuvoice audio list-devices`. {LEGACY_FLAG_WARNING}"),
    )
    parser.add_argument(
        "--wizard",
        action="store_true",
        help=f"[legacy] Equivalent to `shuvoice wizard`. {LEGACY_FLAG_WARNING}",
    )
    parser.add_argument(
        "--control",
        choices=CONTROL_COMMAND_CHOICES,
        default=None,
        help=f"[legacy] Equivalent to `shuvoice control <cmd>`. {LEGACY_FLAG_WARNING}",
    )
    parser.add_argument(
        "--control-wait-sec",
        type=float,
        default=2.0,
        help=(
            "When sending stop/toggle, wait up to this many seconds for post-stop "
            "processing to finish (0 disables wait)."
        ),
    )

    _add_runtime_overrides(parser)

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the speech-to-text overlay")
    _add_runtime_overrides(run_parser)

    control_parser = subparsers.add_parser(
        "control", help="Send control command to running instance"
    )
    control_parser.add_argument(
        "control_command",
        choices=CONTROL_COMMAND_CHOICES,
        help="Control command",
    )
    control_parser.add_argument(
        "--control-wait-sec",
        type=float,
        default=2.0,
        help=(
            "When sending stop/toggle, wait up to this many seconds for post-stop "
            "processing to finish (0 disables wait)."
        ),
    )
    control_parser.add_argument(
        "--control-socket",
        default=None,
        help="Override control socket path (default: $XDG_RUNTIME_DIR/shuvoice/control.sock)",
    )

    preflight_parser = subparsers.add_parser("preflight", help="Run dependency and runtime checks")
    _add_runtime_overrides(preflight_parser)

    setup_parser = subparsers.add_parser(
        "setup",
        help="Bootstrap backend dependencies, model artifacts, and preflight checks",
    )
    _add_runtime_overrides(setup_parser)
    setup_parser.add_argument(
        "--install-missing",
        action="store_true",
        help="Attempt to install missing backend dependencies automatically",
    )
    setup_parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Skip model download step during setup",
    )
    setup_parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip final preflight checks",
    )
    setup_parser.add_argument(
        "--tts-local-voice",
        default=None,
        help=(
            "Curated Local Piper voice to download when tts_backend=local "
            "(for example: en_US-amy-medium)"
        ),
    )
    setup_parser.add_argument(
        "--tts-local-model-dir",
        default=None,
        help=(
            "Directory to store managed Local Piper voices when tts_backend=local "
            "(default: ~/.local/share/shuvoice/models/piper)"
        ),
    )
    setup_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive prompts and use recommended defaults where possible",
    )

    subparsers.add_parser("wizard", help="Launch the setup wizard")

    config_parser = subparsers.add_parser("config", help="Inspect and validate config")
    config_sub = config_parser.add_subparsers(dest="config_command")
    config_sub.add_parser("effective", help="Print merged effective config")
    config_sub.add_parser("path", help="Print active config file path")
    config_sub.add_parser("validate", help="Validate active config")
    config_set_parser = config_sub.add_parser("set", help="Set supported config keys")
    config_set_parser.add_argument(
        "key",
        choices=["typing_final_injection_mode"],
        help="Config key to set",
    )
    config_set_parser.add_argument(
        "value",
        choices=["auto", "clipboard", "direct"],
        help="New value for the selected key",
    )

    model_parser = subparsers.add_parser("model", help="Model management commands")
    _add_runtime_overrides(model_parser)
    model_sub = model_parser.add_subparsers(dest="model_command")
    model_sub.add_parser("download", help="Download model artifacts for active backend")

    audio_parser = subparsers.add_parser("audio", help="Audio utility commands")
    audio_sub = audio_parser.add_subparsers(dest="audio_command")
    audio_sub.add_parser("list-devices", help="List audio input devices")

    diagnostics_parser = subparsers.add_parser("diagnostics", help="Show runtime diagnostics")
    _add_runtime_overrides(diagnostics_parser)
    diagnostics_parser.add_argument(
        "--json",
        action="store_true",
        help="Output diagnostics as JSON",
    )

    return parser


def apply_cli_overrides(args: argparse.Namespace, config: Config) -> None:
    """Apply CLI argument overrides onto a Config instance."""
    if args.asr_backend:
        config.asr_backend = args.asr_backend
    if args.device:
        config.device = args.device
    if args.right_context is not None:
        config.right_context = int(args.right_context)
    if args.sherpa_model_dir is not None:
        config.sherpa_model_dir = args.sherpa_model_dir
    if args.sherpa_model_name is not None:
        config.sherpa_model_name = args.sherpa_model_name
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
    if args.moonshine_provider:
        config.moonshine_provider = args.moonshine_provider
    if args.moonshine_onnx_threads is not None:
        config.moonshine_onnx_threads = int(args.moonshine_onnx_threads)
    if args.audio_device is not None:
        config.audio_device = (
            int(args.audio_device) if str(args.audio_device).isdigit() else args.audio_device
        )
    if args.input_gain is not None:
        config.input_gain = float(args.input_gain)
    if args.output_mode:
        config.output_mode = args.output_mode
    if args.control_socket:
        config.control_socket = args.control_socket


def resolve_command(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[str, list[str]]:
    """Resolve subcommand + legacy flag compatibility routing."""
    warnings: list[str] = []

    command = args.command
    if command == "control":
        args.control_action = args.control_command
        return "control", warnings
    if command == "preflight":
        return "preflight", warnings
    if command == "setup":
        return "setup", warnings
    if command == "wizard":
        return "wizard", warnings
    if command == "run":
        return "run", warnings
    if command == "config":
        if not args.config_command:
            parser.error("config subcommand required: effective | path | validate | set")
        return f"config_{args.config_command}", warnings
    if command == "model":
        if args.model_command != "download":
            parser.error("model subcommand required: download")
        return "model_download", warnings
    if command == "audio":
        if args.audio_command != "list-devices":
            parser.error("audio subcommand required: list-devices")
        return "audio_list_devices", warnings
    if command == "diagnostics":
        return "diagnostics", warnings

    # Legacy path.
    legacy_flags = [
        bool(args.preflight),
        bool(args.wizard),
        bool(args.download_model),
        bool(args.list_audio_devices),
        bool(args.control),
    ]
    if sum(int(v) for v in legacy_flags) > 1:
        parser.error("legacy flags are mutually exclusive")

    if args.control:
        args.control_action = args.control
        warnings.append("`--control` is deprecated; use `shuvoice control <cmd>`")
        return "control", warnings
    if args.preflight:
        warnings.append("`--preflight` is deprecated; use `shuvoice preflight`")
        return "preflight", warnings
    if args.wizard:
        warnings.append("`--wizard` is deprecated; use `shuvoice wizard`")
        return "wizard", warnings
    if args.download_model:
        warnings.append("`--download-model` is deprecated; use `shuvoice model download`")
        return "model_download", warnings
    if args.list_audio_devices:
        warnings.append("`--list-audio-devices` is deprecated; use `shuvoice audio list-devices`")
        return "audio_list_devices", warnings

    return "run", warnings
