"""Top-level CLI entry and command dispatcher."""

from __future__ import annotations

import logging
import os
import sys
from typing import Sequence

from .commands.audio import list_audio_devices
from .commands.common import load_effective_config
from .commands.config import config_effective, config_path, config_validate
from .commands.control import run_control
from .commands.diagnostics import diagnostics
from .commands.model import download_model
from .commands.preflight import run_preflight
from .commands.run import run_app, run_wizard_command
from .parser import apply_cli_overrides, create_parser, resolve_command

__all__ = ["apply_cli_overrides", "main"]


def _configure_logging(verbose: bool) -> None:
    journald = bool(os.environ.get("JOURNAL_STREAM"))
    log_format = (
        "%(levelname)s %(name)s: %(message)s"
        if journald
        else "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=log_format,
    )


def _load_config_or_exit(args):
    try:
        return load_effective_config(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    resolved, legacy_warnings = resolve_command(args, parser)
    for message in legacy_warnings:
        logging.warning(message)

    if resolved == "run":
        return run_app(args)

    if resolved == "wizard":
        return run_wizard_command()

    if resolved == "audio_list_devices":
        return list_audio_devices()

    if resolved == "config_path":
        return config_path()

    if resolved == "config_validate":
        return config_validate()

    if resolved == "config_effective":
        return config_effective()

    config = _load_config_or_exit(args)
    if config is None:
        return 1

    if resolved == "preflight":
        return 0 if run_preflight(config) else 1

    if resolved == "control":
        return run_control(args.control_action, config, wait_sec=float(args.control_wait_sec))

    if resolved == "model_download":
        return download_model(config)

    if resolved == "diagnostics":
        return diagnostics(config, json_output=bool(getattr(args, "json", False)))

    parser.error(f"Unknown command route: {resolved}")
    return 2
