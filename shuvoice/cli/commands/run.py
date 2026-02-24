"""Main runtime command."""

from __future__ import annotations

import sys
from ctypes import CDLL

from ...config import Config
from ..parser import apply_cli_overrides
from .wizard import run_welcome_wizard


def run_app(args) -> int:
    config = Config.load()
    apply_cli_overrides(args, config)

    try:
        config.__post_init__()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Load libgtk4-layer-shell BEFORE any gi imports (required by overlay/app)
    try:
        CDLL("libgtk4-layer-shell.so")
    except OSError:
        print(
            "ERROR: libgtk4-layer-shell.so not found.\nInstall it with: pacman -S gtk4-layer-shell",
            file=sys.stderr,
        )
        return 1

    from ...wizard_state import needs_wizard

    if needs_wizard():
        if not run_welcome_wizard(force_reconfigure=False):
            return 0  # User closed wizard without finishing.
        # Reload config so wizard selections take effect.
        config = Config.load()
        apply_cli_overrides(args, config)
        try:
            config.__post_init__()
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    try:
        from ...app import ShuVoiceApp
    except ModuleNotFoundError as exc:
        if exc.name == "gi":
            print(
                "ERROR: Missing PyGObject (module 'gi').\n"
                "Install Python deps with: uv sync\n"
                "If that fails, install system packages: pacman -S python-gobject gtk4 gtk4-layer-shell",
                file=sys.stderr,
            )
            return 1
        raise

    try:
        app = ShuVoiceApp(config)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Model loading happens asynchronously in do_activate() with a splash screen.
    ret = app.run(None)
    if app._model_load_failed:
        return 1
    return int(ret)


def run_wizard_command() -> int:
    run_welcome_wizard(force_reconfigure=True)
    return 0
