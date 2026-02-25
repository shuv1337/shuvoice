"""Main runtime command."""

from __future__ import annotations

import sys
from ctypes import CDLL

from ...asr import get_backend_class
from ...config import Config
from ...setup_helpers import (
    DEPENDENCY_EXIT_CODE,
    build_backend_setup_report,
    format_missing_dependency_report,
)
from ..parser import apply_cli_overrides
from .wizard import run_welcome_wizard


def _check_backend_dependencies(config: Config) -> bool:
    report = build_backend_setup_report(config)
    if not report.missing_dependencies:
        return True

    print(
        "ERROR: backend dependency check failed.\n" + format_missing_dependency_report(report),
        file=sys.stderr,
    )
    return False


def _check_backend_startup_guards(config: Config) -> bool:
    """Validate backend/runtime compatibility before GTK startup.

    This catches known fatal combinations (for example incompatible model/runtime
    pairs) early and emits actionable user-facing messages.
    """
    backend_cls = get_backend_class(config.asr_backend)

    for warning in backend_cls.startup_warnings(config, apply_fixes=True):
        print(f"WARNING: {warning}", file=sys.stderr)

    errors = backend_cls.startup_errors(config)
    if not errors:
        return True

    print("ERROR: backend startup guard failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return False


def run_app(args) -> int:
    config = Config.load()
    apply_cli_overrides(args, config)

    try:
        config.__post_init__()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return DEPENDENCY_EXIT_CODE

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
            return DEPENDENCY_EXIT_CODE

    if not _check_backend_dependencies(config):
        return DEPENDENCY_EXIT_CODE

    if not _check_backend_startup_guards(config):
        return DEPENDENCY_EXIT_CODE

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
        return DEPENDENCY_EXIT_CODE

    # Model loading happens asynchronously in do_activate() with a splash screen.
    ret = app.run(None)
    if app._model_load_failed:
        return DEPENDENCY_EXIT_CODE
    return int(ret)


def run_wizard_command() -> int:
    run_welcome_wizard(force_reconfigure=True)
    return 0
