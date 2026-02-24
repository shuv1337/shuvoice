"""Wizard command."""

from __future__ import annotations

import sys
from ctypes import CDLL


def run_welcome_wizard(*, force_reconfigure: bool = False) -> bool:
    """Launch the setup wizard. Returns True when the wizard completed."""
    try:
        CDLL("libgtk4-layer-shell.so")
    except OSError:
        print(
            "ERROR: libgtk4-layer-shell.so not found.\nInstall it with: pacman -S gtk4-layer-shell",
            file=sys.stderr,
        )
        return False

    from ...wizard import WelcomeWizard

    wizard = WelcomeWizard(force_reconfigure=force_reconfigure)
    wizard.run(None)
    return wizard.completed
