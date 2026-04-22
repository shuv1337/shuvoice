"""Wizard command."""

from __future__ import annotations

import sys
from ctypes import CDLL


_SHUVOICE_SERVICE = "shuvoice.service"


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


def maybe_restart_running_service(service: str = _SHUVOICE_SERVICE) -> str:
    """Restart ``service`` if it's currently active, so wizard changes take effect.

    Returns one of:
        - ``"restarted"``:  service was active and was restarted successfully
        - ``"not_active"``: service was not active; nothing to do
        - ``"unavailable"``: systemctl is not usable in this environment
        - ``"failed"``:     restart was attempted but failed (see stderr)

    Emits a user-friendly status line on stdout/stderr in all non-``not_active``
    cases so the user knows why config changes are (or aren't) picked up.
    """
    # Imported lazily to keep test isolation and avoid eager subprocess imports.
    from ...waybar.systemd import service_action, service_active_state

    try:
        state = service_active_state(service)
    except Exception:
        # Defensive: service_active_state already swallows common errors, but
        # if a stub/monkeypatch raises we still want a graceful fallback.
        print(
            f"WARNING: could not query {service} state; restart it manually "
            "for wizard changes to take effect.",
            file=sys.stderr,
        )
        return "unavailable"

    if state == "unknown":
        # Either systemctl isn't present or the unit isn't installed.
        # Nothing to restart; stay silent so manual `shuvoice` users aren't spammed.
        return "unavailable"

    if state not in {"active", "activating", "reloading"}:
        return "not_active"

    try:
        service_action(service, "restart")
    except RuntimeError as exc:
        print(
            f"WARNING: failed to restart {service} automatically: {exc}\n"
            f"         Run `systemctl --user restart {service}` manually for "
            "wizard changes to take effect.",
            file=sys.stderr,
        )
        return "failed"

    print(f"✓ Restarted {service} so wizard changes take effect.")
    return "restarted"
