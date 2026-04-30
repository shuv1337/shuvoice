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
    """Start or restart ``service`` after the wizard completes.

    Returns one of:
        - ``"restarted"``:  service was active and was restarted successfully
        - ``"started"``:    service was inactive/failed and was started successfully
        - ``"not_active"``: service was in an unsupported non-active state; nothing to do
        - ``"unavailable"``: systemctl is not usable in this environment
        - ``"failed"``:     start/restart was attempted but failed (see stderr)

    The function name is kept for compatibility, but first-run wizard launches
    need a start, not just a restart: the Waybar setup flow runs in a separate
    process while ``shuvoice.service`` may still be inactive.
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

    if state in {"active", "activating", "reloading"}:
        action = "restart"
        success_status = "restarted"
        success_verb = "Restarted"
    elif state in {"inactive", "failed", "dead", "deactivating"}:
        action = "start"
        success_status = "started"
        success_verb = "Started"
    else:
        return "not_active"

    try:
        service_action(service, action)
    except RuntimeError as exc:
        print(
            f"WARNING: failed to {action} {service} automatically: {exc}\n"
            f"         Run `systemctl --user {action} {service}` manually for "
            "wizard changes to take effect.",
            file=sys.stderr,
        )
        return "failed"

    print(f"✓ {success_verb} {service} so wizard changes take effect.")
    return success_status
