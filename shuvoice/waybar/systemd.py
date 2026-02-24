"""systemd interaction helpers for Waybar integration."""

from __future__ import annotations

import subprocess


def run_systemctl_user(*args: str, timeout: float = 2.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def service_active_state(service: str) -> str:
    try:
        result = run_systemctl_user("show", "--property=ActiveState", "--value", service)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"

    if result.returncode != 0:
        return "unknown"

    return (result.stdout.strip() or "unknown").lower()


def service_action(service: str, action: str) -> None:
    try:
        result = run_systemctl_user(action, service, timeout=3.0)
    except FileNotFoundError as exc:
        raise RuntimeError("systemctl not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"systemctl {action} timed out") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "unknown error"
        raise RuntimeError(f"systemctl {action} {service} failed: {detail}")
