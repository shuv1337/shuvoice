"""Waybar custom-module helper for ShuVoice status + controls.

Usage:
    shuvoice-waybar status
    shuvoice-waybar toggle-record
    shuvoice-waybar service-toggle

The command prints Waybar JSON on stdout for all commands so Waybar can
refresh immediately after click actions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any

from .config import Config
from .control import send_control_command

DEFAULT_SERVICE = "shuvoice.service"
_STATUS_TIMEOUT_SEC = 0.35


def _run_systemctl_user(*args: str, timeout: float = 2.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _service_active_state(service: str) -> str:
    try:
        result = _run_systemctl_user("show", "--property=ActiveState", "--value", service)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"

    if result.returncode != 0:
        return "unknown"

    return (result.stdout.strip() or "unknown").lower()


def _service_action(service: str, action: str):
    try:
        result = _run_systemctl_user(action, service, timeout=3.0)
    except FileNotFoundError as e:
        raise RuntimeError("systemctl not found") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"systemctl {action} timed out") from e

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "unknown error"
        raise RuntimeError(f"systemctl {action} {service} failed: {detail}")


def _query_control_state(config: Config, timeout: float = _STATUS_TIMEOUT_SEC) -> str:
    response = send_control_command("status", config.control_socket, timeout=timeout)
    if response.startswith("OK "):
        return response[3:].strip()
    return response.strip()


def _query_runtime_state(config: Config, service: str) -> tuple[str, str | None, str | None]:
    try:
        state = _query_control_state(config)
        return state, None, None
    except Exception as e:  # noqa: BLE001 - surfaced in Waybar tooltip
        control_error = str(e)

    service_state = _service_active_state(service)
    if service_state in {"active", "activating", "deactivating", "reloading"}:
        return "starting", service_state, control_error
    if service_state == "failed":
        return "error:service_failed", service_state, control_error
    if service_state in {"inactive", "dead"}:
        return "stopped", service_state, control_error

    if "socket not found" in control_error.lower():
        return "stopped", service_state, control_error
    return "error:control_unreachable", service_state, control_error


def _sanitize_class(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")
    return cleaned or "unknown"


def build_waybar_payload(
    state: str,
    *,
    service_state: str | None = None,
    control_error: str | None = None,
    action_error: str | None = None,
) -> dict[str, Any]:
    base_state = state
    reason = ""
    if ":" in state:
        base_state, reason = state.split(":", 1)

    icons = {
        "recording": "",
        "processing": "",
        "idle": "",
        "starting": "",
        "stopped": "",
        "error": "",
    }

    labels = {
        "recording": "Recording",
        "processing": "Processing",
        "idle": "Ready",
        "starting": "Starting",
        "stopped": "Stopped",
        "error": "Error",
    }

    if base_state not in labels:
        base_state = "error"
        reason = reason or "unknown_state"

    lines = [f"ShuVoice: {labels[base_state]}"]

    if reason:
        lines.append(f"Reason: {reason}")
    if service_state and service_state != "unknown":
        lines.append(f"Service: {service_state}")
    if control_error and base_state in {"starting", "error"}:
        lines.append(f"Control: {control_error}")
    if action_error:
        lines.append(f"Action: {action_error}")

    lines.extend(
        [
            "",
            "Left click: toggle recording",
            "Middle click: toggle service",
            "Right click: restart service",
        ]
    )

    class_name = _sanitize_class(base_state)

    return {
        "text": icons[base_state],
        "alt": base_state,
        "class": class_name,
        "tooltip": "\n".join(lines),
    }


def _wait_for_control_socket(config: Config, timeout_sec: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            send_control_command("ping", config.control_socket, timeout=0.2)
            return True
        except Exception:
            time.sleep(0.08)
    return False


def _ensure_service_running(service: str):
    if _service_active_state(service) == "active":
        return
    _service_action(service, "start")


def _action_toggle_record(config: Config, service: str):
    try:
        state = _query_control_state(config, timeout=0.5)
    except Exception:
        _ensure_service_running(service)
        if not _wait_for_control_socket(config):
            raise RuntimeError("control socket not ready after starting service")
        send_control_command("start", config.control_socket, timeout=1.0)
        return

    if state == "recording":
        send_control_command("stop", config.control_socket, timeout=1.0)
    else:
        send_control_command("start", config.control_socket, timeout=1.0)


def _perform_action(command: str, config: Config, service: str):
    if command == "status":
        return

    if command == "toggle-record":
        _action_toggle_record(config, service)
        return

    if command == "start-record":
        _ensure_service_running(service)
        if not _wait_for_control_socket(config):
            raise RuntimeError("control socket not ready after starting service")
        send_control_command("start", config.control_socket, timeout=1.0)
        return

    if command == "stop-record":
        try:
            send_control_command("stop", config.control_socket, timeout=1.0)
        except Exception:
            # Service may already be stopped; treat as no-op.
            if _service_active_state(service) == "active":
                raise
        return

    if command == "service-start":
        _service_action(service, "start")
        return

    if command == "service-stop":
        _service_action(service, "stop")
        return

    if command == "service-restart":
        _service_action(service, "restart")
        return

    if command == "service-toggle":
        state = _service_active_state(service)
        if state in {"active", "activating", "reloading"}:
            _service_action(service, "stop")
        else:
            _service_action(service, "start")
        return

    raise RuntimeError(f"Unknown command: {command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Waybar status helper for ShuVoice")
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=[
            "status",
            "toggle-record",
            "start-record",
            "stop-record",
            "service-start",
            "service-stop",
            "service-restart",
            "service-toggle",
        ],
        help="Action to run before printing Waybar JSON",
    )
    parser.add_argument(
        "--service",
        default=os.environ.get("SHUVOICE_SERVICE", DEFAULT_SERVICE),
        help="systemd --user service name (default: shuvoice.service)",
    )
    args = parser.parse_args(argv)

    config = Config.load()

    action_error: str | None = None
    exit_code = 0

    try:
        _perform_action(args.command, config, args.service)
    except RuntimeError as e:
        action_error = str(e)
        exit_code = 1

    state, service_state, control_error = _query_runtime_state(config, args.service)
    payload = build_waybar_payload(
        state,
        service_state=service_state,
        control_error=control_error,
        action_error=action_error,
    )
    print(json.dumps(payload, ensure_ascii=False))

    if action_error:
        print(f"ERROR: {action_error}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
