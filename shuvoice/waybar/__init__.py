"""Waybar custom-module helper for ShuVoice status + controls.

Usage:
    shuvoice-waybar status
    shuvoice-waybar toggle-record
    shuvoice-waybar menu
    shuvoice-waybar launch-wizard
    shuvoice-waybar service-toggle

The command prints Waybar JSON on stdout for all commands so Waybar can
refresh immediately after click actions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any

from ..config import Config
from ..control import send_control_command
from .format import (
    build_waybar_payload as _build_waybar_payload_impl,
)
from .format import (
    config_info_lines as _config_info_lines_impl,
)
from .format import (
    sanitize_class as _sanitize_class_impl,
)
from .hyprland import detect_keybind as _detect_keybind_impl
from .systemd import (
    run_systemctl_user as _run_systemctl_user_impl,
)
from .systemd import (
    service_action as _service_action_impl,
)
from .systemd import (
    service_active_state as _service_active_state_impl,
)

DEFAULT_SERVICE = "shuvoice.service"
_STATUS_TIMEOUT_SEC = 0.35
_MENU_LAUNCHERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Omarchy/Walker first: common on Hyprland setups and supports dmenu mode.
    ("omarchy-launch-walker", ("omarchy-launch-walker", "--dmenu", "-p", "{prompt}")),
    ("walker", ("walker", "--dmenu", "-p", "{prompt}")),
    ("wofi", ("wofi", "--dmenu", "--prompt", "{prompt}")),
    ("rofi", ("rofi", "-dmenu", "-p", "{prompt}")),
    ("bemenu", ("bemenu", "-p", "{prompt}")),
    ("dmenu", ("dmenu", "-p", "{prompt}")),
)


def _run_systemctl_user(*args: str, timeout: float = 2.0) -> subprocess.CompletedProcess[str]:
    return _run_systemctl_user_impl(*args, timeout=timeout)


def _service_active_state(service: str) -> str:
    return _service_active_state_impl(service)


def _service_action(service: str, action: str):
    return _service_action_impl(service, action)


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


def config_info_lines(config: Config) -> list[str]:
    return _config_info_lines_impl(config)


def detect_keybind(command: str = "start") -> str | None:
    return _detect_keybind_impl(command)


def _sanitize_class(value: str) -> str:
    return _sanitize_class_impl(value)


def build_waybar_payload(
    state: str,
    *,
    config_lines: list[str] | None = None,
    service_state: str | None = None,
    control_error: str | None = None,
    action_error: str | None = None,
) -> dict[str, Any]:
    return _build_waybar_payload_impl(
        state,
        config_lines=config_lines,
        service_state=service_state,
        control_error=control_error,
        action_error=action_error,
    )


def _wait_for_control_socket(config: Config, timeout_sec: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            send_control_command("ping", config.control_socket, timeout=0.2)
            return True
        except Exception:
            time.sleep(0.08)
    return False


def _prompt_menu_choice(prompt: str, options: list[str]) -> str | None:
    menu_input = "\n".join(options) + "\n"

    for binary, template in _MENU_LAUNCHERS:
        if shutil.which(binary) is None:
            continue

        argv = [arg.format(prompt=prompt) for arg in template]
        try:
            result = subprocess.run(
                argv,
                input=menu_input,
                capture_output=True,
                text=True,
                check=False,
                timeout=20.0,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"{binary} failed: {e}") from e

        if result.returncode != 0:
            # User cancel is common here (Esc / click outside).
            return None

        selection = result.stdout.strip()
        return selection or None

    raise RuntimeError(
        "No menu launcher found (install/use omarchy-launch-walker, walker, wofi, rofi, bemenu, or dmenu)"
    )


def _action_menu(config: Config, service: str):
    runtime_state, _, _ = _query_runtime_state(config, service)
    service_state = _service_active_state(service)

    recording_label = "Stop recording" if runtime_state == "recording" else "Start recording"
    recording_command = "stop-record" if runtime_state == "recording" else "start-record"

    service_label = (
        "Stop service"
        if service_state in {"active", "activating", "reloading"}
        else "Start service"
    )

    options: list[tuple[str, str]] = [
        (recording_label, recording_command),
        ("Toggle recording", "toggle-record"),
        (service_label, "service-toggle"),
        ("Relaunch setup wizard", "launch-wizard"),
        ("Restart service (advanced)", "service-restart"),
    ]

    choice = _prompt_menu_choice("ShuVoice", [label for label, _ in options])
    if not choice:
        return

    action_map = {label: command for label, command in options}
    command = action_map.get(choice)
    if not command:
        return

    _perform_action(command, config, service)


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


def _launch_wizard_detached():
    try:
        subprocess.Popen(
            [sys.executable, "-m", "shuvoice", "--wizard"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:  # noqa: BLE001 - surfaced in Waybar tooltip
        raise RuntimeError(f"failed to launch wizard: {e}") from e


def _perform_action(command: str, config: Config, service: str):
    if command == "status":
        return

    if command == "menu":
        _action_menu(config, service)
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

    if command == "launch-wizard":
        _launch_wizard_detached()
        return

    if command == "service-start":
        _service_action(service, "start")
        if not _wait_for_control_socket(config):
            raise RuntimeError("control socket not ready after starting service")
        return

    if command == "service-stop":
        _service_action(service, "stop")
        return

    if command == "service-restart":
        _service_action(service, "restart")
        if not _wait_for_control_socket(config):
            raise RuntimeError("control socket not ready after restarting service")
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
            "menu",
            "toggle-record",
            "start-record",
            "stop-record",
            "launch-wizard",
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

    info = config_info_lines(config)
    ptt_keybind = detect_keybind("start")
    if ptt_keybind:
        info.append(f"PTT Key:  {ptt_keybind}")
    tts_keybind = detect_keybind("tts_speak")
    if tts_keybind:
        info.append(f"TTS Key:  {tts_keybind}")

    if os.environ.get("SHUVOICE_WAYBAR_DEBUG_METRICS", "").lower() in {"1", "true", "yes"}:
        try:
            metrics = send_control_command("metrics", config.control_socket, timeout=0.3)
            if metrics.startswith("OK "):
                info.append(f"Metrics:  {metrics[3:].strip()}")
        except Exception:
            info.append("Metrics:  unavailable")

    state, service_state, control_error = _query_runtime_state(config, args.service)
    payload = build_waybar_payload(
        state,
        config_lines=info,
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
