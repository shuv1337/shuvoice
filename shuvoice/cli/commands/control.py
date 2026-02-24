"""Control socket CLI command."""

from __future__ import annotations

import sys
import time

from ...config import Config
from ...control import send_control_command


def run_control(command: str, config: Config, *, wait_sec: float = 2.0) -> int:
    status_before = ""
    if command == "toggle" and wait_sec > 0:
        try:
            status_before = send_control_command("status", config.control_socket)
        except Exception:  # noqa: BLE001
            status_before = ""

    try:
        response = send_control_command(command, config.control_socket)

        should_wait = False
        if wait_sec > 0:
            if command == "stop":
                should_wait = True
            elif command == "toggle":
                should_wait = status_before.strip().endswith("recording")

        if should_wait:
            deadline = time.monotonic() + float(wait_sec)
            while time.monotonic() < deadline:
                status = send_control_command("status", config.control_socket)
                state = status[3:].strip() if status.startswith("OK ") else status.strip()
                if state != "processing":
                    break
                time.sleep(0.05)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(response)
    return 0
