"""Diagnostics command."""

from __future__ import annotations

import json

from ...config import Config
from ...control import send_control_command


def diagnostics(config: Config, *, json_output: bool = False) -> int:
    payload: dict[str, str] = {}

    try:
        payload["status"] = send_control_command("status", config.control_socket)
    except Exception as exc:  # noqa: BLE001
        payload["status"] = f"ERROR: {exc}"

    try:
        payload["metrics"] = send_control_command("metrics", config.control_socket)
    except Exception as exc:  # noqa: BLE001
        payload["metrics"] = f"ERROR: {exc}"

    if json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")

    if payload["status"].startswith("ERROR"):
        return 1
    return 0
