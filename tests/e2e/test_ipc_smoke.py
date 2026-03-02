from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from shuvoice.control import ControlServer

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]


def _run_control_cli(
    command: str, socket_path: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "shuvoice",
            "--control",
            command,
            "--control-socket",
            str(socket_path),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.fixture
def ipc_server(tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    socket_path = runtime_dir / "shuvoice" / "control.sock"
    state = {"recording": False, "tts": "idle"}

    def handle_tts(command: str) -> str:
        if command == "tts_speak":
            state["tts"] = "playing"
            return "OK tts speaking"
        if command == "tts_stop":
            state["tts"] = "idle"
            return "OK tts stopped"
        if command == "tts_status":
            return f"OK {state['tts']}"
        return "ERROR unsupported tts command"

    server = ControlServer(
        socket_path=str(socket_path),
        on_start=lambda: state.__setitem__("recording", True),
        on_stop=lambda: state.__setitem__("recording", False),
        on_toggle=lambda: state.__setitem__("recording", not state["recording"]),
        on_status=lambda: "recording" if state["recording"] else "idle",
        on_tts_command=handle_tts,
    )
    server.start()

    try:
        for _ in range(80):
            if socket_path.exists():
                break
            time.sleep(0.025)
        assert socket_path.exists(), "control socket did not appear"
        yield socket_path, state, runtime_dir
    finally:
        server.stop()


def test_control_cli_start_status_stop_flow(ipc_server):
    socket_path, state, runtime_dir = ipc_server

    env = dict(os.environ)
    env["XDG_RUNTIME_DIR"] = str(runtime_dir)
    env["XDG_CONFIG_HOME"] = str(runtime_dir / "xdg-config")

    result = _run_control_cli("status", socket_path, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK idle"

    result = _run_control_cli("start", socket_path, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK started"
    assert state["recording"] is True

    result = _run_control_cli("status", socket_path, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK recording"

    result = _run_control_cli("stop", socket_path, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK stopped"
    assert state["recording"] is False

    result = _run_control_cli("status", socket_path, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK idle"


def test_control_cli_toggle_and_ping_flow(ipc_server):
    socket_path, state, runtime_dir = ipc_server

    env = dict(os.environ)
    env["XDG_RUNTIME_DIR"] = str(runtime_dir)
    env["XDG_CONFIG_HOME"] = str(runtime_dir / "xdg-config")

    result = _run_control_cli("ping", socket_path, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK pong"

    result = _run_control_cli("toggle", socket_path, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK toggled"
    assert state["recording"] is True

    result = _run_control_cli("toggle", socket_path, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK toggled"
    assert state["recording"] is False


def test_control_cli_tts_round_trip(ipc_server):
    socket_path, state, runtime_dir = ipc_server

    env = dict(os.environ)
    env["XDG_RUNTIME_DIR"] = str(runtime_dir)
    env["XDG_CONFIG_HOME"] = str(runtime_dir / "xdg-config")

    result = _run_control_cli("tts_status", socket_path, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK idle"

    result = _run_control_cli("tts_speak", socket_path, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK tts speaking"
    assert state["tts"] == "playing"

    result = _run_control_cli("tts_stop", socket_path, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK tts stopped"
    assert state["tts"] == "idle"


def test_control_cli_returns_error_when_socket_not_running(tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    socket_path = runtime_dir / "shuvoice" / "control.sock"

    env = dict(os.environ)
    env["XDG_RUNTIME_DIR"] = str(runtime_dir)
    env["XDG_CONFIG_HOME"] = str(runtime_dir / "xdg-config")

    result = _run_control_cli("status", socket_path, env)
    assert result.returncode == 1
    assert "Control socket not found" in result.stderr
