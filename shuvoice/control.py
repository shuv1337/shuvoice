"""Local control socket for Hyprland bind/bindr integration.

This provides a Unix-domain socket command channel so a running ShuVoice
instance can be controlled by short-lived CLI invocations, e.g.:

  shuvoice --control start
  shuvoice --control stop

Intended for Hyprland fallback hotkeys when evdev access is unavailable.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

VALID_COMMANDS = {"start", "stop", "toggle", "status", "ping"}


def default_control_socket_path() -> Path:
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
    base = runtime_dir / "shuvoice"

    old_umask = os.umask(0o077)
    try:
        base.mkdir(parents=True, exist_ok=True)
    finally:
        os.umask(old_umask)

    return base / "control.sock"


def resolve_control_socket_path(path: str | None) -> Path:
    if path:
        return Path(path)
    return default_control_socket_path()


class ControlServer:
    def __init__(
        self,
        socket_path: str | None,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_toggle: Callable[[], None],
        on_status: Callable[[], str],
    ):
        self.socket_path = resolve_control_socket_path(socket_path)
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_toggle = on_toggle
        self._on_status = on_status

        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._server: socket.socket | None = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        old_umask = os.umask(0o077)
        try:
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        finally:
            os.umask(old_umask)

        self._running.set()
        self._thread = threading.Thread(
            target=self._run,
            name="control-socket",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running.clear()

        # Wake accept() so the server can terminate promptly.
        try:
            send_control_command("ping", str(self.socket_path), timeout=0.2)
        except Exception:
            pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        self._cleanup_socket_file()

    def _cleanup_socket_file(self):
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()
        except Exception:
            log.debug("Failed to remove control socket %s", self.socket_path)

    def _run(self):
        self._cleanup_socket_file()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server = server

        try:
            old_umask = os.umask(0o077)
            try:
                server.bind(str(self.socket_path))
            finally:
                os.umask(old_umask)

            server.listen(8)
            server.settimeout(0.5)
            log.info("Control socket listening: %s", self.socket_path)

            while self._running.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._running.is_set():
                        log.exception("Control socket accept failed")
                    break

                with conn:
                    response = "ERROR invalid request"
                    try:
                        payload = conn.recv(1024)
                        command = payload.decode("utf-8", errors="replace").strip().lower()
                        response = self._handle_command(command)
                    except Exception as e:
                        response = f"ERROR {e}"

                    try:
                        conn.sendall((response + "\n").encode("utf-8"))
                    except OSError:
                        pass
        finally:
            try:
                server.close()
            except Exception:
                pass
            self._cleanup_socket_file()
            log.info("Control socket stopped")

    def _handle_command(self, command: str) -> str:
        if command == "start":
            self._on_start()
            return "OK started"
        if command == "stop":
            self._on_stop()
            return "OK stopped"
        if command == "toggle":
            self._on_toggle()
            return "OK toggled"
        if command == "status":
            return f"OK {self._on_status()}"
        if command == "ping":
            return "OK pong"
        return f"ERROR unknown command: {command}"


def send_control_command(
    command: str,
    socket_path: str | None = None,
    timeout: float = 1.5,
) -> str:
    command = command.strip().lower()
    if command not in VALID_COMMANDS:
        raise ValueError(f"Invalid control command: {command}")

    path = resolve_control_socket_path(socket_path)

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)

    try:
        client.connect(str(path))
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Control socket not found at {path}. Is shuvoice running?"
        ) from e
    except OSError as e:
        raise RuntimeError(f"Failed to connect to control socket {path}: {e}") from e

    with client:
        client.sendall((command + "\n").encode("utf-8"))
        try:
            client.shutdown(socket.SHUT_WR)
        except OSError:
            pass

        response = client.recv(4096).decode("utf-8", errors="replace").strip()
        if not response:
            raise RuntimeError("Empty response from control socket")
        if response.startswith("ERROR"):
            raise RuntimeError(response)
        return response
