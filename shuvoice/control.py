"""Local control socket for Hyprland bind/bindr integration.

This provides a Unix-domain socket command channel so a running ShuVoice
instance can be controlled by short-lived CLI invocations, e.g.:

  shuvoice --control start
  shuvoice --control stop

Intended for Hyprland push-to-talk control via bind/bindr commands.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

VALID_COMMANDS = {"start", "stop", "toggle", "status", "ping", "metrics"}


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _allowed_control_roots() -> list[Path]:
    roots: list[Path] = [Path("/tmp").resolve()]
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        roots.insert(0, Path(runtime_dir).resolve())
    return roots


def _ensure_secure_directory(path: Path):
    old_umask = os.umask(0o077)
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    finally:
        os.umask(old_umask)

    try:
        path.chmod(0o700)
    except OSError:
        log.debug("Could not chmod %s to 0700", path)

    if path.stat().st_uid != os.getuid():
        raise RuntimeError(f"Control socket directory {path} is not owned by current user")


def default_control_socket_path() -> Path:
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")).resolve()
    base = runtime_dir / "shuvoice"
    _ensure_secure_directory(base)
    return base / "control.sock"


def resolve_control_socket_path(path: str | None) -> Path:
    if not path:
        return default_control_socket_path()

    raw = Path(path)
    if not raw.is_absolute():
        raise ValueError("Control socket path must be absolute")
    if path.endswith(os.sep) or (raw.exists() and raw.is_dir()):
        raise ValueError("Control socket path must be a .sock file, not a directory")
    if raw.suffix != ".sock":
        raise ValueError("Control socket path must end with '.sock'")

    parent = raw.parent.resolve()
    allowed_roots = _allowed_control_roots()
    if not any(_is_within(parent, root) for root in allowed_roots):
        roots_text = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(f"Control socket parent must live under: {roots_text}")

    _ensure_secure_directory(parent)
    return parent / raw.name


class ControlServer:
    def __init__(
        self,
        socket_path: str | None,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_toggle: Callable[[], None],
        on_status: Callable[[], str],
        on_metrics: Callable[[], str] | None = None,
    ):
        self.socket_path = resolve_control_socket_path(socket_path)
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_toggle = on_toggle
        self._on_status = on_status
        self._on_metrics = on_metrics or (lambda: "metrics unavailable")

        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._server: socket.socket | None = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        _ensure_secure_directory(self.socket_path.parent)

        self._running.set()
        self._thread = threading.Thread(
            target=self._run,
            name="control-socket",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running.clear()

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

                conn.settimeout(1.0)
                with conn:
                    response = "ERROR invalid request"
                    try:
                        payload = conn.recv(1024)
                        command = payload.decode("utf-8", errors="replace").strip().lower()
                        response = self._handle_command(command)
                    except Exception:
                        log.exception("Error handling control command")
                        response = "ERROR internal error"

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
        if command == "metrics":
            return f"OK {self._on_metrics()}"
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
        raise RuntimeError(f"Control socket not found at {path}. Is shuvoice running?") from e
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
