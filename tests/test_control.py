from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest

from shuvoice.control import (
    ControlServer,
    default_control_socket_path,
    resolve_control_socket_path,
)


def noop(*args, **kwargs):
    return None


@pytest.fixture
def temp_control_socket(tmp_path: Path):
    return str(tmp_path / "control.sock")


def _assert_user_only_mode(path: Path):
    mode = os.stat(path).st_mode
    assert not (mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP))
    assert not (mode & (stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH))


def test_control_socket_permissions(temp_control_socket: str):
    socket_path = Path(temp_control_socket)
    server = ControlServer(
        socket_path=str(socket_path),
        on_start=noop,
        on_stop=noop,
        on_toggle=noop,
        on_status=lambda: "ok",
    )

    server.start()
    try:
        for _ in range(50):
            if socket_path.exists():
                break
            time.sleep(0.05)

        assert socket_path.exists(), "Socket file not created"
        _assert_user_only_mode(socket_path)
    finally:
        server.stop()


def test_control_server_creates_secure_directory(tmp_path: Path):
    socket_path = tmp_path / "subdir" / "control.sock"
    server = ControlServer(
        socket_path=str(socket_path),
        on_start=noop,
        on_stop=noop,
        on_toggle=noop,
        on_status=lambda: "ok",
    )

    server.start()
    try:
        parent_dir = socket_path.parent
        assert parent_dir.exists()
        _assert_user_only_mode(parent_dir)
    finally:
        server.stop()


def test_default_socket_path_creation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    path = default_control_socket_path()

    assert path.name == "control.sock"
    assert path.parent.name == "shuvoice"
    assert path.parent.parent == tmp_path
    _assert_user_only_mode(path.parent)


def test_resolve_control_socket_path_rejects_relative_path():
    with pytest.raises(ValueError, match="absolute"):
        resolve_control_socket_path("control.sock")


def test_resolve_control_socket_path_rejects_directory(tmp_path: Path):
    sock_dir = tmp_path / "sockdir"
    sock_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="directory"):
        resolve_control_socket_path(str(sock_dir))


def test_resolve_control_socket_path_rejects_outside_allowed_roots():
    with pytest.raises(ValueError, match="under"):
        resolve_control_socket_path("/var/lib/shuvoice/control.sock")


def test_resolve_control_socket_path_requires_sock_suffix(tmp_path: Path):
    with pytest.raises(ValueError, match=".sock"):
        resolve_control_socket_path(str(tmp_path / "control.socket"))


def test_resolve_control_socket_path_accepts_runtime_dir(monkeypatch, tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))

    resolved = resolve_control_socket_path(str(runtime_dir / "nested" / "control.sock"))

    assert resolved == (runtime_dir / "nested" / "control.sock").resolve()
    assert resolved.parent.exists()
    _assert_user_only_mode(resolved.parent)


def test_resolve_control_socket_path_accepts_tmp_path(tmp_path: Path):
    candidate = tmp_path / "shuvoice" / "custom.sock"
    resolved = resolve_control_socket_path(str(candidate))

    assert resolved == candidate.resolve()
    assert resolved.parent.exists()
