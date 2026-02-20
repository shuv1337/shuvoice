
import os
import stat
import time
import pytest
from pathlib import Path
from shuvoice.control import ControlServer, default_control_socket_path

def noop(*args, **kwargs):
    pass

@pytest.fixture
def temp_control_socket(tmp_path):
    socket_path = tmp_path / "control.sock"
    return str(socket_path)

def test_control_socket_permissions(temp_control_socket):
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
        # Wait for socket to be created
        for _ in range(50):
            if socket_path.exists():
                break
            time.sleep(0.05)

        assert socket_path.exists(), "Socket file not created"

        st = os.stat(socket_path)
        mode = st.st_mode

        # Check permissions: should be user-only access
        # Ensure group and others have NO permissions
        assert not (mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP)), f"Group has permissions: {oct(mode)}"
        assert not (mode & (stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH)), f"Others have permissions: {oct(mode)}"

        # Verify directory permissions (created by ControlServer.start if missing)
        # In this case tmp_path is created by pytest, but ControlServer.start attempts to create parent if missing.
        # Since tmp_path exists, it might just use it.
        # But let's check a nested path to verify creation logic.

    finally:
        server.stop()

def test_control_server_creates_secure_directory(tmp_path):
    # Use a nested path to force directory creation
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

        st_dir = os.stat(parent_dir)
        mode_dir = st_dir.st_mode

        assert not (mode_dir & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP)), f"Directory group has permissions: {oct(mode_dir)}"
        assert not (mode_dir & (stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH)), f"Directory others have permissions: {oct(mode_dir)}"
    finally:
        server.stop()

def test_default_socket_path_creation(tmp_path, monkeypatch):
    # Mock XDG_RUNTIME_DIR to point to tmp_path
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    # This function creates the directory immediately
    path = default_control_socket_path()

    assert path.name == "control.sock"
    assert path.parent.name == "shuvoice"
    assert path.parent.parent == tmp_path

    # Check directory permissions
    st_dir = os.stat(path.parent)
    mode_dir = st_dir.st_mode

    assert not (mode_dir & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP)), f"Default dir group has permissions: {oct(mode_dir)}"
    assert not (mode_dir & (stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH)), f"Default dir others have permissions: {oct(mode_dir)}"
