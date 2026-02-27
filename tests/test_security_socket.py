import socket
import time
from pathlib import Path

import pytest

from shuvoice.control import ControlServer


def noop(*args, **kwargs):
    return None

def test_control_socket_timeout_dos_protection(tmp_path: Path):
    """
    Verify that the control server times out idle connections to prevent DoS.
    """
    socket_path = tmp_path / "control.sock"
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

        # Connect a malicious client that sends no data
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(socket_path))

        # We expect the server to close the connection or send an error after timeout (1.0s)
        # So we wait slightly longer than the timeout
        start_time = time.monotonic()
        client.settimeout(2.0)

        try:
            # The server should close the connection or send a timeout response
            # Since our implementation sends "ERROR timeout" then closes, we should receive that.
            data = client.recv(1024)
            elapsed = time.monotonic() - start_time

            # Verify we didn't wait indefinitely
            assert elapsed >= 1.0, "Server closed connection too quickly (should wait for timeout)"
            assert elapsed < 3.0, "Server took too long to timeout"

            response = data.decode("utf-8").strip()
            assert "ERROR timeout" in response or response == "", \
                f"Unexpected response: {response}"

        except socket.timeout:
            pytest.fail("Client timed out waiting for server to close connection")
        finally:
            client.close()

    finally:
        server.stop()
