from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from shuvoice.waybar.systemd import service_action, service_active_state


def test_service_active_state_returns_unknown_on_timeout(monkeypatch):
    def boom(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="systemctl", timeout=1.0)

    monkeypatch.setattr("shuvoice.waybar.systemd.run_systemctl_user", boom)

    assert service_active_state("shuvoice.service") == "unknown"


def test_service_action_raises_with_detail(monkeypatch):
    monkeypatch.setattr(
        "shuvoice.waybar.systemd.run_systemctl_user",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr="boom", stdout=""),
    )

    with pytest.raises(RuntimeError, match="boom"):
        service_action("shuvoice.service", "restart")
