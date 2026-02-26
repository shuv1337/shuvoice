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


def test_service_action_restart_stops_failed_service(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_run(*args, **_kwargs):
        calls.append(tuple(args))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("shuvoice.waybar.systemd.run_systemctl_user", fake_run)
    monkeypatch.setattr("shuvoice.waybar.systemd.service_active_state", lambda _svc: "failed")

    with pytest.raises(RuntimeError, match="restart loop"):
        service_action("shuvoice.service", "restart")

    assert ("restart", "shuvoice.service") in calls
    assert ("stop", "shuvoice.service") in calls
