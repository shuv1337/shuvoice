from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from shuvoice.selection import SelectionError, capture_selection


def test_capture_selection_prefers_primary(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if "--primary" in cmd:
            return SimpleNamespace(stdout="primary text")
        return SimpleNamespace(stdout="clipboard text")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert capture_selection() == "primary text"
    assert calls == [["wl-paste", "--no-newline", "--primary"]]


def test_capture_selection_falls_back_to_clipboard(monkeypatch):
    def fake_run(cmd, **_kwargs):
        if "--primary" in cmd:
            return SimpleNamespace(stdout="   ")
        return SimpleNamespace(stdout="clipboard text")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert capture_selection() == "clipboard text"


def test_capture_selection_raises_when_both_sources_empty(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda _cmd, **_kwargs: SimpleNamespace(stdout=""))

    with pytest.raises(SelectionError, match="No selected text"):
        capture_selection()


def test_capture_selection_handles_primary_timeout(monkeypatch):
    def fake_run(cmd, **_kwargs):
        if "--primary" in cmd:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=2)
        return SimpleNamespace(stdout="clipboard text")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert capture_selection() == "clipboard text"
