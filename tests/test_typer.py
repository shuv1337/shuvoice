from __future__ import annotations

import subprocess
from types import SimpleNamespace

from shuvoice.typer import StreamingTyper


def test_run_retries_until_success(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(args, check, timeout, **kwargs):
        calls.append(args)
        if len(calls) < 3:
            raise subprocess.SubprocessError("temporary failure")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr("shuvoice.typer.subprocess.run", fake_run)

    typer = StreamingTyper(retry_attempts=3, retry_delay_ms=0)
    assert typer._run(["wtype", "--", "hello"], "type") is True
    assert len(calls) == 3


def test_update_partial_batches_backspaces(monkeypatch):
    typer = StreamingTyper(retry_attempts=1, retry_delay_ms=0)
    typer.last_partial_len = 120

    calls: list[tuple[list[str], str, int | None]] = []

    def fake_run(args: list[str], op: str, attempts: int | None = None) -> bool:
        calls.append((args, op, attempts))
        return True

    monkeypatch.setattr(typer, "_run", fake_run)

    typer.update_partial("abc")

    assert len(calls) == 4
    assert calls[0][0].count("BackSpace") == 50
    assert calls[1][0].count("BackSpace") == 50
    assert calls[2][0].count("BackSpace") == 20
    assert calls[3][0] == ["wtype", "--", "abc"]
    assert typer.last_partial_len == 3


def test_commit_final_falls_back_to_direct_type_and_restores_clipboard(monkeypatch):
    typer = StreamingTyper(preserve_clipboard=True)
    events: list[object] = []

    monkeypatch.setattr(typer, "_capture_clipboard", lambda: (True, "orig"))
    monkeypatch.setattr(typer, "_backspace_partial", lambda: events.append("backspace") or True)
    monkeypatch.setattr(
        typer,
        "_paste_via_clipboard",
        lambda text: events.append(("paste", text)) or False,
    )
    monkeypatch.setattr(
        typer,
        "_type_direct",
        lambda text: events.append(("direct", text)) or True,
    )
    monkeypatch.setattr(
        typer,
        "_restore_clipboard",
        lambda had, content: events.append(("restore", had, content)),
    )

    typer.commit_final("hello")

    assert ("direct", "hello") in events
    assert ("restore", True, "orig") in events
    assert typer.last_partial_len == 0


def test_restore_clipboard_clear_when_no_prior_content(monkeypatch):
    typer = StreamingTyper(preserve_clipboard=True)
    calls: list[list[str]] = []

    def fake_run(args: list[str], op: str, attempts: int | None = None) -> bool:
        calls.append(args)
        return True

    monkeypatch.setattr(typer, "_run", fake_run)

    typer._restore_clipboard(False, "")

    assert calls == [["wl-copy", "--clear"]]


def test_reset_clears_partial_state():
    typer = StreamingTyper()
    typer.last_partial_len = 99
    typer.reset()
    assert typer.last_partial_len == 0
