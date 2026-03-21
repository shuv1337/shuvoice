from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

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
    typer.last_partial_text = "x" * 120

    calls: list[tuple[list[str], str, int | None]] = []

    def fake_run(args: list[str], op: str, attempts: int | None = None) -> bool:
        calls.append((args, op, attempts))
        return True

    monkeypatch.setattr(typer, "_prefer_xdotool", lambda: False)
    monkeypatch.setattr(typer, "_prefer_ydotool", lambda: False)
    monkeypatch.setattr(typer, "_run", fake_run)

    typer.update_partial("abc")

    assert len(calls) == 4
    assert calls[0][0].count("BackSpace") == 50
    assert calls[1][0].count("BackSpace") == 50
    assert calls[2][0].count("BackSpace") == 20
    assert calls[3][0] == ["wtype", "--", "abc"]
    assert typer.last_partial_len == 3


def test_update_partial_uses_common_prefix_for_small_suffix_edits(monkeypatch):
    typer = StreamingTyper(retry_attempts=1, retry_delay_ms=0)
    typer.last_partial_text = "hello world"
    typer.last_partial_len = len(typer.last_partial_text)

    calls: list[list[str]] = []

    def fake_run(args: list[str], op: str, attempts: int | None = None) -> bool:
        calls.append(args)
        return True

    monkeypatch.setattr(typer, "_prefer_xdotool", lambda: False)
    monkeypatch.setattr(typer, "_prefer_ydotool", lambda: False)
    monkeypatch.setattr(typer, "_run", fake_run)

    typer.update_partial("hello there")

    # Only remove the differing suffix ("world" -> "there"), not full retype.
    assert len(calls) == 2
    assert calls[0].count("BackSpace") == 5
    assert calls[1] == ["wtype", "--", "there"]


def test_commit_final_falls_back_to_direct_type_and_restores_clipboard(monkeypatch):
    typer = StreamingTyper(preserve_clipboard=True, final_injection_mode="clipboard")
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


def test_commit_final_auto_mode_prefers_direct_when_watchers_detected(monkeypatch):
    typer = StreamingTyper(final_injection_mode="auto", preserve_clipboard=True)
    typer.last_partial_len = 7
    typer.last_partial_text = "partial"
    events: list[object] = []

    monkeypatch.setattr(typer, "_prefer_xdotool", lambda: False)
    monkeypatch.setattr(typer, "_detect_clipboard_watchers", lambda: True)
    monkeypatch.setattr(typer, "update_partial", lambda text: events.append(("update", text)))
    monkeypatch.setattr(
        typer, "_capture_clipboard", lambda: events.append("capture") or (True, "x")
    )
    monkeypatch.setattr(typer, "_backspace_partial", lambda: events.append("backspace") or True)
    monkeypatch.setattr(
        typer,
        "_paste_via_clipboard",
        lambda text: events.append(("paste", text)) or True,
    )

    typer.commit_final("hello")

    assert events == [("update", "hello")]
    assert typer.last_partial_len == 0
    assert typer.last_partial_text == ""


def test_commit_final_auto_mode_uses_clipboard_when_no_watchers(monkeypatch):
    typer = StreamingTyper(final_injection_mode="auto", preserve_clipboard=False)
    events: list[object] = []

    monkeypatch.setattr(typer, "_detect_clipboard_watchers", lambda: False)
    monkeypatch.setattr(typer, "_backspace_partial", lambda: events.append("backspace") or True)
    monkeypatch.setattr(
        typer,
        "_paste_via_clipboard",
        lambda text: events.append(("paste", text)) or True,
    )
    monkeypatch.setattr(
        typer,
        "_type_direct",
        lambda text: events.append(("direct", text)) or True,
    )

    typer.commit_final("hello")

    assert "backspace" in events
    assert ("paste", "hello") in events
    assert ("direct", "hello") not in events


def test_paste_via_clipboard_applies_settle_delay(monkeypatch):
    typer = StreamingTyper(clipboard_settle_delay_ms=40, retry_attempts=1, retry_delay_ms=0)
    calls: list[list[str]] = []
    sleeps: list[float] = []

    monkeypatch.setattr(typer, "_prefer_xdotool", lambda: False)
    monkeypatch.setattr(typer, "_prefer_ydotool", lambda: False)
    monkeypatch.setattr(typer, "_ydotool_installed", lambda: False)
    monkeypatch.setattr(
        typer,
        "_run",
        lambda args, op, attempts=None: calls.append(args) or True,
    )
    monkeypatch.setattr("shuvoice.typer.time.sleep", lambda value: sleeps.append(value))

    assert typer._paste_via_clipboard("hello") is True

    assert calls[0] == ["wl-copy", "--", "hello"]
    assert calls[1] == ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"]
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(0.04)


def test_type_direct_prefers_ydotool_for_xwayland(monkeypatch):
    typer = StreamingTyper(retry_attempts=1, retry_delay_ms=0)
    calls: list[list[str]] = []

    monkeypatch.setattr(typer, "_prefer_xdotool", lambda: False)
    monkeypatch.setattr(typer, "_prefer_ydotool", lambda: True)
    monkeypatch.setattr(
        typer,
        "_run",
        lambda args, op, attempts=None: calls.append(args) or True,
    )

    assert typer._type_direct("hello") is True
    assert calls == [
        ["ydotool", "type", "--key-delay", "0", "--key-hold", "0", "--", "hello"]
    ]


def test_type_direct_falls_back_to_ydotool_when_wtype_fails(monkeypatch):
    typer = StreamingTyper(retry_attempts=1, retry_delay_ms=0)
    calls: list[list[str]] = []

    monkeypatch.setattr(typer, "_prefer_xdotool", lambda: False)
    monkeypatch.setattr(typer, "_prefer_ydotool", lambda: False)
    monkeypatch.setattr(typer, "_ydotool_installed", lambda: True)
    monkeypatch.setattr(typer, "_active_window_is_xwayland", lambda: False)

    def fake_run(args, op, attempts=None):
        calls.append(args)
        return args[0] == "ydotool"

    monkeypatch.setattr(typer, "_run", fake_run)

    assert typer._type_direct("hello") is True
    assert calls == [
        ["wtype", "--", "hello"],
        ["ydotool", "type", "--key-delay", "0", "--key-hold", "0", "--", "hello"],
    ]


def test_paste_via_clipboard_prefers_ydotool_for_xwayland(monkeypatch):
    typer = StreamingTyper(clipboard_settle_delay_ms=40, retry_attempts=1, retry_delay_ms=0)
    calls: list[list[str]] = []
    sleeps: list[float] = []

    monkeypatch.setattr(typer, "_prefer_xdotool", lambda: False)
    monkeypatch.setattr(typer, "_prefer_ydotool", lambda: True)
    monkeypatch.setattr(
        typer,
        "_run",
        lambda args, op, attempts=None: calls.append(args) or True,
    )
    monkeypatch.setattr("shuvoice.typer.time.sleep", lambda value: sleeps.append(value))

    assert typer._paste_via_clipboard("hello") is True
    assert calls == [
        ["wl-copy", "--", "hello"],
        ["ydotool", "key", "-d", "0", "29:1", "47:1", "47:0", "29:0"],
    ]
    assert sleeps == [pytest.approx(0.04)]


def test_type_direct_prefers_xdotool_for_xwayland(monkeypatch):
    typer = StreamingTyper(retry_attempts=1, retry_delay_ms=0)
    calls: list[list[str]] = []

    monkeypatch.setattr(typer, "_prefer_xdotool", lambda: True)
    monkeypatch.setattr(typer, "_active_xdotool_window_id", lambda: "123")
    monkeypatch.setattr(
        typer,
        "_run",
        lambda args, op, attempts=None: calls.append(args) or True,
    )

    assert typer._type_direct("hello") is True
    assert calls == [["xdotool", "type", "--clearmodifiers", "--delay", "0", "--window", "123", "hello"]]


def test_paste_via_clipboard_prefers_xdotool_for_xwayland(monkeypatch):
    typer = StreamingTyper(clipboard_settle_delay_ms=40, retry_attempts=1, retry_delay_ms=0)
    calls: list[list[str]] = []
    sleeps: list[float] = []

    monkeypatch.setattr(typer, "_prefer_xdotool", lambda: True)
    monkeypatch.setattr(typer, "_active_xdotool_window_id", lambda: "123")
    monkeypatch.setattr(
        typer,
        "_run",
        lambda args, op, attempts=None: calls.append(args) or True,
    )
    monkeypatch.setattr("shuvoice.typer.time.sleep", lambda value: sleeps.append(value))

    assert typer._paste_via_clipboard("hello") is True
    assert calls == [
        ["wl-copy", "--", "hello"],
        ["xdotool", "key", "--clearmodifiers", "--delay", "0", "--window", "123", "ctrl+v"],
    ]
    assert sleeps == [pytest.approx(0.04)]


def test_commit_final_auto_mode_prefers_clipboard_for_xwayland(monkeypatch):
    typer = StreamingTyper(final_injection_mode="auto", preserve_clipboard=False)
    events: list[object] = []

    monkeypatch.setattr(typer, "_prefer_xdotool", lambda: True)
    monkeypatch.setattr(typer, "_detect_clipboard_watchers", lambda: True)
    monkeypatch.setattr(typer, "_backspace_partial", lambda: events.append("backspace") or True)
    monkeypatch.setattr(
        typer,
        "_paste_via_clipboard",
        lambda text: events.append(("paste", text)) or True,
    )
    monkeypatch.setattr(
        typer,
        "_type_direct",
        lambda text: events.append(("direct", text)) or True,
    )

    typer.commit_final("hello")

    assert "backspace" in events
    assert ("paste", "hello") in events
    assert ("direct", "hello") not in events


def test_detect_active_window_reports_xwayland_when_present(monkeypatch):
    typer = StreamingTyper(retry_attempts=1, retry_delay_ms=0)

    def fake_run(args, check, capture_output, text, timeout):
        assert args == ["hyprctl", "activewindow", "-j"]
        return SimpleNamespace(stdout='{"xwayland": true}')

    monkeypatch.setattr("shuvoice.typer.subprocess.run", fake_run)

    assert typer._active_window_is_xwayland() is True


def test_send_backspaces_prefers_xdotool_for_xwayland(monkeypatch):
    typer = StreamingTyper(retry_attempts=1, retry_delay_ms=0)
    calls: list[list[str]] = []

    monkeypatch.setattr(typer, "_prefer_xdotool", lambda: True)
    monkeypatch.setattr(typer, "_active_xdotool_window_id", lambda: "123")
    monkeypatch.setattr(
        typer,
        "_run",
        lambda args, op, attempts=None: calls.append(args) or True,
    )

    assert typer._send_backspaces(55, "backspace") is True
    assert calls == [
        [
            "xdotool",
            "key",
            "--clearmodifiers",
            "--delay",
            "0",
            "--window",
            "123",
            "--repeat",
            "50",
            "--repeat-delay",
            "0",
            "BackSpace",
        ],
        [
            "xdotool",
            "key",
            "--clearmodifiers",
            "--delay",
            "0",
            "--window",
            "123",
            "--repeat",
            "5",
            "--repeat-delay",
            "0",
            "BackSpace",
        ],
    ]


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
