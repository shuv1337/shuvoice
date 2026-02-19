from __future__ import annotations

import pytest

from shuvoice.hotkey import (
    _HELD,
    _IDLE,
    _STOPPING,
    _TOGGLED,
    HotkeyListener,
)


def test_invalid_hotkey_name_raises():
    with pytest.raises(ValueError):
        HotkeyListener("KEY_THIS_DOES_NOT_EXIST")


def test_tap_toggle_flow(monkeypatch):
    events: list[str] = []

    listener = HotkeyListener("KEY_RIGHTCTRL", hold_threshold_ms=300)
    listener.set_callbacks(
        on_start=lambda: events.append("start"),
        on_stop=lambda: events.append("stop"),
    )

    # press at t=10.0
    monkeypatch.setattr("shuvoice.hotkey.time.monotonic", lambda: 10.0)
    listener._on_key_down()
    assert listener._state == _HELD
    assert events == ["start"]

    # release at t=10.1 (tap)
    monkeypatch.setattr("shuvoice.hotkey.time.monotonic", lambda: 10.1)
    listener._on_key_up()
    assert listener._state == _TOGGLED
    assert events == ["start"]

    # second tap down should stop
    listener._on_key_down()
    assert listener._state == _STOPPING
    assert events == ["start", "stop"]

    # release after stop tap returns idle
    listener._on_key_up()
    assert listener._state == _IDLE


def test_hold_push_to_talk_flow(monkeypatch):
    events: list[str] = []

    listener = HotkeyListener("KEY_RIGHTCTRL", hold_threshold_ms=300)
    listener.set_callbacks(
        on_start=lambda: events.append("start"),
        on_stop=lambda: events.append("stop"),
    )

    times = iter([100.0, 100.5])
    monkeypatch.setattr("shuvoice.hotkey.time.monotonic", lambda: next(times))

    listener._on_key_down()  # start
    assert listener._state == _HELD
    listener._on_key_up()  # held 0.5s => stop

    assert events == ["start", "stop"]
    assert listener._state == _IDLE
