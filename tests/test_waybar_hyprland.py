from __future__ import annotations

import json
from types import SimpleNamespace

from shuvoice.waybar.hyprland import clear_keybind_cache, detect_keybind, detect_keybinds


def test_detect_keybind_uses_cache(monkeypatch):
    calls = {"count": 0}

    def fake_run(*_args, **_kwargs):
        calls["count"] += 1
        payload = [
            {
                "arg": "shuvoice --control start",
                "key": "V",
                "modmask": 64,
            },
            {
                "arg": "shuvoice control tts_speak --control-wait-sec 0",
                "key": "S",
                "modmask": 68,
            },
        ]
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload))

    monkeypatch.setattr("shuvoice.waybar.hyprland.subprocess.run", fake_run)
    clear_keybind_cache()

    first = detect_keybind(ttl_sec=10.0)
    second = detect_keybind(ttl_sec=10.0)
    tts = detect_keybind("tts_speak", ttl_sec=10.0)

    assert first == "Super + V"
    assert second == "Super + V"
    assert tts == "Super + Ctrl + S"
    assert calls["count"] == 1


def test_detect_keybind_matches_lua_dispatched_binds(monkeypatch):
    """Lua-dispatched binds expose the command via `description`, not `arg`."""

    payload = [
        {
            "arg": "194",
            "key": "Control_R",
            "modmask": 0,
            "release": False,
            "dispatcher": "__lua",
            "description": "ShuVoice start",
        },
        {
            "arg": "198",
            "key": "Control_R",
            "modmask": 0,
            "release": True,
            "dispatcher": "__lua",
            "description": "ShuVoice stop",
        },
        {
            "arg": "206",
            "key": "S",
            "modmask": 68,
            "release": False,
            "dispatcher": "__lua",
            "description": "ShuVoice TTS speak",
        },
    ]

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload))

    monkeypatch.setattr("shuvoice.waybar.hyprland.subprocess.run", fake_run)
    clear_keybind_cache()

    assert detect_keybind("start", ttl_sec=10.0) == "Control_R"
    assert detect_keybind("tts_speak", ttl_sec=10.0) == "Super + Ctrl + S"


def test_detect_keybind_skips_release_binds_for_start(monkeypatch):
    """Release-only `ShuVoice stop` must not be returned as the `start` bind."""

    payload = [
        {
            "arg": "198",
            "key": "Control_R",
            "modmask": 0,
            "release": True,
            "dispatcher": "__lua",
            "description": "ShuVoice stop",
        },
    ]

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload))

    monkeypatch.setattr("shuvoice.waybar.hyprland.subprocess.run", fake_run)
    clear_keybind_cache()

    assert detect_keybind("start", ttl_sec=10.0) is None


def test_detect_keybind_does_not_collide_tts_speak_and_clipboard(monkeypatch):
    payload = [
        {
            "arg": "shuvoice control tts_speak_clipboard --control-wait-sec 0",
            "key": "S",
            "modmask": 69,
            "release": False,
        },
        {
            "arg": "shuvoice control tts_speak --control-wait-sec 0",
            "key": "S",
            "modmask": 68,
            "release": False,
        },
    ]

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload))

    monkeypatch.setattr("shuvoice.waybar.hyprland.subprocess.run", fake_run)
    clear_keybind_cache()

    detected = detect_keybinds(ttl_sec=10.0)

    assert detected["tts_speak"] == "Super + Ctrl + S"
    assert detected["tts_speak_clipboard"] == "Super + Ctrl + Shift + S"


def test_detect_keybind_matches_clipboard_lua_description(monkeypatch):
    payload = [
        {
            "arg": "207",
            "key": "S",
            "modmask": 69,
            "release": False,
            "dispatcher": "__lua",
            "description": "ShuVoice TTS speak clipboard",
        },
    ]

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload))

    monkeypatch.setattr("shuvoice.waybar.hyprland.subprocess.run", fake_run)
    clear_keybind_cache()

    assert detect_keybind("tts_speak", ttl_sec=10.0) is None
    assert detect_keybind("tts_speak_clipboard", ttl_sec=10.0) == "Super + Ctrl + Shift + S"
