from __future__ import annotations

import json
from types import SimpleNamespace

from shuvoice.waybar.hyprland import clear_keybind_cache, detect_keybind


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
