from __future__ import annotations

import time

import numpy as np

from shuvoice.tts_player import TTSPlayer


class _FakeOutputStream:
    instances: list["_FakeOutputStream"] = []

    def __init__(self, **_kwargs):
        self.active = False
        self.writes: list[np.ndarray] = []
        self.stopped = 0
        self.started = 0
        self.aborted = 0
        self.closed = 0
        _FakeOutputStream.instances.append(self)

    def start(self):
        self.active = True
        self.started += 1

    def stop(self):
        self.active = False
        self.stopped += 1

    def abort(self):
        self.active = False
        self.aborted += 1

    def close(self):
        self.active = False
        self.closed += 1

    def write(self, data):
        self.writes.append(np.array(data, copy=True))


class _Backend:
    def __init__(self, chunks: list[bytes], *, delay_sec: float = 0.0, fail: bool = False):
        self._chunks = chunks
        self._delay_sec = delay_sec
        self._fail = fail
        self.calls: list[tuple[str, str, str]] = []

    def synthesize_stream(self, text: str, voice_id: str, model_id: str):
        self.calls.append((text, voice_id, model_id))
        if self._fail:
            raise RuntimeError("boom")
        for chunk in self._chunks:
            if self._delay_sec:
                time.sleep(self._delay_sec)
            yield chunk


def _wait_until(predicate, timeout_sec: float = 2.0):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_tts_player_basic_state_flow(monkeypatch):
    _FakeOutputStream.instances.clear()
    monkeypatch.setattr("shuvoice.tts_player.sd.OutputStream", _FakeOutputStream)

    backend = _Backend([b"\x00\x00" * 200, b"\x01\x00" * 200])
    events: list[str] = []

    player = TTSPlayer(backend, on_state_change=lambda state, _info: events.append(state))

    interrupted = player.speak("hello", "voice", "model")

    assert interrupted is False
    assert _wait_until(lambda: player.state == "idle")
    assert "synthesizing" in events
    assert "playing" in events
    assert events[-1] == "idle"
    assert _FakeOutputStream.instances
    assert _FakeOutputStream.instances[0].writes


def test_tts_player_applies_playback_speed_to_output(monkeypatch):
    _FakeOutputStream.instances.clear()
    monkeypatch.setattr("shuvoice.tts_player.sd.OutputStream", _FakeOutputStream)

    backend = _Backend([b"\x00\x00" * 200])
    player = TTSPlayer(backend, playback_speed=2.0)

    player.speak("hello", "voice", "model")

    assert _wait_until(lambda: player.state == "idle")
    written = _FakeOutputStream.instances[0].writes[0]
    assert written.shape == (100, 1)
    assert player.status_payload()["playback_speed"] == 2.0


def test_tts_player_set_playback_speed_clamps_value(monkeypatch):
    _FakeOutputStream.instances.clear()
    monkeypatch.setattr("shuvoice.tts_player.sd.OutputStream", _FakeOutputStream)

    player = TTSPlayer(_Backend([b"\x00\x00" * 10]))

    assert player.set_playback_speed(5.0) == 2.0
    assert player.playback_speed == 2.0


def test_tts_player_pause_resume(monkeypatch):
    _FakeOutputStream.instances.clear()
    monkeypatch.setattr("shuvoice.tts_player.sd.OutputStream", _FakeOutputStream)

    backend = _Backend([b"\x00\x00" * 120] * 8, delay_sec=0.02)
    player = TTSPlayer(backend)

    player.speak("hello", "voice", "model")
    assert _wait_until(lambda: player.state == "playing")

    assert player.pause() is True
    assert player.state == "paused"

    assert player.resume() is True
    assert _wait_until(lambda: player.state == "idle")

    # Pause/resume should not call stream.stop()/start() on existing handles.
    assert all(instance.stopped == 0 for instance in _FakeOutputStream.instances)


def test_tts_player_recovers_from_transient_portaudio_write_error(monkeypatch):
    class _RecoveringOutputStream(_FakeOutputStream):
        fail_next_write = True

        def write(self, data):
            if _RecoveringOutputStream.fail_next_write:
                _RecoveringOutputStream.fail_next_write = False
                raise RuntimeError("transient host error")
            super().write(data)

    _RecoveringOutputStream.instances.clear()
    _RecoveringOutputStream.fail_next_write = True

    monkeypatch.setattr("shuvoice.tts_player.sd.OutputStream", _RecoveringOutputStream)
    monkeypatch.setattr("shuvoice.tts_player.sd.PortAudioError", RuntimeError)

    backend = _Backend([b"\x00\x00" * 200, b"\x01\x00" * 200])
    player = TTSPlayer(backend)

    player.speak("hello", "voice", "model")

    assert _wait_until(lambda: player.state == "idle")
    assert len(_RecoveringOutputStream.instances) >= 2


def test_tts_player_interrupt_semantics(monkeypatch):
    _FakeOutputStream.instances.clear()
    monkeypatch.setattr("shuvoice.tts_player.sd.OutputStream", _FakeOutputStream)

    backend = _Backend([b"\x00\x00" * 80] * 20, delay_sec=0.02)
    player = TTSPlayer(backend)

    player.speak("first", "voice-a", "model")
    assert _wait_until(lambda: player.state in {"playing", "paused", "synthesizing"})

    interrupted = player.speak("second", "voice-b", "model")
    assert interrupted is True

    assert _wait_until(lambda: player.state == "idle")
    assert backend.calls[-1][0] == "second"


def test_tts_player_stop_from_active_state(monkeypatch):
    _FakeOutputStream.instances.clear()
    monkeypatch.setattr("shuvoice.tts_player.sd.OutputStream", _FakeOutputStream)

    backend = _Backend([b"\x00\x00" * 80] * 20, delay_sec=0.03)
    player = TTSPlayer(backend)

    player.speak("hello", "voice", "model")
    assert _wait_until(lambda: player.state in {"synthesizing", "playing"})

    assert player.stop() is True
    assert player.state == "idle"


def test_tts_player_error_transition(monkeypatch):
    _FakeOutputStream.instances.clear()
    monkeypatch.setattr("shuvoice.tts_player.sd.OutputStream", _FakeOutputStream)

    backend = _Backend([], fail=True)
    player = TTSPlayer(backend)

    player.speak("hello", "voice", "model")

    assert _wait_until(lambda: player.state == "error")
    assert player.stop() is True
    assert player.state == "idle"


def test_tts_player_restart_uses_last_text(monkeypatch):
    _FakeOutputStream.instances.clear()
    monkeypatch.setattr("shuvoice.tts_player.sd.OutputStream", _FakeOutputStream)

    backend = _Backend([b"\x00\x00" * 120])
    player = TTSPlayer(backend)

    assert player.restart() is False

    player.speak("hello", "voice", "model")
    assert _wait_until(lambda: player.state == "idle")

    assert player.restart() is True
    assert _wait_until(lambda: player.state == "idle")
    assert [call[0] for call in backend.calls] == ["hello", "hello"]
