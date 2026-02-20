from __future__ import annotations

import numpy as np

from shuvoice.audio import AudioCapture, audio_rms


def test_downsample_integer_ratio_drops_samples():
    audio = AudioCapture()

    source = np.array([0, 3, 6, 9, 12, 15], dtype=np.float32)
    out = audio._downsample_integer_ratio(source, 3)

    assert np.allclose(out, np.array([0.0, 9.0], dtype=np.float32))


def test_downsample_integer_ratio_keeps_carry_between_calls():
    audio = AudioCapture()

    out1 = audio._downsample_integer_ratio(np.array([1, 2], dtype=np.float32), 3)
    assert out1.size == 0

    out2 = audio._downsample_integer_ratio(np.array([4, 5, 7, 8], dtype=np.float32), 3)
    assert np.allclose(out2, np.array([1.0, 5.0], dtype=np.float32))


def test_queue_overflow_drops_oldest_and_tracks_drops():
    audio = AudioCapture(audio_queue_max_size=2)

    for value in (0.1, 0.2, 0.3):
        indata = np.full((4, 1), value, dtype=np.float32)
        audio._callback(indata, 4, None, None)

    drained = audio.drain_pending_chunks()

    assert len(drained) == 2
    assert np.allclose(drained[0], np.full(4, 0.2, dtype=np.float32))
    assert np.allclose(drained[1], np.full(4, 0.3, dtype=np.float32))
    assert audio._dropped_chunks == 1


def test_clear_empties_queue():
    audio = AudioCapture(audio_queue_max_size=3)

    audio.queue.put_nowait(np.array([0.1, 0.2], dtype=np.float32))
    audio.queue.put_nowait(np.array([0.3], dtype=np.float32))

    audio.clear()

    assert audio.queue.empty()


def test_select_input_device_prefers_pulse(monkeypatch):
    devices = [
        {"name": "alsa_input", "max_input_channels": 2},
        {"name": "pulse", "max_input_channels": 2},
        {"name": "pipewire", "max_input_channels": 2},
    ]
    monkeypatch.setattr("shuvoice.audio.sd.query_devices", lambda: devices)

    audio = AudioCapture(device=None)
    assert audio._select_input_device() == 1


def test_select_input_device_falls_back_to_pipewire(monkeypatch):
    devices = [
        {"name": "alsa_input", "max_input_channels": 2},
        {"name": "My PipeWire Source", "max_input_channels": 1},
    ]
    monkeypatch.setattr("shuvoice.audio.sd.query_devices", lambda: devices)

    audio = AudioCapture(device=None)
    assert audio._select_input_device() == 1


def test_select_input_device_respects_explicit_setting():
    audio = AudioCapture(device=4)
    assert audio._select_input_device() == 4


def test_audio_rms_handles_edge_cases():
    assert audio_rms(np.array([], dtype=np.float32)) == 0.0
    assert audio_rms(np.zeros(4, dtype=np.float32)) == 0.0

    known = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    assert np.isclose(audio_rms(known), 1.0)
