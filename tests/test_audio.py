from __future__ import annotations

import numpy as np

from shuvoice.audio import AudioCapture


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


def test_drain_pending_chunks_returns_and_clears_queue():
    audio = AudioCapture()

    c1 = np.array([0.1, 0.2], dtype=np.float32)
    c2 = np.array([0.3], dtype=np.float32)
    audio.queue.put_nowait(c1)
    audio.queue.put_nowait(c2)

    drained = audio.drain_pending_chunks()

    assert len(drained) == 2
    assert np.array_equal(drained[0], c1)
    assert np.array_equal(drained[1], c2)
    assert audio.queue.empty()
