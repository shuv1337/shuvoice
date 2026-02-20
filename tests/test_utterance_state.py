from __future__ import annotations

import numpy as np

from shuvoice.utterance_state import _UtteranceState


def test_reset_clears_all_fields():
    state = _UtteranceState(
        buffer=[np.array([1, 2], dtype=np.float32)],
        total=2,
        last_text="hello",
        speech_samples=5,
        peak_rms=0.5,
        utterance_gain=2.0,
        utterance_rms_threshold=0.1,
    )

    state.reset(rms_threshold=0.33)

    assert state.buffer == []
    assert state.total == 0
    assert state.last_text == ""
    assert state.speech_samples == 0
    assert state.peak_rms == 0.0
    assert state.utterance_gain == 1.0
    assert state.utterance_rms_threshold == 0.33


def test_add_chunk_increments_total():
    state = _UtteranceState()

    state.add_chunk(np.array([1, 2, 3], dtype=np.float32))
    state.add_chunk(np.array([4], dtype=np.float32))

    assert state.total == 4
    assert len(state.buffer) == 2


def test_consume_native_chunk_returns_chunk_remainder_and_has_more_false():
    state = _UtteranceState()
    state.buffer = [
        np.array([0, 1, 2, 3, 4], dtype=np.float32),
        np.array([5, 6, 7, 8], dtype=np.float32),
    ]
    state.total = 9

    to_process, has_more = state.consume_native_chunk(6)

    assert np.array_equal(to_process, np.array([0, 1, 2, 3, 4, 5], dtype=np.float32))
    assert has_more is False
    assert state.total == 3
    assert len(state.buffer) == 1
    assert np.array_equal(state.buffer[0], np.array([6, 7, 8], dtype=np.float32))


def test_consume_native_chunk_returns_has_more_true_when_enough_remainder():
    state = _UtteranceState()
    state.buffer = [np.arange(10, dtype=np.float32)]
    state.total = 10

    to_process, has_more = state.consume_native_chunk(4)

    assert np.array_equal(to_process, np.array([0, 1, 2, 3], dtype=np.float32))
    assert has_more is True
    assert state.total == 6
