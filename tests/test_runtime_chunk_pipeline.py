from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from shuvoice.runtime.chunk_pipeline import apply_utterance_gain
from shuvoice.utterance_state import _UtteranceState


def test_apply_utterance_gain_scales_and_clips():
    audio = np.array([0.2, -0.5, 0.9], dtype=np.float32)
    out = apply_utterance_gain(audio, 2.0)

    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.array([0.4, -1.0, 1.0], dtype=np.float32))


def test_apply_utterance_gain_noop_for_small_gain():
    audio = np.array([0.2, -0.5], dtype=np.float32)
    out = apply_utterance_gain(audio, 1.01)

    assert out is audio


def test_begin_utterance_prepends_recording_preroll():
    preroll = [np.full(160, 0.02, dtype=np.float32)]
    state = _UtteranceState()
    app = SimpleNamespace(
        _asr_lock=threading.Lock(),
        _asr_disabled_event=threading.Event(),
        asr=SimpleNamespace(reset=Mock(), wants_raw_audio=True),
        _recover_asr_after_failure=Mock(),
        _speech_rms_threshold=0.008,
        _noise_floor_rms=0.001,
        _speech_rms_multiplier=1.8,
        _take_recording_preroll=Mock(return_value=preroll),
    )

    from shuvoice.runtime.chunk_pipeline import begin_utterance

    begin_utterance(app, state)

    assert state.total == 160
    assert state.speech_samples == 160
    app._take_recording_preroll.assert_called_once()
