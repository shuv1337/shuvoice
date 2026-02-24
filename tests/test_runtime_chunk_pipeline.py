from __future__ import annotations

import numpy as np

from shuvoice.runtime.chunk_pipeline import apply_utterance_gain


def test_apply_utterance_gain_scales_and_clips():
    audio = np.array([0.2, -0.5, 0.9], dtype=np.float32)
    out = apply_utterance_gain(audio, 2.0)

    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.array([0.4, -1.0, 1.0], dtype=np.float32))


def test_apply_utterance_gain_noop_for_small_gain():
    audio = np.array([0.2, -0.5], dtype=np.float32)
    out = apply_utterance_gain(audio, 1.01)

    assert out is audio
