from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from shuvoice.runtime.flush_policy import make_flush_noise


def test_make_flush_noise_respects_bounds():
    app = SimpleNamespace(
        _noise_floor_rms=0.01,
        _FLUSH_NOISE_MIN_RMS=0.005,
        _FLUSH_NOISE_MAX_RMS=0.08,
    )

    noise = make_flush_noise(app, 4096)

    assert noise.dtype == np.float32
    assert np.max(np.abs(noise)) <= 1.0
