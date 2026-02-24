from __future__ import annotations

from shuvoice.asr import get_backend_class
from shuvoice.config import Config


def test_backend_registry_exposes_capabilities():
    for name in ("nemo", "sherpa", "moonshine"):
        backend_cls = get_backend_class(name)
        caps = backend_cls.capabilities
        assert caps.expected_chunking in {"streaming", "windowed"}


def test_wants_raw_audio_passthrough_matches_capabilities(tmp_path):
    nemo = get_backend_class("nemo")()
    sherpa = get_backend_class("sherpa")(
        Config(
            asr_backend="sherpa",
            sherpa_model_dir=str(tmp_path),
        )
    )

    assert nemo.wants_raw_audio == nemo.capabilities.wants_raw_audio
    assert sherpa.wants_raw_audio == sherpa.capabilities.wants_raw_audio
