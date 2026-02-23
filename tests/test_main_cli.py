from __future__ import annotations

from argparse import Namespace

from shuvoice.__main__ import _apply_cli_overrides
from shuvoice.config import Config


def _args(**overrides) -> Namespace:
    values = {
        "asr_backend": None,
        "device": None,
        "right_context": None,
        "sherpa_model_dir": None,
        "sherpa_provider": None,
        "sherpa_num_threads": None,
        "sherpa_chunk_ms": None,
        "moonshine_model_name": None,
        "moonshine_model_dir": None,
        "moonshine_model_precision": None,
        "moonshine_chunk_ms": None,
        "moonshine_max_window_sec": None,
        "moonshine_max_tokens": None,
        "moonshine_provider": None,
        "moonshine_onnx_threads": None,
        "audio_device": None,
        "input_gain": None,
        "output_mode": None,
        "control_socket": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_apply_cli_overrides_moonshine_provider_and_threads() -> None:
    config = Config()

    _apply_cli_overrides(
        _args(
            asr_backend="moonshine",
            moonshine_provider="cuda",
            moonshine_onnx_threads=4,
            audio_device="2",
        ),
        config,
    )

    assert config.asr_backend == "moonshine"
    assert config.moonshine_provider == "cuda"
    assert config.moonshine_onnx_threads == 4
    assert config.audio_device == 2
