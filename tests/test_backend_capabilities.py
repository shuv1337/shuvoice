from __future__ import annotations

from pathlib import Path

import pytest

from shuvoice.asr import get_backend_class
from shuvoice.config import Config


def _make_sherpa_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "sherpa-model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "tokens.txt").write_text("<blk>\na\n")
    for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx"):
        (model_dir / name).write_bytes(b"onnx")
    return model_dir


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


@pytest.mark.parametrize(
    ("instant_mode", "decode_mode", "expect_error"),
    [
        (False, "auto", True),
        (True, "auto", False),
        (False, "offline_instant", False),
        (True, "streaming", True),
    ],
)
def test_sherpa_startup_guard_matrix_for_parakeet(
    tmp_path: Path,
    instant_mode: bool,
    decode_mode: str,
    expect_error: bool,
):
    model_dir = _make_sherpa_model_dir(tmp_path)
    cfg = Config(
        asr_backend="sherpa",
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        sherpa_model_dir=str(model_dir),
        instant_mode=instant_mode,
        sherpa_decode_mode=decode_mode,
    )

    sherpa_cls = get_backend_class("sherpa")
    errors = sherpa_cls.startup_errors(cfg)

    if expect_error:
        assert errors
        joined = "\n".join(errors)
        assert "offline" in joined.lower()
        assert "sherpa_decode_mode" in joined
    else:
        assert errors == []


def test_sherpa_startup_guard_allows_non_parakeet_streaming(tmp_path: Path):
    model_dir = _make_sherpa_model_dir(tmp_path)
    cfg = Config(
        asr_backend="sherpa",
        sherpa_model_name="sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
        sherpa_model_dir=str(model_dir),
        sherpa_decode_mode="streaming",
    )

    sherpa_cls = get_backend_class("sherpa")
    assert sherpa_cls.startup_errors(cfg) == []


def test_sherpa_startup_warning_cuda_fallback_applies_in_streaming_mode(monkeypatch, tmp_path: Path):
    model_dir = _make_sherpa_model_dir(tmp_path)
    cfg = Config(
        asr_backend="sherpa",
        sherpa_model_dir=str(model_dir),
        sherpa_decode_mode="streaming",
        sherpa_provider="cuda",
    )

    sherpa_cls = get_backend_class("sherpa")
    monkeypatch.setattr(
        sherpa_cls,
        "_cuda_provider_available",
        staticmethod(lambda: (False, "missing CUDA provider library")),
    )

    warnings = sherpa_cls.startup_warnings(cfg, apply_fixes=True)

    assert warnings
    assert cfg.sherpa_provider == "cpu"
    assert "CUDAExecutionProvider" in warnings[0]


def test_sherpa_startup_warning_cuda_fallback_applies_in_offline_mode(monkeypatch, tmp_path: Path):
    model_dir = _make_sherpa_model_dir(tmp_path)
    cfg = Config(
        asr_backend="sherpa",
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        sherpa_model_dir=str(model_dir),
        sherpa_decode_mode="offline_instant",
        sherpa_provider="cuda",
    )

    sherpa_cls = get_backend_class("sherpa")
    monkeypatch.setattr(
        sherpa_cls,
        "_cuda_provider_available",
        staticmethod(lambda: (False, "missing CUDA provider library")),
    )

    warnings = sherpa_cls.startup_warnings(cfg, apply_fixes=True)

    assert warnings
    assert cfg.sherpa_provider == "cpu"
    assert "CUDAExecutionProvider" in warnings[0]
