from __future__ import annotations

import builtins
import types
from pathlib import Path

import numpy as np
import pytest

from shuvoice.asr import ASREngine, create_backend, get_backend_class
from shuvoice.config import Config


def test_dependency_errors_when_dependencies_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing torch")
        if name == "nemo.collections.asr":
            raise ImportError("missing nemo")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    errors = ASREngine.dependency_errors()

    assert any("PyTorch" in err for err in errors)
    assert any("NeMo ASR" in err for err in errors)


def test_dependency_errors_when_dependencies_present(monkeypatch):
    original_import = builtins.__import__
    dummy_module = types.ModuleType("dummy")

    def fake_import(name, *args, **kwargs):
        if name in {"torch", "nemo.collections.asr"}:
            return dummy_module
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert ASREngine.dependency_errors() == []


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (None, ""),
        ("plain", "plain"),
        (types.SimpleNamespace(text="from-attr"), "from-attr"),
        (types.SimpleNamespace(text=123), ""),
        (types.SimpleNamespace(no_text="x"), ""),
    ],
)
def test_normalize_transcript_item_matrix(item, expected):
    assert ASREngine._normalize_transcript_item(item) == expected


def test_process_chunk_and_reset_raise_when_model_unloaded():
    engine = ASREngine()

    with pytest.raises(RuntimeError, match="not loaded"):
        engine.reset()

    with pytest.raises(RuntimeError, match="not loaded"):
        engine.process_chunk(np.zeros(32, dtype=np.float32))


def test_nemo_native_chunk_samples_scaling():
    assert ASREngine(right_context=0).native_chunk_samples == 1280
    assert ASREngine(right_context=1).native_chunk_samples == 2560
    assert ASREngine(right_context=6).native_chunk_samples == 8960
    assert ASREngine(right_context=13).native_chunk_samples == 17920
    assert ASREngine(right_context=999).native_chunk_samples == 17920


def test_nemo_debug_step_num_property():
    engine = ASREngine()
    engine._step_num = 7
    assert engine.debug_step_num == 7


def test_wants_raw_audio_by_backend(tmp_path: Path):
    nemo = create_backend("nemo", Config())
    assert nemo.wants_raw_audio is True

    moonshine = create_backend("moonshine", Config(asr_backend="moonshine"))
    assert moonshine.wants_raw_audio is True

    sherpa = create_backend(
        "sherpa",
        Config(asr_backend="sherpa", sherpa_model_dir=str(tmp_path)),
    )
    assert sherpa.wants_raw_audio is False


def test_get_backend_class_resolves_known_backends():
    assert get_backend_class("nemo").__name__ == "NemoBackend"
    assert get_backend_class("sherpa").__name__ == "SherpaBackend"
    assert get_backend_class("moonshine").__name__ == "MoonshineBackend"


def test_resolving_nemo_backend_does_not_import_other_backend_dependencies(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"sherpa_onnx", "moonshine_onnx"}:
            raise AssertionError(f"{name} should not be imported when resolving nemo")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    backend_cls = get_backend_class("nemo")
    assert backend_cls.__name__ == "NemoBackend"


def test_get_backend_class_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown ASR backend"):
        get_backend_class("unknown-backend")


def test_create_backend_factory_returns_expected_classes(tmp_path: Path):
    nemo = create_backend("nemo", Config())
    assert nemo.__class__.__name__ == "NemoBackend"

    sherpa_cfg = Config(asr_backend="sherpa", sherpa_model_dir=str(tmp_path))
    sherpa = create_backend("sherpa", sherpa_cfg)
    assert sherpa.__class__.__name__ == "SherpaBackend"

    moonshine_cfg = Config(asr_backend="moonshine")
    moonshine = create_backend("moonshine", moonshine_cfg)
    assert moonshine.__class__.__name__ == "MoonshineBackend"


def test_sherpa_dependency_errors_when_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sherpa_onnx":
            raise ImportError("missing sherpa")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    sherpa_cls = get_backend_class("sherpa")
    errors = sherpa_cls.dependency_errors()
    assert any("sherpa-onnx" in err for err in errors)


def test_moonshine_dependency_errors_when_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "moonshine_onnx":
            raise ImportError("missing moonshine")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    moonshine_cls = get_backend_class("moonshine")
    errors = moonshine_cls.dependency_errors()
    assert any("Moonshine ONNX" in err for err in errors)


def test_sherpa_load_requires_model_dir():
    cfg = Config(asr_backend="sherpa", sherpa_model_dir=None)
    backend = create_backend("sherpa", cfg)

    with pytest.raises(ValueError, match="sherpa_model_dir"):
        backend.load()


def test_sherpa_load_requires_transducer_artifacts(tmp_path: Path):
    (tmp_path / "tokens.txt").write_text("<blk>\na\n")

    cfg = Config(asr_backend="sherpa", sherpa_model_dir=str(tmp_path))
    backend = create_backend("sherpa", cfg)

    with pytest.raises(ValueError, match="encoder"):
        backend.load()


def test_moonshine_load_requires_local_artifacts_when_model_dir_is_set(tmp_path: Path):
    cfg = Config(asr_backend="moonshine", moonshine_model_dir=str(tmp_path))
    backend = create_backend("moonshine", cfg)

    with pytest.raises(ValueError, match="encoder_model.onnx"):
        backend.load()
