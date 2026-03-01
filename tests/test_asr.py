from __future__ import annotations

import builtins
import types
from pathlib import Path

import numpy as np
import pytest

from shuvoice.asr import create_backend, get_backend_class
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

    backend_cls = get_backend_class("nemo")
    errors = backend_cls.dependency_errors()

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

    backend_cls = get_backend_class("nemo")
    assert backend_cls.dependency_errors() == []


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
    backend_cls = get_backend_class("nemo")
    assert backend_cls._normalize_transcript_item(item) == expected


def test_process_chunk_and_reset_raise_when_model_unloaded():
    nemo_cls = get_backend_class("nemo")
    engine = nemo_cls()

    with pytest.raises(RuntimeError, match="not loaded"):
        engine.reset()

    with pytest.raises(RuntimeError, match="not loaded"):
        engine.process_chunk(np.zeros(32, dtype=np.float32))


def test_nemo_native_chunk_samples_scaling():
    nemo_cls = get_backend_class("nemo")
    assert nemo_cls(right_context=0).native_chunk_samples == 1280
    assert nemo_cls(right_context=1).native_chunk_samples == 2560
    assert nemo_cls(right_context=6).native_chunk_samples == 8960
    assert nemo_cls(right_context=13).native_chunk_samples == 17920
    assert nemo_cls(right_context=999).native_chunk_samples == 17920


def test_nemo_debug_step_num_property():
    nemo_cls = get_backend_class("nemo")
    engine = nemo_cls()
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


def test_deprecated_asrengine_alias_is_lazy_and_points_to_nemo(monkeypatch):
    import shuvoice.asr as asr_module

    with pytest.warns(DeprecationWarning):
        alias = asr_module.ASREngine

    assert alias is get_backend_class("nemo")


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


def test_sherpa_auto_downloads_default_model_when_model_dir_missing(
    monkeypatch,
    tmp_path: Path,
):
    cfg = Config(asr_backend="sherpa", sherpa_model_dir=None)
    backend = create_backend("sherpa", cfg)

    sherpa_cls = get_backend_class("sherpa")
    auto_dir = tmp_path / "auto-sherpa-model"

    monkeypatch.setattr(
        sherpa_cls,
        "_default_model_dir",
        classmethod(lambda cls, model_name=None: auto_dir),
    )

    def fake_download_model(cls, model_name=None, model_dir=None, **_):
        target = Path(model_dir).expanduser() if model_dir else auto_dir
        target.mkdir(parents=True, exist_ok=True)
        (target / "tokens.txt").write_text("<blk>\na\n")
        for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx"):
            (target / name).write_bytes(b"onnx")

    monkeypatch.setattr(sherpa_cls, "download_model", classmethod(fake_download_model))

    backend._validate_runtime_config()

    assert cfg.sherpa_model_dir == str(auto_dir)
    assert backend._model_files is not None
    assert backend._model_files["tokens"] == auto_dir / "tokens.txt"


def test_sherpa_auto_download_uses_configured_model_name(monkeypatch, tmp_path: Path):
    model_name = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    cfg = Config(asr_backend="sherpa", sherpa_model_name=model_name, sherpa_model_dir=None)
    backend = create_backend("sherpa", cfg)

    sherpa_cls = get_backend_class("sherpa")
    auto_dir = tmp_path / model_name

    monkeypatch.setattr(
        sherpa_cls,
        "_default_model_dir",
        classmethod(lambda cls, model_name=None: tmp_path / str(model_name)),
    )

    captured: dict[str, str | None] = {"model_name": None}

    def fake_download_model(cls, model_name=None, model_dir=None, **_):
        captured["model_name"] = model_name
        target = Path(model_dir).expanduser() if model_dir else auto_dir
        target.mkdir(parents=True, exist_ok=True)
        (target / "tokens.txt").write_text("<blk>\na\n")
        for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx"):
            (target / name).write_bytes(b"onnx")

    monkeypatch.setattr(sherpa_cls, "download_model", classmethod(fake_download_model))

    backend._validate_runtime_config()

    assert captured["model_name"] == model_name
    assert cfg.sherpa_model_dir == str(auto_dir)


def test_sherpa_startup_errors_block_parakeet_streaming_by_default():
    cfg = Config(
        asr_backend="sherpa", sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    )

    sherpa_cls = get_backend_class("sherpa")
    errors = sherpa_cls.startup_errors(cfg)

    assert errors
    assert any("Parakeet" in error for error in errors)
    assert any("offline" in error.lower() for error in errors)
    assert any("sherpa_enable_parakeet_streaming" in error for error in errors)


def test_sherpa_startup_warnings_downgrade_cuda_when_runtime_is_cpu_only(monkeypatch):
    cfg = Config(asr_backend="sherpa", sherpa_provider="cuda")

    sherpa_cls = get_backend_class("sherpa")
    monkeypatch.setattr(
        sherpa_cls,
        "_cuda_provider_available",
        staticmethod(lambda: (False, "missing CUDA provider library")),
    )

    warnings = sherpa_cls.startup_warnings(cfg, apply_fixes=True)

    assert warnings
    assert cfg.sherpa_provider == "cpu"
    assert any("sherpa_provider='cuda'" in warning for warning in warnings)
    assert any("Falling back to sherpa_provider='cpu'" in warning for warning in warnings)


def test_sherpa_download_format_helpers():
    sherpa_cls = get_backend_class("sherpa")

    assert sherpa_cls._format_bytes(0) == "0 B"
    assert sherpa_cls._format_bytes(1536) == "1.5 KiB"
    assert sherpa_cls._format_bytes(3 * 1024 * 1024) == "3.0 MiB"

    assert sherpa_cls._format_eta(None) == "--:--"
    assert sherpa_cls._format_eta(65.0) == "01:05"
    assert sherpa_cls._format_eta(3661.0) == "1:01:01"


def test_sherpa_download_progress_includes_bytes_and_eta(monkeypatch, tmp_path: Path):
    sherpa_cls = get_backend_class("sherpa")

    source_dir = tmp_path / "source-model"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "tokens.txt").write_text("<blk>\na\n")
    for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx"):
        (source_dir / name).write_bytes(b"onnx")

    monkeypatch.setattr(
        "shuvoice.asr_sherpa.urllib.request.urlretrieve",
        lambda _url, filename, reporthook=None: (
            reporthook and reporthook(1, 1024 * 1024, 10 * 1024 * 1024),
            reporthook and reporthook(5, 1024 * 1024, 10 * 1024 * 1024),
            reporthook and reporthook(10, 1024 * 1024, 10 * 1024 * 1024),
            Path(filename).write_bytes(b"archive"),
        ),
    )
    monkeypatch.setattr(
        sherpa_cls,
        "_safe_extract_tar",
        staticmethod(lambda _archive_path, _target_dir: None),
    )
    monkeypatch.setattr(
        sherpa_cls,
        "_find_extracted_model_dir",
        classmethod(lambda cls, _root: source_dir),
    )

    timeline = iter([0.0, 1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr("shuvoice.asr_sherpa.time.monotonic", lambda: next(timeline, 4.0))

    events: list[tuple[float | None, str]] = []
    target_dir = tmp_path / "downloaded-model"
    sherpa_cls.download_model(
        model_name="fake-model",
        model_dir=str(target_dir),
        progress_callback=lambda fraction, text: events.append((fraction, text)),
    )

    download_messages = [text for _fraction, text in events if "Downloading model archive" in text]
    assert download_messages
    assert any("ETA" in text and "/" in text for text in download_messages)


def test_sherpa_load_requires_transducer_artifacts(tmp_path: Path):
    (tmp_path / "tokens.txt").write_text("<blk>\na\n")

    cfg = Config(asr_backend="sherpa", sherpa_model_dir=str(tmp_path))
    backend = create_backend("sherpa", cfg)

    with pytest.raises(ValueError, match="missing required artifacts"):
        backend.load()


def test_moonshine_load_requires_local_artifacts_when_model_dir_is_set(tmp_path: Path):
    cfg = Config(asr_backend="moonshine", moonshine_model_dir=str(tmp_path))
    backend = create_backend("moonshine", cfg)

    with pytest.raises(ValueError, match="encoder_model.onnx"):
        backend.load()
