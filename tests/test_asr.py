from __future__ import annotations

import builtins
import types

import numpy as np
import pytest

from shuvoice.asr import ASREngine


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
