"""Tests for Sherpa offline instant mode recognizer path."""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shuvoice.asr import create_backend, get_backend_class
from shuvoice.config import Config


def _make_model_dir(tmp_path: Path) -> Path:
    """Create a minimal valid model directory structure."""
    model_dir = tmp_path / "sherpa-model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "tokens.txt").write_text("<blk>\na\n")
    (model_dir / "encoder.onnx").write_bytes(b"onnx-window_size")
    for name in ("decoder.onnx", "joiner.onnx"):
        (model_dir / name).write_bytes(b"onnx")
    return model_dir


class TestSherpaOfflineMode:
    """Tests for _is_offline_mode property and mode-based branching."""

    def test_is_offline_mode_returns_true_when_offline_instant(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
        )
        backend = create_backend("sherpa", cfg)
        assert backend._is_offline_mode is True

    def test_is_offline_mode_returns_false_when_streaming(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="streaming",
        )
        backend = create_backend("sherpa", cfg)
        assert backend._is_offline_mode is False

    def test_is_offline_mode_auto_resolves_for_parakeet_with_instant(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="auto",
            instant_mode=True,
        )
        backend = create_backend("sherpa", cfg)
        assert backend._is_offline_mode is True

    def test_is_offline_mode_auto_resolves_streaming_for_non_parakeet(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-streaming-zipformer-en-kroko",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="auto",
            instant_mode=True,
        )
        backend = create_backend("sherpa", cfg)
        assert backend._is_offline_mode is False


class TestSherpaStartupErrors:
    """Tests for startup_errors with offline mode consideration."""

    def test_parakeet_blocked_in_streaming_mode(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="streaming",
        )

        sherpa_cls = get_backend_class("sherpa")
        errors = sherpa_cls.startup_errors(cfg)

        assert errors
        assert any("Parakeet" in error for error in errors)
        assert any("offline instant mode" in error for error in errors)

    def test_parakeet_allowed_in_offline_instant_mode(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
        )

        sherpa_cls = get_backend_class("sherpa")
        errors = sherpa_cls.startup_errors(cfg)

        assert not errors

    def test_parakeet_allowed_with_auto_and_instant_mode(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="auto",
            instant_mode=True,
        )

        sherpa_cls = get_backend_class("sherpa")
        errors = sherpa_cls.startup_errors(cfg)

        assert not errors


class TestSherpaProcessChunk:
    """Tests for process_chunk behavior in streaming vs offline mode."""

    def test_process_chunk_raises_in_offline_mode(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
        )
        backend = create_backend("sherpa", cfg)

        # Mock the offline recognizer to avoid real loading
        backend._offline_recognizer = MagicMock()

        audio = np.zeros(1600, dtype=np.float32)

        with pytest.raises(RuntimeError, match="process_chunk.*not supported in offline"):
            backend.process_chunk(audio)


class TestSherpaProcessUtterance:
    """Tests for process_utterance behavior."""

    def test_process_utterance_raises_in_streaming_mode(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="streaming",
        )
        backend = create_backend("sherpa", cfg)

        # Mock the recognizer to avoid real loading
        backend._recognizer = MagicMock()
        backend._stream = MagicMock()

        audio = np.zeros(16000, dtype=np.float32)

        with pytest.raises(RuntimeError, match="process_utterance.*only supported in offline"):
            backend.process_utterance(audio)

    def test_process_utterance_raises_when_not_loaded(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
        )
        backend = create_backend("sherpa", cfg)
        # Don't load or mock the recognizer

        audio = np.zeros(16000, dtype=np.float32)

        with pytest.raises(RuntimeError, match="not loaded"):
            backend.process_utterance(audio)

    def test_process_utterance_returns_text_from_result(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
        )
        backend = create_backend("sherpa", cfg)

        # Create mock result with text attribute
        mock_result = types.SimpleNamespace(text=" hello world ")
        mock_stream = MagicMock()
        mock_stream.result = mock_result

        mock_recognizer = MagicMock()
        mock_recognizer.create_stream.return_value = mock_stream

        backend._offline_recognizer = mock_recognizer

        audio = np.zeros(16000, dtype=np.float32)
        result = backend.process_utterance(audio)

        assert result == "hello world"
        mock_stream.accept_waveform.assert_called_once()
        mock_recognizer.decode_stream.assert_called_once_with(mock_stream)

    def test_process_utterance_handles_multidimensional_audio(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
        )
        backend = create_backend("sherpa", cfg)

        mock_result = types.SimpleNamespace(text="test")
        mock_stream = MagicMock()
        mock_stream.result = mock_result

        mock_recognizer = MagicMock()
        mock_recognizer.create_stream.return_value = mock_stream

        backend._offline_recognizer = mock_recognizer

        # 2D audio array (e.g., from accidental reshape)
        audio = np.zeros((100, 160), dtype=np.float32)
        result = backend.process_utterance(audio)

        # Should flatten and still work
        assert result == "test"


class TestSherpaReset:
    """Tests for reset behavior in streaming vs offline mode."""

    def test_reset_raises_when_streaming_not_loaded(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="streaming",
        )
        backend = create_backend("sherpa", cfg)

        with pytest.raises(RuntimeError, match="not loaded"):
            backend.reset()

    def test_reset_raises_when_offline_not_loaded(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
        )
        backend = create_backend("sherpa", cfg)

        with pytest.raises(RuntimeError, match="not loaded"):
            backend.reset()

    def test_reset_succeeds_in_offline_mode_when_loaded(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
        )
        backend = create_backend("sherpa", cfg)
        backend._offline_recognizer = MagicMock()

        # Should not raise
        backend.reset()


class TestSherpaLoadBranching:
    """Tests for load() mode branching."""

    def test_load_calls_online_recognizer_for_streaming(self, tmp_path: Path, monkeypatch):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="streaming",
        )
        backend = create_backend("sherpa", cfg)

        online_called = {"value": False}
        offline_called = {"value": False}

        def fake_load_online(self):
            online_called["value"] = True
            self._recognizer = MagicMock()
            self._stream = MagicMock()

        def fake_load_offline(self):
            offline_called["value"] = True
            self._offline_recognizer = MagicMock()

        # Mock dependency_errors to return empty (no real sherpa-onnx needed)
        monkeypatch.setattr(type(backend), "dependency_errors", staticmethod(lambda: []))
        monkeypatch.setattr(type(backend), "_load_online_recognizer", fake_load_online)
        monkeypatch.setattr(type(backend), "_load_offline_recognizer", fake_load_offline)

        backend.load()

        assert online_called["value"] is True
        assert offline_called["value"] is False

    def test_load_calls_offline_recognizer_for_offline_instant(self, tmp_path: Path, monkeypatch):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
        )
        backend = create_backend("sherpa", cfg)

        online_called = {"value": False}
        offline_called = {"value": False}

        def fake_load_online(self):
            online_called["value"] = True
            self._recognizer = MagicMock()
            self._stream = MagicMock()

        def fake_load_offline(self):
            offline_called["value"] = True
            self._offline_recognizer = MagicMock()

        # Mock dependency_errors to return empty (no real sherpa-onnx needed)
        monkeypatch.setattr(type(backend), "dependency_errors", staticmethod(lambda: []))
        monkeypatch.setattr(type(backend), "_load_online_recognizer", fake_load_online)
        monkeypatch.setattr(type(backend), "_load_offline_recognizer", fake_load_offline)

        backend.load()

        assert online_called["value"] is False
        assert offline_called["value"] is True


class TestSherpaOfflineRecognizerInit:
    """Tests for OfflineRecognizer initialization details."""

    def test_offline_recognizer_uses_nemo_transducer_model_type(self, tmp_path: Path, monkeypatch):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
        )
        backend = create_backend("sherpa", cfg)

        # Mock sherpa_onnx module
        mock_sherpa = MagicMock()
        captured_kwargs = {}

        def capture_from_transducer(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        mock_offline_cls = MagicMock()
        mock_offline_cls.from_transducer = capture_from_transducer
        mock_sherpa.OfflineRecognizer = mock_offline_cls

        with patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa}):
            backend._model_files = {
                "tokens": model_dir / "tokens.txt",
                "encoder": model_dir / "encoder.onnx",
                "decoder": model_dir / "decoder.onnx",
                "joiner": model_dir / "joiner.onnx",
            }
            backend._load_offline_recognizer()

        assert captured_kwargs.get("model_type") == "nemo_transducer"

    def test_offline_recognizer_uses_configured_provider(self, tmp_path: Path, monkeypatch):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
            sherpa_provider="cuda",
        )
        backend = create_backend("sherpa", cfg)

        mock_sherpa = MagicMock()
        captured_kwargs = {}

        def capture_from_transducer(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        mock_offline_cls = MagicMock()
        mock_offline_cls.from_transducer = capture_from_transducer
        mock_sherpa.OfflineRecognizer = mock_offline_cls

        with patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa}):
            backend._model_files = {
                "tokens": model_dir / "tokens.txt",
                "encoder": model_dir / "encoder.onnx",
                "decoder": model_dir / "decoder.onnx",
                "joiner": model_dir / "joiner.onnx",
            }
            backend._load_offline_recognizer()

        assert captured_kwargs.get("provider") == "cuda"


class TestSherpaOnlineRecognizerInit:
    """Tests for OnlineRecognizer initialization details."""

    def test_online_parakeet_streaming_uses_nemo_transducer_model_type(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="streaming",
            sherpa_enable_parakeet_streaming=True,
        )
        backend = create_backend("sherpa", cfg)

        mock_sherpa = MagicMock()
        captured_kwargs = {}

        def capture_from_transducer(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        mock_online_cls = MagicMock()
        mock_online_cls.from_transducer = capture_from_transducer
        mock_sherpa.OnlineRecognizer = mock_online_cls

        with patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa}):
            backend._model_files = {
                "tokens": model_dir / "tokens.txt",
                "encoder": model_dir / "encoder.onnx",
                "decoder": model_dir / "decoder.onnx",
                "joiner": model_dir / "joiner.onnx",
            }
            backend._load_online_recognizer()

        assert captured_kwargs.get("model_type") == "nemo_transducer"

    def test_online_parakeet_streaming_fails_fast_without_window_size_metadata(
        self, tmp_path: Path
    ):
        model_dir = tmp_path / "sherpa-model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "tokens.txt").write_text("<blk>\na\n")
        (model_dir / "encoder.onnx").write_bytes(b"onnx")
        for name in ("decoder.onnx", "joiner.onnx"):
            (model_dir / name).write_bytes(b"onnx")

        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="streaming",
            sherpa_enable_parakeet_streaming=True,
        )
        backend = create_backend("sherpa", cfg)
        mock_sherpa = MagicMock()

        with patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa}):
            backend._model_files = {
                "tokens": model_dir / "tokens.txt",
                "encoder": model_dir / "encoder.onnx",
                "decoder": model_dir / "decoder.onnx",
                "joiner": model_dir / "joiner.onnx",
            }

            with pytest.raises(RuntimeError, match="window_size"):
                backend._load_online_recognizer()
