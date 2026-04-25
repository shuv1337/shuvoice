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

        with pytest.raises(RuntimeError, match=r"process_chunk.*not supported in offline"):
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

        with pytest.raises(RuntimeError, match=r"process_utterance.*only supported in offline"):
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
        backend._model_files = {
            "tokens": model_dir / "tokens.txt",
            "encoder": model_dir / "encoder.onnx",
            "decoder": model_dir / "decoder.onnx",
            "joiner": model_dir / "joiner.onnx",
        }

        with pytest.raises(RuntimeError, match="window_size"):
            backend._load_online_recognizer()


# -- CUDA OOM detection + CPU fallback ---------------------------------------


class TestLooksLikeCudaOomError:
    def test_detects_cublas_alloc_failed(self):
        from shuvoice.asr_sherpa import looks_like_cuda_oom_error

        exc = RuntimeError(
            "cuda_call.cc:129 ... CUBLAS failure 3: CUBLAS_STATUS_ALLOC_FAILED ; "
            "expr=cublasCreate(&cublas_handle_);"
        )
        assert looks_like_cuda_oom_error(exc) is True

    def test_detects_cudnn_internal_error(self):
        from shuvoice.asr_sherpa import looks_like_cuda_oom_error

        exc = RuntimeError(
            "CUDNN failure 4000: CUDNN_STATUS_INTERNAL_ERROR ; expr=cudnnCreate(&cudnn_handle_);"
        )
        assert looks_like_cuda_oom_error(exc) is True

    def test_detects_plain_out_of_memory(self):
        from shuvoice.asr_sherpa import looks_like_cuda_oom_error

        assert looks_like_cuda_oom_error(RuntimeError("CUDA error: out of memory")) is True

    def test_does_not_false_positive_on_unrelated_errors(self):
        from shuvoice.asr_sherpa import looks_like_cuda_oom_error

        assert looks_like_cuda_oom_error(RuntimeError("tokens.txt not found")) is False
        assert looks_like_cuda_oom_error(ValueError("bad shape")) is False


class TestTryFallbackToCpu:
    def test_fallback_noop_when_already_cpu(self, tmp_path: Path):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
            sherpa_provider="cpu",
        )
        backend = create_backend("sherpa", cfg)

        ok, detail = backend.try_fallback_to_cpu()

        assert ok is False
        assert "cpu" in detail.lower()
        assert backend.cpu_fallback_applied is False

    def test_fallback_switches_provider_and_reloads_offline_recognizer(
        self, tmp_path: Path, monkeypatch
    ):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
            sherpa_provider="cuda",
        )
        backend = create_backend("sherpa", cfg)
        backend._model_files = {
            "tokens": model_dir / "tokens.txt",
            "encoder": model_dir / "encoder.onnx",
            "decoder": model_dir / "decoder.onnx",
            "joiner": model_dir / "joiner.onnx",
        }

        reload_calls: list[str] = []

        def fake_load_offline(self):
            reload_calls.append(self.config.sherpa_provider)
            self._offline_recognizer = MagicMock()

        monkeypatch.setattr(type(backend), "_load_offline_recognizer", fake_load_offline)

        ok, detail = backend.try_fallback_to_cpu()

        assert ok is True
        assert backend.config.sherpa_provider == "cpu"
        assert backend.cpu_fallback_applied is True
        assert reload_calls == ["cpu"]
        assert "cpu" in detail.lower()

    def test_fallback_is_idempotent(self, tmp_path: Path, monkeypatch):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
            sherpa_provider="cuda",
        )
        backend = create_backend("sherpa", cfg)
        backend._model_files = {
            "tokens": model_dir / "tokens.txt",
            "encoder": model_dir / "encoder.onnx",
            "decoder": model_dir / "decoder.onnx",
            "joiner": model_dir / "joiner.onnx",
        }

        monkeypatch.setattr(
            type(backend),
            "_load_offline_recognizer",
            lambda self: setattr(self, "_offline_recognizer", MagicMock()),
        )

        first_ok, _ = backend.try_fallback_to_cpu()
        second_ok, second_detail = backend.try_fallback_to_cpu()

        assert first_ok is True
        assert second_ok is False
        assert "already" in second_detail.lower()

    def test_fallback_reload_failure_leaves_flag_unset(self, tmp_path: Path, monkeypatch):
        model_dir = _make_model_dir(tmp_path)
        cfg = Config(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
            sherpa_provider="cuda",
        )
        backend = create_backend("sherpa", cfg)
        backend._model_files = {
            "tokens": model_dir / "tokens.txt",
            "encoder": model_dir / "encoder.onnx",
            "decoder": model_dir / "decoder.onnx",
            "joiner": model_dir / "joiner.onnx",
        }

        def failing_load(self):
            raise RuntimeError("boom")

        monkeypatch.setattr(type(backend), "_load_offline_recognizer", failing_load)

        ok, detail = backend.try_fallback_to_cpu()

        assert ok is False
        assert backend.cpu_fallback_applied is False
        # Provider was flipped before the reload attempt; that's fine — a later
        # retry will still see the CUDA-was-requested->CPU transition.
        assert backend.config.sherpa_provider == "cpu"
        assert "reload failed" in detail.lower()


# -- Regression: ORT BFCArena / MemcpyFromHost OOM detection ---------------
#
# The 2026-04-25 production failure on shuvdev (RTX 5080, 28 GiB allocation
# request from a stuck push-to-talk) surfaced as:
#   "Non-zero status code returned while running MemcpyFromHost node.
#    Name:'Memcpy_token_530' Status Message: bfc_arena.cc:359 ...
#    AllocateRawInternal ... Failed to allocate memory for requested
#    buffer of size 30817320960"
# None of the original markers matched, so the per-call fallback never fired.
# These tests pin the broadened markers in place while avoiding false positives
# on generic memcpy failures that are not allocation failures.


class TestLooksLikeCudaOomErrorBfcArena:
    def test_detects_ort_failed_to_allocate_memory(self):
        from shuvoice.asr_sherpa import looks_like_cuda_oom_error

        exc = RuntimeError(
            "Non-zero status code returned while running MemcpyFromHost node. "
            "Name:'Memcpy_token_530' Status Message: "
            "/onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:359 "
            "void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, "
            "onnxruntime::Stream*) Failed to allocate memory for requested "
            "buffer of size 30817320960"
        )
        assert looks_like_cuda_oom_error(exc) is True

    def test_detects_bfc_arena_substring(self):
        from shuvoice.asr_sherpa import looks_like_cuda_oom_error

        assert looks_like_cuda_oom_error(
            RuntimeError("bfc_arena.cc:359 ... AllocateRawInternal ...")
        ) is True

    def test_detects_failed_to_allocate_memory_substring(self):
        from shuvoice.asr_sherpa import looks_like_cuda_oom_error

        assert looks_like_cuda_oom_error(
            RuntimeError("Failed to allocate memory for requested buffer of size 123")
        ) is True

    def test_does_not_treat_generic_memcpy_node_as_oom(self):
        from shuvoice.asr_sherpa import looks_like_cuda_oom_error

        assert looks_like_cuda_oom_error(
            RuntimeError("running MemcpyFromHost node. Name:'Memcpy_token_530'")
        ) is False
        assert looks_like_cuda_oom_error(
            RuntimeError("running MemcpyToHost node. Name:'Memcpy_out'")
        ) is False


# -- Regression: utterance length cap in offline_instant mode ---------------
#
# A stuck PTT (or runaway recording) can push 10+ minutes of audio into the
# offline transducer.  Encoder/joiner activations grow with audio length, so
# this can request many GiB of GPU/CPU memory.  process_utterance() now
# enforces ``sherpa_offline_max_utterance_sec`` by truncating to the
# trailing window before decode.


class TestSherpaOfflineMaxUtteranceCap:
    def _make_loaded_backend(self, tmp_path: Path, **cfg_overrides):
        model_dir = _make_model_dir(tmp_path)
        cfg_kwargs = dict(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            sherpa_model_dir=str(model_dir),
            sherpa_decode_mode="offline_instant",
        )
        cfg_kwargs.update(cfg_overrides)
        cfg = Config(**cfg_kwargs)
        backend = create_backend("sherpa", cfg)

        mock_result = types.SimpleNamespace(text="ok")
        mock_stream = MagicMock()
        mock_stream.result = mock_result

        mock_recognizer = MagicMock()
        mock_recognizer.create_stream.return_value = mock_stream
        backend._offline_recognizer = mock_recognizer
        return backend, mock_stream, mock_recognizer

    def test_truncates_audio_above_cap_to_trailing_window(self, tmp_path: Path, caplog):
        # Cap = 5s @ 16 kHz, audio = 12s
        backend, mock_stream, _ = self._make_loaded_backend(
            tmp_path, sherpa_offline_max_utterance_sec=5.0
        )
        sample_rate = backend.config.sample_rate
        audio = np.arange(12 * sample_rate, dtype=np.float32)

        with caplog.at_level("WARNING", logger="shuvoice.asr_sherpa"):
            backend.process_utterance(audio)

        # accept_waveform should have been called with only the trailing 5s.
        call_args = mock_stream.accept_waveform.call_args
        passed_sample_rate, passed_waveform = call_args.args
        assert passed_sample_rate == sample_rate
        assert passed_waveform.shape == (5 * sample_rate,)
        # Trailing window: last sample of 12s audio is 12*sr-1 (np.arange).
        assert int(passed_waveform[-1]) == 12 * sample_rate - 1
        assert int(passed_waveform[0]) == 7 * sample_rate
        # Warning was emitted explaining the truncation.
        assert any(
            "too long" in record.message.lower() and "truncating" in record.message.lower()
            for record in caplog.records
        )

    def test_does_not_truncate_audio_within_cap(self, tmp_path: Path):
        backend, mock_stream, _ = self._make_loaded_backend(
            tmp_path, sherpa_offline_max_utterance_sec=60.0
        )
        sample_rate = backend.config.sample_rate
        audio = np.arange(3 * sample_rate, dtype=np.float32)

        backend.process_utterance(audio)

        passed_waveform = mock_stream.accept_waveform.call_args.args[1]
        assert passed_waveform.shape == (3 * sample_rate,)
        # Identical to input.
        assert int(passed_waveform[0]) == 0
        assert int(passed_waveform[-1]) == 3 * sample_rate - 1

    def test_zero_cap_disables_truncation(self, tmp_path: Path):
        backend, mock_stream, _ = self._make_loaded_backend(
            tmp_path, sherpa_offline_max_utterance_sec=0.0
        )
        sample_rate = backend.config.sample_rate
        # 200s — well past any reasonable PTT — must pass through untouched.
        audio = np.arange(200 * sample_rate, dtype=np.float32)

        backend.process_utterance(audio)

        passed_waveform = mock_stream.accept_waveform.call_args.args[1]
        assert passed_waveform.shape == (200 * sample_rate,)

    def test_default_cap_is_60_seconds(self, tmp_path: Path):
        """Lock the default in place; lowering it is a behavior change."""
        backend, _, _ = self._make_loaded_backend(tmp_path)
        assert backend.config.sherpa_offline_max_utterance_sec == 60.0
