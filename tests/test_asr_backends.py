"""Integration tests for Sherpa and Moonshine ASR backends.

These tests mock the external sherpa_onnx and moonshine_onnx libraries and
exercise the full load -> process_chunk -> reset lifecycle, catching
integration issues without requiring real model files.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from shuvoice.config import Config


# ---------------------------------------------------------------------------
# Sherpa backend tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def sherpa_model_dir(tmp_path: Path) -> Path:
    """Create a minimal fake Sherpa model directory."""
    (tmp_path / "tokens.txt").write_text("<blk>\na\nb\n")
    (tmp_path / "encoder.onnx").write_bytes(b"\x00")
    (tmp_path / "decoder.onnx").write_bytes(b"\x00")
    (tmp_path / "joiner.onnx").write_bytes(b"\x00")
    return tmp_path


@pytest.fixture()
def fake_sherpa_module() -> types.ModuleType:
    """Build a fake ``sherpa_onnx`` module with a usable OnlineRecognizer."""
    mod = types.ModuleType("sherpa_onnx")

    fake_stream = MagicMock(name="OnlineStream")
    fake_stream.accept_waveform = MagicMock()

    fake_recognizer = MagicMock(name="OnlineRecognizer")
    fake_recognizer.create_stream.return_value = fake_stream
    fake_recognizer.is_ready.return_value = False
    fake_recognizer.decode_stream = MagicMock()
    fake_recognizer.get_result.return_value = ""

    recognizer_cls = MagicMock(name="OnlineRecognizerClass")
    recognizer_cls.from_transducer = MagicMock(return_value=fake_recognizer)

    mod.OnlineRecognizer = recognizer_cls  # type: ignore[attr-defined]

    # Attach inner mocks for assertion convenience.
    mod._fake_recognizer = fake_recognizer  # type: ignore[attr-defined]
    mod._fake_stream = fake_stream  # type: ignore[attr-defined]

    return mod


@pytest.fixture()
def sherpa_backend(monkeypatch, sherpa_model_dir, fake_sherpa_module):
    """Return a loaded SherpaBackend backed by fakes."""
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa_module)

    from shuvoice.asr import create_backend

    cfg = Config(asr_backend="sherpa", sherpa_model_dir=str(sherpa_model_dir))
    backend = create_backend("sherpa", cfg)
    backend.load()
    return backend


class TestSherpaBackendLoad:
    """Verify that load() initialises the recognizer and creates a stream."""

    def test_load_creates_recognizer_via_from_transducer(
        self, sherpa_backend, fake_sherpa_module
    ):
        recognizer_cls = fake_sherpa_module.OnlineRecognizer
        recognizer_cls.from_transducer.assert_called_once()

    def test_load_creates_initial_stream(self, sherpa_backend, fake_sherpa_module):
        fake_recognizer = fake_sherpa_module._fake_recognizer
        fake_recognizer.create_stream.assert_called_once()

    def test_stream_is_set_after_load(self, sherpa_backend):
        assert sherpa_backend._stream is not None
        assert sherpa_backend._recognizer is not None


class TestSherpaBackendProcessChunk:
    """Verify process_chunk drives the accept/decode/result loop."""

    def test_process_chunk_calls_accept_waveform(
        self, sherpa_backend, fake_sherpa_module
    ):
        fake_recognizer = fake_sherpa_module._fake_recognizer
        fake_stream = fake_sherpa_module._fake_stream

        fake_recognizer.is_ready.return_value = False
        fake_recognizer.get_result.return_value = ""

        audio = np.zeros(1600, dtype=np.float32)
        sherpa_backend.process_chunk(audio)

        fake_stream.accept_waveform.assert_called_once()
        args = fake_stream.accept_waveform.call_args
        assert args[0][0] == 16000  # sample_rate

    def test_process_chunk_calls_decode_stream_while_ready(
        self, sherpa_backend, fake_sherpa_module
    ):
        fake_recognizer = fake_sherpa_module._fake_recognizer
        fake_stream = fake_sherpa_module._fake_stream

        # Simulate three ready rounds then stop.
        fake_recognizer.is_ready.side_effect = [True, True, True, False]
        fake_recognizer.get_result.return_value = "decoded"

        audio = np.ones(1600, dtype=np.float32)
        sherpa_backend.process_chunk(audio)

        assert fake_recognizer.decode_stream.call_count == 3

    def test_process_chunk_calls_get_result(
        self, sherpa_backend, fake_sherpa_module
    ):
        fake_recognizer = fake_sherpa_module._fake_recognizer

        fake_recognizer.is_ready.return_value = False
        fake_recognizer.get_result.return_value = "final"

        audio = np.zeros(1600, dtype=np.float32)
        result = sherpa_backend.process_chunk(audio)

        fake_recognizer.get_result.assert_called()
        assert result == "final"

    def test_process_chunk_returns_stripped_text(
        self, sherpa_backend, fake_sherpa_module
    ):
        """Trailing/leading whitespace should be stripped from the result."""
        fake_recognizer = fake_sherpa_module._fake_recognizer

        fake_recognizer.is_ready.side_effect = [True, False]
        fake_recognizer.get_result.return_value = "  hello world  "

        audio = np.zeros(1600, dtype=np.float32)
        result = sherpa_backend.process_chunk(audio)

        assert result == "hello world"

    def test_process_chunk_handles_result_object_with_text_attr(
        self, sherpa_backend, fake_sherpa_module
    ):
        """When get_result returns an object with a .text attribute, use it."""
        fake_recognizer = fake_sherpa_module._fake_recognizer

        fake_recognizer.is_ready.return_value = False
        result_obj = types.SimpleNamespace(text="from attribute")
        fake_recognizer.get_result.return_value = result_obj

        text = sherpa_backend.process_chunk(np.zeros(1600, dtype=np.float32))
        assert text == "from attribute"


class TestSherpaBackendReset:
    """Verify reset() creates a fresh stream."""

    def test_reset_creates_new_stream(self, sherpa_backend, fake_sherpa_module):
        fake_recognizer = fake_sherpa_module._fake_recognizer

        new_stream = MagicMock(name="NewStream")
        fake_recognizer.create_stream.return_value = new_stream

        sherpa_backend.reset()

        assert sherpa_backend._stream is new_stream
        # create_stream called once during load() + once during reset().
        assert fake_recognizer.create_stream.call_count == 2


# ---------------------------------------------------------------------------
# Moonshine backend tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_moonshine_module() -> types.ModuleType:
    """Build a fake ``moonshine_onnx`` module."""
    mod = types.ModuleType("moonshine_onnx")

    fake_model = MagicMock(name="MoonshineOnnxModel")
    model_cls = MagicMock(name="MoonshineOnnxModelClass", return_value=fake_model)
    mod.MoonshineOnnxModel = model_cls  # type: ignore[attr-defined]

    mod.transcribe = MagicMock(return_value=["transcribed text"])  # type: ignore[attr-defined]

    # Keep reference for assertion convenience.
    mod._fake_model = fake_model  # type: ignore[attr-defined]

    return mod


@pytest.fixture()
def moonshine_backend(monkeypatch, fake_moonshine_module):
    """Return a loaded MoonshineBackend backed by fakes."""
    monkeypatch.setitem(sys.modules, "moonshine_onnx", fake_moonshine_module)

    from shuvoice.asr import create_backend

    cfg = Config(asr_backend="moonshine")
    backend = create_backend("moonshine", cfg)
    backend.load()
    return backend


class TestMoonshineBackendLoad:
    """Verify that load() initialises the model and module reference."""

    def test_load_creates_model(self, moonshine_backend, fake_moonshine_module):
        fake_moonshine_module.MoonshineOnnxModel.assert_called_once()
        assert moonshine_backend._model is not None

    def test_moonshine_ref_is_set_after_load(self, moonshine_backend, fake_moonshine_module):
        assert moonshine_backend._moonshine is fake_moonshine_module

    def test_load_resets_audio_buffer(self, moonshine_backend):
        """After load(), the audio buffer should be empty."""
        assert moonshine_backend._audio_buffer.size == 0
        assert moonshine_backend._last_text == ""
        assert moonshine_backend._step_num == 0


class TestMoonshineBackendProcessChunk:
    """Verify process_chunk accumulates audio and respects minimum segment threshold."""

    def test_process_chunk_below_threshold_returns_empty(
        self, moonshine_backend, fake_moonshine_module
    ):
        """Audio shorter than _MIN_SEGMENT_S should NOT trigger transcription."""
        # _MIN_SEGMENT_S = 0.35 at 16 kHz => 5600 samples minimum.
        short_audio = np.zeros(1000, dtype=np.float32)
        text = moonshine_backend.process_chunk(short_audio)

        fake_moonshine_module.transcribe.assert_not_called()
        assert text == ""

    def test_process_chunk_above_threshold_calls_transcribe(
        self, moonshine_backend, fake_moonshine_module
    ):
        """Once audio exceeds the minimum segment, transcribe should be called."""
        audio = np.ones(6000, dtype=np.float32) * 0.1
        text = moonshine_backend.process_chunk(audio)

        fake_moonshine_module.transcribe.assert_called_once()
        call_kwargs = fake_moonshine_module.transcribe.call_args
        assert call_kwargs[1]["model"] is fake_moonshine_module._fake_model
        assert text == "transcribed text"

    def test_process_chunk_accumulates_across_calls(
        self, moonshine_backend, fake_moonshine_module
    ):
        """Multiple small chunks should be concatenated until threshold is met."""
        chunk = np.zeros(2000, dtype=np.float32)

        # First two chunks: 2000 + 2000 = 4000 < 5600 threshold.
        moonshine_backend.process_chunk(chunk)
        moonshine_backend.process_chunk(chunk)
        fake_moonshine_module.transcribe.assert_not_called()

        # Third chunk: 4000 + 2000 = 6000 > 5600 => transcription fires.
        moonshine_backend.process_chunk(chunk)
        fake_moonshine_module.transcribe.assert_called_once()

        # Verify the buffer passed to transcribe has the right length.
        buffer_arg = fake_moonshine_module.transcribe.call_args[0][0]
        assert len(buffer_arg) == 6000

    def test_process_chunk_returns_empty_for_empty_waveform(
        self, moonshine_backend, fake_moonshine_module
    ):
        """An empty waveform should return _last_text without transcribing."""
        empty_audio = np.zeros(0, dtype=np.float32)
        result = moonshine_backend.process_chunk(empty_audio)

        assert result == ""
        fake_moonshine_module.transcribe.assert_not_called()

    def test_process_chunk_increments_step_num(
        self, moonshine_backend, fake_moonshine_module
    ):
        """Each successful transcription should bump debug_step_num."""
        audio = np.ones(6000, dtype=np.float32)

        moonshine_backend.process_chunk(audio)
        assert moonshine_backend.debug_step_num == 1

        fake_moonshine_module.transcribe.return_value = ["second pass"]
        moonshine_backend.process_chunk(audio)
        assert moonshine_backend.debug_step_num == 2


class TestMoonshineBackendReset:
    """Verify reset() clears the audio buffer and internal state."""

    def test_reset_clears_audio_buffer(self, moonshine_backend):
        """reset() should empty the audio buffer."""
        audio = np.ones(6000, dtype=np.float32)
        moonshine_backend.process_chunk(audio)
        assert moonshine_backend._audio_buffer.size > 0

        moonshine_backend.reset()

        assert moonshine_backend._audio_buffer.size == 0

    def test_reset_clears_last_text(self, moonshine_backend, fake_moonshine_module):
        """reset() should clear the cached last text."""
        audio = np.ones(6000, dtype=np.float32)
        moonshine_backend.process_chunk(audio)
        assert moonshine_backend._last_text == "transcribed text"

        moonshine_backend.reset()

        assert moonshine_backend._last_text == ""

    def test_reset_clears_step_num(self, moonshine_backend, fake_moonshine_module):
        """reset() should zero the step counter."""
        audio = np.ones(6000, dtype=np.float32)
        moonshine_backend.process_chunk(audio)
        assert moonshine_backend._step_num == 1

        moonshine_backend.reset()

        assert moonshine_backend._step_num == 0
