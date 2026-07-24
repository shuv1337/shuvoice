"""RealMoonshineEngine tests with a stubbed moonshine_onnx module.

Regression coverage for the token-decode bug: generate() returns integer
token IDs, and the engine must decode them through the tokenizer rather
than stringifying them. Requires numpy; skipped in stdlib-only environments.
"""

from __future__ import annotations

import sys
import types
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - stdlib-only CI
    np = None

from moonshine_asr.engine import EXPECTED_SAMPLE_RATE, MoonshineLoadConfig, RealMoonshineEngine


class StubModel:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.generate_calls = []

    def generate(self, audio, max_len=None):
        self.generate_calls.append({"shape": audio.shape, "max_len": max_len})
        return [[1, 843, 220]]


class StubTokenizer:
    def __init__(self):
        self.decoded = []

    def decode_batch(self, tokens):
        self.decoded.append(tokens)
        return [" hello world "]


def install_stub_moonshine(model_holder: dict, tokenizer: StubTokenizer):
    mod = types.ModuleType("moonshine_onnx")

    def make_model(**kwargs):
        model = StubModel(**kwargs)
        model_holder["model"] = model
        return model

    mod.MoonshineOnnxModel = make_model
    mod.load_tokenizer = lambda: tokenizer
    sys.modules["moonshine_onnx"] = mod
    return mod


@unittest.skipUnless(np is not None, "numpy not available")
class RealMoonshineEngineTests(unittest.TestCase):
    def setUp(self):
        self.holder = {}
        self.tokenizer = StubTokenizer()
        install_stub_moonshine(self.holder, self.tokenizer)
        self.addCleanup(sys.modules.pop, "moonshine_onnx", None)

        self.engine = RealMoonshineEngine()
        self.engine.load(
            MoonshineLoadConfig(model_name="moonshine/tiny", max_tokens=32, max_window_sec=5.0)
        )

    def test_load_passes_precision_and_models_dir(self):
        engine = RealMoonshineEngine()
        engine.load(
            MoonshineLoadConfig(
                model_name="moonshine/base",
                model_precision="quantized",
                model_dir="/models/moonshine",
            )
        )
        kwargs = self.holder["model"].init_kwargs
        self.assertEqual(kwargs["model_name"], "moonshine/base")
        self.assertEqual(kwargs["model_precision"], "quantized")
        self.assertEqual(kwargs["models_dir"], "/models/moonshine")

    def test_process_chunk_decodes_tokens_to_text(self):
        # One second of audio: past the min-segment and throttle thresholds.
        chunk = [0.05] * EXPECTED_SAMPLE_RATE
        text = self.engine.process_chunk(chunk)
        self.assertEqual(text, "hello world")
        self.assertEqual(len(self.tokenizer.decoded), 1)
        self.assertEqual(self.tokenizer.decoded[0], [[1, 843, 220]])

    def test_generate_receives_2d_audio_with_max_len(self):
        chunk = [0.05] * EXPECTED_SAMPLE_RATE
        self.engine.process_chunk(chunk)
        call = self.holder["model"].generate_calls[0]
        self.assertEqual(len(call["shape"]), 2)
        self.assertEqual(call["shape"][0], 1)
        self.assertEqual(call["max_len"], 32)

    def test_finish_decodes_throttled_tail_audio(self):
        # First inference consumes the throttle budget.
        self.engine.process_chunk([0.05] * EXPECTED_SAMPLE_RATE)
        # A short trailing chunk is below the throttle threshold: no new decode.
        calls_before = len(self.holder["model"].generate_calls)
        self.engine.process_chunk([0.05] * (EXPECTED_SAMPLE_RATE // 10))
        self.assertEqual(len(self.holder["model"].generate_calls), calls_before)
        # finish() must decode the tail so final text covers all audio.
        text = self.engine.finish()
        self.assertEqual(text, "hello world")
        self.assertEqual(len(self.holder["model"].generate_calls), calls_before + 1)

    def test_silence_bypasses_throttle(self):
        self.engine.process_chunk([0.05] * EXPECTED_SAMPLE_RATE)
        calls_before = len(self.holder["model"].generate_calls)
        # All-zero tail-flush chunk must trigger a decode despite the throttle.
        self.engine.process_chunk([0.0] * (EXPECTED_SAMPLE_RATE // 10))
        self.assertEqual(len(self.holder["model"].generate_calls), calls_before + 1)

    def test_window_cap_bounds_buffer(self):
        for _ in range(8):
            self.engine.process_chunk([0.05] * EXPECTED_SAMPLE_RATE)
        self.assertLessEqual(self.engine._buffer.size, 5 * EXPECTED_SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()
