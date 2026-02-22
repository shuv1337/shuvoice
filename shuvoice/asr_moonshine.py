"""Moonshine ONNX ASR backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .asr_base import ASRBackend
from .config import Config

log = logging.getLogger(__name__)


class MoonshineBackend(ASRBackend):
    """Moonshine ONNX backend with chunk-wise cumulative decoding.

    Performance notes
    -----------------
    Moonshine is a *batch* encoder-decoder: every ``process_chunk`` call
    re-encodes the full accumulated audio buffer and autoregressively
    decodes up to ``max_tokens`` output tokens.  Three mechanisms keep
    inference viable for real-time streaming:

    1. **Inference throttling** — only run the encoder+decoder when at
       least ``_INFER_INTERVAL_S`` seconds of *new* audio have been
       buffered since the last inference.  Silence chunks (e.g. tail-
       flush padding) bypass the throttle so final accuracy is
       preserved.
    2. **Tokenizer caching** — the HuggingFace tokenizer is loaded once
       at ``load()`` time instead of on every call.
    3. **Repetition guard** — hallucinated repetition loops (a known
       transformer failure mode) are detected and truncated before the
       result is returned.
    """

    _EXPECTED_SAMPLE_RATE = 16000
    # The upstream helper asserts >0.1s, but very short windows often decode as
    # repetitive junk (e.g., repeated "W."). Use a safer minimum segment.
    _MIN_SEGMENT_S = 0.35
    _MAX_SEGMENT_S = 64.0
    _REPO_URL = "https://github.com/moonshine-ai/moonshine"

    # Only run inference after accumulating this much new audio.  Between
    # inference calls the cached ``_last_text`` is returned immediately.
    # Silence chunks (all zeros, used for tail-flush) bypass the throttle.
    _INFER_INTERVAL_S = 0.30

    # Buffer-level normalization target.  Raw microphone audio is
    # typically 0.01–0.05 RMS; Moonshine expects levels closer to what
    # librosa.load produces (~0.1–0.3 RMS).  We normalize the full
    # buffer once at inference time for a consistent signal level.
    _NORM_TARGET_RMS = 0.10
    _NORM_MAX_GAIN = 15.0
    _NORM_MIN_RMS = 0.001  # below this the buffer is silence — skip normalization

    # Repetition guard limits
    _MAX_WORDS_PER_SEC = 6.0  # generous cap; typical speech ≈ 2-3 wps
    _REPETITION_THRESHOLD = 4  # consecutive pattern repeats to trigger cut

    def __init__(self, config: Config):
        self.config = config

        self._model: Any = None
        self._tokenizer: Any = None

        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._last_text = ""
        self._step_num = 0
        self._samples_since_infer = 0

    @property
    def wants_raw_audio(self) -> bool:
        return True

    @property
    def native_chunk_samples(self) -> int:
        return int(self.config.sample_rate) * int(self.config.moonshine_chunk_ms) // 1000

    @property
    def debug_step_num(self) -> int | None:
        return self._step_num

    @staticmethod
    def dependency_errors() -> list[str]:
        errors: list[str] = []

        try:
            import moonshine_onnx  # noqa: F401
        except Exception as e:
            errors.append(
                "Missing Moonshine ONNX dependency: "
                f"{e}. Install with: pip install useful-moonshine-onnx"
            )

        return errors

    @classmethod
    def download_model(cls, **kwargs) -> None:
        raise NotImplementedError(
            "Moonshine model pre-download is not implemented. "
            "Models are fetched lazily on first load from Hugging Face via useful-moonshine-onnx. "
            f"See: {cls._REPO_URL}"
        )

    def _validate_runtime_config(self) -> None:
        if int(self.config.sample_rate) != self._EXPECTED_SAMPLE_RATE:
            raise ValueError(
                "Moonshine backend currently requires sample_rate=16000 "
                f"(got {self.config.sample_rate})."
            )

        if not str(self.config.moonshine_model_name).strip():
            raise ValueError("moonshine_model_name must not be empty")

        model_dir_raw = self.config.moonshine_model_dir
        if model_dir_raw:
            model_dir = Path(model_dir_raw).expanduser()
            if not model_dir.is_dir():
                raise ValueError(
                    f"moonshine_model_dir does not exist or is not a directory: {model_dir}"
                )

            required = [
                model_dir / "encoder_model.onnx",
                model_dir / "decoder_model_merged.onnx",
            ]
            missing = [str(p.name) for p in required if not p.is_file()]
            if missing:
                raise ValueError(
                    "moonshine_model_dir is missing required model artifacts: " + ", ".join(missing)
                )

        max_window = float(self.config.moonshine_max_window_sec)
        if max_window <= 0 or max_window > self._MAX_SEGMENT_S:
            raise ValueError(
                "moonshine_max_window_sec must be > 0 and <= 64 "
                f"(got {self.config.moonshine_max_window_sec})."
            )

    def load(self) -> None:
        self._validate_runtime_config()

        errors = self.dependency_errors()
        if errors:
            raise RuntimeError("\n".join(errors))

        import moonshine_onnx

        model_name = str(self.config.moonshine_model_name).strip()
        model_precision = str(self.config.moonshine_model_precision).strip()
        model_dir = self.config.moonshine_model_dir

        kwargs: dict[str, Any] = {
            "model_name": model_name,
            "model_precision": model_precision,
        }
        if model_dir:
            kwargs["models_dir"] = str(Path(model_dir).expanduser())

        try:
            self._model = moonshine_onnx.MoonshineOnnxModel(**kwargs)
            self._tokenizer = moonshine_onnx.load_tokenizer()
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize Moonshine ONNX backend. "
                "Check moonshine_model_name/model_precision/model_dir configuration."
            ) from e

        self.reset()

    def reset(self) -> None:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("ASR backend is not loaded. Call load() first.")

        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._last_text = ""
        self._step_num = 0
        self._samples_since_infer = 0

    def process_chunk(self, audio_chunk: np.ndarray) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("ASR backend is not loaded. Call load() first.")

        waveform = np.asarray(audio_chunk, dtype=np.float32)
        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)

        if waveform.size == 0:
            return self._last_text

        # --- buffer management ---
        self._audio_buffer = np.concatenate([self._audio_buffer, waveform])

        max_samples = int(
            float(self.config.moonshine_max_window_sec) * int(self.config.sample_rate)
        )
        if self._audio_buffer.size > max_samples:
            self._audio_buffer = self._audio_buffer[-max_samples:]

        min_samples = int(self._MIN_SEGMENT_S * int(self.config.sample_rate))
        if self._audio_buffer.size <= min_samples:
            return self._last_text

        # --- inference throttle ---
        # Moonshine re-encodes the full buffer each call, so running on
        # every 100ms chunk causes a cascading queue backup.  Skip
        # inference until enough new audio has accumulated.  Silence
        # chunks (tail-flush padding from app.py) always run so that the
        # decoder can finalize output at utterance boundaries.
        self._samples_since_infer += waveform.size
        is_silence = float(np.max(np.abs(waveform))) == 0.0
        min_infer_samples = int(self._INFER_INTERVAL_S * int(self.config.sample_rate))
        if not is_silence and self._samples_since_infer < min_infer_samples:
            return self._last_text
        self._samples_since_infer = 0

        # --- buffer-level normalization ---
        # Raw microphone audio is very quiet (~0.01 RMS).  Normalize the
        # full buffer to a consistent level so the encoder sees a signal
        # similar to librosa.load output (what the model was trained on).
        # This replaces the app's per-chunk gain which creates wildly
        # inconsistent levels across the buffer (40× on early noise, 5×
        # on speech) and degrades quality for batch re-encoding models.
        audio_for_model = self._normalize_buffer(self._audio_buffer)

        # --- run encoder + decoder ---
        # Call model.generate() directly instead of going through
        # moonshine_onnx.transcribe() which reloads the tokenizer from
        # disk on every call.
        try:
            audio_2d = audio_for_model[np.newaxis, :]
            max_tokens = int(self.config.moonshine_max_tokens)
            tokens = self._model.generate(audio_2d, max_len=max_tokens)
            text = self._tokenizer.decode_batch(tokens)[0].strip()
        except Exception:
            log.exception("Moonshine inference failed")
            raise RuntimeError("Moonshine inference failed")

        # --- repetition guard ---
        audio_seconds = self._audio_buffer.size / float(self.config.sample_rate)
        text = self._guard_repetition(text, audio_seconds)

        self._last_text = text
        self._step_num += 1
        return text

    @classmethod
    def _normalize_buffer(cls, buf: np.ndarray) -> np.ndarray:
        """Uniform RMS normalization of the full audio buffer.

        Returns a copy scaled so that the overall RMS matches
        ``_NORM_TARGET_RMS``, capped at ``_NORM_MAX_GAIN`` to avoid
        amplifying silence.
        """
        rms = float(np.sqrt(np.mean(buf * buf)))
        if rms < cls._NORM_MIN_RMS:
            return buf  # pure silence — nothing to normalize
        gain = min(cls._NORM_TARGET_RMS / rms, cls._NORM_MAX_GAIN)
        if gain <= 1.05:
            return buf  # already at target level
        return np.clip(buf * gain, -1.0, 1.0).astype(np.float32)

    @classmethod
    def _guard_repetition(cls, text: str, audio_seconds: float) -> str:
        """Detect and truncate repetitive hallucination output.

        Two checks:
        1. **Word-count cap** — at most ~6 words per second of audio.
           Hallucination loops easily exceed this (e.g. 200 words for 2s
           of audio).
        2. **Pattern detection** — any 1-4 word n-gram repeating ≥ 4
           consecutive times is truncated to a single occurrence.
        """
        words = text.split()
        if len(words) <= 5:
            return text

        # 1. Hard cap: prevent returning enormous hallucinated strings
        max_words = max(10, int(audio_seconds * cls._MAX_WORDS_PER_SEC) + 5)
        if len(words) > max_words:
            log.debug(
                "Repetition guard: word count %d exceeds cap %d for %.1fs audio",
                len(words),
                max_words,
                audio_seconds,
            )
            words = words[:max_words]

        # 2. N-gram repetition: find patterns (1–4 words) repeating 4+ times
        threshold = cls._REPETITION_THRESHOLD
        for plen in range(1, 5):
            if len(words) < plen * threshold:
                continue
            # Check starting positions within the first few words
            for start in range(min(len(words) - plen * threshold + 1, 8)):
                pattern = tuple(w.lower().strip(".,!?;:'\"") for w in words[start : start + plen])
                count = 0
                pos = start
                while pos + plen <= len(words):
                    candidate = tuple(w.lower().strip(".,!?;:'\"") for w in words[pos : pos + plen])
                    if candidate == pattern:
                        count += 1
                        pos += plen
                    else:
                        break
                if count >= threshold:
                    # Keep everything before the run + one instance
                    kept = words[: start + plen]
                    log.debug(
                        "Repetition guard: pattern %r repeated %d× at word %d, truncating",
                        pattern,
                        count,
                        start,
                    )
                    return " ".join(kept)

        return " ".join(words)
