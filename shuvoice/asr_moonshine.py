"""Moonshine ONNX ASR backend."""

from __future__ import annotations

import logging
import re
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
    Moonshine is a *batch* encoder-decoder: every inference re-encodes
    the full accumulated audio buffer and autoregressively decodes up to
    ``max_tokens`` output tokens.  Four mechanisms keep inference viable
    for real-time streaming:

    1. **Inference throttling** — only run the encoder+decoder when at
       least ``_INFER_INTERVAL_S`` seconds of *new* audio have been
       buffered since the last inference.  Silence chunks (e.g. tail-
       flush padding) bypass the throttle so final accuracy is
       preserved.
    2. **Tokenizer caching** — the HuggingFace tokenizer is loaded once
       at ``load()`` time instead of on every call.
    3. **Deferred buffer concatenation** — incoming chunks are queued and
       coalesced only when an inference step is due, avoiding repeated
       O(n) array copies on every 100 ms chunk.
    4. **Repetition guard** — hallucinated repetition loops (a known
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
    _INFER_INTERVAL_S = 0.50

    # Buffer-level normalization target.  Raw microphone audio is
    # typically 0.01–0.05 RMS; Moonshine expects levels closer to what
    # librosa.load produces (~0.1–0.3 RMS).  We normalize the full
    # buffer once at inference time for a consistent signal level.
    _NORM_TARGET_RMS = 0.10
    _NORM_MAX_GAIN = 15.0
    _NORM_MIN_RMS = 0.001  # below this the buffer is silence — skip normalization

    # Repetition guard limits
    _MAX_WORDS_PER_SEC = 6.0  # generous cap; typical speech ≈ 2-3 wps
    _MAX_CHARS_PER_SEC = 40.0  # generous cap; typical speech ≈ 15-20 chars/s
    _REPETITION_THRESHOLD = 4  # consecutive pattern repeats to trigger cut
    _LONG_REPETITION_THRESHOLD = 2
    _MAX_PATTERN_WORDS = 12
    _MAX_PATTERN_STARTS = 20
    _TOKEN_SPAN_RE = re.compile(r"\S+")
    # Detect token-local repetition (including hyphen-delimited loops) before
    # word-level checks. Examples: "hake-hake-hake-hake", "127127127127".
    # The pattern unit must be ≥2 chars so that single-char runs in normal
    # numbers (e.g. "100000") do not false-positive.
    _TOKEN_REPETITION_RE = re.compile(r"(.{2,10}?)(?:-?\1){3,}")
    # Single-char runs are only pathological at higher repeat counts (≥8).
    # Example: "1270000000000..." — the "0" repeats ≥8 times.
    _SINGLE_CHAR_RUN_RE = re.compile(r"(.)\1{7,}")

    def __init__(self, config: Config):
        self.config = config

        self._model: Any = None
        self._tokenizer: Any = None

        self._sample_rate = int(self.config.sample_rate)
        self._max_window_samples = int(
            float(self.config.moonshine_max_window_sec) * self._sample_rate
        )
        self._min_segment_samples = int(self._MIN_SEGMENT_S * self._sample_rate)
        self._min_infer_samples = int(self._INFER_INTERVAL_S * self._sample_rate)

        # Committed cumulative buffer used for model inference.
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        # Newly arrived chunks since last inference; merged lazily.
        self._pending_chunks: list[np.ndarray] = []
        self._pending_samples = 0

        self._last_text = ""
        self._step_num = 0
        self._samples_since_infer = 0

    @property
    def wants_raw_audio(self) -> bool:
        return True

    @property
    def native_chunk_samples(self) -> int:
        return self._sample_rate * int(self.config.moonshine_chunk_ms) // 1000

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
                f"{e}. Install with: uv sync --extra asr-moonshine"
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

        # --- Track B: ONNX session options tuning ---
        self._tune_onnx_sessions()

        # --- Track D: startup warnings ---
        self._emit_startup_warnings()

        self.reset()

    def _tune_onnx_sessions(self) -> None:
        """Replace upstream ONNX sessions with tuned versions.

        Track B: thread/parallelism tuning.
        Track C: GPU execution provider when ``moonshine_provider = "cuda"``.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            log.debug("onnxruntime not directly importable; skipping session tuning")
            return

        provider = str(self.config.moonshine_provider).strip().lower()
        threads = int(self.config.moonshine_onnx_threads)

        # Determine execution providers.
        if provider == "cuda":
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                log.info("Moonshine: using CUDAExecutionProvider")
            else:
                log.warning(
                    "Moonshine: CUDA provider requested but not available "
                    "(available: %s). Falling back to CPU.",
                    available,
                )
                providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # Build session options.
        import os

        sess_opts = ort.SessionOptions()
        cpu_count = os.cpu_count() or 4
        intra = threads if threads > 0 else max(2, cpu_count // 2)
        sess_opts.intra_op_num_threads = intra
        sess_opts.inter_op_num_threads = max(1, cpu_count // 4)
        sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        # Replace encoder and decoder sessions.
        # The upstream MoonshineOnnxModel stores sessions as self.encoder
        # and self.decoder (onnxruntime.InferenceSession objects).
        replaced = 0
        for attr in ("encoder", "decoder", "uncached_decoder"):
            session = getattr(self._model, attr, None)
            if session is None or not isinstance(session, ort.InferenceSession):
                continue
            try:
                model_path = session._model_path  # noqa: SLF001
                new_session = ort.InferenceSession(
                    model_path,
                    sess_options=sess_opts,
                    providers=providers,
                )
                setattr(self._model, attr, new_session)
                replaced += 1
            except Exception:
                log.debug("Could not replace %s session; keeping default", attr, exc_info=True)

        if replaced:
            log.info(
                "Moonshine: replaced %d ONNX session(s) (threads=%d, provider=%s)",
                replaced,
                intra,
                providers[0],
            )

    def _emit_startup_warnings(self) -> None:
        """Emit actionable startup warnings for slow Moonshine configurations."""
        model_name = str(self.config.moonshine_model_name).lower()
        provider = str(self.config.moonshine_provider).lower()
        max_window = float(self.config.moonshine_max_window_sec)

        if "base" in model_name and provider == "cpu":
            log.warning(
                "Moonshine 'base' model on CPU is very slow for interactive use "
                "(~8s per phrase). Consider switching to moonshine_model_name = "
                "'moonshine/tiny' or setting moonshine_provider = 'cuda' for "
                "better latency."
            )
        elif provider == "cpu":
            log.warning(
                "Moonshine runs on CPU and is slower than NeMo (CUDA) or Sherpa. "
                "Best suited for short utterances (<5s) on systems without GPU support."
            )

        if max_window > 5.0 and provider == "cpu":
            log.warning(
                "moonshine_max_window_sec=%.1f with CPU provider may cause high "
                "latency on long utterances. Consider reducing to 3.0–5.0.",
                max_window,
            )

    def reset(self) -> None:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("ASR backend is not loaded. Call load() first.")

        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._pending_chunks = []
        self._pending_samples = 0
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
        # Queue chunk and coalesce only when inference is actually due.
        # This avoids repeated O(n) concatenations on every chunk.
        self._pending_chunks.append(waveform.copy())
        self._pending_samples += waveform.size

        total_buffered = self._audio_buffer.size + self._pending_samples
        if total_buffered <= self._min_segment_samples:
            return self._last_text

        # --- inference throttle ---
        # Moonshine re-encodes the full buffer each call, so running on
        # every 100ms chunk causes a cascading queue backup.  Skip
        # inference until enough new audio has accumulated.  Silence
        # chunks (tail-flush padding from app.py) always run so that the
        # decoder can finalize output at utterance boundaries.
        self._samples_since_infer += waveform.size
        is_silence = not np.any(waveform)
        if not is_silence and self._samples_since_infer < self._min_infer_samples:
            return self._last_text
        self._samples_since_infer = 0

        self._commit_pending_audio()

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
        audio_seconds = self._audio_buffer.size / float(self._sample_rate)
        text = self._guard_repetition(text, audio_seconds)

        self._last_text = text
        self._step_num += 1
        return text

    def _commit_pending_audio(self) -> None:
        """Merge queued chunks into the committed inference buffer."""
        if not self._pending_chunks:
            return

        if len(self._pending_chunks) == 1:
            pending = self._pending_chunks[0]
        else:
            pending = np.concatenate(self._pending_chunks)

        if self._audio_buffer.size == 0:
            merged = pending
        else:
            merged = np.concatenate([self._audio_buffer, pending])

        if merged.size > self._max_window_samples:
            merged = merged[-self._max_window_samples :]

        self._audio_buffer = merged
        self._pending_chunks = []
        self._pending_samples = 0

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

        Checks are applied in this order:
        1. **Token-local repetition** — catches loops inside a single token,
           such as ``hake-hake-hake-hake`` or ``127127127127``.
        2. **Character-count cap** — bounds huge single-token outputs that can
           bypass word-count limits.
        3. **Word-count cap** — at most ~6 words per second of audio.
        4. **N-gram repetition** — catches repeated clauses in 1–12 word windows.
        """
        if not text:
            return text

        # 0. Token-level repetition guard (before word split / short-text return).
        for token_match in cls._TOKEN_SPAN_RE.finditer(text):
            token = token_match.group(0)

            # Try multi-char pattern first (≥2 char units repeated ≥4 times).
            repeated = cls._TOKEN_REPETITION_RE.search(token)
            if repeated is not None:
                kept_len = repeated.start() + len(repeated.group(1))
                kept_token = token[:kept_len]
                text = f"{text[: token_match.start()]}{kept_token}{text[token_match.end() :]}"
                log.debug(
                    "Repetition guard: token has repeated multi-char pattern (len=%d), truncating",
                    len(repeated.group(1)),
                )
                break

            # Try single-char run (e.g. "0" × 8+).
            char_run = cls._SINGLE_CHAR_RUN_RE.search(token)
            if char_run is not None:
                # Keep prefix + one instance of the repeated char.
                kept_len = char_run.start() + 1
                kept_token = token[:kept_len]
                text = f"{text[: token_match.start()]}{kept_token}{text[token_match.end() :]}"
                log.debug(
                    "Repetition guard: token has single-char run (char=%r, count=%d), truncating",
                    char_run.group(1),
                    len(char_run.group(0)),
                )
                break

        # 1. Character-count cap catches long single-token runs.
        max_chars = max(100, int(audio_seconds * cls._MAX_CHARS_PER_SEC) + 20)
        if len(text) > max_chars:
            truncated = text[:max_chars]
            if " " in truncated:
                boundary = truncated.rsplit(" ", 1)[0]
                if boundary:
                    truncated = boundary
            if not truncated:
                truncated = text[:max_chars]

            log.debug(
                "Repetition guard: char count %d exceeds cap %d for %.1fs audio",
                len(text),
                max_chars,
                audio_seconds,
            )
            text = truncated

        words = text.split()
        if len(words) <= 5:
            return text

        # 2. Hard cap: prevent returning enormous hallucinated strings.
        max_words = max(10, int(audio_seconds * cls._MAX_WORDS_PER_SEC) + 5)
        if len(words) > max_words:
            log.debug(
                "Repetition guard: word count %d exceeds cap %d for %.1fs audio",
                len(words),
                max_words,
                audio_seconds,
            )
            words = words[:max_words]

        # 3. N-gram repetition: find repeated 1–12 word patterns.
        # Pre-compute normalized words once (lowercase, strip punctuation).
        _PUNCT_STRIP = str.maketrans("", "", ".,!?;:'\"")
        norm_words = [w.lower().translate(_PUNCT_STRIP) for w in words]

        for plen in range(1, cls._MAX_PATTERN_WORDS + 1):
            threshold = cls._REPETITION_THRESHOLD if plen <= 4 else cls._LONG_REPETITION_THRESHOLD
            min_words = plen * threshold
            if len(words) < min_words:
                continue

            # Scan from all possible start positions (not capped) for long
            # patterns ≥5 words to catch clause loops that start late.
            if plen >= 5:
                start_limit = len(words) - min_words + 1
            else:
                start_limit = min(len(words) - min_words + 1, cls._MAX_PATTERN_STARTS)

            for start in range(start_limit):
                pattern = tuple(norm_words[start : start + plen])
                # Skip patterns that are all-empty after normalization.
                if not any(pattern):
                    continue
                count = 0
                pos = start
                while pos + plen <= len(words):
                    candidate = tuple(norm_words[pos : pos + plen])
                    if candidate == pattern:
                        count += 1
                        pos += plen
                    else:
                        break

                if count >= threshold:
                    kept = words[: start + plen]
                    log.debug(
                        "Repetition guard: %d-word pattern repeated %d× at word %d "
                        "(threshold=%d), truncating",
                        plen,
                        count,
                        start,
                        threshold,
                    )
                    return " ".join(kept)

        return " ".join(words)
