"""Moonshine ASR engine boundary — optional worker with fake for tests."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

EXPECTED_SAMPLE_RATE = 16000

# Decode pacing / signal conditioning, ported from the pre-rewrite backend
# (shuvoice/asr_moonshine.py). Moonshine is a batch encoder-decoder: every
# inference re-encodes the full accumulated buffer, so inference is throttled
# and the buffer is RMS-normalized once per inference call.
_MIN_SEGMENT_S = 0.35
_INFER_INTERVAL_S = 0.50
_NORM_TARGET_RMS = 0.10
_NORM_MAX_GAIN = 15.0
_NORM_MIN_RMS = 0.001  # below this the buffer is silence — skip normalization

# Repetition guard limits (hallucinated repetition loops are a known
# transformer failure mode; detect and truncate before returning).
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

_MIN_SEGMENT_SAMPLES = int(_MIN_SEGMENT_S * EXPECTED_SAMPLE_RATE)
_MIN_INFER_SAMPLES = int(_INFER_INTERVAL_S * EXPECTED_SAMPLE_RATE)


def dependency_errors() -> list[str]:
    errors: list[str] = []
    try:
        import moonshine_onnx  # noqa: F401
    except Exception:
        try:
            import useful_moonshine_onnx  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"Missing Moonshine ONNX dependency: {type(exc).__name__}. "
                "Install useful-moonshine-onnx."
            )
    return errors


@dataclass
class MoonshineLoadConfig:
    model_name: str = "moonshine/tiny"
    max_tokens: int = 64
    max_window_sec: float = 5.0
    provider: str = "cpu"
    num_threads: int = 0
    model_precision: str = "float"
    model_dir: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> MoonshineLoadConfig:
        raw = raw or {}
        max_window = raw.get("max_window_sec", raw.get("moonshine_max_window_sec", 5.0))
        try:
            max_window_f = float(max_window)
        except (TypeError, ValueError):
            max_window_f = 5.0
        if max_window_f <= 0:
            max_window_f = 5.0
        # Hard cap to avoid pathological buffers even if host misconfigures.
        max_window_f = min(max_window_f, 30.0)
        max_tokens = raw.get("max_tokens", raw.get("moonshine_max_tokens", 64))
        try:
            max_tokens_i = int(max_tokens)
        except (TypeError, ValueError):
            max_tokens_i = 64
        threads = raw.get("num_threads", raw.get("moonshine_onnx_threads", 0))
        try:
            threads_i = int(threads)
        except (TypeError, ValueError):
            threads_i = 0
        model_dir = raw.get("model_dir", raw.get("moonshine_model_dir"))
        return cls(
            model_name=str(
                raw.get("model_name") or raw.get("moonshine_model_name") or cls.model_name
            ),
            max_tokens=max(1, max_tokens_i),
            max_window_sec=max_window_f,
            provider=str(raw.get("provider") or raw.get("moonshine_provider") or "cpu"),
            num_threads=max(0, threads_i),
            model_precision=str(
                raw.get("model_precision") or raw.get("moonshine_model_precision") or "float"
            ),
            model_dir=str(model_dir) if model_dir else None,
        )

    @property
    def max_window_samples(self) -> int:
        return max(1, int(self.max_window_sec * EXPECTED_SAMPLE_RATE))


class AsrEngine(Protocol):
    def load(self, config: MoonshineLoadConfig) -> None: ...
    def reset(self) -> None: ...
    def process_chunk(self, samples: list[float]) -> str: ...
    def finish(self, timeout_ms: object | None = None) -> str: ...
    @property
    def loaded(self) -> bool: ...
    @property
    def model_name(self) -> str: ...
    @property
    def max_window_sec(self) -> float: ...


class FakeMoonshineEngine:
    def __init__(self) -> None:
        self._loaded = False
        self._cfg = MoonshineLoadConfig()
        self._samples = 0
        self._last = ""

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._cfg.model_name

    @property
    def max_window_sec(self) -> float:
        return self._cfg.max_window_sec

    def load(self, config: MoonshineLoadConfig) -> None:
        self._cfg = config
        self._loaded = True
        self.reset()

    def reset(self) -> None:
        if not self._loaded:
            raise RuntimeError("not_loaded")
        self._samples = 0
        self._last = ""

    def process_chunk(self, samples: list[float]) -> str:
        if not self._loaded:
            raise RuntimeError("not_loaded")
        # Honor max window: count only the retained trailing window.
        self._samples = min(
            self._samples + len(samples),
            self._cfg.max_window_samples,
        )
        self._last = f"moon-samples-{self._samples}"
        return self._last

    def finish(self, timeout_ms: object | None = None) -> str:
        if not self._loaded:
            raise RuntimeError("not_loaded")
        _ = timeout_ms
        return self._last


def _normalize_buffer(buf: Any) -> Any:
    """Uniform RMS normalization of the full audio buffer.

    Returns a copy scaled so that the overall RMS matches
    ``_NORM_TARGET_RMS``, capped at ``_NORM_MAX_GAIN`` to avoid
    amplifying silence.
    """
    import numpy as np

    rms = float(np.sqrt(np.dot(buf, buf) / buf.size))
    if rms < _NORM_MIN_RMS:
        return buf  # pure silence — nothing to normalize
    gain = min(_NORM_TARGET_RMS / rms, _NORM_MAX_GAIN)
    if gain <= 1.05:
        return buf  # already at target level
    return np.clip(buf * gain, -1.0, 1.0).astype(np.float32)


def _guard_repetition(text: str, audio_seconds: float) -> str:
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
    for token_match in _TOKEN_SPAN_RE.finditer(text):
        token = token_match.group(0)

        # Try multi-char pattern first (≥2 char units repeated ≥4 times).
        repeated = _TOKEN_REPETITION_RE.search(token)
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
        char_run = _SINGLE_CHAR_RUN_RE.search(token)
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
    max_chars = max(100, int(audio_seconds * _MAX_CHARS_PER_SEC) + 20)
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
    max_words = max(10, int(audio_seconds * _MAX_WORDS_PER_SEC) + 5)
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
    punct_strip = str.maketrans("", "", ".,!?;:'\"")
    norm_words = [w.lower().translate(punct_strip) for w in words]

    for plen in range(1, _MAX_PATTERN_WORDS + 1):
        threshold = _REPETITION_THRESHOLD if plen <= 4 else _LONG_REPETITION_THRESHOLD
        min_words = plen * threshold
        if len(words) < min_words:
            continue

        # Scan from all possible start positions (not capped) for long
        # patterns ≥5 words to catch clause loops that start late.
        if plen >= 5:
            start_limit = len(words) - min_words + 1
        else:
            start_limit = min(len(words) - min_words + 1, _MAX_PATTERN_STARTS)

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


class RealMoonshineEngine:
    """Adapter around useful-moonshine-onnx, ported from the pre-rewrite backend."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._cfg = MoonshineLoadConfig()
        # Committed cumulative buffer used for model inference (np.float32).
        self._buffer: Any = None
        # Newly arrived chunks since last inference; merged lazily to avoid
        # repeated O(n) concatenations on every chunk.
        self._pending: list[Any] = []
        self._pending_samples = 0
        self._samples_since_infer = 0
        self._dirty = False
        self._last = ""

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def model_name(self) -> str:
        return self._cfg.model_name

    @property
    def max_window_sec(self) -> float:
        return self._cfg.max_window_sec

    def load(self, config: MoonshineLoadConfig) -> None:
        errors = dependency_errors()
        if errors:
            raise RuntimeError("deps:" + errors[0])
        try:
            import moonshine_onnx as moonshine  # type: ignore
        except Exception:
            import useful_moonshine_onnx as moonshine  # type: ignore

        kwargs: dict[str, Any] = {
            "model_name": config.model_name,
            "model_precision": config.model_precision,
        }
        if config.model_dir:
            kwargs["models_dir"] = str(Path(config.model_dir).expanduser())
        try:
            self._model = moonshine.MoonshineOnnxModel(**kwargs)
            # generate() returns token IDs; the tokenizer is required to
            # produce text. Loaded once here — moonshine_onnx.transcribe()
            # would reload it from disk on every call.
            self._tokenizer = moonshine.load_tokenizer()
        except Exception as exc:
            self._model = None
            self._tokenizer = None
            raise RuntimeError(
                "Failed to initialize Moonshine ONNX backend. "
                "Check model_name/model_precision/model_dir configuration."
            ) from exc
        self._cfg = config
        self._tune_onnx_sessions()
        self.reset()

    def _tune_onnx_sessions(self) -> None:
        """Replace upstream ONNX sessions with tuned thread/provider settings."""
        try:
            import onnxruntime as ort
        except ImportError:
            log.debug("onnxruntime not directly importable; skipping session tuning")
            return

        provider = str(self._cfg.provider).strip().lower()
        threads = int(self._cfg.num_threads)

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

        import os

        sess_opts = ort.SessionOptions()
        cpu_count = os.cpu_count() or 4
        intra = threads if threads > 0 else max(2, cpu_count // 2)
        sess_opts.intra_op_num_threads = intra
        sess_opts.inter_op_num_threads = max(1, cpu_count // 4)
        sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        # The upstream MoonshineOnnxModel stores sessions as attributes
        # (onnxruntime.InferenceSession objects).
        replaced = 0
        for attr in ("encoder", "decoder", "uncached_decoder"):
            session = getattr(self._model, attr, None)
            if session is None or not isinstance(session, ort.InferenceSession):
                continue
            try:
                model_path = session._model_path
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

    def reset(self) -> None:
        if not self.loaded:
            raise RuntimeError("not_loaded")
        import numpy as np

        self._buffer = np.zeros(0, dtype=np.float32)
        self._pending = []
        self._pending_samples = 0
        self._samples_since_infer = 0
        self._dirty = False
        self._last = ""

    def process_chunk(self, samples: list[float]) -> str:
        if not self.loaded:
            raise RuntimeError("not_loaded")
        import numpy as np

        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            return self._last

        self._pending.append(waveform)
        self._pending_samples += waveform.size
        self._dirty = True

        total_buffered = self._buffer.size + self._pending_samples
        if total_buffered <= _MIN_SEGMENT_SAMPLES:
            return self._last

        # Inference throttle: Moonshine re-encodes the full buffer each call,
        # so running on every chunk causes a cascading queue backup. Silence
        # chunks (tail-flush padding) always run so the decoder can finalize
        # output at utterance boundaries.
        self._samples_since_infer += waveform.size
        is_silence = not np.any(waveform)
        if not is_silence and self._samples_since_infer < _MIN_INFER_SAMPLES:
            return self._last
        self._samples_since_infer = 0

        self._commit_pending()
        return self._infer()

    def finish(self, timeout_ms: object | None = None) -> str:
        if not self.loaded:
            raise RuntimeError("not_loaded")
        _ = timeout_ms
        self._commit_pending()
        # Decode audio the throttle skipped so the final text reflects the
        # complete utterance.
        if self._dirty and self._buffer.size > _MIN_SEGMENT_SAMPLES:
            self._samples_since_infer = 0
            return self._infer()
        return self._last

    def _commit_pending(self) -> None:
        """Merge queued chunks into the committed inference buffer."""
        if not self._pending:
            return
        import numpy as np

        pending = self._pending[0] if len(self._pending) == 1 else np.concatenate(self._pending)
        merged = pending if self._buffer.size == 0 else np.concatenate([self._buffer, pending])
        max_n = self._cfg.max_window_samples
        if merged.size > max_n:
            merged = merged[-max_n:]
        self._buffer = merged
        self._pending = []
        self._pending_samples = 0

    def _infer(self) -> str:
        import numpy as np

        if self._buffer.size == 0:
            return self._last
        audio = _normalize_buffer(self._buffer)
        # Residual: ONNX generate() is not preemptible mid-call; hosts may kill
        # the worker process to bound runaway inference.
        try:
            tokens = self._model.generate(  # type: ignore[attr-defined]
                audio[np.newaxis, :], max_len=self._cfg.max_tokens
            )
            text = self._tokenizer.decode_batch(tokens)[0].strip()
        except Exception as exc:
            log.exception("Moonshine inference failed")
            raise RuntimeError("Moonshine inference failed") from exc
        text = _guard_repetition(text, self._buffer.size / float(EXPECTED_SAMPLE_RATE))
        self._dirty = False
        self._last = text
        return text


def create_engine(*, fake: bool = False) -> AsrEngine:
    if fake:
        return FakeMoonshineEngine()
    return RealMoonshineEngine()
