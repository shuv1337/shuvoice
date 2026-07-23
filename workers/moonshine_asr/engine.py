"""Moonshine ASR engine boundary — optional worker with fake for tests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

log = logging.getLogger(__name__)

EXPECTED_SAMPLE_RATE = 16000


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


class RealMoonshineEngine:
    """Best-effort adapter around useful-moonshine-onnx (lazy)."""

    def __init__(self) -> None:
        self._model: Any = None
        self._cfg = MoonshineLoadConfig()
        self._buffer: list[float] = []
        self._last = ""

    @property
    def loaded(self) -> bool:
        return self._model is not None

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
            from moonshine_onnx import MoonshineOnnxModel  # type: ignore
        except Exception:
            from useful_moonshine_onnx import MoonshineOnnxModel  # type: ignore

        # model_dir / provider / threads are accepted for host parity; the public
        # ONNX wrapper may ignore some of them depending on package version.
        kwargs: dict[str, Any] = {"model_name": config.model_name}
        if config.model_dir:
            kwargs["model_dir"] = config.model_dir
        try:
            self._model = MoonshineOnnxModel(**kwargs)
        except TypeError:
            self._model = MoonshineOnnxModel(model_name=config.model_name)
        self._cfg = config
        self.reset()

    def reset(self) -> None:
        if self._model is None:
            raise RuntimeError("not_loaded")
        self._buffer.clear()
        self._last = ""

    def process_chunk(self, samples: list[float]) -> str:
        if self._model is None:
            raise RuntimeError("not_loaded")
        import numpy as np

        self._buffer.extend(samples)
        max_n = self._cfg.max_window_samples
        if len(self._buffer) > max_n:
            self._buffer = self._buffer[-max_n:]
        audio = np.asarray(self._buffer, dtype=np.float32)
        # Residual: ONNX generate() is not preemptible mid-call; hosts may kill
        # the worker process to bound runaway inference.
        tokens = self._model.generate(audio)  # type: ignore[attr-defined]
        text = " ".join(str(t) for t in tokens) if not isinstance(tokens, str) else tokens
        self._last = text
        return text

    def finish(self, timeout_ms: object | None = None) -> str:
        if self._model is None:
            raise RuntimeError("not_loaded")
        _ = timeout_ms
        return self._last


def create_engine(*, fake: bool = False) -> AsrEngine:
    if fake:
        return FakeMoonshineEngine()
    return RealMoonshineEngine()
