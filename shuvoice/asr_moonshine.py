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
    """Moonshine ONNX backend with chunk-wise cumulative decoding."""

    _EXPECTED_SAMPLE_RATE = 16000
    # The upstream helper asserts >0.1s, but very short windows often decode as
    # repetitive junk (e.g., repeated "W."). Use a safer minimum segment.
    _MIN_SEGMENT_S = 0.35
    _MAX_SEGMENT_S = 64.0
    _REPO_URL = "https://github.com/moonshine-ai/moonshine"

    def __init__(self, config: Config):
        self.config = config

        self._model: Any = None
        self._moonshine: Any = None

        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._last_text = ""
        self._step_num = 0

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
                    "moonshine_model_dir is missing required model artifacts: "
                    + ", ".join(missing)
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
            self._moonshine = moonshine_onnx
            self._model = moonshine_onnx.MoonshineOnnxModel(**kwargs)
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize Moonshine ONNX backend. "
                "Check moonshine_model_name/model_precision/model_dir configuration."
            ) from e

        self.reset()

    def reset(self) -> None:
        if self._model is None or self._moonshine is None:
            raise RuntimeError("ASR backend is not loaded. Call load() first.")

        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._last_text = ""
        self._step_num = 0

    def process_chunk(self, audio_chunk: np.ndarray) -> str:
        if self._model is None or self._moonshine is None:
            raise RuntimeError("ASR backend is not loaded. Call load() first.")

        waveform = np.asarray(audio_chunk, dtype=np.float32)
        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)

        if waveform.size == 0:
            return self._last_text

        self._audio_buffer = np.concatenate([self._audio_buffer, waveform])

        max_samples = int(float(self.config.moonshine_max_window_sec) * int(self.config.sample_rate))
        if self._audio_buffer.size > max_samples:
            self._audio_buffer = self._audio_buffer[-max_samples:]

        min_samples = int(self._MIN_SEGMENT_S * int(self.config.sample_rate))
        if self._audio_buffer.size <= min_samples:
            return self._last_text

        try:
            decoded = self._moonshine.transcribe(self._audio_buffer, model=self._model)
            text = decoded[0].strip() if decoded else ""
        except Exception:
            log.exception("Moonshine inference failed")
            raise RuntimeError("Moonshine inference failed")

        self._last_text = text
        self._step_num += 1
        return text
