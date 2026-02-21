"""Sherpa ONNX streaming ASR backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .asr_base import ASRBackend
from .config import Config

log = logging.getLogger(__name__)


class SherpaBackend(ASRBackend):
    """Sherpa ONNX streaming transducer backend (v1 contract)."""

    _EXPECTED_SAMPLE_RATE = 16000
    _RELEASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases"

    def __init__(self, config: Config):
        self.config = config
        self._recognizer: Any = None
        self._stream: Any = None
        self._model_files: dict[str, Path] | None = None

    @property
    def native_chunk_samples(self) -> int:
        return int(self.config.sample_rate) * int(self.config.sherpa_chunk_ms) // 1000

    @staticmethod
    def dependency_errors() -> list[str]:
        errors: list[str] = []

        try:
            import sherpa_onnx  # noqa: F401
        except Exception as e:
            errors.append(
                f"Missing sherpa-onnx dependency: {e}. Install with: pip install sherpa-onnx"
            )

        return errors

    @classmethod
    def download_model(cls, **kwargs) -> None:
        raise NotImplementedError(
            "Sherpa model auto-download is not implemented. "
            f"Download a streaming transducer model manually from: {cls._RELEASE_URL}"
        )

    def _validate_runtime_config(self) -> None:
        if int(self.config.sample_rate) != self._EXPECTED_SAMPLE_RATE:
            raise ValueError(
                "Sherpa backend currently requires sample_rate=16000 "
                f"(got {self.config.sample_rate})."
            )

        model_dir_raw = self.config.sherpa_model_dir
        if not model_dir_raw:
            raise ValueError(
                "sherpa_model_dir is required when asr_backend='sherpa'. "
                f"See Sherpa releases: {self._RELEASE_URL}"
            )

        model_dir = Path(model_dir_raw).expanduser()
        if not model_dir.is_dir():
            raise ValueError(f"sherpa_model_dir does not exist or is not a directory: {model_dir}")

        tokens = model_dir / "tokens.txt"
        if not tokens.is_file():
            raise ValueError(
                "Sherpa model directory is missing required file: tokens.txt "
                "(streaming transducer contract)."
            )

        encoder = self._pick_model_onnx(model_dir, "encoder")
        decoder = self._pick_model_onnx(model_dir, "decoder")
        joiner = self._pick_model_onnx(model_dir, "joiner")

        self._model_files = {
            "tokens": tokens,
            "encoder": encoder,
            "decoder": decoder,
            "joiner": joiner,
        }

    @staticmethod
    def _pick_model_onnx(model_dir: Path, name: str) -> Path:
        exact = model_dir / f"{name}.onnx"
        if exact.is_file():
            return exact

        matches = sorted(p for p in model_dir.glob(f"{name}*.onnx") if p.is_file())
        if not matches:
            raise ValueError(
                "Sherpa model directory is missing required streaming transducer artifact: "
                f"{name}*.onnx"
            )

        return matches[0]

    def load(self) -> None:
        self._validate_runtime_config()

        errors = self.dependency_errors()
        if errors:
            raise RuntimeError("\n".join(errors))

        if self._model_files is None:
            raise RuntimeError("Sherpa model validation failed unexpectedly")

        import sherpa_onnx

        model_files = self._model_files

        try:
            recognizer_cls = sherpa_onnx.OnlineRecognizer
            if hasattr(recognizer_cls, "from_transducer"):
                self._recognizer = recognizer_cls.from_transducer(
                    encoder=str(model_files["encoder"]),
                    decoder=str(model_files["decoder"]),
                    joiner=str(model_files["joiner"]),
                    tokens=str(model_files["tokens"]),
                    num_threads=int(self.config.sherpa_num_threads),
                    provider=self.config.sherpa_provider,
                    sample_rate=int(self.config.sample_rate),
                    feature_dim=80,
                )
            else:
                feat_config = sherpa_onnx.FeatureConfig(
                    sample_rate=int(self.config.sample_rate),
                    feature_dim=80,
                )
                transducer_config = sherpa_onnx.OnlineTransducerModelConfig(
                    encoder=str(model_files["encoder"]),
                    decoder=str(model_files["decoder"]),
                    joiner=str(model_files["joiner"]),
                )
                model_config = sherpa_onnx.OnlineModelConfig(
                    transducer=transducer_config,
                    tokens=str(model_files["tokens"]),
                    num_threads=int(self.config.sherpa_num_threads),
                    provider=self.config.sherpa_provider,
                )
                recognizer_config = sherpa_onnx.OnlineRecognizerConfig(
                    feat_config=feat_config,
                    model_config=model_config,
                    decoding_method="greedy_search",
                )
                self._recognizer = recognizer_cls(recognizer_config)
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize Sherpa streaming recognizer. "
                "Ensure sherpa_model_dir points to a supported streaming transducer model."
            ) from e

        self.reset()

    def reset(self) -> None:
        if self._recognizer is None:
            raise RuntimeError("ASR backend is not loaded. Call load() first.")

        self._stream = self._recognizer.create_stream()

    def process_chunk(self, audio_chunk: np.ndarray) -> str:
        if self._recognizer is None or self._stream is None:
            raise RuntimeError("ASR backend is not loaded. Call load() first.")

        waveform = np.asarray(audio_chunk, dtype=np.float32)
        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)

        self._stream.accept_waveform(int(self.config.sample_rate), waveform)

        decode_stream = getattr(self._recognizer, "decode_stream", None)
        decode_streams = getattr(self._recognizer, "decode_streams", None)

        while self._recognizer.is_ready(self._stream):
            if callable(decode_stream):
                decode_stream(self._stream)
            elif callable(decode_streams):
                decode_streams([self._stream])
            else:
                raise RuntimeError("Unsupported sherpa-onnx recognizer API: decode method missing")

        result = self._recognizer.get_result(self._stream)
        if isinstance(result, str):
            return result.strip()

        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text.strip()

        return str(result).strip()
