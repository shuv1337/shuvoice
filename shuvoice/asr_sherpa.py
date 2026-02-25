"""Sherpa ONNX streaming ASR backend."""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .asr_base import ASRBackend, ASRCapabilities
from .config import Config

log = logging.getLogger(__name__)


class SherpaBackend(ASRBackend):
    """Sherpa ONNX streaming transducer backend (v1 contract)."""

    capabilities = ASRCapabilities(
        supports_gpu=True,
        supports_model_download=True,
        wants_raw_audio=False,
        expected_chunking="streaming",
    )

    _EXPECTED_SAMPLE_RATE = 16000
    _RELEASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases"
    _RELEASE_DOWNLOAD_ROOT = f"{_RELEASE_URL}/download/asr-models"
    _DEFAULT_MODEL_NAME = "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"

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
                f"Missing sherpa-onnx dependency: {e}. Install with: uv sync --extra asr-sherpa"
            )

        return errors

    @classmethod
    def _model_archive_url(cls, model_name: str) -> str:
        return f"{cls._RELEASE_DOWNLOAD_ROOT}/{model_name}.tar.bz2"

    @classmethod
    def _default_model_dir(cls, model_name: str | None = None) -> Path:
        name = (model_name or cls._DEFAULT_MODEL_NAME).strip()
        return Config.data_dir() / "models" / "sherpa" / name

    @staticmethod
    def _is_model_dir_complete(model_dir: Path) -> bool:
        if not model_dir.is_dir():
            return False

        if not (model_dir / "tokens.txt").is_file():
            return False

        for stem in ("encoder", "decoder", "joiner"):
            if not any(p.is_file() for p in model_dir.glob(f"{stem}*.onnx")):
                return False

        return True

    @staticmethod
    def _safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
        root = target_dir.resolve()
        with tarfile.open(archive_path, mode="r:bz2") as tf:
            for member in tf.getmembers():
                member_path = (target_dir / member.name).resolve()
                if not member_path.is_relative_to(root):
                    raise RuntimeError(
                        f"Unsafe path {member.name!r} while extracting Sherpa model archive"
                    )
            tf.extractall(path=target_dir)

    @classmethod
    def _find_extracted_model_dir(cls, root: Path) -> Path | None:
        if cls._is_model_dir_complete(root):
            return root

        for path in root.rglob("*"):
            if path.is_dir() and cls._is_model_dir_complete(path):
                return path

        return None

    @classmethod
    def download_model(
        cls,
        model_name: str | None = None,
        model_dir: str | None = None,
        progress_callback: Callable[[float | None, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        **_: Any,
    ) -> None:
        """Download a Sherpa streaming transducer model archive and extract it."""
        resolved_model_name = (model_name or cls._DEFAULT_MODEL_NAME).strip()
        target_dir = (
            Path(model_dir).expanduser()
            if model_dir
            else cls._default_model_dir(model_name=resolved_model_name)
        )

        def _emit_progress(fraction: float | None, message: str) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(fraction, message)
            except Exception:  # noqa: BLE001
                log.debug("Sherpa progress callback failed", exc_info=True)

        def _check_cancel() -> None:
            if cancel_check is not None and cancel_check():
                _emit_progress(None, "Model download cancelled")
                raise RuntimeError("Model download cancelled")

        _check_cancel()

        if cls._is_model_dir_complete(target_dir):
            log.info("Sherpa model already available: %s", target_dir)
            _emit_progress(1.0, "Sherpa model already available")
            return

        if target_dir.exists() and not target_dir.is_dir():
            raise RuntimeError(f"Sherpa model target exists and is not a directory: {target_dir}")

        target_dir.parent.mkdir(parents=True, exist_ok=True)

        archive_url = cls._model_archive_url(resolved_model_name)
        log.info("Downloading Sherpa model %s", resolved_model_name)
        log.info("Source: %s", archive_url)
        log.info("Destination: %s", target_dir)

        try:
            with tempfile.TemporaryDirectory(prefix="shuvoice-sherpa-") as tmp:
                tmp_dir = Path(tmp)
                archive_path = tmp_dir / f"{resolved_model_name}.tar.bz2"
                extracted_dir = tmp_dir / "extracted"
                extracted_dir.mkdir(parents=True, exist_ok=True)

                _emit_progress(0.0, f"Downloading {resolved_model_name}")
                last_fraction = -1.0

                def _reporthook(block_count: int, block_size: int, total_size: int) -> None:
                    nonlocal last_fraction
                    _check_cancel()
                    if total_size <= 0:
                        if last_fraction < 0.0:
                            _emit_progress(None, "Downloading model archive…")
                            last_fraction = 0.0
                        return

                    downloaded = min(block_count * block_size, total_size)
                    archive_fraction = downloaded / total_size
                    ui_fraction = min(0.9, archive_fraction * 0.9)

                    if ui_fraction >= 0.9 or ui_fraction - last_fraction >= 0.01:
                        last_fraction = ui_fraction
                        percent = int(archive_fraction * 100)
                        _emit_progress(ui_fraction, f"Downloading model archive… {percent}%")

                try:
                    urllib.request.urlretrieve(archive_url, archive_path, reporthook=_reporthook)
                except (urllib.error.URLError, TimeoutError) as e:
                    raise RuntimeError(
                        f"Failed to download Sherpa model archive. URL: {archive_url}. Error: {e}"
                    ) from e

                _check_cancel()
                _emit_progress(
                    0.93,
                    "Extracting model archive… this can take 10–60s on slower disks",
                )
                cls._safe_extract_tar(archive_path, extracted_dir)
                _check_cancel()
                source_dir = cls._find_extracted_model_dir(extracted_dir)
                if source_dir is None:
                    raise RuntimeError(
                        "Downloaded Sherpa archive did not contain required artifacts "
                        "(tokens.txt + encoder/decoder/joiner ONNX files)."
                    )

                _emit_progress(0.97, "Finalizing model files… almost done")
                _check_cancel()
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.copytree(source_dir, target_dir)
                _check_cancel()

            if not cls._is_model_dir_complete(target_dir):
                raise RuntimeError(
                    f"Sherpa model download completed but artifacts are incomplete: {target_dir}"
                )
        except RuntimeError as exc:
            if "cancelled" in str(exc).lower() and target_dir.exists():
                try:
                    if not cls._is_model_dir_complete(target_dir):
                        shutil.rmtree(target_dir)
                except Exception:  # noqa: BLE001
                    log.debug("Failed to clean partial Sherpa model directory", exc_info=True)
            raise

        _emit_progress(1.0, "Sherpa model ready")
        log.info("Sherpa model ready: %s", target_dir)

    def _resolve_model_dir(self) -> Path:
        model_name = str(self.config.sherpa_model_name).strip() or self._DEFAULT_MODEL_NAME

        configured = self.config.sherpa_model_dir
        if configured:
            model_dir = Path(configured).expanduser()
            if self._is_model_dir_complete(model_dir):
                return model_dir

            if model_dir.exists():
                raise ValueError(
                    "Sherpa model directory exists but is missing required artifacts. "
                    "Expected tokens.txt and encoder/decoder/joiner ONNX files: "
                    f"{model_dir}"
                )

            log.info(
                "Configured sherpa_model_dir does not exist. Downloading Sherpa model %s to %s",
                model_name,
                model_dir,
            )
            self.download_model(model_name=model_name, model_dir=str(model_dir))
            if not self._is_model_dir_complete(model_dir):
                raise RuntimeError(
                    f"Sherpa auto-download failed to populate model directory: {model_dir}"
                )

            self.config.sherpa_model_dir = str(model_dir)
            return model_dir

        default_dir = self._default_model_dir(model_name=model_name)
        if not self._is_model_dir_complete(default_dir):
            log.info(
                "sherpa_model_dir is not set. Downloading Sherpa model %s to %s",
                model_name,
                default_dir,
            )
            self.download_model(model_name=model_name, model_dir=str(default_dir))

        if not self._is_model_dir_complete(default_dir):
            raise RuntimeError(
                "Sherpa model auto-download failed and no valid local model directory is "
                "available. "
                f"Expected artifacts under: {default_dir}"
            )

        self.config.sherpa_model_dir = str(default_dir)
        return default_dir

    def _validate_runtime_config(self) -> None:
        if int(self.config.sample_rate) != self._EXPECTED_SAMPLE_RATE:
            raise ValueError(
                "Sherpa backend currently requires sample_rate=16000 "
                f"(got {self.config.sample_rate})."
            )

        model_dir = self._resolve_model_dir()

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
