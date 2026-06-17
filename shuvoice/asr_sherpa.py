"""Sherpa ONNX ASR backend with streaming and offline modes."""

from __future__ import annotations

import logging
import math
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .asr_base import ASRBackend, ASRCapabilities
from .config import Config
from .sherpa_cuda import (
    cuda_provider_runtime_status,
    prepare_cuda_runtime,
    prepare_import_runtime,
    sherpa_lib_dir,
)

log = logging.getLogger(__name__)


# Substrings identifying CUDA/cuBLAS/cuDNN allocation / workspace failures that
# typically mean the GPU is out of memory (rather than a genuine logic bug).
# Matched case-insensitively against str(exc). Kept in one place so app.py and
# the backend stay in sync.
_CUDA_OOM_ERROR_MARKERS: tuple[str, ...] = (
    "cublas_status_alloc_failed",
    "cublascreate",
    "cudnn_status_internal_error",
    "cudnn_status_alloc_failed",
    "cudnncreate",
    "cuda error: out of memory",
    "cuda out of memory",
    "out of memory",  # onnxruntime sometimes stringifies generic CUDA alloc failures this way
    # ONNX Runtime's BFCArena failure path (CUDA EP) emits messages like:
    #   "bfc_arena.cc:359 ... AllocateRawInternal ... Failed to allocate memory
    #    for requested buffer of size N"
    # which may be wrapped by a "MemcpyFromHost" / "MemcpyToHost" node error.
    # Match the allocation-specific text rather than the generic memcpy node
    # names to avoid treating unrelated copy failures as OOM.
    "failed to allocate memory",
    "bfc_arena",
)


def looks_like_cuda_oom_error(exc: BaseException) -> bool:
    """Heuristic: does ``exc`` look like a CUDA/cuBLAS/cuDNN out-of-memory error.

    Used by both the Sherpa backend itself and by the app's circuit breaker to
    decide whether to auto-fall back to CPU.
    """
    text = str(exc).lower()
    return any(marker in text for marker in _CUDA_OOM_ERROR_MARKERS)


class SherpaBackend(ASRBackend):
    """Sherpa ONNX transducer backend with streaming and offline modes."""

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
        self._recognizer: Any = None  # OnlineRecognizer for streaming mode
        self._offline_recognizer: Any = None  # OfflineRecognizer for offline mode
        self._stream: Any = None
        self._model_files: dict[str, Path] | None = None
        self._cpu_fallback_applied = False

    @property
    def cpu_fallback_applied(self) -> bool:
        """True when this session auto-fell back to CPU after a CUDA failure."""
        return self._cpu_fallback_applied

    def try_fallback_to_cpu(self) -> tuple[bool, str]:
        """Swap the runtime to CPU after a CUDA failure and reload the recognizer.

        Returns ``(ok, detail)`` where ``ok`` is ``True`` when the backend is now
        running on CPU. Idempotent: if already on CPU or a previous fallback
        already succeeded, returns ``(False, reason)`` so the caller can stop
        attempting.
        """
        if self._cpu_fallback_applied:
            return False, "CPU fallback already applied earlier this session"

        provider = str(getattr(self.config, "sherpa_provider", "cpu")).strip().lower()
        if provider != "cuda":
            return False, f"sherpa_provider is already '{provider}'; no fallback to apply"

        log.warning("Falling back Sherpa runtime from CUDA to CPU after repeated GPU errors")
        self.config.sherpa_provider = "cpu"

        # Drop previous recognizer/stream so garbage collection can release any
        # GPU memory we still hold.
        self._recognizer = None
        self._offline_recognizer = None
        self._stream = None

        try:
            if self._is_offline_mode:
                self._load_offline_recognizer()
            else:
                self._load_online_recognizer()
                self._stream = self._recognizer.create_stream()
        except Exception as exc:
            log.exception("Sherpa CPU fallback failed to reload recognizer")
            return False, f"CPU fallback reload failed: {exc}"

        self._cpu_fallback_applied = True
        log.info("Sherpa runtime now running on CPU (CUDA fallback applied)")
        return True, "Sherpa runtime switched to CPU after CUDA failure"

    @property
    def native_chunk_samples(self) -> int:
        return int(self.config.sample_rate) * int(self.config.sherpa_chunk_ms) // 1000

    @property
    def _is_offline_mode(self) -> bool:
        """Check if backend is configured for offline instant mode."""
        resolved = self.config.resolved_sherpa_decode_mode
        return resolved == "offline_instant"

    _PARAKEET_MODEL_MARKERS = (
        "parakeet-tdt",
        "nemo-parakeet",
    )

    @staticmethod
    def dependency_errors() -> list[str]:
        errors: list[str] = []

        try:
            import sherpa_onnx
        except Exception:
            lib_dir = sherpa_lib_dir()
            if lib_dir is not None and lib_dir.is_dir():
                repaired, _detail = prepare_import_runtime(lib_dir)
                if repaired:
                    try:
                        import sherpa_onnx  # noqa: F401

                        return errors
                    except Exception as repaired_exc:
                        e = repaired_exc

            errors.append(
                f"Missing sherpa-onnx dependency: {e}. Install with: uv sync --extra asr-sherpa"
            )

        return errors

    @classmethod
    def startup_errors(cls, config: Config) -> list[str]:
        errors: list[str] = []

        if cls._looks_like_parakeet_model(config):
            resolved_mode = config.resolved_sherpa_decode_mode
            allow_parakeet_streaming = bool(
                getattr(config, "sherpa_enable_parakeet_streaming", False)
            )

            if resolved_mode == "offline_instant":
                return errors

            if resolved_mode == "streaming" and allow_parakeet_streaming:
                compatible, detail = cls._parakeet_streaming_model_compatible(config)
                if compatible:
                    return errors
                errors.append(
                    "Configured Sherpa model appears to be Parakeet TDT with streaming "
                    "override enabled, but the model/runtime combination looks incompatible "
                    f"with Sherpa online decoding ({detail}). "
                    "Use offline instant mode (instant_mode=true with "
                    "sherpa_decode_mode='auto', or sherpa_decode_mode='offline_instant')."
                )
                return errors

            message = (
                "Configured Sherpa model appears to be Parakeet TDT, but ShuVoice is "
                "configured for streaming mode. By default, Parakeet remains blocked in "
                "streaming mode to avoid startup/runtime instability with incompatible model "
                "metadata. "
                "Use offline instant mode (instant_mode=true with sherpa_decode_mode='auto', "
                "or sherpa_decode_mode='offline_instant') for the stable path. "
                "To force streaming anyway, set "
                "sherpa_enable_parakeet_streaming=true and sherpa_decode_mode='streaming'."
            )
            errors.append(message)

        return errors

    @classmethod
    def startup_warnings(cls, config: Config, *, apply_fixes: bool = False) -> list[str]:
        warnings: list[str] = []

        provider = str(getattr(config, "sherpa_provider", "cpu")).strip().lower()
        if provider != "cuda":
            return warnings

        cuda_ok, detail = cls._cuda_provider_available()
        if cuda_ok:
            return warnings

        warning = (
            "sherpa_provider='cuda' requested, but installed sherpa-onnx runtime does not "
            f"expose CUDAExecutionProvider ({detail})."
        )
        if apply_fixes:
            config.sherpa_provider = "cpu"
            warning += " Falling back to sherpa_provider='cpu' for this run."
        else:
            warning += ""
        warning += (
            " Install a CUDA-enabled sherpa-onnx build (for example, a GPU-enabled wheel "
            "or local source build) to restore GPU inference."
        )
        warnings.append(warning)

        return warnings

    @classmethod
    def _looks_like_parakeet_model(cls, config: Config) -> bool:
        model_name = str(getattr(config, "sherpa_model_name", "")).strip().lower()
        model_dir = str(getattr(config, "sherpa_model_dir", "") or "").strip().lower()
        candidates = [model_name, model_dir]
        return any(
            marker in candidate
            for candidate in candidates
            for marker in cls._PARAKEET_MODEL_MARKERS
        )

    @staticmethod
    def _cuda_provider_available() -> tuple[bool, str]:
        try:
            import sherpa_onnx
        except Exception as exc:
            lib_dir = sherpa_lib_dir()
            if lib_dir is not None and lib_dir.is_dir():
                repaired, repair_detail = prepare_import_runtime(lib_dir)
                if repaired:
                    try:
                        import sherpa_onnx  # noqa: F401
                    except Exception as retry_exc:
                        return False, f"failed to import sherpa_onnx ({retry_exc})"
                else:
                    return False, repair_detail
            else:
                return False, f"failed to import sherpa_onnx ({exc})"

        lib_dir = sherpa_lib_dir()
        if lib_dir is None or not lib_dir.is_dir():
            return False, "runtime lib directory not found for sherpa_onnx"

        status_ok, detail = cuda_provider_runtime_status(lib_dir)
        if status_ok:
            return True, detail

        repaired, repair_detail = prepare_cuda_runtime(lib_dir)
        if repaired:
            return cuda_provider_runtime_status(lib_dir)
        return False, repair_detail or detail

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

    @staticmethod
    def _format_bytes(num_bytes: int | float) -> str:
        value = max(0.0, float(num_bytes))
        units = ("B", "KiB", "MiB", "GiB", "TiB")
        unit_index = 0

        while value >= 1024.0 and unit_index < len(units) - 1:
            value /= 1024.0
            unit_index += 1

        unit = units[unit_index]
        if unit == "B":
            return f"{int(value)} {unit}"
        return f"{value:.1f} {unit}"

    @staticmethod
    def _format_eta(seconds: float | None) -> str:
        if seconds is None or not math.isfinite(seconds):
            return "--:--"

        clamped = max(0, round(seconds))
        minutes, sec = divmod(clamped, 60)
        hours, minutes = divmod(minutes, 60)

        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"

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
            except Exception:
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
                download_started_monotonic = time.monotonic()

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

                        elapsed = max(0.001, time.monotonic() - download_started_monotonic)
                        remaining_bytes = max(0, total_size - downloaded)
                        bytes_per_second = downloaded / elapsed
                        eta_seconds = (
                            remaining_bytes / bytes_per_second if bytes_per_second > 0 else None
                        )

                        downloaded_text = cls._format_bytes(downloaded)
                        total_text = cls._format_bytes(total_size)
                        eta_text = cls._format_eta(eta_seconds)

                        _emit_progress(
                            ui_fraction,
                            "Downloading model archive… "
                            f"{percent}% ({downloaded_text}/{total_text}, ETA {eta_text})",
                        )

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
                except Exception:
                    log.debug("Failed to clean partial Sherpa model directory", exc_info=True)
            raise

        _emit_progress(1.0, "Sherpa model ready")
        log.info("Sherpa model ready: %s", target_dir)

    def _resolve_model_dir(
        self,
        *,
        progress_callback: Callable[[float | None, str], None] | None = None,
    ) -> Path:
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
            self.download_model(
                model_name=model_name,
                model_dir=str(model_dir),
                progress_callback=progress_callback,
            )
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
            self.download_model(
                model_name=model_name,
                model_dir=str(default_dir),
                progress_callback=progress_callback,
            )

        if not self._is_model_dir_complete(default_dir):
            raise RuntimeError(
                "Sherpa model auto-download failed and no valid local model directory is "
                "available. "
                f"Expected artifacts under: {default_dir}"
            )

        self.config.sherpa_model_dir = str(default_dir)
        return default_dir

    def _validate_runtime_config(
        self,
        *,
        progress_callback: Callable[[float | None, str], None] | None = None,
    ) -> None:
        if int(self.config.sample_rate) != self._EXPECTED_SAMPLE_RATE:
            raise ValueError(
                "Sherpa backend currently requires sample_rate=16000 "
                f"(got {self.config.sample_rate})."
            )

        model_dir = self._resolve_model_dir(progress_callback=progress_callback)

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

    @staticmethod
    def _onnx_file_contains_token(path: Path, token: bytes) -> bool:
        """Check whether an ONNX file contains a metadata token string."""
        overlap = max(0, len(token) - 1)
        tail = b""

        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    return False

                data = tail + chunk
                if token in data:
                    return True

                tail = data[-overlap:] if overlap else b""

    @classmethod
    def _parakeet_streaming_model_compatible(cls, config: Config) -> tuple[bool, str]:
        """Best-effort compatibility check for Parakeet streaming mode.

        Returns (compatible, detail). If model files are not available yet,
        this returns compatible=True and defers final check to load().
        """
        if not cls._looks_like_parakeet_model(config):
            return True, "not a Parakeet model"

        if config.resolved_sherpa_decode_mode != "streaming":
            return True, "not in streaming mode"

        if not bool(getattr(config, "sherpa_enable_parakeet_streaming", False)):
            return True, "Parakeet streaming override disabled"

        configured = getattr(config, "sherpa_model_dir", None)
        model_name = str(getattr(config, "sherpa_model_name", "") or "").strip() or None
        model_dir = (
            Path(configured).expanduser() if configured else cls._default_model_dir(model_name)
        )

        if not model_dir.is_dir():
            return True, f"model directory not present yet ({model_dir})"

        try:
            encoder = cls._pick_model_onnx(model_dir, "encoder")
        except Exception as exc:
            return False, f"unable to locate encoder ONNX file ({exc})"

        try:
            has_window_size = cls._onnx_file_contains_token(encoder, b"window_size")
        except Exception as exc:
            return False, f"failed to inspect encoder metadata ({exc})"

        if not has_window_size:
            return (
                False,
                "encoder metadata appears to be missing 'window_size' required by "
                f"Sherpa online decoder ({encoder})",
            )

        return True, f"encoder metadata includes 'window_size' ({encoder})"

    def _load_online_recognizer(self) -> None:
        """Load OnlineRecognizer for streaming mode."""
        import sherpa_onnx

        model_files = self._model_files
        assert model_files is not None

        use_nemo_transducer = self._looks_like_parakeet_model(self.config)
        if use_nemo_transducer:
            compatible, detail = self._parakeet_streaming_model_compatible(self.config)
            if not compatible:
                raise RuntimeError(
                    "Parakeet streaming model is incompatible with Sherpa online decoding "
                    f"on this runtime ({detail}). "
                    "Use sherpa_decode_mode='offline_instant' for this model."
                )
            log.info("Sherpa streaming Parakeet mode enabled (model_type='nemo_transducer')")

        recognizer_cls = sherpa_onnx.OnlineRecognizer
        if hasattr(recognizer_cls, "from_transducer"):
            recognizer_kwargs: dict[str, Any] = {
                "encoder": str(model_files["encoder"]),
                "decoder": str(model_files["decoder"]),
                "joiner": str(model_files["joiner"]),
                "tokens": str(model_files["tokens"]),
                "num_threads": int(self.config.sherpa_num_threads),
                "provider": self.config.sherpa_provider,
                "sample_rate": int(self.config.sample_rate),
                "feature_dim": 80,
            }
            if use_nemo_transducer:
                recognizer_kwargs["model_type"] = "nemo_transducer"

            try:
                self._recognizer = recognizer_cls.from_transducer(**recognizer_kwargs)
            except TypeError as exc:
                if not use_nemo_transducer or "model_type" not in str(exc):
                    raise
                recognizer_kwargs.pop("model_type", None)
                self._recognizer = recognizer_cls.from_transducer(**recognizer_kwargs)
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
            model_kwargs: dict[str, Any] = {
                "transducer": transducer_config,
                "tokens": str(model_files["tokens"]),
                "num_threads": int(self.config.sherpa_num_threads),
                "provider": self.config.sherpa_provider,
            }
            if use_nemo_transducer:
                model_kwargs["model_type"] = "nemo_transducer"

            try:
                model_config = sherpa_onnx.OnlineModelConfig(**model_kwargs)
            except TypeError as exc:
                if not use_nemo_transducer or "model_type" not in str(exc):
                    raise
                model_kwargs.pop("model_type", None)
                model_config = sherpa_onnx.OnlineModelConfig(**model_kwargs)

            recognizer_config = sherpa_onnx.OnlineRecognizerConfig(
                feat_config=feat_config,
                model_config=model_config,
                decoding_method="greedy_search",
            )
            self._recognizer = recognizer_cls(recognizer_config)

    def _load_offline_recognizer(self) -> None:
        """Load OfflineRecognizer for offline instant mode (Parakeet support)."""
        import sherpa_onnx

        model_files = self._model_files
        assert model_files is not None

        recognizer_cls = sherpa_onnx.OfflineRecognizer
        if hasattr(recognizer_cls, "from_transducer"):
            # Use from_transducer with model_type="nemo_transducer" for Parakeet
            self._offline_recognizer = recognizer_cls.from_transducer(
                encoder=str(model_files["encoder"]),
                decoder=str(model_files["decoder"]),
                joiner=str(model_files["joiner"]),
                tokens=str(model_files["tokens"]),
                num_threads=int(self.config.sherpa_num_threads),
                provider=self.config.sherpa_provider,
                sample_rate=int(self.config.sample_rate),
                feature_dim=80,
                model_type="nemo_transducer",
            )
        else:
            # Fallback for older sherpa-onnx versions
            transducer_config = sherpa_onnx.OfflineTransducerModelConfig(
                encoder=str(model_files["encoder"]),
                decoder=str(model_files["decoder"]),
                joiner=str(model_files["joiner"]),
            )
            model_config = sherpa_onnx.OfflineModelConfig(
                transducer=transducer_config,
                tokens=str(model_files["tokens"]),
                num_threads=int(self.config.sherpa_num_threads),
                provider=self.config.sherpa_provider,
                model_type="nemo_transducer",
            )
            recognizer_config = sherpa_onnx.OfflineRecognizerConfig(
                model_config=model_config,
                decoding_method="greedy_search",
            )
            self._offline_recognizer = recognizer_cls(recognizer_config)

    def load(self, progress_callback: Callable[[float | None, str], None] | None = None) -> None:
        def _emit_progress(fraction: float | None, message: str) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(fraction, message)
            except Exception:
                log.debug("Sherpa load progress callback failed", exc_info=True)

        _emit_progress(None, "Validating Sherpa model…")
        self._validate_runtime_config(progress_callback=progress_callback)

        errors = self.dependency_errors()
        if errors:
            raise RuntimeError("\n".join(errors))

        if self._model_files is None:
            raise RuntimeError("Sherpa model validation failed unexpectedly")

        decode_mode = self.config.resolved_sherpa_decode_mode
        provider = self.config.sherpa_provider

        log.info(
            "Loading Sherpa backend: decode_mode=%s, provider=%s, model=%s",
            decode_mode,
            provider,
            self.config.sherpa_model_name,
        )
        _emit_progress(0.98, "Initializing Sherpa recognizer…")

        try:
            if self._is_offline_mode:
                self._load_offline_recognizer()
                log.info("Sherpa OfflineRecognizer loaded successfully (offline_instant mode)")
            else:
                self._load_online_recognizer()
                log.info("Sherpa OnlineRecognizer loaded successfully (streaming mode)")
        except Exception as e:
            mode_desc = "offline" if self._is_offline_mode else "streaming"
            raise RuntimeError(
                f"Failed to initialize Sherpa {mode_desc} recognizer. "
                "Ensure sherpa_model_dir points to a supported transducer model."
            ) from e

        self.reset()
        _emit_progress(1.0, "Sherpa model ready")

    def reset(self) -> None:
        if self._is_offline_mode:
            if self._offline_recognizer is None:
                raise RuntimeError("ASR backend is not loaded. Call load() first.")
            # Offline mode doesn't use streams; nothing to reset
            return

        if self._recognizer is None:
            raise RuntimeError("ASR backend is not loaded. Call load() first.")

        self._stream = self._recognizer.create_stream()

    def process_chunk(self, audio_chunk: np.ndarray) -> str:
        if self._is_offline_mode:
            raise RuntimeError(
                "process_chunk() is not supported in offline instant mode. "
                "Use process_utterance() instead."
            )

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

    def process_utterance(self, audio: np.ndarray) -> str:
        """Process a complete utterance in offline mode.

        Args:
            audio: Complete utterance audio as float32 samples at 16kHz.

        Returns:
            Transcribed text for the utterance.

        Raises:
            RuntimeError: If not in offline mode or backend not loaded.
        """
        if not self._is_offline_mode:
            raise RuntimeError(
                "process_utterance() is only supported in offline instant mode. "
                "Use process_chunk() for streaming mode."
            )

        if self._offline_recognizer is None:
            raise RuntimeError("ASR backend is not loaded. Call load() first.")

        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)

        # Hard cap utterance length: the offline transducer allocates
        # encoder/joiner activation buffers that scale with audio duration,
        # so a stuck push-to-talk that captures, say, 15 minutes of audio can
        # request many GiB of GPU/CPU memory and blow up the process.
        # When over the cap, keep only the trailing window so the most recent
        # speech still gets transcribed.
        sample_rate = int(self.config.sample_rate)
        max_seconds = float(getattr(self.config, "sherpa_offline_max_utterance_sec", 0.0) or 0.0)
        if max_seconds > 0 and waveform.size > sample_rate * max_seconds:
            max_samples = max(1, int(sample_rate * max_seconds))
            log.warning(
                "Sherpa offline utterance too long (%.1fs > %.1fs cap); "
                "truncating to last %.1fs to avoid memory blow-up.",
                waveform.size / sample_rate,
                max_seconds,
                max_seconds,
            )
            waveform = waveform[-max_samples:]

        # Create a stream for this utterance
        stream = self._offline_recognizer.create_stream()
        stream.accept_waveform(sample_rate, waveform)

        # Decode the complete utterance
        self._offline_recognizer.decode_stream(stream)

        # Extract result
        result = stream.result
        if hasattr(result, "text"):
            return result.text.strip()

        return str(result).strip()
