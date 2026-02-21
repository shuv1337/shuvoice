"""NeMo streaming ASR backend."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .asr_base import ASRBackend
from .config import Config

log = logging.getLogger(__name__)


class NemoBackend(ASRBackend):
    """NeMo Nemotron backend preserving legacy ASREngine behavior."""

    def __init__(
        self,
        model_name: str | Config = "nvidia/nemotron-speech-streaming-en-0.6b",
        right_context: int = 13,
        device: str = "cuda",
        use_cuda_graph_decoder: bool = False,
    ):
        if isinstance(model_name, Config):
            cfg = model_name
            self.model_name = cfg.model_name
            self.right_context = int(cfg.right_context)
            self.device = cfg.device
            self.use_cuda_graph_decoder = bool(cfg.use_cuda_graph_decoder)
        else:
            self.model_name = model_name
            self.right_context = int(right_context)
            self.device = device
            self.use_cuda_graph_decoder = bool(use_cuda_graph_decoder)

        self.model: Any = None
        self._torch: Any = None
        self._nemo_asr: Any = None

        self._cache_last_channel = None
        self._cache_last_time = None
        self._cache_last_channel_len = None
        self._pre_encode_cache = None
        self._previous_hypotheses = None
        self._pred_out_stream = None
        self._step_num = 0

    @property
    def native_chunk_samples(self) -> int:
        """Return expected audio samples per chunk based on right-context."""
        match self.right_context:
            case 0:
                return 1280
            case 1:
                return 2560
            case 6:
                return 8960
            case _:
                return 17920

    @property
    def debug_step_num(self) -> int | None:
        return self._step_num

    @staticmethod
    def dependency_errors() -> list[str]:
        errors: list[str] = []

        try:
            import torch  # noqa: F401
        except Exception as e:
            errors.append(
                f"Missing PyTorch dependency: {e}. Install torch (or python-pytorch-cuda on Arch)."
            )

        try:
            import nemo.collections.asr  # noqa: F401
        except Exception as e:
            errors.append(
                f"Missing NeMo ASR dependency: {e}. "
                "Install nemo-toolkit[asr] (or NeMo from git main)."
            )

        return errors

    def _ensure_dependencies(self):
        if self._torch is None:
            try:
                import torch
            except Exception as e:
                raise RuntimeError(
                    "PyTorch is required for ASR. "
                    "Install torch (or python-pytorch-cuda on Arch Linux)."
                ) from e
            self._torch = torch

        if self._nemo_asr is None:
            try:
                import nemo.collections.asr as nemo_asr
            except Exception as e:
                raise RuntimeError(
                    "NeMo ASR is required for ASR. "
                    "Install nemo-toolkit[asr] or "
                    "pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
                ) from e
            self._nemo_asr = nemo_asr

    def _configure_decoding_strategy(self):
        """Apply runtime decoding overrides for better compatibility/stability."""
        if self.model is None:
            return

        try:
            from omegaconf import OmegaConf

            decoding_cfg = OmegaConf.to_container(self.model.cfg.decoding, resolve=True)
            if not isinstance(decoding_cfg, dict):
                return

            greedy_cfg = decoding_cfg.setdefault("greedy", {})
            if isinstance(greedy_cfg, dict):
                greedy_cfg["use_cuda_graph_decoder"] = bool(self.use_cuda_graph_decoder)

            self.model.change_decoding_strategy(OmegaConf.create(decoding_cfg))
            log.info(
                "RNNT greedy CUDA graph decoder: %s",
                "enabled" if self.use_cuda_graph_decoder else "disabled",
            )
        except Exception:
            log.warning("Failed to apply decoding strategy overrides", exc_info=True)

    @staticmethod
    def _normalize_transcript_item(item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, str):
            return item

        text = getattr(item, "text", None)
        if isinstance(text, str):
            return text

        # Unknown transcript type (eg. Hypothesis without text) — ignore safely.
        return ""

    def load(self):
        self._ensure_dependencies()

        log.info("Loading model: %s", self.model_name)
        self.model = self._nemo_asr.models.ASRModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

        self._configure_decoding_strategy()

        # Configure streaming latency
        self.model.encoder.set_default_att_context_size([70, self.right_context])
        self.model.encoder.setup_streaming_params()

        self.reset()
        log.info(
            "Model loaded and streaming configured (right_context=%d)",
            self.right_context,
        )

    def reset(self):
        """Reset streaming state for a new utterance."""
        if self.model is None:
            raise RuntimeError("ASR model is not loaded. Call load() first.")

        cache_last_channel, cache_last_time, cache_last_channel_len = (
            self.model.encoder.get_initial_cache_state(batch_size=1)
        )

        # Ensure state tensors are actually moved to the correct device.
        if hasattr(cache_last_channel, "to"):
            self._cache_last_channel = cache_last_channel.to(self.device).clone()
        else:
            self._cache_last_channel = cache_last_channel

        if hasattr(cache_last_time, "to"):
            self._cache_last_time = cache_last_time.to(self.device).clone()
        else:
            self._cache_last_time = cache_last_time

        if hasattr(cache_last_channel_len, "to"):
            self._cache_last_channel_len = cache_last_channel_len.to(self.device).clone()
        else:
            self._cache_last_channel_len = cache_last_channel_len

        # NeMo API compatibility: older versions expose feat_out, newer ones use nfilt.
        featurizer = self.model.preprocessor.featurizer
        if hasattr(featurizer, "feat_out"):
            num_features = int(featurizer.feat_out)
        elif hasattr(featurizer, "nfilt"):
            num_features = int(featurizer.nfilt)
        else:
            raise RuntimeError(
                "Could not determine feature dimension from NeMo featurizer "
                f"type={type(featurizer)!r}"
            )

        pre_encode_size = self.model.encoder.streaming_cfg.pre_encode_cache_size[1]
        self._pre_encode_cache = self._torch.zeros(
            (1, num_features, pre_encode_size), device=self.device
        )

        self._previous_hypotheses = None
        self._pred_out_stream = None
        self._step_num = 0

    def process_chunk(self, audio_chunk: np.ndarray) -> str:
        """Process one native chunk and return cumulative transcription."""
        if self.model is None:
            raise RuntimeError("ASR model is not loaded. Call load() first.")

        with self._torch.inference_mode():
            audio_tensor = self._torch.from_numpy(audio_chunk).unsqueeze(0).to(self.device)
            audio_len = self._torch.tensor([audio_tensor.shape[1]], device=self.device)

            # Preprocess to mel spectrogram
            processed_signal, processed_signal_length = self.model.preprocessor(
                input_signal=audio_tensor, length=audio_len
            )

            # Prepend pre-encode cache
            pre_encode_size = self._pre_encode_cache.shape[-1]
            processed_signal = self._torch.cat([self._pre_encode_cache, processed_signal], dim=-1)
            processed_signal_length += pre_encode_size
            self._pre_encode_cache = processed_signal[:, :, -pre_encode_size:].clone()

            drop = (
                0
                if self._step_num == 0
                else self.model.encoder.streaming_cfg.drop_extra_pre_encoded
            )

            try:
                (
                    self._pred_out_stream,
                    transcribed_texts,
                    self._cache_last_channel,
                    self._cache_last_time,
                    self._cache_last_channel_len,
                    self._previous_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=self._cache_last_channel,
                    cache_last_time=self._cache_last_time,
                    cache_last_channel_len=self._cache_last_channel_len,
                    keep_all_outputs=False,
                    previous_hypotheses=self._previous_hypotheses,
                    previous_pred_out=self._pred_out_stream,
                    drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )
            except Exception as e:
                # If the cache gets corrupted due to race conditions or bad data,
                # NeMo will throw cryptic shape errors here. Re-raise cleanly.
                log.exception("NeMo stream step failed. Model state may be corrupted.")
                raise e

            self._step_num += 1
            if not transcribed_texts:
                return ""

            text = self._normalize_transcript_item(transcribed_texts[0])
            return text

    @classmethod
    def download_model(cls, model_name: str):
        """Pre-download the model and exit."""
        errors = cls.dependency_errors()
        if errors:
            raise RuntimeError("\n".join(errors))

        import nemo.collections.asr as nemo_asr

        log.info("Downloading model: %s", model_name)
        nemo_asr.models.ASRModel.from_pretrained(model_name)
        log.info("Download complete.")
