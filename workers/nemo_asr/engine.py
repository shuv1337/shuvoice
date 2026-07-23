"""NeMo ASR engine boundary — real NeMo is lazy-imported; fakes used in tests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

log = logging.getLogger(__name__)

# NeMo's native streaming chunk size for each supported right-context profile.
_RIGHT_CONTEXT_CHUNK_SAMPLES: dict[int, int] = {
    0: 1280,
    1: 2560,
    6: 8960,
    13: 17920,
}


def native_chunk_samples(right_context: int) -> int:
    return _RIGHT_CONTEXT_CHUNK_SAMPLES.get(int(right_context), 17920)


def dependency_errors() -> list[str]:
    """Actionable missing-dep diagnostics (no heavy import side effects beyond probe)."""
    errors: list[str] = []
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        errors.append(
            f"Missing PyTorch dependency: {type(exc).__name__}. "
            "Install torch (or python-pytorch-cuda on Arch)."
        )
    try:
        import nemo.collections.asr  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        errors.append(
            f"Missing NeMo ASR dependency: {type(exc).__name__}. "
            "Install nemo-toolkit[asr] (or NeMo from git main)."
        )
    return errors


@dataclass
class NemoLoadConfig:
    model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    right_context: int = 13
    device: str = "cuda"
    use_cuda_graph_decoder: bool = False

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> NemoLoadConfig:
        raw = raw or {}
        return cls(
            model_name=str(raw.get("model_name") or cls.model_name),
            right_context=int(raw.get("right_context", 13)),
            device=str(raw.get("device") or "cuda"),
            use_cuda_graph_decoder=bool(raw.get("use_cuda_graph_decoder", False)),
        )


class AsrEngine(Protocol):
    def load(self, config: NemoLoadConfig) -> None: ...
    def reset(self) -> None: ...
    def process_chunk(self, samples: list[float]) -> str: ...
    def finish(self, timeout_ms: object | None = None) -> str: ...
    @property
    def right_context(self) -> int: ...
    @property
    def model_name(self) -> str: ...
    @property
    def device(self) -> str: ...
    @property
    def loaded(self) -> bool: ...


class FakeNemoEngine:
    """Deterministic stand-in for unit tests (no NeMo/torch)."""

    def __init__(self) -> None:
        self._cfg = NemoLoadConfig()
        self._loaded = False
        self._step = 0
        self._last = ""
        self.chunks: list[int] = []

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def right_context(self) -> int:
        return self._cfg.right_context

    @property
    def model_name(self) -> str:
        return self._cfg.model_name

    @property
    def device(self) -> str:
        return self._cfg.device

    def load(self, config: NemoLoadConfig) -> None:
        self._cfg = config
        self._loaded = True
        self.reset()

    def reset(self) -> None:
        if not self._loaded:
            raise RuntimeError("not_loaded")
        self._step = 0
        self._last = ""
        self.chunks.clear()

    def process_chunk(self, samples: list[float]) -> str:
        if not self._loaded:
            raise RuntimeError("not_loaded")
        self.chunks.append(len(samples))
        self._step += 1
        # Cumulative fake transcript based on step count only (no audio content leak in errors).
        self._last = f"step-{self._step}"
        return self._last

    def finish(self, timeout_ms: object | None = None) -> str:
        if not self._loaded:
            raise RuntimeError("not_loaded")
        _ = timeout_ms
        # Fake keeps last partial; real engine flushes trailing context with silence.
        return self._last


class RealNemoEngine:
    """Owns the standalone NeMo streaming implementation for this worker."""

    def __init__(self) -> None:
        self._backend: Any = None
        self._cfg = NemoLoadConfig()
        self._last = ""

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    @property
    def right_context(self) -> int:
        return self._cfg.right_context

    @property
    def model_name(self) -> str:
        return self._cfg.model_name

    @property
    def device(self) -> str:
        return self._cfg.device

    def load(self, config: NemoLoadConfig) -> None:
        errors = dependency_errors()
        if errors:
            raise RuntimeError("deps:" + "|".join(errors))
        self._load_standalone(config)

    def _load_standalone(self, config: NemoLoadConfig) -> None:
        import numpy as np

        # Keep a tiny shim object with the same methods as NemoBackend.
        from types import SimpleNamespace

        import nemo.collections.asr as nemo_asr
        import torch

        model = nemo_asr.models.ASRModel.from_pretrained(config.model_name)
        model.eval()
        model.to(config.device)
        model.encoder.set_default_att_context_size([70, config.right_context])
        model.encoder.setup_streaming_params()

        state = SimpleNamespace(
            model=model,
            torch=torch,
            np=np,
            device=config.device,
            right_context=config.right_context,
            cache_last_channel=None,
            cache_last_time=None,
            cache_last_channel_len=None,
            pre_encode_cache=None,
            previous_hypotheses=None,
            pred_out_stream=None,
            step_num=0,
        )
        self._backend = ("standalone", state)
        self._cfg = config
        self._reset_standalone(state)
        self._last = ""

    def _reset_standalone(self, state: Any) -> None:
        model = state.model
        torch = state.torch
        cache_last_channel, cache_last_time, cache_last_channel_len = (
            model.encoder.get_initial_cache_state(batch_size=1)
        )
        state.cache_last_channel = cache_last_channel.to(state.device).clone()
        state.cache_last_time = cache_last_time.to(state.device).clone()
        state.cache_last_channel_len = cache_last_channel_len.to(state.device).clone()
        featurizer = model.preprocessor.featurizer
        if hasattr(featurizer, "feat_out"):
            num_features = int(featurizer.feat_out)
        elif hasattr(featurizer, "nfilt"):
            num_features = int(featurizer.nfilt)
        else:
            raise RuntimeError("feature_dim_unknown")
        pre_encode_size = model.encoder.streaming_cfg.pre_encode_cache_size[1]
        state.pre_encode_cache = torch.zeros(
            (1, num_features, pre_encode_size), device=state.device
        )
        state.previous_hypotheses = None
        state.pred_out_stream = None
        state.step_num = 0

    def reset(self) -> None:
        if self._backend is None:
            raise RuntimeError("not_loaded")
        if isinstance(self._backend, tuple):
            self._reset_standalone(self._backend[1])
        else:
            self._backend.reset()
        self._last = ""

    def process_chunk(self, samples: list[float]) -> str:
        if self._backend is None:
            raise RuntimeError("not_loaded")
        import numpy as np

        audio = np.asarray(samples, dtype=np.float32)
        if isinstance(self._backend, tuple):
            text = self._process_standalone(self._backend[1], audio)
        else:
            text = self._backend.process_chunk(audio)
        self._last = text or self._last
        return text or ""

    def _process_standalone(self, state: Any, audio_chunk: Any) -> str:
        torch = state.torch
        model = state.model
        with torch.inference_mode():
            audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0).to(state.device)
            audio_len = torch.tensor([audio_tensor.shape[1]], device=state.device)
            processed_signal, processed_signal_length = model.preprocessor(
                input_signal=audio_tensor, length=audio_len
            )
            pre_encode_size = state.pre_encode_cache.shape[-1]
            processed_signal = torch.cat([state.pre_encode_cache, processed_signal], dim=-1)
            processed_signal_length += pre_encode_size
            state.pre_encode_cache = processed_signal[:, :, -pre_encode_size:].clone()
            drop = 0 if state.step_num == 0 else model.encoder.streaming_cfg.drop_extra_pre_encoded
            (
                state.pred_out_stream,
                transcribed_texts,
                state.cache_last_channel,
                state.cache_last_time,
                state.cache_last_channel_len,
                state.previous_hypotheses,
            ) = model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=state.cache_last_channel,
                cache_last_time=state.cache_last_time,
                cache_last_channel_len=state.cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=state.previous_hypotheses,
                previous_pred_out=state.pred_out_stream,
                drop_extra_pre_encoded=drop,
                return_transcription=True,
            )
            state.step_num += 1
            if not transcribed_texts:
                return ""
            item = transcribed_texts[0]
            if isinstance(item, str):
                return item
            text = getattr(item, "text", None)
            return text if isinstance(text, str) else ""

    def finish(self, timeout_ms: object | None = None) -> str:
        if self._backend is None:
            raise RuntimeError("not_loaded")
        _ = timeout_ms
        # Flush trailing right-context by feeding silence native chunks.
        # Residual: NeMo conformer_stream_step is not preemptible mid-call; the
        # host may kill the worker process to bound runaway inference.
        n = native_chunk_samples(self._cfg.right_context)
        for _ in range(2):
            text = self.process_chunk([0.0] * n)
            if text:
                self._last = text
        return self._last


def create_engine(*, fake: bool = False) -> AsrEngine:
    if fake:
        return FakeNemoEngine()
    return RealNemoEngine()
