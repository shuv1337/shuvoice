"""Streaming ASR engine using NeMo Nemotron."""

import logging

import numpy as np
import torch

log = logging.getLogger(__name__)


class ASREngine:
    def __init__(
        self,
        model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b",
        right_context: int = 1,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.right_context = right_context
        self.device = device
        self.model = None
        self._cache_last_channel = None
        self._cache_last_time = None
        self._cache_last_channel_len = None
        self._pre_encode_cache = None
        self._previous_hypotheses = None
        self._pred_out_stream = None
        self._step_num = 0

    def load(self):
        import nemo.collections.asr as nemo_asr

        log.info("Loading model: %s", self.model_name)
        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

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
        cache_last_channel, cache_last_time, cache_last_channel_len = (
            self.model.encoder.get_initial_cache_state(batch_size=1)
        )
        self._cache_last_channel = cache_last_channel
        self._cache_last_time = cache_last_time
        self._cache_last_channel_len = cache_last_channel_len

        num_features = self.model.preprocessor.featurizer.feat_out
        pre_encode_size = self.model.encoder.streaming_cfg.pre_encode_cache_size[1]
        self._pre_encode_cache = torch.zeros(
            (1, num_features, pre_encode_size), device=self.device
        )

        self._previous_hypotheses = None
        self._pred_out_stream = None
        self._step_num = 0

    @torch.inference_mode()
    def process_chunk(self, audio_chunk: np.ndarray) -> str:
        """Process one 1120ms audio chunk, return cumulative transcription."""
        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0).to(self.device)
        audio_len = torch.tensor([audio_tensor.shape[1]], device=self.device)

        # Preprocess to mel spectrogram
        processed_signal, processed_signal_length = self.model.preprocessor(
            input_signal=audio_tensor, length=audio_len
        )

        # Prepend pre-encode cache
        pre_encode_size = self._pre_encode_cache.shape[-1]
        processed_signal = torch.cat(
            [self._pre_encode_cache, processed_signal], dim=-1
        )
        processed_signal_length += pre_encode_size
        self._pre_encode_cache = processed_signal[:, :, -pre_encode_size:].clone()

        drop = (
            0
            if self._step_num == 0
            else self.model.encoder.streaming_cfg.drop_extra_pre_encoded
        )

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

        self._step_num += 1
        return transcribed_texts[0] if transcribed_texts else ""

    @staticmethod
    def download_model(model_name: str):
        """Pre-download the model and exit."""
        import nemo.collections.asr as nemo_asr

        log.info("Downloading model: %s", model_name)
        nemo_asr.models.ASRModel.from_pretrained(model_name)
        log.info("Download complete.")
