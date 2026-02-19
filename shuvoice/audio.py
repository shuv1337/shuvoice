"""Audio capture via sounddevice with PipeWire."""

from __future__ import annotations

import logging
import queue

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


class AudioCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_samples: int = 1600,
        fallback_sample_rate: int = 48000,
        device: str | int | None = None,
        input_gain: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.fallback_sample_rate = fallback_sample_rate
        self.device = device
        self.input_gain = float(input_gain)

        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stream: sd.InputStream | None = None
        self._resampling = False
        self._resample_ratio: int | None = None
        self._resample_carry = np.empty(0, dtype=np.float32)
        self._dropped_chunks = 0

    def _downsample_integer_ratio(self, audio: np.ndarray, ratio: int) -> np.ndarray:
        """Downsample by integer ratio with simple anti-alias averaging.

        This performs a box-filter before decimation (average every N samples)
        rather than naive sample dropping. It's lightweight and yields cleaner
        ASR input than raw `audio[::ratio]`.
        """
        if ratio <= 1:
            return audio

        if self._resample_carry.size:
            audio = np.concatenate((self._resample_carry, audio), axis=0)

        usable = (len(audio) // ratio) * ratio
        if usable == 0:
            self._resample_carry = audio
            return np.empty(0, dtype=np.float32)

        out = audio[:usable].reshape(-1, ratio).mean(axis=1, dtype=np.float32)
        self._resample_carry = audio[usable:]
        return out.astype(np.float32, copy=False)

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.warning("Audio status: %s", status)

        # Mono float32, must copy — sounddevice reuses callback buffer.
        audio = indata[:, 0].copy()

        if self._resampling:
            ratio = self._resample_ratio or (self.fallback_sample_rate // self.sample_rate)
            audio = self._downsample_integer_ratio(audio, ratio)
            if audio.size == 0:
                return

        if self.input_gain != 1.0:
            audio = np.clip(audio * self.input_gain, -1.0, 1.0)

        try:
            self.queue.put_nowait(audio)
        except queue.Full:
            # Keep latency low by dropping oldest data and retaining freshest chunk.
            self._dropped_chunks += 1
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.queue.put_nowait(audio)
            except queue.Full:
                pass

            if self._dropped_chunks % 50 == 1:
                log.warning(
                    "Audio queue overflow: dropped %d chunks (queue size=%d)",
                    self._dropped_chunks,
                    self.queue.qsize(),
                )

    def start(self):
        self._resampling = False
        self._resample_ratio = None
        self._resample_carry = np.empty(0, dtype=np.float32)

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_samples,
                channels=1,
                dtype="float32",
                callback=self._callback,
                latency="low",
                device=self.device,
            )
            self._stream.start()
            log.info(
                "Audio capture started at %d Hz (device=%s, gain=%.2f)",
                self.sample_rate,
                self.device if self.device is not None else "default",
                self.input_gain,
            )
        except sd.PortAudioError:
            log.warning(
                "Failed at %d Hz, falling back to %d Hz",
                self.sample_rate,
                self.fallback_sample_rate,
            )
            if self.fallback_sample_rate % self.sample_rate != 0:
                raise RuntimeError(
                    "fallback_sample_rate must be an integer multiple of sample_rate "
                    f"(got {self.fallback_sample_rate} and {self.sample_rate})"
                )

            self._resampling = True
            self._resample_ratio = self.fallback_sample_rate // self.sample_rate
            self._resample_carry = np.empty(0, dtype=np.float32)
            self._stream = sd.InputStream(
                samplerate=self.fallback_sample_rate,
                blocksize=self.chunk_samples * self._resample_ratio,
                channels=1,
                dtype="float32",
                callback=self._callback,
                latency="low",
                device=self.device,
            )
            self._stream.start()
            log.info(
                "Audio capture started at %d Hz (resampling to %d Hz, ratio=%dx, device=%s, gain=%.2f)",
                self.fallback_sample_rate,
                self.sample_rate,
                self._resample_ratio,
                self.device if self.device is not None else "default",
                self.input_gain,
            )

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._resample_carry = np.empty(0, dtype=np.float32)

        if self._dropped_chunks:
            log.info("Audio dropped chunks total: %d", self._dropped_chunks)

    def drain_pending_chunks(self) -> list[np.ndarray]:
        chunks: list[np.ndarray] = []
        while not self.queue.empty():
            try:
                chunks.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    def clear(self):
        self.drain_pending_chunks()

    def get_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
