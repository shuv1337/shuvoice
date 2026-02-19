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
        self._dropped_chunks = 0

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.warning("Audio status: %s", status)

        # Mono float32, must copy — sounddevice reuses callback buffer.
        audio = indata[:, 0].copy()

        if self._resampling:
            ratio = self.fallback_sample_rate // self.sample_rate
            audio = audio[::ratio]  # Simple decimation for 48k->16k fallback mode

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
            self._resampling = True
            ratio = self.fallback_sample_rate // self.sample_rate
            self._stream = sd.InputStream(
                samplerate=self.fallback_sample_rate,
                blocksize=self.chunk_samples * ratio,
                channels=1,
                dtype="float32",
                callback=self._callback,
                latency="low",
                device=self.device,
            )
            self._stream.start()
            log.info(
                "Audio capture started at %d Hz (resampling to %d Hz, device=%s, gain=%.2f)",
                self.fallback_sample_rate,
                self.sample_rate,
                self.device if self.device is not None else "default",
                self.input_gain,
            )

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._dropped_chunks:
            log.info("Audio dropped chunks total: %d", self._dropped_chunks)

    def clear(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def get_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
