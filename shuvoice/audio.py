"""Audio capture via sounddevice with PipeWire."""

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
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.fallback_sample_rate = fallback_sample_rate
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stream: sd.InputStream | None = None
        self._resampling = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.warning("Audio status: %s", status)
        audio = indata[:, 0].copy()  # Mono float32, must copy — buffer is reused
        if self._resampling:
            ratio = self.fallback_sample_rate // self.sample_rate
            audio = audio[::ratio]  # Simple decimation (fix #3: PipeWire fallback)
        try:
            self.queue.put_nowait(audio)
        except queue.Full:
            pass  # Drop oldest implicitly by not blocking

    def start(self):
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_samples,
                channels=1,
                dtype="float32",
                callback=self._callback,
                latency="low",
            )
            self._stream.start()
            log.info("Audio capture started at %d Hz", self.sample_rate)
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
            )
            self._stream.start()
            log.info(
                "Audio capture started at %d Hz (resampling to %d Hz)",
                self.fallback_sample_rate,
                self.sample_rate,
            )

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

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
