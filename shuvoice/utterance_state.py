"""Pure utterance buffering state for ASR loop orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _UtteranceState:
    """Mutable utterance state container (private by convention)."""

    buffer: list[np.ndarray] = field(default_factory=list)
    total: int = 0
    last_text: str = ""
    speech_samples: int = 0
    peak_rms: float = 0.0
    utterance_gain: float = 1.0
    utterance_rms_threshold: float = 0.0
    unchanged_steps: int = 0
    last_chunk_rms: float = 0.0

    def reset(self, rms_threshold: float = 0.0):
        self.buffer.clear()
        self.total = 0
        self.last_text = ""
        self.speech_samples = 0
        self.peak_rms = 0.0
        self.utterance_gain = 1.0
        self.utterance_rms_threshold = rms_threshold
        self.unchanged_steps = 0
        self.last_chunk_rms = 0.0

    def add_chunk(self, chunk: np.ndarray):
        if chunk.size == 0:
            return
        self.buffer.append(chunk)
        self.total += len(chunk)

    def consume_native_chunk(self, native: int) -> tuple[np.ndarray, bool]:
        if native <= 0:
            raise ValueError("native must be >= 1")
        if not self.buffer:
            return np.empty(0, dtype=np.float32), False

        if len(self.buffer) == 1 and len(self.buffer[0]) >= native:
            audio_data = self.buffer[0]
        else:
            audio_data = np.concatenate(self.buffer)

        to_process = audio_data[:native]
        remainder = audio_data[native:]
        self.buffer = [remainder] if len(remainder) > 0 else []
        self.total = len(remainder)
        return to_process, self.total >= native
