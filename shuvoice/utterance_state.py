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
    speech_chunks_seen: int = 0
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
        self.speech_chunks_seen = 0
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

        # Fast path: first chunk is large enough
        if len(self.buffer[0]) >= native:
            first_chunk = self.buffer[0]
            to_process = first_chunk[:native]
            remainder = first_chunk[native:]

            if remainder.size > 0:
                self.buffer[0] = remainder
            else:
                self.buffer.pop(0)

            self.total -= native
            return to_process, True  # Since we checked len >= native

        # Slow path: consume multiple chunks until we satisfy `native`
        extracted_chunks = []
        samples_extracted = 0

        while self.buffer and samples_extracted < native:
            chunk = self.buffer[0]
            needed = native - samples_extracted

            if len(chunk) <= needed:
                extracted_chunks.append(chunk)
                samples_extracted += len(chunk)
                self.buffer.pop(0)
            else:
                extracted_chunks.append(chunk[:needed])
                self.buffer[0] = chunk[needed:]
                samples_extracted += needed

        if not extracted_chunks:
            return np.empty(0, dtype=np.float32), False

        to_process = np.concatenate(extracted_chunks)
        self.total -= len(to_process)

        return to_process, len(to_process) >= native
