"""Streaming health heuristics for ASR loop stability."""

from __future__ import annotations


def should_trigger_stall_flush(
    *,
    unchanged_steps: int,
    chunk_rms: float,
    utterance_threshold: float,
    stall_chunks: int,
    stall_rms_ratio: float,
) -> bool:
    """Return True when the stream appears stalled despite active speech.

    A stall is considered likely when:
    - transcript has not changed for N consecutive native chunks
    - current chunk energy is still above a fraction of the speech threshold
    """
    if unchanged_steps < max(1, int(stall_chunks)):
        return False

    rms_ratio = max(0.0, float(stall_rms_ratio))
    threshold = float(utterance_threshold)
    chunk_rms = float(chunk_rms)

    if threshold <= 0.0:
        return chunk_rms > 0.0

    return chunk_rms >= (threshold * rms_ratio)
