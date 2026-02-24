"""Runtime orchestration helpers."""

from .chunk_pipeline import (
    append_recording_chunk,
    apply_utterance_gain,
    begin_utterance,
    drain_and_buffer,
    process_recording_chunks,
    transcribe_native_chunk,
    update_noise_floor,
)
from .flush_policy import flush_streaming_stall, flush_tail_silence, make_flush_noise
from .state_machine import (
    on_recording_start,
    on_recording_stop,
    on_recording_toggle,
    recording_status,
)

__all__ = [
    "append_recording_chunk",
    "apply_utterance_gain",
    "begin_utterance",
    "drain_and_buffer",
    "flush_streaming_stall",
    "flush_tail_silence",
    "make_flush_noise",
    "on_recording_start",
    "on_recording_stop",
    "on_recording_toggle",
    "process_recording_chunks",
    "recording_status",
    "transcribe_native_chunk",
    "update_noise_floor",
]
