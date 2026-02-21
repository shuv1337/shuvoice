from __future__ import annotations

from shuvoice.streaming_health import should_trigger_stall_flush


def test_should_trigger_stall_flush_false_before_chunk_limit():
    assert not should_trigger_stall_flush(
        unchanged_steps=2,
        chunk_rms=0.12,
        utterance_threshold=0.05,
        stall_chunks=4,
        stall_rms_ratio=0.7,
    )


def test_should_trigger_stall_flush_false_when_rms_low():
    assert not should_trigger_stall_flush(
        unchanged_steps=5,
        chunk_rms=0.02,
        utterance_threshold=0.05,
        stall_chunks=4,
        stall_rms_ratio=0.7,
    )


def test_should_trigger_stall_flush_true_when_stalled_and_active_speech():
    assert should_trigger_stall_flush(
        unchanged_steps=5,
        chunk_rms=0.05,
        utterance_threshold=0.05,
        stall_chunks=4,
        stall_rms_ratio=0.7,
    )


def test_should_trigger_stall_flush_threshold_zero_uses_positive_rms():
    assert should_trigger_stall_flush(
        unchanged_steps=4,
        chunk_rms=0.001,
        utterance_threshold=0.0,
        stall_chunks=4,
        stall_rms_ratio=0.7,
    )
