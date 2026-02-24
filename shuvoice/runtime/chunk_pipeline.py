"""Chunk buffering + gain + ASR dispatch helpers."""

from __future__ import annotations

import logging

import numpy as np

from ..audio import audio_rms
from ..streaming_health import should_trigger_stall_flush
from ..transcript import prefer_transcript

log = logging.getLogger("shuvoice.app")


def apply_utterance_gain(audio: np.ndarray, gain: float) -> np.ndarray:
    if gain <= 1.05 or audio.size == 0:
        return audio
    # Use a single pre-allocated output buffer to avoid temporary arrays in this hot path.
    result = np.empty_like(audio, dtype=np.float32)
    np.multiply(audio, gain, out=result)
    np.clip(result, -1.0, 1.0, out=result)
    return result


def update_noise_floor(app, chunk_rms: float) -> None:
    if chunk_rms <= 0.0:
        return
    if app._noise_floor_rms <= 0.0:
        app._noise_floor_rms = chunk_rms
    else:
        app._noise_floor_rms = 0.98 * app._noise_floor_rms + 0.02 * chunk_rms


def begin_utterance(app, state) -> None:
    # Defensive reset: ensure ASR model is clean after any potential contamination
    # from tail flush race conditions.
    with app._asr_lock:
        if not app._asr_disabled:
            try:
                app.asr.reset()
            except Exception:  # noqa: BLE001
                log.exception("ASR reset failed at utterance start")
                app._recover_asr_after_failure("ASR reset at utterance start")

    threshold = max(
        app._speech_rms_threshold,
        app._noise_floor_rms * app._speech_rms_multiplier,
    )
    state.reset(rms_threshold=threshold)
    log.debug(
        "Recording energy threshold: %.4f (noise_floor=%.4f, floor=%.4f, x%.2f)",
        threshold,
        app._noise_floor_rms,
        app._speech_rms_threshold,
        app._speech_rms_multiplier,
    )


def append_recording_chunk(app, state, chunk: np.ndarray) -> None:
    state.add_chunk(chunk)
    chunk_rms = audio_rms(chunk)
    state.last_chunk_rms = chunk_rms
    state.peak_rms = max(state.peak_rms, chunk_rms)

    if chunk_rms >= state.utterance_rms_threshold:
        state.speech_samples += len(chunk)
        state.speech_chunks_seen += 1

    # Backends with internal normalization (e.g. NeMo/Moonshine) bypass app-side gain entirely.
    if app.asr.wants_raw_audio:
        return

    if state.speech_chunks_seen < app._auto_gain_settle_chunks:
        return

    if state.peak_rms > 0.003:
        state.utterance_gain = min(
            app._auto_gain_target_peak / state.peak_rms,
            app._auto_gain_max,
        )


def transcribe_native_chunk(app, state, error_context: str) -> bool:
    to_process, has_more = state.consume_native_chunk(app.asr.native_chunk_samples)
    if to_process.size == 0:
        return False

    if not app.asr.wants_raw_audio:
        to_process = apply_utterance_gain(to_process, state.utterance_gain)

    try:
        text = app._process_chunk_safe(to_process)
    except Exception:  # noqa: BLE001
        app._recover_asr_after_failure(error_context)
        return False

    metrics = getattr(app, "metrics", None)
    if metrics is not None:
        metrics.observe_chunk(audio_rms(to_process), app.audio.queue.qsize())

    log.debug(
        "ASR step=%s queue_size=%d raw_text_len=%d chunk_rms=%.4f gain=%.1f",
        app.asr.debug_step_num,
        app.audio.queue.qsize(),
        len(text),
        audio_rms(to_process),
        state.utterance_gain,
    )

    merged = prefer_transcript(state.last_text, text)
    if merged != state.last_text:
        log.debug("Transcript updated: len %d -> %d", len(state.last_text), len(merged))
        state.last_text = merged
        state.unchanged_steps = 0
        if hasattr(app, "_on_transcript_update"):
            app._on_transcript_update(state.last_text)
    else:
        state.unchanged_steps += 1

    return has_more


def process_recording_chunks(app, state) -> None:
    while (
        app._recording.is_set()
        and not app._asr_disabled
        and state.total >= app.asr.native_chunk_samples
    ):
        has_more = transcribe_native_chunk(app, state, "ASR chunk processing failed")
        if not has_more:
            break

        if app._streaming_stall_guard and should_trigger_stall_flush(
            unchanged_steps=state.unchanged_steps,
            chunk_rms=state.last_chunk_rms,
            utterance_threshold=state.utterance_rms_threshold,
            stall_chunks=app._streaming_stall_chunks,
            stall_rms_ratio=app._streaming_stall_rms_ratio,
        ):
            log.debug(
                "Streaming stall guard triggered (unchanged_steps=%d, chunk_rms=%.4f, threshold=%.4f)",
                state.unchanged_steps,
                state.last_chunk_rms,
                state.utterance_rms_threshold,
            )
            app._flush_streaming_stall(state)


def drain_and_buffer(app, state) -> None:
    drained = app.audio.drain_pending_chunks()
    if drained:
        for drained_chunk in drained:
            append_recording_chunk(app, state, drained_chunk)
        log.debug(
            "Drained %d queued audio chunk(s) on stop (%d samples buffered)",
            len(drained),
            state.total,
        )

    tail_chunk = app.audio.get_chunk(timeout=0.12)
    if tail_chunk is not None:
        append_recording_chunk(app, state, tail_chunk)

        drained_after_tail = app.audio.drain_pending_chunks()
        if drained_after_tail:
            for drained_chunk in drained_after_tail:
                append_recording_chunk(app, state, drained_chunk)
            log.debug(
                "Drained %d queued audio chunk(s) after stop grace (%d samples buffered)",
                len(drained_after_tail),
                state.total,
            )
