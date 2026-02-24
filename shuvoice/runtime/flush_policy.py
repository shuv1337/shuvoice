"""Streaming tail flush/noise policy helpers."""

from __future__ import annotations

import logging

import numpy as np

from ..transcript import prefer_transcript
from .chunk_pipeline import apply_utterance_gain

log = logging.getLogger(__name__)


def make_flush_noise(app, n_samples: int, escalation: float = 1.0) -> np.ndarray:
    """Generate low-amplitude noise for flushing streaming transducers."""
    base_rms = max(app._noise_floor_rms, app._FLUSH_NOISE_MIN_RMS)
    rms = min(base_rms * escalation, app._FLUSH_NOISE_MAX_RMS)
    noise = np.random.default_rng().normal(0.0, rms, size=n_samples).astype(np.float32)
    return np.clip(noise, -1.0, 1.0)


def flush_streaming_stall(app, state) -> None:
    if app._asr_disabled:
        return

    native = app.asr.native_chunk_samples
    # Intentionally feed raw digital silence here; this path is a short stall
    # nudge and should not apply utterance gain.
    silence = np.zeros(native, dtype=np.float32)

    for _ in range(app._streaming_stall_flush_chunks):
        try:
            text = app._process_chunk_safe(silence)
        except Exception:  # noqa: BLE001
            app._recover_asr_after_failure("ASR stall-guard flush failed")
            break

        merged = prefer_transcript(state.last_text, text)
        if merged != state.last_text:
            log.debug(
                "Transcript updated after stall flush: len %d -> %d",
                len(state.last_text),
                len(merged),
            )
            state.last_text = merged
            if hasattr(app, "_on_transcript_update"):
                app._on_transcript_update(state.last_text)

    state.unchanged_steps = 0


def flush_tail_silence(app, state) -> None:
    if app._asr_disabled:
        return

    native = app.asr.native_chunk_samples
    stable_steps = 0
    ever_had_text = bool(state.last_text.strip())
    # Generous budget: streaming transducers (e.g. Sherpa) may need many
    # silence frames to flush internally-buffered hypotheses after
    # the user stops speaking.
    max_flush = 20
    stable_required = 5
    # Track consecutive stalled steps so we can escalate noise amplitude.
    stalled_consecutive = 0

    for i in range(max_flush):
        # Abort tail flush if a new recording has started to avoid contaminating
        # the fresh ASR state with noise.
        if app._recording.is_set():
            log.debug("Aborting tail flush: new recording started")
            break

        escalation = app._FLUSH_NOISE_ESCALATION**stalled_consecutive
        if hasattr(app, "_make_flush_noise"):
            flush_audio = app._make_flush_noise(native, escalation=escalation)
        else:
            flush_audio = make_flush_noise(app, native, escalation=escalation)
        if not app.asr.wants_raw_audio:
            flush_audio = apply_utterance_gain(flush_audio, state.utterance_gain)

        try:
            text = app._process_chunk_safe(flush_audio)
        except Exception:  # noqa: BLE001
            app._recover_asr_after_failure("ASR tail flush failed")
            break

        merged = prefer_transcript(state.last_text, text)
        if merged != state.last_text:
            log.debug(
                "Tail flush step %d: len %d -> %d (escalation=%.2f)",
                i,
                len(state.last_text),
                len(merged),
                escalation,
            )
            state.last_text = merged
            stable_steps = 0
            stalled_consecutive = 0
            ever_had_text = True
            if hasattr(app, "_on_transcript_update"):
                app._on_transcript_update(state.last_text)
        else:
            stable_steps += 1
            stalled_consecutive += 1
            # Once we've seen text, converge quickly; otherwise keep flushing
            # longer in case the model is still buffering.
            needed = stable_required if ever_had_text else 5
            if stable_steps >= needed:
                break
