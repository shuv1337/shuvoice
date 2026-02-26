"""Recording state-machine handlers for ``ShuVoiceApp``."""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

# Guard against spurious immediate re-starts caused by key bounce / compositor
# ordering quirks on press+release keybinds (notably bare modifier keys).
_PTT_REARM_GRACE_SEC = 0.35


def on_recording_start(app) -> None:
    if app._recording.is_set():
        log.debug("Recording already active; ignoring start")
        return

    if not app._asr_thread_alive:
        app._show_overlay_error("⚠ ASR thread crashed — restart ShuVoice")
        return

    last_stop_monotonic = float(getattr(app, "_last_stop_monotonic", 0.0) or 0.0)
    since_stop = time.monotonic() - last_stop_monotonic
    if getattr(app, "_processing", None) is not None and app._processing.is_set():
        if 0.0 <= since_stop < _PTT_REARM_GRACE_SEC:
            log.debug(
                "Ignoring start during processing rearm window (%.3fs < %.3fs)",
                since_stop,
                _PTT_REARM_GRACE_SEC,
            )
            return

    with app._asr_lock:
        if app._asr_disabled:
            log.warning("ASR disabled; attempting one-shot reset on recording start")
            try:
                app.asr.reset()
            except Exception:  # noqa: BLE001
                log.exception("ASR recovery reset failed; still disabled")
                app._show_overlay_error("⚠ ASR error — restart ShuVoice")
                return
            app._asr_disabled = False
            app._consecutive_asr_failures = 0

        app.audio.clear()

        try:
            app.asr.reset()
        except Exception:  # noqa: BLE001
            app._consecutive_asr_failures += 1
            failures = app._consecutive_asr_failures
            log.exception(
                "ASR reset failed on recording start (%d/%d)",
                failures,
                app._ASR_MAX_FAILURES,
            )
            if failures >= app._ASR_MAX_FAILURES:
                app._disable_asr("ASR disabled after repeated reset failures")
            else:
                app._show_overlay_error("⚠ ASR error — restart ShuVoice")
            return

        app.audio.clear()

    app._processing.clear()
    app._recording.set()
    metrics = getattr(app, "metrics", None)
    if metrics is not None:
        metrics.recording_started()
    log.info("Recording started")

    if app.overlay:
        app.overlay.show()
        app.overlay.set_state("listening")
        app.overlay.set_text("Listening…")

    app._play_feedback_tone(is_start=True)


def on_recording_stop(app) -> None:
    if not app._recording.is_set():
        log.debug("Recording already stopped; ignoring stop")
        return

    log.info("Recording stopped")
    app._recording.clear()
    app._processing.set()
    app._last_stop_monotonic = time.monotonic()
    metrics = getattr(app, "metrics", None)
    if metrics is not None:
        metrics.recording_stopped()
    app._play_feedback_tone(is_start=False)

    if app.overlay:
        app.overlay.set_state("processing")


def on_recording_toggle(app) -> None:
    if app._recording.is_set():
        on_recording_stop(app)
    else:
        on_recording_start(app)


def recording_status(app) -> str:
    if app._asr_disabled:
        return "error:asr_disabled"
    if not app._asr_thread_alive:
        return "error:asr_thread_dead"
    if app._recording.is_set():
        return "recording"
    if app._processing.is_set():
        return "processing"
    return "idle"
