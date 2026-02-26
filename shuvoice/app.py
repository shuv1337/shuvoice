"""Main ShuVoice application — ties all components together.

IMPORTANT: ctypes.CDLL('libgtk4-layer-shell.so') must be called
before do_activate() imports .overlay. See __main__.py.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import TYPE_CHECKING

import gi
import numpy as np

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from .asr import create_backend
from .audio import AudioCapture, audio_rms
from .config import Config
from .control import ControlServer
from .diagnostics import metrics_to_json
from .feedback import play_tone
from .metrics import MetricsCollector
from .postprocess import apply_text_replacements, capitalize_first
from .runtime import (
    append_recording_chunk,
    apply_utterance_gain,
    begin_utterance,
    drain_and_buffer,
    flush_streaming_stall,
    flush_tail_silence,
    make_flush_noise,
    on_recording_start,
    on_recording_stop,
    on_recording_toggle,
    process_recording_chunks,
    recording_status,
    transcribe_native_chunk,
    update_noise_floor,
)
from .transcript import prefer_transcript
from .typer import StreamingTyper
from .utterance_state import _UtteranceState

if TYPE_CHECKING:
    from .overlay import CaptionOverlay

log = logging.getLogger(__name__)


class ShuVoiceApp(Gtk.Application):
    """GTK4 application that orchestrates streaming speech-to-text."""

    _ASR_MAX_FAILURES = 10
    _MIN_SPLASH_VISIBLE_SEC = 2.0

    def __init__(self, config: Config):
        super().__init__(application_id="io.github.shuv1337.shuvoice")
        self.config = config

        self.asr = create_backend(config.asr_backend, config)
        self.audio = AudioCapture(
            config.sample_rate,
            config.chunk_samples,
            config.fallback_sample_rate,
            device=config.audio_device,
            input_gain=config.input_gain,
            audio_queue_max_size=config.audio_queue_max_size,
        )
        self.typer = StreamingTyper(
            final_injection_mode=config.typing_final_injection_mode,
            preserve_clipboard=config.preserve_clipboard,
            clipboard_settle_delay_ms=config.typing_clipboard_settle_delay_ms,
            retry_attempts=config.typing_retry_attempts,
            retry_delay_ms=config.typing_retry_delay_ms,
        )
        self.metrics = MetricsCollector()

        if config.output_mode not in {"final_only", "streaming_partial"}:
            raise ValueError(
                f"Invalid output_mode '{config.output_mode}'. "
                "Expected one of: final_only, streaming_partial"
            )

        self.control = ControlServer(
            socket_path=config.control_socket,
            on_start=self._on_recording_start,
            on_stop=self._on_recording_stop,
            on_toggle=self._on_recording_toggle,
            on_status=self._recording_status,
            on_metrics=self._metrics_status,
        )

        self.overlay: CaptionOverlay | None = None

        self._recording = threading.Event()
        self._processing = threading.Event()
        self._running = threading.Event()
        self._running.set()

        self._asr_lock = threading.Lock()
        self._asr_thread_alive = True

        self._consecutive_asr_failures = 0
        self._asr_disabled = False
        self._model_load_failed = False
        self._splash_started_monotonic: float | None = None

        self._speech_rms_threshold = max(0.0, float(self.config.silence_rms_threshold))
        self._speech_rms_multiplier = max(1.0, float(self.config.silence_rms_multiplier))
        self._min_speech_samples = max(
            0,
            self.config.sample_rate * max(0, int(self.config.min_speech_ms)) // 1000,
        )
        self._auto_gain_target_peak = max(1e-4, float(self.config.auto_gain_target_peak))
        self._auto_gain_max = max(1.0, float(self.config.auto_gain_max))
        self._auto_gain_settle_chunks = max(1, int(self.config.auto_gain_settle_chunks))
        self._noise_floor_rms = 0.0

        self._streaming_stall_guard = bool(self.config.streaming_stall_guard)
        self._streaming_stall_chunks = max(1, int(self.config.streaming_stall_chunks))
        self._streaming_stall_rms_ratio = max(0.0, float(self.config.streaming_stall_rms_ratio))
        self._streaming_stall_flush_chunks = max(1, int(self.config.streaming_stall_flush_chunks))

        self._last_stop_monotonic = 0.0
        self._next_metrics_log_monotonic = time.monotonic() + 10.0

    def load_model(self):
        """Load the ASR model synchronously (legacy helper).

        Prefer letting ``do_activate`` load the model asynchronously with a
        splash screen.  Call this only when you need the model ready before
        ``run()`` (e.g. in tests).
        """
        self.asr.load()
        self._model_loaded = True

    # -- GTK lifecycle ------------------------------------------------------

    def do_activate(self):
        GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGINT, self._on_sigint)

        if getattr(self, "_model_loaded", False):
            # Model was pre-loaded (legacy / test path) — skip splash.
            self._finish_activation()
            return

        # Show splash while model loads in the background.
        from .splash import SplashOverlay

        self._splash_started_monotonic = time.monotonic()
        self._splash = SplashOverlay(self)
        threading.Thread(target=self._load_model_async, name="model-loader", daemon=True).start()

    @staticmethod
    def _remaining_splash_ms(
        shown_at_monotonic: float | None,
        min_visible_sec: float,
        *,
        now_monotonic: float | None = None,
    ) -> int:
        """Return remaining splash visibility time in milliseconds."""
        if shown_at_monotonic is None or min_visible_sec <= 0:
            return 0

        now = time.monotonic() if now_monotonic is None else now_monotonic
        elapsed = max(0.0, now - shown_at_monotonic)
        remaining = max(0.0, min_visible_sec - elapsed)
        return int(remaining * 1000)

    def _complete_model_loaded_startup(self):
        splash = getattr(self, "_splash", None)
        if splash:
            splash.dismiss()
            self._splash = None

        self._splash_started_monotonic = None
        self._finish_activation()
        return GLib.SOURCE_REMOVE

    def _report_model_progress(self, fraction: float | None, message: str):
        splash = getattr(self, "_splash", None)
        if splash is None:
            return

        set_progress = getattr(splash, "set_progress", None)
        if callable(set_progress):
            set_progress(fraction, message)
            return

        set_status = getattr(splash, "set_status", None)
        if callable(set_status):
            set_status(message)

    def _load_model_async(self):
        """Background thread: load the ASR model, then signal the main thread."""
        try:
            self._report_model_progress(None, "Loading model runtime…")
            try:
                self.asr.load(progress_callback=self._report_model_progress)
            except TypeError as exc:
                # Backward-compatible fallback for backends that don't accept
                # progress_callback in load().
                if "progress_callback" not in str(exc):
                    raise
                self.asr.load()

            GLib.idle_add(self._on_model_loaded)
        except Exception as exc:
            error_msg = str(exc)
            GLib.idle_add(self._on_model_load_failed, error_msg)

    def _on_model_loaded(self):
        self._model_loaded = True

        splash = getattr(self, "_splash", None)
        shown_at_monotonic = None
        if splash is not None:
            ShuVoiceApp._report_model_progress(self, 1.0, "Model ready. Starting ShuVoice…")
            shown_at_monotonic = getattr(splash, "shown_monotonic", None)

        remaining_visible_ms = ShuVoiceApp._remaining_splash_ms(
            shown_at_monotonic or getattr(self, "_splash_started_monotonic", None),
            self._MIN_SPLASH_VISIBLE_SEC,
        )
        min_post_load_ms = int(self._MIN_SPLASH_VISIBLE_SEC * 1000)
        delay_ms = max(remaining_visible_ms, min_post_load_ms)

        if delay_ms > 0:
            log.debug(
                "Holding splash for %dms after model load (remaining visible=%dms)",
                delay_ms,
                remaining_visible_ms,
            )
            GLib.timeout_add(delay_ms, self._complete_model_loaded_startup)
        else:
            self._complete_model_loaded_startup()

        return GLib.SOURCE_REMOVE

    def _on_model_load_failed(self, error_msg: str):
        log.critical("Model loading failed: %s", error_msg)
        self._model_load_failed = True
        splash = getattr(self, "_splash", None)
        if splash:
            splash.set_status(f"Error: {error_msg}")
        GLib.timeout_add(3000, self.quit)
        return GLib.SOURCE_REMOVE

    def _finish_activation(self):
        """Complete app activation after the model is ready."""
        from .overlay import CaptionOverlay

        self.overlay = CaptionOverlay(self, self.config)

        self.audio.start()
        self.control.start()

        threading.Thread(target=self._asr_worker, name="asr", daemon=True).start()

        log.info("Ready — use Hyprland bind/bindr with `shuvoice control start|stop`")
        log.info("Control socket: %s", self.control.socket_path)

    def do_shutdown(self):
        log.info("Shutting down…")
        self._running.clear()
        self._recording.clear()
        self._processing.clear()
        self.control.stop()
        self.audio.stop()
        Gtk.Application.do_shutdown(self)

    def _on_sigint(self):
        log.info("SIGINT received, quitting")
        self.quit()
        return GLib.SOURCE_REMOVE

    # -- Runtime safety helpers --------------------------------------------

    def _show_overlay_error(self, text: str):
        if not self.overlay:
            return
        self.overlay.show()
        self.overlay.set_state("error")
        self.overlay.set_text(text)

    def _disable_asr(self, reason: str):
        self._asr_disabled = True
        self._recording.clear()
        self._processing.clear()
        log.critical(reason)
        self._show_overlay_error("⚠ ASR error — restart ShuVoice")

    def _recover_asr_after_failure(self, context: str):
        if self._asr_disabled:
            return

        with self._asr_lock:
            if self._asr_disabled:
                return
            try:
                self.asr.reset()
                metrics = getattr(self, "metrics", None)
                if metrics is not None:
                    metrics.observe_recovery_reset()
            except Exception:
                self._consecutive_asr_failures += 1
                failures = self._consecutive_asr_failures
                if failures >= self._ASR_MAX_FAILURES:
                    self._disable_asr("ASR disabled after repeated recovery reset failures")
                else:
                    log.exception(
                        "%s; ASR reset failed (%d/%d)",
                        context,
                        failures,
                        self._ASR_MAX_FAILURES,
                    )

    @property
    def _is_offline_instant_mode(self) -> bool:
        return (
            self.config.asr_backend == "sherpa"
            and self.config.resolved_sherpa_decode_mode == "offline_instant"
        )

    def _process_chunk_safe(self, audio_data: np.ndarray) -> str:
        """Serialize access to mutable ASR streaming state."""
        if self._asr_disabled:
            return ""

        with self._asr_lock:
            if self._asr_disabled:
                return ""

            try:
                text = self.asr.process_chunk(audio_data)
            except Exception:
                self._consecutive_asr_failures += 1
                failures = self._consecutive_asr_failures
                if failures >= self._ASR_MAX_FAILURES:
                    log.critical(
                        "ASR process_chunk failed %d times; disabling ASR",
                        failures,
                        exc_info=True,
                    )
                    self._disable_asr("ASR disabled after repeated chunk failures")
                else:
                    log.exception(
                        "ASR chunk processing failed (%d/%d)",
                        failures,
                        self._ASR_MAX_FAILURES,
                    )
                raise

            self._consecutive_asr_failures = 0
            return text

    def _process_utterance_safe(self, audio_data: np.ndarray) -> str:
        """Serialize access to one-shot ASR utterance decoding state."""
        if self._asr_disabled:
            return ""

        with self._asr_lock:
            if self._asr_disabled:
                return ""

            try:
                process_utterance = getattr(self.asr, "process_utterance", None)
                if not callable(process_utterance):
                    raise RuntimeError("ASR backend does not implement process_utterance()")
                text = process_utterance(audio_data)
            except Exception:
                self._consecutive_asr_failures += 1
                failures = self._consecutive_asr_failures
                if failures >= self._ASR_MAX_FAILURES:
                    log.critical(
                        "ASR process_utterance failed %d times; disabling ASR",
                        failures,
                        exc_info=True,
                    )
                    self._disable_asr("ASR disabled after repeated utterance decode failures")
                else:
                    log.exception(
                        "ASR utterance decode failed (%d/%d)",
                        failures,
                        self._ASR_MAX_FAILURES,
                    )
                raise

            self._consecutive_asr_failures = 0
            return text

    # -- Recording state (called from control socket threads) ---------------

    def _play_feedback_tone(self, is_start: bool):
        if not self.config.audio_feedback:
            return

        freq = self.config.feedback_start_freq if is_start else self.config.feedback_stop_freq
        play_tone(
            freq=freq,
            duration_ms=self.config.feedback_duration_ms,
            volume=self.config.feedback_volume,
            sample_rate=self.config.sample_rate,
        )

    def _on_recording_start(self):
        return on_recording_start(self)

    def _on_recording_stop(self):
        return on_recording_stop(self)

    def _on_recording_toggle(self):
        return on_recording_toggle(self)

    def _recording_status(self) -> str:
        return recording_status(self)

    def _metrics_status(self) -> str:
        return metrics_to_json(self.metrics.snapshot())

    def _on_transcript_update(self, text: str):
        rendered_text = self._render_transcript_text(text)
        if self.overlay:
            self.overlay.set_text(rendered_text)
        if self.config.output_mode == "streaming_partial" and not self._is_offline_instant_mode:
            self.typer.update_partial(rendered_text)
            metrics = getattr(self, "metrics", None)
            if metrics is not None:
                metrics.observe_partial_update()

    def _log_metrics_if_due(self):
        if not log.isEnabledFor(logging.INFO):
            return
        now = time.monotonic()
        if now < self._next_metrics_log_monotonic:
            return
        self._next_metrics_log_monotonic = now + 10.0
        log.info(self.metrics.summary_line())

    def _render_transcript_text(self, text: str) -> str:
        """Render transcript text for preview/final output consistency."""
        if not text:
            return text

        rendered = apply_text_replacements(
            text,
            self.config.text_replacements,
            compiled_replacements=getattr(self.config, "compiled_text_replacements", None),
        )
        if not rendered:
            return rendered

        if self.config.auto_capitalize:
            rendered = capitalize_first(rendered)

        return rendered

    # -- ASR processing helpers ---------------------------------------------

    def _apply_utterance_gain(self, audio: np.ndarray, gain: float) -> np.ndarray:
        return apply_utterance_gain(audio, gain)

    def _update_noise_floor(self, chunk_rms: float):
        return update_noise_floor(self, chunk_rms)

    def _begin_utterance(self, state: _UtteranceState):
        return begin_utterance(self, state)

    def _append_recording_chunk(self, state: _UtteranceState, chunk: np.ndarray):
        return append_recording_chunk(self, state, chunk)

    def _transcribe_native_chunk(self, state: _UtteranceState, error_context: str) -> bool:
        return transcribe_native_chunk(self, state, error_context)

    def _flush_streaming_stall(self, state: _UtteranceState):
        metrics = getattr(self, "metrics", None)
        if metrics is not None:
            metrics.observe_stall_flush()
        return flush_streaming_stall(self, state)

    def _process_recording_chunks(self, state: _UtteranceState):
        if self._is_offline_instant_mode:
            return None
        return process_recording_chunks(self, state)

    def _drain_and_buffer(self, state: _UtteranceState):
        return drain_and_buffer(self, state)

    # Minimum RMS for tail-flush noise.  In very quiet rooms the measured
    # noise floor can be ~0.001 which, even after utterance gain, is too
    # quiet for streaming transducers (e.g. Sherpa) to emit buffered tokens.
    # 0.005 pre-gain produces ~0.05–0.10 RMS after typical gain (10–20×),
    # matching the ambient level that reliably triggers flushing.
    _FLUSH_NOISE_MIN_RMS = 0.005
    # Escalation factor applied per stalled flush step.  When the model
    # doesn't emit new text, each subsequent step increases the noise RMS
    # by this multiplier, up to ``_FLUSH_NOISE_MAX_RMS``.
    _FLUSH_NOISE_ESCALATION = 1.4
    _FLUSH_NOISE_MAX_RMS = 0.08

    def _make_flush_noise(self, n_samples: int, escalation: float = 1.0) -> np.ndarray:
        return make_flush_noise(self, n_samples, escalation)

    def _flush_tail_silence(self, state: _UtteranceState):
        return flush_tail_silence(self, state)

    def _commit_utterance(self, state: _UtteranceState):
        final_text = state.last_text.strip()
        if not final_text:
            return

        final_text = self._render_transcript_text(final_text)
        if not final_text:
            return

        log.info("Final: len=%d", len(final_text))
        if self.overlay:
            self.overlay.set_text(final_text)

        # Delegate fully to typer, which resolves the correct final injection mode
        self.typer.commit_final(final_text)

        metrics = getattr(self, "metrics", None)
        if metrics is not None:
            metrics.observe_final_commit()

    def _decode_offline_utterance(self, state: _UtteranceState):
        if state.total <= 0 or self._asr_disabled:
            return

        audio_data = state.buffer[0] if len(state.buffer) == 1 else np.concatenate(state.buffer)
        if not self.asr.wants_raw_audio and state.utterance_gain > 1.05:
            audio_data = self._apply_utterance_gain(audio_data, state.utterance_gain)

        try:
            text = self._process_utterance_safe(audio_data)
        except Exception:
            self._recover_asr_after_failure("ASR offline utterance decode failed")
        else:
            state.last_text = prefer_transcript(state.last_text, text)

    def _handle_recording_stop(self, state: _UtteranceState):
        if self.overlay:
            self.overlay.set_state("processing")

        self._drain_and_buffer(state)

        log.debug(
            "Utterance energy: peak_rms=%.4f speech_samples=%d/%d threshold=%.4f",
            state.peak_rms,
            state.speech_samples,
            self._min_speech_samples,
            state.utterance_rms_threshold,
        )

        has_speech = (
            self._min_speech_samples == 0 or state.speech_samples >= self._min_speech_samples
        )
        if not has_speech:
            log.info(
                "Ignoring silent utterance (peak_rms=%.4f, speech_samples=%d/%d)",
                state.peak_rms,
                state.speech_samples,
                self._min_speech_samples,
            )
            if self.overlay:
                self.overlay.hide()
            state.reset(rms_threshold=self._speech_rms_threshold)
            self.typer.reset()
            return

        if self._is_offline_instant_mode:
            self._decode_offline_utterance(state)
            self._commit_utterance(state)

            if self.overlay:
                self.overlay.hide()
            state.reset(rms_threshold=self._speech_rms_threshold)
            self.typer.reset()
            return

        while state.total >= self.asr.native_chunk_samples and not self._asr_disabled:
            has_more = self._transcribe_native_chunk(state, "ASR buffered final chunk failed")
            if not has_more:
                break

        if state.total > 0 and not self._asr_disabled:
            audio_data = state.buffer[0] if len(state.buffer) == 1 else np.concatenate(state.buffer)
            padded = np.zeros(self.asr.native_chunk_samples, dtype=np.float32)
            padded[: len(audio_data)] = audio_data
            if not self.asr.wants_raw_audio and state.utterance_gain > 1.05:
                padded[: len(audio_data)] = self._apply_utterance_gain(
                    padded[: len(audio_data)],
                    state.utterance_gain,
                )

            try:
                text = self._process_chunk_safe(padded)
            except Exception:
                self._recover_asr_after_failure("ASR final flush failed")
            else:
                state.last_text = prefer_transcript(state.last_text, text)

        self._flush_tail_silence(state)

        log.info(
            "Post-processing: last_text_len=%d total_remaining=%d step_num=%s",
            len(state.last_text),
            state.total,
            self.asr.debug_step_num,
        )

        self._commit_utterance(state)

        if self.overlay:
            self.overlay.hide()
        state.reset(rms_threshold=self._speech_rms_threshold)
        self.typer.reset()

    # -- ASR processing thread ----------------------------------------------

    def _asr_loop(self):
        state = _UtteranceState()
        state.reset(rms_threshold=self._speech_rms_threshold)
        was_recording = False

        while self._running.is_set():
            chunk = self.audio.get_chunk(timeout=0.05)
            is_recording = self._recording.is_set()

            if is_recording and not was_recording:
                self._begin_utterance(state)

            if chunk is not None:
                if is_recording:
                    self._append_recording_chunk(state, chunk)
                else:
                    self._update_noise_floor(audio_rms(chunk))

            if is_recording and not self._asr_disabled and not self._is_offline_instant_mode:
                self._process_recording_chunks(state)

            if was_recording and not is_recording:
                try:
                    self._handle_recording_stop(state)
                finally:
                    self._processing.clear()

            self._log_metrics_if_due()
            was_recording = is_recording

    def _asr_worker(self):
        try:
            self._asr_loop()
        except Exception:
            self._asr_thread_alive = False
            log.critical("ASR worker crashed", exc_info=True)
            self._recording.clear()
            self._show_overlay_error("⚠ ASR thread crashed — restart ShuVoice")
        else:
            if self._running.is_set():
                self._asr_thread_alive = False
                log.critical("ASR worker exited unexpectedly")
                self._show_overlay_error("⚠ ASR thread crashed — restart ShuVoice")
