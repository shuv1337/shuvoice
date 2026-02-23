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
from .feedback import play_tone
from .postprocess import apply_text_replacements, capitalize_first
from .streaming_health import should_trigger_stall_flush
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
            preserve_clipboard=config.preserve_clipboard,
            retry_attempts=config.typing_retry_attempts,
            retry_delay_ms=config.typing_retry_delay_ms,
        )

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

    def _load_model_async(self):
        """Background thread: load the ASR model, then signal the main thread."""
        try:
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

        log.info("Ready — use Hyprland bind/bindr with shuvoice --control start/stop")
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
        if self._recording.is_set():
            log.debug("Recording already active; ignoring start")
            return

        if not self._asr_thread_alive:
            self._show_overlay_error("⚠ ASR thread crashed — restart ShuVoice")
            return

        with self._asr_lock:
            if self._asr_disabled:
                log.warning("ASR disabled; attempting one-shot reset on recording start")
                try:
                    self.asr.reset()
                except Exception:
                    log.exception("ASR recovery reset failed; still disabled")
                    self._show_overlay_error("⚠ ASR error — restart ShuVoice")
                    return
                self._asr_disabled = False
                self._consecutive_asr_failures = 0

            self.audio.clear()

            try:
                self.asr.reset()
            except Exception:
                self._consecutive_asr_failures += 1
                failures = self._consecutive_asr_failures
                log.exception(
                    "ASR reset failed on recording start (%d/%d)",
                    failures,
                    self._ASR_MAX_FAILURES,
                )
                if failures >= self._ASR_MAX_FAILURES:
                    self._disable_asr("ASR disabled after repeated reset failures")
                else:
                    self._show_overlay_error("⚠ ASR error — restart ShuVoice")
                return

            self.audio.clear()

        self._processing.clear()
        self._recording.set()
        log.info("Recording started")

        if self.overlay:
            self.overlay.show()
            self.overlay.set_state("listening")
            self.overlay.set_text("Listening…")

        self._play_feedback_tone(is_start=True)

    def _on_recording_stop(self):
        if not self._recording.is_set():
            log.debug("Recording already stopped; ignoring stop")
            return

        log.info("Recording stopped")
        self._recording.clear()
        self._processing.set()
        self._play_feedback_tone(is_start=False)

        if self.overlay:
            self.overlay.set_state("processing")

    def _on_recording_toggle(self):
        if self._recording.is_set():
            self._on_recording_stop()
        else:
            self._on_recording_start()

    def _recording_status(self) -> str:
        if self._asr_disabled:
            return "error:asr_disabled"
        if not self._asr_thread_alive:
            return "error:asr_thread_dead"
        if self._recording.is_set():
            return "recording"
        if self._processing.is_set():
            return "processing"
        return "idle"

    def _render_transcript_text(self, text: str) -> str:
        """Render transcript text for preview/final output consistency."""
        if not text:
            return text

        rendered = apply_text_replacements(text, self.config.text_replacements)
        if not rendered:
            return rendered

        if self.config.auto_capitalize:
            rendered = capitalize_first(rendered)

        return rendered

    # -- ASR processing helpers ---------------------------------------------

    def _apply_utterance_gain(self, audio: np.ndarray, gain: float) -> np.ndarray:
        if gain <= 1.05 or audio.size == 0:
            return audio
        return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

    def _update_noise_floor(self, chunk_rms: float):
        if chunk_rms <= 0.0:
            return
        if self._noise_floor_rms <= 0.0:
            self._noise_floor_rms = chunk_rms
        else:
            self._noise_floor_rms = 0.98 * self._noise_floor_rms + 0.02 * chunk_rms

    def _begin_utterance(self, state: _UtteranceState):
        # Defensive reset: ensure ASR model is clean after any potential
        # contamination from tail flush race conditions.
        with self._asr_lock:
            if not self._asr_disabled:
                try:
                    self.asr.reset()
                except Exception:
                    log.exception("ASR reset failed at utterance start")
                    self._recover_asr_after_failure("ASR reset at utterance start")

        threshold = max(
            self._speech_rms_threshold,
            self._noise_floor_rms * self._speech_rms_multiplier,
        )
        state.reset(rms_threshold=threshold)
        log.debug(
            "Recording energy threshold: %.4f (noise_floor=%.4f, floor=%.4f, x%.2f)",
            threshold,
            self._noise_floor_rms,
            self._speech_rms_threshold,
            self._speech_rms_multiplier,
        )

    def _append_recording_chunk(self, state: _UtteranceState, chunk: np.ndarray):
        state.add_chunk(chunk)
        chunk_rms = audio_rms(chunk)
        state.last_chunk_rms = chunk_rms
        state.peak_rms = max(state.peak_rms, chunk_rms)

        if chunk_rms >= state.utterance_rms_threshold:
            state.speech_samples += len(chunk)
            state.speech_chunks_seen += 1

        # Backends with internal normalization (e.g. NeMo/Moonshine) bypass
        # app-side gain entirely.
        if self.asr.wants_raw_audio:
            return

        if state.speech_chunks_seen < self._auto_gain_settle_chunks:
            return

        if state.peak_rms > 0.003:
            state.utterance_gain = min(
                self._auto_gain_target_peak / state.peak_rms,
                self._auto_gain_max,
            )

    def _transcribe_native_chunk(self, state: _UtteranceState, error_context: str) -> bool:
        to_process, has_more = state.consume_native_chunk(self.asr.native_chunk_samples)
        if to_process.size == 0:
            return False

        if not self.asr.wants_raw_audio:
            to_process = self._apply_utterance_gain(to_process, state.utterance_gain)

        try:
            text = self._process_chunk_safe(to_process)
        except Exception:
            self._recover_asr_after_failure(error_context)
            return False

        log.debug(
            "ASR step=%s queue_size=%d raw_text=%r chunk_rms=%.4f gain=%.1f",
            self.asr.debug_step_num,
            self.audio.queue.qsize(),
            text,
            audio_rms(to_process),
            state.utterance_gain,
        )

        merged = prefer_transcript(state.last_text, text)
        if merged != state.last_text:
            log.debug("Transcript updated: %r -> %r", state.last_text, merged)
            state.last_text = merged
            state.unchanged_steps = 0
            rendered_text = self._render_transcript_text(state.last_text)
            if self.overlay:
                self.overlay.set_text(rendered_text)
            if self.config.output_mode == "streaming_partial":
                self.typer.update_partial(rendered_text)
        else:
            state.unchanged_steps += 1

        return has_more

    def _flush_streaming_stall(self, state: _UtteranceState):
        if self._asr_disabled:
            return

        native = self.asr.native_chunk_samples
        # Intentionally feed raw digital silence here; this path is a short
        # stall nudge and should not apply utterance gain.
        silence = np.zeros(native, dtype=np.float32)

        for _ in range(self._streaming_stall_flush_chunks):
            try:
                text = self._process_chunk_safe(silence)
            except Exception:
                self._recover_asr_after_failure("ASR stall-guard flush failed")
                break

            merged = prefer_transcript(state.last_text, text)
            if merged != state.last_text:
                log.debug("Transcript updated after stall flush: %r -> %r", state.last_text, merged)
                state.last_text = merged
                rendered_text = self._render_transcript_text(state.last_text)
                if self.overlay:
                    self.overlay.set_text(rendered_text)
                if self.config.output_mode == "streaming_partial":
                    self.typer.update_partial(rendered_text)

        state.unchanged_steps = 0

    def _process_recording_chunks(self, state: _UtteranceState):
        while (
            self._recording.is_set()
            and not self._asr_disabled
            and state.total >= self.asr.native_chunk_samples
        ):
            has_more = self._transcribe_native_chunk(state, "ASR chunk processing failed")
            if not has_more:
                break

            if self._streaming_stall_guard and should_trigger_stall_flush(
                unchanged_steps=state.unchanged_steps,
                chunk_rms=state.last_chunk_rms,
                utterance_threshold=state.utterance_rms_threshold,
                stall_chunks=self._streaming_stall_chunks,
                stall_rms_ratio=self._streaming_stall_rms_ratio,
            ):
                log.debug(
                    "Streaming stall guard triggered (unchanged_steps=%d, chunk_rms=%.4f, threshold=%.4f)",
                    state.unchanged_steps,
                    state.last_chunk_rms,
                    state.utterance_rms_threshold,
                )
                self._flush_streaming_stall(state)

    def _drain_and_buffer(self, state: _UtteranceState):
        drained = self.audio.drain_pending_chunks()
        if drained:
            for drained_chunk in drained:
                self._append_recording_chunk(state, drained_chunk)
            log.debug(
                "Drained %d queued audio chunk(s) on stop (%d samples buffered)",
                len(drained),
                state.total,
            )

        tail_chunk = self.audio.get_chunk(timeout=0.12)
        if tail_chunk is not None:
            self._append_recording_chunk(state, tail_chunk)

            drained_after_tail = self.audio.drain_pending_chunks()
            if drained_after_tail:
                for drained_chunk in drained_after_tail:
                    self._append_recording_chunk(state, drained_chunk)
                log.debug(
                    "Drained %d queued audio chunk(s) after stop grace (%d samples buffered)",
                    len(drained_after_tail),
                    state.total,
                )

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
        """Generate low-amplitude noise for flushing streaming transducers.

        Streaming transducers (e.g. Sherpa) may not flush buffered hypotheses
        when fed perfect digital silence.  Ambient-level noise better
        simulates the end-of-speech condition the model was trained on.

        The ``escalation`` multiplier (≥1.0) is applied on top of the base
        RMS so that stalled flush steps progressively increase amplitude,
        ensuring even very quiet environments eventually trigger emission.
        """
        base_rms = max(self._noise_floor_rms, self._FLUSH_NOISE_MIN_RMS)
        rms = min(base_rms * escalation, self._FLUSH_NOISE_MAX_RMS)
        noise = np.random.default_rng().normal(0.0, rms, size=n_samples).astype(np.float32)
        return np.clip(noise, -1.0, 1.0)

    def _flush_tail_silence(self, state: _UtteranceState):
        if self._asr_disabled:
            return

        native = self.asr.native_chunk_samples
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
            # Abort tail flush if a new recording has started to avoid
            # contaminating the fresh ASR state with noise.
            if self._recording.is_set():
                log.debug("Aborting tail flush: new recording started")
                break

            # Use ambient-level noise at the same gain scale the model has
            # been seeing, so streaming transducers recognise end-of-speech.
            # Escalate amplitude on stalled steps so even very quiet
            # environments eventually produce enough energy to trigger
            # token emission.
            escalation = self._FLUSH_NOISE_ESCALATION**stalled_consecutive
            flush_audio = self._make_flush_noise(native, escalation=escalation)
            if not self.asr.wants_raw_audio:
                flush_audio = self._apply_utterance_gain(flush_audio, state.utterance_gain)

            try:
                text = self._process_chunk_safe(flush_audio)
            except Exception:
                self._recover_asr_after_failure("ASR tail flush failed")
                break

            merged = prefer_transcript(state.last_text, text)
            if merged != state.last_text:
                log.debug(
                    "Tail flush step %d: %r -> %r (escalation=%.2f)",
                    i,
                    state.last_text,
                    merged,
                    escalation,
                )
                state.last_text = merged
                stable_steps = 0
                stalled_consecutive = 0
                ever_had_text = True
                if self.overlay:
                    self.overlay.set_text(self._render_transcript_text(state.last_text))
            else:
                stable_steps += 1
                stalled_consecutive += 1
                # Once we've seen text, converge quickly; otherwise keep
                # flushing longer in case the model is still buffering.
                needed = stable_required if ever_had_text else 5
                if stable_steps >= needed:
                    break

    def _commit_utterance(self, state: _UtteranceState):
        final_text = state.last_text.strip()
        if not final_text:
            return

        final_text = self._render_transcript_text(final_text)
        if not final_text:
            return

        log.info("Final: %s", final_text)
        if self.overlay:
            self.overlay.set_text(final_text)

        if self.config.use_clipboard_for_final:
            self.typer.commit_final(final_text)
        else:
            self.typer.update_partial(final_text)
            self.typer.reset()

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
            "Post-processing: last_text=%r total_remaining=%d step_num=%s",
            state.last_text,
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

            if is_recording and not self._asr_disabled:
                self._process_recording_chunks(state)

            if was_recording and not is_recording:
                try:
                    self._handle_recording_stop(state)
                finally:
                    self._processing.clear()

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

