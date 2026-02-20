"""Main ShuVoice application — ties all components together.

IMPORTANT: ctypes.CDLL('libgtk4-layer-shell.so') must be called
before this module is imported. See __main__.py.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import threading

import numpy as np

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from .asr import ASREngine
from .audio import AudioCapture
from .config import Config
from .control import ControlServer
from .hotkey import HotkeyListener
from .overlay import CaptionOverlay
from .typer import StreamingTyper

log = logging.getLogger(__name__)


def _prefer_transcript(previous: str, candidate: str) -> str:
    """Prefer stable cumulative transcript growth over regressions.

    Streaming RNNT hypotheses can occasionally jump backward (shorter text) between
    steps. Keep the older hypothesis unless the new one clearly extends it.
    """
    if not candidate:
        return previous
    if not previous:
        return candidate

    prev = previous.strip()
    new = candidate.strip()
    if not new:
        return previous

    # Normal growth path: cumulative hypothesis extends prior text.
    if new.startswith(prev):
        return candidate

    # Decoder regressed to a shorter prefix — keep the fuller prior text.
    if prev.startswith(new):
        return previous

    # Streaming models with finite context windows can drop the prefix of a
    # long utterance. Try to stitch the candidate to the previous text if there
    # is a significant overlap (at least 5 chars to avoid false positives).
    min_len = min(len(prev), len(new))
    for i in range(min_len, 5, -1):
        if prev.endswith(new[:i]):
            stitched = prev + new[i:]
            return stitched

    # The model frequently rewrites earlier words as it gains context
    # (e.g. "Quick brown" -> "The quick brown dog jumped over").
    # Accept the new hypothesis if it carries more content.
    return candidate if len(new) >= len(prev) else previous


class ShuVoiceApp(Gtk.Application):
    """GTK4 application that orchestrates streaming speech-to-text.

    Threading model:
      - Main thread:    GTK4 event loop (Gtk.Application.run)
      - Audio thread:   managed by sounddevice callback
      - ASR thread:     consumes audio queue, runs inference, updates overlay
      - Hotkey thread:  optional asyncio event loop reading evdev
      - Control thread: local Unix socket server for IPC start/stop/toggle
    """

    def __init__(self, config: Config):
        super().__init__(application_id="dev.shuv.shuvoice")
        self.config = config

        # Components (created here, started in do_activate)
        self.asr = ASREngine(
            config.model_name,
            config.right_context,
            config.device,
            config.use_cuda_graph_decoder,
        )
        self.audio = AudioCapture(
            config.sample_rate,
            config.chunk_samples,
            config.fallback_sample_rate,
            device=config.audio_device,
            input_gain=config.input_gain,
        )
        self.typer = StreamingTyper(
            preserve_clipboard=config.preserve_clipboard,
            retry_attempts=config.typing_retry_attempts,
            retry_delay_ms=config.typing_retry_delay_ms,
        )

        if config.hotkey_backend not in {"evdev", "ipc"}:
            raise ValueError(
                f"Invalid hotkey_backend '{config.hotkey_backend}'. "
                "Expected one of: evdev, ipc"
            )
        if config.output_mode not in {"final_only", "streaming_partial"}:
            raise ValueError(
                f"Invalid output_mode '{config.output_mode}'. "
                "Expected one of: final_only, streaming_partial"
            )

        self.hotkey: HotkeyListener | None = None
        if config.hotkey_backend == "evdev":
            self.hotkey = HotkeyListener(
                config.hotkey,
                config.hold_threshold_ms,
                config.hotkey_device,
                listen_all_devices=config.hotkey_listen_all_devices,
            )

        self.control = ControlServer(
            socket_path=config.control_socket,
            on_start=self._on_recording_start,
            on_stop=self._on_recording_stop,
            on_toggle=self._on_recording_toggle,
            on_status=self._recording_status,
        )

        self.overlay: CaptionOverlay | None = None

        # Shared state
        self._recording = threading.Event()
        self._running = threading.Event()
        self._running.set()

        # Synchronize ASR state transitions across hotkey and ASR threads.
        self._asr_lock = threading.Lock()

    def load_model(self):
        """Load the ASR model. Call before run() — this blocks."""
        self.asr.load()

    # -- GTK lifecycle ------------------------------------------------------

    def do_activate(self):
        self.overlay = CaptionOverlay(self, self.config)

        # Start audio capture (runs continuously, chunks go to queue)
        self.audio.start()

        # Always expose local control socket (for Hyprland bind/bindr fallback)
        self.control.start()

        # Configure optional evdev hotkey callbacks
        if self.hotkey:
            self.hotkey.set_callbacks(
                on_start=self._on_recording_start,
                on_stop=self._on_recording_stop,
            )

        # Start background workers
        threading.Thread(target=self._asr_loop, name="asr", daemon=True).start()
        if self.hotkey:
            threading.Thread(target=self._hotkey_loop, name="hotkey", daemon=True).start()

        # Handle SIGINT gracefully inside GTK main loop
        GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGINT, self._on_sigint)

        if self.hotkey:
            log.info("Ready — press %s to talk", self.config.hotkey)
        else:
            log.info(
                "Ready — hotkey backend=%s. Use control socket commands instead.",
                self.config.hotkey_backend,
            )
        log.info("Control socket: %s", self.control.socket_path)

    def do_shutdown(self):
        log.info("Shutting down…")
        self._running.clear()
        self._recording.clear()
        self.control.stop()
        self.audio.stop()
        Gtk.Application.do_shutdown(self)

    def _on_sigint(self):
        log.info("SIGINT received, quitting")
        self.quit()
        return GLib.SOURCE_REMOVE

    # -- Recording state (called from hotkey/control threads) ---------------

    def _on_recording_start(self):
        if self._recording.is_set():
            log.debug("Recording already active; ignoring start")
            return

        log.info("Recording started")

        self.audio.clear()
        
        # Taking the lock to reset ASR prevents a race condition
        # where the ASR thread is still flushing the last chunk
        # of the *previous* utterance.
        with self._asr_lock:
            self.asr.reset()
            
        self._recording.set()

        if self.overlay:
            self.overlay.show()
            self.overlay.set_text("Listening…")

    def _on_recording_stop(self):
        if not self._recording.is_set():
            log.debug("Recording already stopped; ignoring stop")
            return

        log.info("Recording stopped")
        self._recording.clear()

    def _on_recording_toggle(self):
        if self._recording.is_set():
            self._on_recording_stop()
        else:
            self._on_recording_start()

    def _recording_status(self) -> str:
        return "recording" if self._recording.is_set() else "idle"

    def _process_chunk_safe(self, audio_data: np.ndarray) -> str:
        """Serialize access to mutable ASR streaming state."""
        with self._asr_lock:
            return self.asr.process_chunk(audio_data)

    # -- ASR processing thread ----------------------------------------------

    def _asr_loop(self):
        buffer: list[np.ndarray] = []
        total = 0
        last_text = ""
        was_recording = False
        native = self.config.native_chunk_samples

        speech_rms_threshold = max(0.0, float(self.config.silence_rms_threshold))
        speech_rms_multiplier = max(1.0, float(self.config.silence_rms_multiplier))
        min_speech_samples = max(
            0,
            self.config.sample_rate * max(0, int(self.config.min_speech_ms)) // 1000,
        )
        speech_samples = 0
        peak_rms = 0.0
        noise_floor_rms = 0.0
        utterance_gain = 1.0
        utterance_rms_threshold = speech_rms_threshold

        while self._running.is_set():
            chunk = self.audio.get_chunk(timeout=0.05)
            is_recording = self._recording.is_set()

            if is_recording and not was_recording:
                speech_samples = 0
                peak_rms = 0.0
                utterance_gain = 1.0
                utterance_rms_threshold = max(
                    speech_rms_threshold,
                    noise_floor_rms * speech_rms_multiplier,
                )
                log.debug(
                    "Recording energy threshold: %.4f (noise_floor=%.4f, floor=%.4f, x%.2f)",
                    utterance_rms_threshold,
                    noise_floor_rms,
                    speech_rms_threshold,
                    speech_rms_multiplier,
                )

            # Accumulate audio while recording
            if chunk is not None:
                if is_recording:
                    buffer.append(chunk)
                    total += len(chunk)

                    if chunk.size:
                        rms = float(np.sqrt(np.mean(chunk * chunk)))
                        peak_rms = max(peak_rms, rms)
                        if rms >= utterance_rms_threshold:
                            speech_samples += len(chunk)
                        # Compute a single gain factor for the utterance based on the
                        # loudest speech seen so far. This brings quiet mic input up
                        # to a level NeMo was trained on (~0.1-0.3 RMS) while keeping
                        # one gain for ALL chunks (preserving relative dynamics).
                        if peak_rms > 0.003:
                            # Empirically this model performs much better with
                            # stronger vocal peaks (~0.25-0.35).
                            target_peak = 0.3
                            utterance_gain = min(target_peak / peak_rms, 40.0)
                elif chunk.size:
                    rms = float(np.sqrt(np.mean(chunk * chunk)))
                    if noise_floor_rms <= 0.0:
                        noise_floor_rms = rms
                    else:
                        # Track ambient level slowly while idle.
                        noise_floor_rms = 0.98 * noise_floor_rms + 0.02 * rms

            # Process complete 1120 ms chunks
            while is_recording and total >= native:
                audio_data = np.concatenate(buffer)
                to_process = audio_data[:native]
                remainder = audio_data[native:]
                buffer = [remainder] if len(remainder) > 0 else []
                total = len(remainder)

                # Apply utterance-level gain (uniform across all chunks to
                # preserve relative dynamics — unlike per-chunk normalization
                # which destroys the volume envelope the model relies on).
                if utterance_gain > 1.05:
                    to_process = np.clip(
                        to_process * utterance_gain, -1.0, 1.0
                    ).astype(np.float32)

                try:
                    text = self._process_chunk_safe(to_process)
                except Exception:
                    log.exception("ASR chunk processing failed")
                    text = ""
                    with self._asr_lock:
                        self.asr.reset()
                    break

                log.debug(
                    "ASR step=%d queue_size=%d raw_text=%r chunk_rms=%.4f gain=%.1f",
                    self.asr._step_num,
                    self.audio.queue.qsize(),
                    text,
                    float(np.sqrt(np.mean(to_process * to_process))) if to_process.size else 0.0,
                    utterance_gain,
                )

                # When streaming, the model can sometimes output trailing whitespace
                # which causes our string match to think it regressed.
                merged = _prefer_transcript(last_text, text)
                if merged != last_text:
                    log.debug("Transcript updated: %r -> %r", last_text, merged)
                    last_text = merged
                    if self.overlay:
                        self.overlay.set_text(last_text)
                    if self.config.output_mode == "streaming_partial":
                        self.typer.update_partial(last_text)

            # Detect recording→stopped transition
            if was_recording and not is_recording:
                # Drain queued audio captured just before stop to avoid losing
                # late words when ASR is slightly behind real-time.
                drained = self.audio.drain_pending_chunks()
                if drained:
                    buffer.extend(drained)
                    total += sum(len(chunk) for chunk in drained)
                    for drained_chunk in drained:
                        if drained_chunk.size:
                            rms = float(np.sqrt(np.mean(drained_chunk * drained_chunk)))
                            peak_rms = max(peak_rms, rms)
                            if rms >= utterance_rms_threshold:
                                speech_samples += len(drained_chunk)
                    log.debug(
                        "Drained %d queued audio chunk(s) on stop (%d samples buffered)",
                        len(drained),
                        total,
                    )

                # Small grace window for in-flight callback audio that can
                # arrive right after key release.
                tail_chunk = self.audio.get_chunk(timeout=0.12)
                if tail_chunk is not None:
                    buffer.append(tail_chunk)
                    total += len(tail_chunk)
                    if tail_chunk.size:
                        rms = float(np.sqrt(np.mean(tail_chunk * tail_chunk)))
                        peak_rms = max(peak_rms, rms)
                        if rms >= utterance_rms_threshold:
                            speech_samples += len(tail_chunk)

                    drained = self.audio.drain_pending_chunks()
                    if drained:
                        buffer.extend(drained)
                        total += sum(len(chunk) for chunk in drained)
                        for drained_chunk in drained:
                            if drained_chunk.size:
                                rms = float(np.sqrt(np.mean(drained_chunk * drained_chunk)))
                                peak_rms = max(peak_rms, rms)
                                if rms >= utterance_rms_threshold:
                                    speech_samples += len(drained_chunk)
                        log.debug(
                            "Drained %d queued audio chunk(s) after stop grace (%d samples buffered)",
                            len(drained),
                            total,
                        )

                log.debug(
                    "Utterance energy: peak_rms=%.4f speech_samples=%d/%d threshold=%.4f",
                    peak_rms,
                    speech_samples,
                    min_speech_samples,
                    utterance_rms_threshold,
                )

                has_speech = min_speech_samples == 0 or speech_samples >= min_speech_samples
                if not has_speech:
                    log.info(
                        "Ignoring silent utterance (peak_rms=%.4f, speech_samples=%d/%d)",
                        peak_rms,
                        speech_samples,
                        min_speech_samples,
                    )
                    if self.overlay:
                        self.overlay.hide()
                    buffer.clear()
                    total = 0
                    last_text = ""
                    self.typer.reset()
                    speech_samples = 0
                    peak_rms = 0.0
                    utterance_gain = 1.0
                    utterance_rms_threshold = speech_rms_threshold
                    was_recording = is_recording
                    continue

                # Process any complete buffered 1120 ms chunks.
                while total >= native:
                    audio_data = np.concatenate(buffer)
                    to_process = audio_data[:native]
                    remainder = audio_data[native:]
                    buffer = [remainder] if len(remainder) > 0 else []
                    total = len(remainder)

                    if utterance_gain > 1.05:
                        to_process = np.clip(
                            to_process * utterance_gain, -1.0, 1.0
                        ).astype(np.float32)

                    try:
                        text = self._process_chunk_safe(to_process)
                    except Exception:
                        log.exception("ASR buffered final chunk failed")
                        text = ""
                        # Re-initialize ASR engine cache if crashed so it recovers for the next phrase
                        with self._asr_lock:
                            self.asr.reset()
                        break

                    last_text = _prefer_transcript(last_text, text)

                # Process any remaining audio (pad with silence)
                if total > 0:
                    audio_data = np.concatenate(buffer)
                    padded = np.zeros(native, dtype=np.float32)
                    padded[: len(audio_data)] = audio_data
                    if utterance_gain > 1.05:
                        # Only amplify the real audio portion, not the silence padding
                        padded[:len(audio_data)] = np.clip(
                            padded[:len(audio_data)] * utterance_gain, -1.0, 1.0
                        ).astype(np.float32)
                    try:
                        text = self._process_chunk_safe(padded)
                    except Exception:
                        log.exception("ASR final flush failed")
                        text = ""
                        with self._asr_lock:
                            self.asr.reset()
                    last_text = _prefer_transcript(last_text, text)

                # Flush a couple extra silent chunks to release decoder tail
                # tokens (helps avoid clipped last words on stop).
                silence = np.zeros(native, dtype=np.float32)
                stable_steps = 0
                for _ in range(5):
                    try:
                        text = self._process_chunk_safe(silence)
                    except Exception:
                        log.exception("ASR tail flush failed")
                        with self._asr_lock:
                            self.asr.reset()
                        break

                    merged = _prefer_transcript(last_text, text)
                    if merged == last_text:
                        stable_steps += 1
                        if stable_steps >= 2:
                            break
                    else:
                        stable_steps = 0
                        last_text = merged

                # Commit final text
                log.info(
                    "Post-processing: last_text=%r total_remaining=%d step_num=%d",
                    last_text,
                    total,
                    self.asr._step_num,
                )
                if last_text:
                    log.info("Final: %s", last_text)
                    if self.overlay:
                        self.overlay.set_text(last_text)
                    if self.config.use_clipboard_for_final:
                        self.typer.commit_final(last_text)
                    else:
                        self.typer.update_partial(last_text)
                        self.typer.reset()

                # Clean up for next utterance
                if self.overlay:
                    self.overlay.hide()
                buffer.clear()
                total = 0
                last_text = ""
                self.typer.reset()
                speech_samples = 0
                peak_rms = 0.0
                utterance_gain = 1.0
                utterance_rms_threshold = speech_rms_threshold

            was_recording = is_recording

    # -- Hotkey event loop thread -------------------------------------------

    def _hotkey_loop(self):
        if not self.hotkey:
            return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.hotkey.run())
        except Exception:
            log.exception("Hotkey listener error")
