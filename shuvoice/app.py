"""Main ShuVoice application — ties all components together.

IMPORTANT: ctypes.CDLL('libgtk4-layer-shell.so') must be called
before this module is imported. See __main__.py.
"""

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
from .hotkey import HotkeyListener
from .overlay import CaptionOverlay
from .typer import StreamingTyper

log = logging.getLogger(__name__)


class ShuVoiceApp(Gtk.Application):
    """GTK4 application that orchestrates streaming speech-to-text.

    Threading model:
      - Main thread:   GTK4 event loop (Gtk.Application.run)
      - Audio thread:  managed by sounddevice (callback puts chunks in a queue)
      - ASR thread:    consumes audio queue, runs inference, updates overlay
      - Hotkey thread: asyncio event loop reading evdev
    """

    def __init__(self, config: Config):
        super().__init__(application_id="dev.shuv.shuvoice")
        self.config = config

        # Components (created here, started in do_activate)
        self.asr = ASREngine(config.model_name, config.right_context, config.device)
        self.audio = AudioCapture(
            config.sample_rate, config.chunk_samples, config.fallback_sample_rate
        )
        self.typer = StreamingTyper()
        self.hotkey = HotkeyListener(config.hotkey, config.hold_threshold_ms)

        self.overlay: CaptionOverlay | None = None

        # Shared state
        self._recording = threading.Event()
        self._running = threading.Event()
        self._running.set()

    def load_model(self):
        """Load the ASR model. Call before run() — this blocks."""
        self.asr.load()

    # -- GTK lifecycle ------------------------------------------------------

    def do_activate(self):
        self.overlay = CaptionOverlay(self, self.config)

        # Start audio capture (runs continuously, chunks go to queue)
        self.audio.start()

        # Configure hotkey callbacks
        self.hotkey.set_callbacks(
            on_start=self._on_recording_start,
            on_stop=self._on_recording_stop,
        )

        # Start background threads
        threading.Thread(target=self._asr_loop, name="asr", daemon=True).start()
        threading.Thread(target=self._hotkey_loop, name="hotkey", daemon=True).start()

        # Handle SIGINT gracefully inside GTK main loop
        GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGINT, self._on_sigint)

        log.info("Ready — press %s to talk", self.config.hotkey)

    def do_shutdown(self):
        log.info("Shutting down…")
        self._running.clear()
        self._recording.clear()
        self.audio.stop()
        Gtk.Application.do_shutdown(self)

    def _on_sigint(self):
        log.info("SIGINT received, quitting")
        self.quit()
        return GLib.SOURCE_REMOVE

    # -- Recording state (called from hotkey thread) ------------------------

    def _on_recording_start(self):
        log.info("Recording started")
        self.asr.reset()
        self.audio.clear()
        self._recording.set()
        if self.overlay:
            self.overlay.show()
            self.overlay.set_text("Listening…")

    def _on_recording_stop(self):
        log.info("Recording stopped")
        self._recording.clear()

    # -- ASR processing thread ----------------------------------------------

    def _asr_loop(self):
        buffer: list[np.ndarray] = []
        total = 0
        last_text = ""
        was_recording = False
        native = self.config.native_chunk_samples

        while self._running.is_set():
            chunk = self.audio.get_chunk(timeout=0.05)
            is_recording = self._recording.is_set()

            # Accumulate audio while recording
            if chunk is not None and is_recording:
                buffer.append(chunk)
                total += len(chunk)

            # Process complete 1120 ms chunks
            while is_recording and total >= native:
                audio_data = np.concatenate(buffer)
                to_process = audio_data[:native]
                remainder = audio_data[native:]
                buffer = [remainder] if len(remainder) > 0 else []
                total = len(remainder)

                text = self.asr.process_chunk(to_process)
                if text and text != last_text:
                    last_text = text
                    if self.overlay:
                        self.overlay.set_text(text)

            # Detect recording→stopped transition
            if was_recording and not is_recording:
                # Process any remaining audio (pad with silence)
                if total > 0:
                    audio_data = np.concatenate(buffer)
                    padded = np.zeros(native, dtype=np.float32)
                    padded[: len(audio_data)] = audio_data
                    text = self.asr.process_chunk(padded)
                    if text:
                        last_text = text

                # Commit final text
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

            was_recording = is_recording

    # -- Hotkey event loop thread -------------------------------------------

    def _hotkey_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.hotkey.run())
        except Exception:
            log.exception("Hotkey listener error")
