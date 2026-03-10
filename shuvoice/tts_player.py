"""Streaming TTS playback state machine."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from queue import Empty, Full, Queue
from typing import Any

import numpy as np
import sounddevice as sd

from .tts_base import TTSBackend, TTSSpeedApplyError, TTSSynthesisRequest
from .tts_speed import normalize_tts_playback_speed

log = logging.getLogger(__name__)


class TTSPlayer:
    """Thread-safe TTS synthesis + playback coordinator.

    The player is intentionally transport-only: it snapshots the selected speed
    into a synthesis request and plays backend PCM exactly as produced. Speed
    changes must be applied provider-side or not at all.
    """

    ACTIVE_STATES = {"synthesizing", "playing", "paused"}

    def __init__(
        self,
        backend: TTSBackend,
        *,
        output_device: str | int | None = None,
        sample_rate: int | None = None,
        output_format: str = "pcm_24000",
        playback_speed: float = 1.0,
        on_state_change: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self._backend = backend
        self._output_device = output_device
        self._sample_rate = int(sample_rate) if sample_rate is not None else int(
            getattr(backend, "sample_rate_hz", lambda: self._parse_sample_rate(output_format))()
        )
        self._on_state_change = on_state_change

        self._lock = threading.RLock()
        self._state = "idle"
        self._generation = 0
        self._playback_speed = normalize_tts_playback_speed(playback_speed)

        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()

        self._queue: Queue[bytes | None] = Queue(maxsize=256)
        self._synth_thread: threading.Thread | None = None
        self._play_thread: threading.Thread | None = None
        self._stream: sd.OutputStream | None = None

        self._last_request: TTSSynthesisRequest | None = None
        self._active_request: TTSSynthesisRequest | None = None
        self._synth_started_at: float | None = None
        self._play_started_at: float | None = None

    @staticmethod
    def _parse_sample_rate(output_format: str) -> int:
        # Raw PCM format style: "pcm_24000"
        text = str(output_format).strip().lower()
        if text.startswith("pcm_"):
            maybe_rate = text.split("_", 1)[1]
            if maybe_rate.isdigit() and int(maybe_rate) > 0:
                return int(maybe_rate)
        return 24000

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def playback_speed(self) -> float:
        with self._lock:
            return self._playback_speed

    def set_playback_speed(self, speed: float) -> float:
        normalized = normalize_tts_playback_speed(speed)
        with self._lock:
            self._playback_speed = normalized
        return normalized

    def is_active(self) -> bool:
        with self._lock:
            return self._state in self.ACTIVE_STATES

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            last_request = self._last_request
            active_request = self._active_request
            return {
                "state": self._state,
                "voice_id": last_request.voice_id if last_request else "",
                "model_id": last_request.model_id if last_request else "",
                "text_len": len(last_request.text) if last_request else 0,
                "playback_speed": self._playback_speed,
                "selected_playback_speed": self._playback_speed,
                "active_request_speed": (
                    active_request.playback_speed if active_request is not None else None
                ),
            }

    def _transition(self, state: str, **info: Any) -> None:
        with self._lock:
            changed = state != self._state
            self._state = state

        if not changed and not info:
            return

        callback = self._on_state_change
        if callback is None:
            return

        try:
            callback(state, dict(info))
        except Exception:  # noqa: BLE001
            log.debug("TTS state callback failed", exc_info=True)

    def _current_generation(self) -> int:
        with self._lock:
            return self._generation

    def _is_generation_current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation

    def _clear_active_request(self, generation: int) -> None:
        with self._lock:
            if generation == self._generation:
                self._active_request = None

    def speak(self, text: str, voice_id: str, model_id: str) -> bool:
        """Start speaking text. Returns True when an active session was interrupted."""
        text_value = str(text).strip()
        if not text_value:
            raise ValueError("TTS text must not be empty")

        interrupted = False
        if self.is_active():
            interrupted = self.stop()

        with self._lock:
            request = TTSSynthesisRequest(
                text=text_value,
                voice_id=str(voice_id).strip(),
                model_id=str(model_id).strip(),
                playback_speed=self._playback_speed,
            )

            self._generation += 1
            generation = self._generation

            self._last_request = request
            self._active_request = request

            self._cancel_event = threading.Event()
            self._pause_event = threading.Event()
            self._queue = Queue(maxsize=256)
            self._synth_started_at = time.monotonic()
            self._play_started_at = None

            self._synth_thread = threading.Thread(
                target=self._run_synthesis,
                args=(generation, request),
                name="tts-synth",
                daemon=True,
            )
            self._play_thread = threading.Thread(
                target=self._run_playback,
                args=(generation, request),
                name="tts-playback",
                daemon=True,
            )

        self._transition(
            "synthesizing",
            request_playback_speed=request.playback_speed,
            voice_id=request.voice_id,
            model_id=request.model_id,
        )

        self._synth_thread.start()
        self._play_thread.start()

        return interrupted

    def _run_synthesis(self, generation: int, request: TTSSynthesisRequest) -> None:
        first_chunk = True

        try:
            for chunk in self._backend.synthesize_stream(request):
                if not self._is_generation_current(generation) or self._cancel_event.is_set():
                    break
                if not chunk:
                    continue

                if first_chunk:
                    first_chunk = False
                    synth_started_at = self._synth_started_at
                    latency = 0.0
                    if synth_started_at is not None:
                        latency = max(0.0, time.monotonic() - synth_started_at)
                    self._transition(
                        "playing",
                        synth_latency_sec=latency,
                        request_playback_speed=request.playback_speed,
                    )

                while not self._cancel_event.is_set():
                    try:
                        self._queue.put(chunk, timeout=0.1)
                        break
                    except Full:
                        continue
        except Exception as exc:  # noqa: BLE001
            if self._cancel_event.is_set() or not self._is_generation_current(generation):
                pass
            else:
                self._clear_active_request(generation)
                self._transition(
                    "error",
                    error_class=type(exc).__name__,
                    message=str(exc),
                    request_playback_speed=request.playback_speed,
                    speed_apply_failure=isinstance(exc, TTSSpeedApplyError),
                )
        finally:
            while self._is_generation_current(generation):
                try:
                    self._queue.put(None, timeout=0.1)
                    break
                except Full:
                    if self._cancel_event.is_set():
                        break

    def _ensure_stream(self) -> sd.OutputStream:
        with self._lock:
            if self._stream is not None:
                return self._stream

        stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            device=self._output_device,
        )
        stream.start()

        with self._lock:
            self._stream = stream

        return stream

    def _close_stream(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None

        if stream is None:
            return

        try:
            stream.abort()
        except Exception:  # noqa: BLE001
            pass
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _chunk_to_samples(raw_chunk: bytes, carry: bytes) -> tuple[np.ndarray, bytes]:
        chunk = carry + raw_chunk
        usable_len = len(chunk) - (len(chunk) % 2)
        if usable_len <= 0:
            return np.empty((0, 1), dtype=np.int16), chunk
        usable = chunk[:usable_len]
        next_carry = chunk[usable_len:]
        samples = np.frombuffer(usable, dtype="<i2").reshape(-1, 1)
        return samples, next_carry

    def _write_samples_with_recovery(self, samples: np.ndarray) -> None:
        """Write PCM samples with a one-time stream recreation retry."""
        for attempt in (1, 2):
            stream = self._ensure_stream()
            try:
                stream.write(samples)
                return
            except sd.PortAudioError:
                if attempt >= 2:
                    raise
                log.warning("TTS playback write failed; recreating output stream")
                self._close_stream()
                time.sleep(0.03)

    def _run_playback(self, generation: int, request: TTSSynthesisRequest) -> None:
        carry = b""

        try:
            while self._is_generation_current(generation):
                if self._cancel_event.is_set():
                    return

                if self._pause_event.is_set():
                    # Keep pause/resume fully software-driven on the playback thread.
                    # Avoid calling stream.stop()/start() from control threads; some
                    # host stacks return transient PortAudio host errors in that path.
                    self._close_stream()
                    time.sleep(0.03)
                    continue

                try:
                    item = self._queue.get(timeout=0.1)
                except Empty:
                    continue

                if item is None:
                    break

                samples, carry = self._chunk_to_samples(item, carry)
                if samples.size == 0:
                    continue

                if self._play_started_at is None:
                    self._play_started_at = time.monotonic()
                self._write_samples_with_recovery(samples)

            if self._cancel_event.is_set() or not self._is_generation_current(generation):
                return
            if self.state == "error":
                return

            duration = 0.0
            if self._play_started_at is not None:
                duration = max(0.0, time.monotonic() - self._play_started_at)
            self._clear_active_request(generation)
            self._transition(
                "idle",
                playback_duration_sec=duration,
                request_playback_speed=request.playback_speed,
            )
        except Exception as exc:  # noqa: BLE001
            if self._cancel_event.is_set() or not self._is_generation_current(generation):
                return
            self._clear_active_request(generation)
            self._transition(
                "error",
                error_class=type(exc).__name__,
                message=str(exc),
                request_playback_speed=request.playback_speed,
                speed_apply_failure=False,
            )
        finally:
            self._close_stream()

    def pause(self) -> bool:
        with self._lock:
            if self._state != "playing":
                return False
            self._pause_event.set()

        self._transition("paused")
        return True

    def resume(self) -> bool:
        with self._lock:
            if self._state != "paused":
                return False
            self._pause_event.clear()

        self._transition("playing")
        return True

    def toggle_pause(self) -> bool:
        if self.state == "paused":
            return self.resume()
        return self.pause()

    def restart(self) -> bool:
        with self._lock:
            last_request = self._last_request

        if last_request is None or not last_request.text:
            return False

        self.speak(last_request.text, last_request.voice_id, last_request.model_id)
        return True

    def stop(self) -> bool:
        with self._lock:
            was_active = self._state in self.ACTIVE_STATES or self._state == "error"
            if not was_active:
                return False
            self._cancel_event.set()
            self._pause_event.clear()
            self._active_request = None
            synth_thread = self._synth_thread
            play_thread = self._play_thread

        for worker in (synth_thread, play_thread):
            if worker and worker.is_alive():
                worker.join(timeout=1.0)

        self._close_stream()

        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

        self._transition("idle")
        return True
