"""Lightweight in-memory runtime metrics."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from statistics import fmean
from typing import Any


class MetricsCollector:
    """Low-overhead process-local counters + rolling timings."""

    def __init__(self, *, timing_window: int = 128):
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._timings: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=timing_window))

        self._recording_started_at: float | None = None

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += int(amount)

    def observe_timing(self, name: str, seconds: float) -> None:
        if seconds < 0:
            return
        with self._lock:
            self._timings[name].append(float(seconds))

    def recording_started(self) -> None:
        with self._lock:
            self._counters["recording_start_count"] += 1
            self._recording_started_at = time.monotonic()

    def recording_stopped(self) -> None:
        with self._lock:
            self._counters["recording_stop_count"] += 1
            started = self._recording_started_at
            self._recording_started_at = None

        if started is not None:
            self.observe_timing("utterance_duration_sec", time.monotonic() - started)

    def observe_chunk(self, chunk_rms: float, queue_depth: int) -> None:
        with self._lock:
            self._counters["chunks_processed"] += 1
            self._timings["chunk_rms"].append(float(chunk_rms))
            self._timings["queue_depth"].append(float(queue_depth))
            if queue_depth > self._counters["queue_depth_max"]:
                self._counters["queue_depth_max"] = int(queue_depth)

    def observe_partial_update(self) -> None:
        self.increment("partial_updates")

    def observe_final_commit(self) -> None:
        self.increment("final_commits")

    def observe_commit_failure(self) -> None:
        self.increment("commit_failures")

    def observe_stall_flush(self) -> None:
        self.increment("stall_flushes")

    def observe_recovery_reset(self) -> None:
        self.increment("recovery_resets")

    # -- TTS telemetry helpers ------------------------------------------------

    def observe_tts_speak(self) -> None:
        self.increment("tts_speak_count")

    def observe_tts_interrupt(self) -> None:
        self.increment("tts_interrupt_count")

    def observe_tts_synth_failure(self) -> None:
        self.increment("tts_synth_failures")

    def observe_tts_playback_completion(self) -> None:
        self.increment("tts_playback_completions")

    def observe_tts_pause(self) -> None:
        self.increment("tts_pause_count")

    def observe_tts_selection_failure(self) -> None:
        self.increment("tts_selection_failures")

    def observe_tts_synth_latency(self, seconds: float) -> None:
        self.observe_timing("tts_synth_latency_sec", seconds)

    def observe_tts_playback_duration(self, seconds: float) -> None:
        self.observe_timing("tts_playback_duration_sec", seconds)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            timings = {k: list(v) for k, v in self._timings.items()}

        timing_summary: dict[str, dict[str, float]] = {
            name: {
                "count": len(values),
                "avg": fmean(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }
            for name, values in timings.items()
        }

        summary: dict[str, Any] = {
            "counters": counters,
            "timings": timing_summary,
            "tts": {
                "speak_count": counters.get("tts_speak_count", 0),
                "interrupt_count": counters.get("tts_interrupt_count", 0),
                "synth_failures": counters.get("tts_synth_failures", 0),
                "playback_completions": counters.get("tts_playback_completions", 0),
                "pause_count": counters.get("tts_pause_count", 0),
                "selection_failures": counters.get("tts_selection_failures", 0),
                "synth_latency_sec": timing_summary.get(
                    "tts_synth_latency_sec",
                    {"count": 0, "avg": 0.0, "max": 0.0},
                ),
                "playback_duration_sec": timing_summary.get(
                    "tts_playback_duration_sec",
                    {"count": 0, "avg": 0.0, "max": 0.0},
                ),
            },
        }
        return summary

    def summary_line(self) -> str:
        snap = self.snapshot()
        counters: Mapping[str, int] = snap["counters"]
        timings: Mapping[str, Mapping[str, float]] = snap["timings"]

        return (
            "metrics "
            f"chunks={counters.get('chunks_processed', 0)} "
            f"starts={counters.get('recording_start_count', 0)} "
            f"stops={counters.get('recording_stop_count', 0)} "
            f"partials={counters.get('partial_updates', 0)} "
            f"commits={counters.get('final_commits', 0)} "
            f"tts_speaks={counters.get('tts_speak_count', 0)} "
            f"tts_done={counters.get('tts_playback_completions', 0)} "
            f"queue_max={counters.get('queue_depth_max', 0)} "
            f"utt_avg={timings.get('utterance_duration_sec', {}).get('avg', 0.0):.2f}s"
        )
