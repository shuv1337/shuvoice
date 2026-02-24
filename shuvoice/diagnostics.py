"""Diagnostics formatting helpers."""

from __future__ import annotations

import json
from typing import Any


def metrics_to_json(metrics: dict[str, Any]) -> str:
    return json.dumps(metrics, ensure_ascii=False, sort_keys=True)


def metrics_to_human(metrics: dict[str, Any]) -> str:
    counters = metrics.get("counters", {})
    timings = metrics.get("timings", {})
    queue_depth = timings.get("queue_depth", {}).get("avg", 0.0)
    utt_avg = timings.get("utterance_duration_sec", {}).get("avg", 0.0)
    return (
        f"chunks={counters.get('chunks_processed', 0)} "
        f"starts={counters.get('recording_start_count', 0)} "
        f"stops={counters.get('recording_stop_count', 0)} "
        f"partials={counters.get('partial_updates', 0)} "
        f"commits={counters.get('final_commits', 0)} "
        f"queue_avg={queue_depth:.2f} "
        f"utterance_avg_sec={utt_avg:.2f}"
    )
