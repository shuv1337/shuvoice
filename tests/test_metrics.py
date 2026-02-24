from __future__ import annotations

from shuvoice.metrics import MetricsCollector


def test_metrics_collector_counts_and_timings():
    metrics = MetricsCollector()

    metrics.recording_started()
    metrics.observe_chunk(0.2, 3)
    metrics.observe_partial_update()
    metrics.observe_final_commit()
    metrics.recording_stopped()

    snap = metrics.snapshot()

    assert snap["counters"]["recording_start_count"] == 1
    assert snap["counters"]["recording_stop_count"] == 1
    assert snap["counters"]["chunks_processed"] == 1
    assert snap["counters"]["partial_updates"] == 1
    assert snap["counters"]["final_commits"] == 1
    assert snap["timings"]["utterance_duration_sec"]["count"] >= 1


def test_metrics_summary_line_has_no_transcript_content():
    metrics = MetricsCollector()
    line = metrics.summary_line()

    # Guardrail: only numeric/system counters, no free-form transcript text.
    assert "transcript" not in line.lower()
