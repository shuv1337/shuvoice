//! Lightweight in-memory runtime metrics.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Mutex;
use std::time::Instant;

use serde::Serialize;
use serde_json::{Value, json};

const DEFAULT_TIMING_WINDOW: usize = 128;

/// Timing summary block in snapshots.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TimingSummary {
    pub count: usize,
    pub avg: f64,
    pub max: f64,
}

impl TimingSummary {
    fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                count: 0,
                avg: 0.0,
                max: 0.0,
            };
        }
        let sum: f64 = values.iter().sum();
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Self {
            count: values.len(),
            avg: sum / values.len() as f64,
            max,
        }
    }
}

/// Full metrics snapshot with the stable public key layout.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MetricsSnapshot {
    pub counters: BTreeMap<String, i64>,
    pub timings: BTreeMap<String, TimingSummary>,
    pub runtime: RuntimeSnapshot,
    pub tts: TtsMetricsSnapshot,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RuntimeSnapshot {
    pub pid: u32,
    pub recording_active: bool,
    pub recording_duration_sec: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TtsMetricsSnapshot {
    pub speak_count: i64,
    pub interrupt_count: i64,
    pub synth_failures: i64,
    pub playback_completions: i64,
    pub pause_count: i64,
    pub selection_failures: i64,
    pub speed_change_count: i64,
    pub speed_restart_count: i64,
    pub speed_unsupported_count: i64,
    pub speed_apply_failure_count: i64,
    pub synth_latency_sec: TimingSummary,
    pub playback_duration_sec: TimingSummary,
}

/// Low-overhead process-local counters + rolling timings.
#[derive(Debug)]
pub struct MetricsCollector {
    inner: Mutex<Inner>,
}

#[derive(Debug)]
struct Inner {
    counters: HashMap<String, i64>,
    timings: HashMap<String, VecDeque<f64>>,
    timing_window: usize,
    recording_started_at: Option<Instant>,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self::with_timing_window(DEFAULT_TIMING_WINDOW)
    }

    pub fn with_timing_window(timing_window: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                counters: HashMap::new(),
                timings: HashMap::new(),
                timing_window: timing_window.max(1),
                recording_started_at: None,
            }),
        }
    }

    pub fn increment(&self, name: &str, amount: i64) {
        let mut g = self.inner.lock().expect("metrics lock");
        *g.counters.entry(name.to_string()).or_insert(0) += amount;
    }

    pub fn observe_timing(&self, name: &str, seconds: f64) {
        if seconds < 0.0 {
            return;
        }
        let mut g = self.inner.lock().expect("metrics lock");
        let window = g.timing_window;
        let entry = g
            .timings
            .entry(name.to_string())
            .or_insert_with(|| VecDeque::with_capacity(window));
        if entry.len() == window {
            entry.pop_front();
        }
        entry.push_back(seconds);
    }

    pub fn recording_started(&self) {
        let mut g = self.inner.lock().expect("metrics lock");
        *g.counters
            .entry("recording_start_count".into())
            .or_insert(0) += 1;
        g.recording_started_at = Some(Instant::now());
    }

    pub fn recording_stopped(&self) {
        let started = {
            let mut g = self.inner.lock().expect("metrics lock");
            *g.counters.entry("recording_stop_count".into()).or_insert(0) += 1;
            g.recording_started_at.take()
        };
        if let Some(started) = started {
            self.observe_timing("utterance_duration_sec", started.elapsed().as_secs_f64());
        }
    }

    pub fn observe_chunk(&self, chunk_rms: f64, queue_depth: usize) {
        let mut g = self.inner.lock().expect("metrics lock");
        *g.counters.entry("chunks_processed".into()).or_insert(0) += 1;
        let window = g.timing_window;
        for (name, value) in [
            ("chunk_rms", chunk_rms),
            ("queue_depth", queue_depth as f64),
        ] {
            let entry = g
                .timings
                .entry(name.to_string())
                .or_insert_with(|| VecDeque::with_capacity(window));
            if entry.len() == window {
                entry.pop_front();
            }
            entry.push_back(value);
        }
        let max_entry = g.counters.entry("queue_depth_max".into()).or_insert(0);
        if queue_depth as i64 > *max_entry {
            *max_entry = queue_depth as i64;
        }
    }

    pub fn observe_partial_update(&self) {
        self.increment("partial_updates", 1);
    }
    pub fn observe_final_commit(&self) {
        self.increment("final_commits", 1);
    }
    pub fn observe_commit_failure(&self) {
        self.increment("commit_failures", 1);
    }
    /// Utterance dropped by the silence gate before reaching the ASR.
    ///
    /// Counted so that "recorded N times, transcribed 0" is visible in
    /// `control metrics` — success-only counters render a total failure as
    /// missing data rather than as a signal.
    pub fn observe_silent_discard(&self) {
        self.increment("silent_discards", 1);
    }
    pub fn observe_stall_flush(&self) {
        self.increment("stall_flushes", 1);
    }
    pub fn observe_recovery_reset(&self) {
        self.increment("recovery_resets", 1);
    }

    pub fn observe_tts_speak(&self) {
        self.increment("tts_speak_count", 1);
    }
    pub fn observe_tts_interrupt(&self) {
        self.increment("tts_interrupt_count", 1);
    }
    pub fn observe_tts_synth_failure(&self) {
        self.increment("tts_synth_failures", 1);
    }
    pub fn observe_tts_playback_completion(&self) {
        self.increment("tts_playback_completions", 1);
    }
    pub fn observe_tts_pause(&self) {
        self.increment("tts_pause_count", 1);
    }
    pub fn observe_tts_selection_failure(&self) {
        self.increment("tts_selection_failures", 1);
    }
    pub fn observe_tts_speed_change(&self) {
        self.increment("tts_speed_change_count", 1);
    }
    pub fn observe_tts_speed_restart(&self) {
        self.increment("tts_speed_restart_count", 1);
    }
    pub fn observe_tts_speed_unsupported(&self) {
        self.increment("tts_speed_unsupported_count", 1);
    }
    pub fn observe_tts_speed_apply_failure(&self) {
        self.increment("tts_speed_apply_failure_count", 1);
    }
    pub fn observe_tts_synth_latency(&self, seconds: f64) {
        self.observe_timing("tts_synth_latency_sec", seconds);
    }
    pub fn observe_tts_playback_duration(&self, seconds: f64) {
        self.observe_timing("tts_playback_duration_sec", seconds);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let g = self.inner.lock().expect("metrics lock");
        let counters: BTreeMap<String, i64> =
            g.counters.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let mut timings = BTreeMap::new();
        for (name, values) in &g.timings {
            let vals: Vec<f64> = values.iter().copied().collect();
            timings.insert(name.clone(), TimingSummary::from_values(&vals));
        }
        let recording_active = g.recording_started_at.is_some();
        let recording_duration_sec = g
            .recording_started_at
            .map(|t| t.elapsed().as_secs_f64().max(0.0))
            .unwrap_or(0.0);

        let speak_count = counters.get("tts_speak_count").copied().unwrap_or(0);
        let interrupt_count = counters.get("tts_interrupt_count").copied().unwrap_or(0);
        let synth_failures = counters.get("tts_synth_failures").copied().unwrap_or(0);
        let playback_completions = counters
            .get("tts_playback_completions")
            .copied()
            .unwrap_or(0);
        let pause_count = counters.get("tts_pause_count").copied().unwrap_or(0);
        let selection_failures = counters.get("tts_selection_failures").copied().unwrap_or(0);
        let speed_change_count = counters.get("tts_speed_change_count").copied().unwrap_or(0);
        let speed_restart_count = counters
            .get("tts_speed_restart_count")
            .copied()
            .unwrap_or(0);
        let speed_unsupported_count = counters
            .get("tts_speed_unsupported_count")
            .copied()
            .unwrap_or(0);
        let speed_apply_failure_count = counters
            .get("tts_speed_apply_failure_count")
            .copied()
            .unwrap_or(0);
        let empty_timing = TimingSummary {
            count: 0,
            avg: 0.0,
            max: 0.0,
        };
        let synth_latency_sec = timings
            .get("tts_synth_latency_sec")
            .cloned()
            .unwrap_or(empty_timing.clone());
        let playback_duration_sec = timings
            .get("tts_playback_duration_sec")
            .cloned()
            .unwrap_or(empty_timing);

        MetricsSnapshot {
            counters,
            timings,
            runtime: RuntimeSnapshot {
                pid: std::process::id(),
                recording_active,
                recording_duration_sec,
            },
            tts: TtsMetricsSnapshot {
                speak_count,
                interrupt_count,
                synth_failures,
                playback_completions,
                pause_count,
                selection_failures,
                speed_change_count,
                speed_restart_count,
                speed_unsupported_count,
                speed_apply_failure_count,
                synth_latency_sec,
                playback_duration_sec,
            },
        }
    }

    pub fn summary_line(&self) -> String {
        let snap = self.snapshot();
        let c = |k: &str| snap.counters.get(k).copied().unwrap_or(0);
        let utt_avg = snap
            .timings
            .get("utterance_duration_sec")
            .map(|t| t.avg)
            .unwrap_or(0.0);
        format!(
            "metrics chunks={} starts={} stops={} partials={} commits={} silent_discards={} tts_speaks={} tts_done={} tts_speed_changes={} tts_speed_restarts={} queue_max={} utt_avg={:.2}s",
            c("chunks_processed"),
            c("recording_start_count"),
            c("recording_stop_count"),
            c("partial_updates"),
            c("final_commits"),
            c("silent_discards"),
            c("tts_speak_count"),
            c("tts_playback_completions"),
            c("tts_speed_change_count"),
            c("tts_speed_restart_count"),
            c("queue_depth_max"),
            utt_avg
        )
    }
}

/// Serialize metrics with deterministically sorted keys.
pub fn metrics_to_json(snapshot: &MetricsSnapshot) -> String {
    let value = serde_json::to_value(snapshot).unwrap_or_else(|_| json!({}));
    let sorted = sort_value(value);
    serde_json::to_string(&sorted).unwrap_or_else(|_| "{}".into())
}

/// Human-readable metrics one-liner used by diagnostics.
pub fn metrics_to_human(snapshot: &MetricsSnapshot) -> String {
    let c = |k: &str| snapshot.counters.get(k).copied().unwrap_or(0);
    let queue_avg = snapshot
        .timings
        .get("queue_depth")
        .map(|t| t.avg)
        .unwrap_or(0.0);
    let utt_avg = snapshot
        .timings
        .get("utterance_duration_sec")
        .map(|t| t.avg)
        .unwrap_or(0.0);
    format!(
        "chunks={} starts={} stops={} partials={} commits={} silent_discards={} tts_speaks={} tts_done={} tts_speed_changes={} tts_speed_restarts={} queue_avg={:.2} utterance_avg_sec={:.2}",
        c("chunks_processed"),
        c("recording_start_count"),
        c("recording_stop_count"),
        c("partial_updates"),
        c("final_commits"),
        c("silent_discards"),
        snapshot.tts.speak_count,
        snapshot.tts.playback_completions,
        snapshot.tts.speed_change_count,
        snapshot.tts.speed_restart_count,
        queue_avg,
        utt_avg
    )
}

fn sort_value(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<_> = map.keys().cloned().collect();
            keys.sort();
            let mut out = serde_json::Map::new();
            for k in keys {
                if let Some(v) = map.get(&k) {
                    out.insert(k, sort_value(v.clone()));
                }
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(sort_value).collect()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_collector_counts_and_timings() {
        let metrics = MetricsCollector::new();
        metrics.recording_started();
        metrics.observe_chunk(0.2, 3);
        metrics.observe_partial_update();
        metrics.observe_final_commit();
        metrics.recording_stopped();

        metrics.observe_tts_speak();
        metrics.observe_tts_interrupt();
        metrics.observe_tts_pause();
        metrics.observe_tts_selection_failure();
        metrics.observe_tts_speed_change();
        metrics.observe_tts_speed_restart();
        metrics.observe_tts_speed_unsupported();
        metrics.observe_tts_speed_apply_failure();
        metrics.observe_tts_synth_failure();
        metrics.observe_tts_playback_completion();
        metrics.observe_tts_synth_latency(0.42);
        metrics.observe_tts_playback_duration(1.23);

        let snap = metrics.snapshot();
        assert_eq!(snap.counters.get("recording_start_count"), Some(&1));
        assert_eq!(snap.counters.get("recording_stop_count"), Some(&1));
        assert_eq!(snap.counters.get("chunks_processed"), Some(&1));
        assert_eq!(snap.counters.get("partial_updates"), Some(&1));
        assert_eq!(snap.counters.get("final_commits"), Some(&1));
        assert!(snap.timings.get("utterance_duration_sec").unwrap().count >= 1);
        assert_eq!(snap.tts.speak_count, 1);
        assert_eq!(snap.tts.interrupt_count, 1);
        assert_eq!(snap.tts.pause_count, 1);
        assert_eq!(snap.tts.selection_failures, 1);
        assert_eq!(snap.tts.speed_change_count, 1);
        assert_eq!(snap.tts.speed_restart_count, 1);
        assert_eq!(snap.tts.speed_unsupported_count, 1);
        assert_eq!(snap.tts.speed_apply_failure_count, 1);
        assert_eq!(snap.tts.synth_failures, 1);
        assert_eq!(snap.tts.playback_completions, 1);
        assert_eq!(snap.tts.synth_latency_sec.count, 1);
        assert_eq!(snap.tts.playback_duration_sec.count, 1);
    }

    #[test]
    fn metrics_summary_line_has_no_transcript_content() {
        let metrics = MetricsCollector::new();
        let line = metrics.summary_line();
        assert!(!line.to_lowercase().contains("transcript"));
    }
}
