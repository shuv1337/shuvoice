//! Diagnostics formatting + in-process log ring buffer.

use std::collections::VecDeque;
use std::sync::Mutex;

use serde_json::{Map, Value};

/// Serialize metrics map to JSON (`ensure_ascii=false`, sorted keys via BTree insertion).
#[must_use]
pub fn metrics_to_json(metrics: &Value) -> String {
    // Caller should pass an object; we re-emit compact JSON.
    serde_json::to_string(metrics).unwrap_or_else(|_| "{}".into())
}

/// Serialize debug status map to JSON.
#[must_use]
pub fn debug_status_to_json(status: &Value) -> String {
    serde_json::to_string(status).unwrap_or_else(|_| "{}".into())
}

/// Human one-line metrics summary (Python `metrics_to_human` parity).
#[must_use]
pub fn metrics_to_human(metrics: &Value) -> String {
    let counters = metrics
        .get("counters")
        .cloned()
        .unwrap_or(Value::Object(Map::new()));
    let timings = metrics
        .get("timings")
        .cloned()
        .unwrap_or(Value::Object(Map::new()));
    let tts = metrics
        .get("tts")
        .cloned()
        .unwrap_or(Value::Object(Map::new()));

    let queue_depth = timings
        .pointer("/queue_depth/avg")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let utt_avg = timings
        .pointer("/utterance_duration_sec/avg")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    let counter = |k: &str| counters.get(k).and_then(Value::as_u64).unwrap_or(0);
    let tts_or = |tk: &str, ck: &str| {
        tts.get(tk)
            .and_then(Value::as_u64)
            .unwrap_or_else(|| counter(ck))
    };

    format!(
        "chunks={} starts={} stops={} partials={} commits={} tts_speaks={} tts_done={} tts_speed_changes={} tts_speed_restarts={} queue_avg={:.2} utterance_avg_sec={:.2}",
        counter("chunks_processed"),
        counter("recording_start_count"),
        counter("recording_stop_count"),
        counter("partial_updates"),
        counter("final_commits"),
        tts_or("speak_count", "tts_speak_count"),
        tts_or("playback_completions", "tts_playback_completions"),
        tts_or("speed_change_count", "tts_speed_change_count"),
        tts_or("speed_restart_count", "tts_speed_restart_count"),
        queue_depth,
        utt_avg,
    )
}

/// Thread-safe ring buffer of formatted log lines (debug overlay).
#[derive(Debug)]
pub struct RecentLogBuffer {
    entries: Mutex<VecDeque<String>>,
    max_entries: usize,
}

impl RecentLogBuffer {
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(VecDeque::new()),
            max_entries: max_entries.max(1),
        }
    }

    pub fn push(&self, line: impl Into<String>) {
        let mut g = self.entries.lock().unwrap();
        if g.len() >= self.max_entries {
            g.pop_front();
        }
        g.push_back(line.into());
    }

    #[must_use]
    pub fn tail(&self, max_lines: usize) -> Vec<String> {
        if max_lines == 0 {
            return Vec::new();
        }
        let g = self.entries.lock().unwrap();
        let n = max_lines.min(g.len());
        g.iter()
            .rev()
            .take(n)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    #[must_use]
    pub fn render(&self, max_lines: usize) -> String {
        self.tail(max_lines).join("\n")
    }
}

impl Default for RecentLogBuffer {
    fn default() -> Self {
        Self::new(400)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn log_buffer_tail() {
        let buf = RecentLogBuffer::new(20);
        buf.push("a");
        buf.push("b");
        buf.push("c");
        assert_eq!(buf.tail(2), vec!["b".to_string(), "c".to_string()]);
        assert!(buf.tail(0).is_empty());
    }

    #[test]
    fn human_metrics_line() {
        let m = json!({
            "counters": {
                "chunks_processed": 3,
                "recording_start_count": 1,
                "recording_stop_count": 1,
                "partial_updates": 2,
                "final_commits": 1
            },
            "timings": {
                "queue_depth": {"avg": 1.5},
                "utterance_duration_sec": {"avg": 0.75}
            },
            "tts": {
                "speak_count": 0,
                "playback_completions": 0,
                "speed_change_count": 0,
                "speed_restart_count": 0
            }
        });
        let line = metrics_to_human(&m);
        assert!(line.contains("chunks=3"));
        assert!(line.contains("queue_avg=1.50"));
    }
}
