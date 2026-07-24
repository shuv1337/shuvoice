//! In-process log capture and debug overlay formatting.

use std::collections::VecDeque;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// Thread-safe ring buffer of formatted log lines.
#[derive(Debug)]
pub struct RecentLogBuffer {
    max_entries: usize,
    entries: Mutex<VecDeque<String>>,
}

impl RecentLogBuffer {
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries: max_entries.max(1),
            entries: Mutex::new(VecDeque::new()),
        }
    }

    pub fn push(&self, line: impl Into<String>) {
        let mut guard = self.entries.lock().expect("log buffer lock");
        if guard.len() >= self.max_entries {
            guard.pop_front();
        }
        guard.push_back(line.into());
    }

    pub fn tail(&self, max_lines: usize) -> Vec<String> {
        let guard = self.entries.lock().expect("log buffer lock");
        if max_lines == 0 {
            return Vec::new();
        }
        let n = max_lines.min(guard.len());
        guard
            .iter()
            .rev()
            .take(n)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    pub fn render(&self, max_lines: usize) -> String {
        self.tail(max_lines).join("\n")
    }
}

impl Default for RecentLogBuffer {
    fn default() -> Self {
        Self::new(400)
    }
}

/// Snapshot inputs for the caption debug panel.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DebugSnapshot {
    pub recording: bool,
    pub processing: bool,
    pub asr_disabled: bool,
    pub thread_alive: bool,
    pub audio_queue_depth: usize,
    pub audio_queue_max: usize,
    pub noise_floor_rms: f64,
    pub speech_rms_threshold: f64,
    pub asr_backend: String,
    pub asr_step: u64,
    pub native_chunk_samples: u64,
    pub wants_raw_audio: bool,
    pub chunks_processed: u64,
    pub partial_updates: u64,
    pub final_commits: u64,
    pub recovery_resets: u64,
    pub utterance_avg_sec: f64,
    pub recording_duration_sec: f64,
    pub queue_avg: f64,
    pub utt_buf: Option<u64>,
    pub speech_samples: Option<u64>,
    pub peak_rms: Option<f64>,
    pub gain: Option<f64>,
    pub unchanged_steps: Option<u64>,
    pub partial_transcript: String,
    pub final_transcript: String,
    pub log_lines: Vec<String>,
}

/// Format the multiline debug overlay text.
pub fn format_debug_overlay_lines(snap: &DebugSnapshot) -> String {
    let mut lines = vec![
        format!(
            "state rec={} proc={} asr_disabled={} thread_alive={}",
            u8::from(snap.recording),
            u8::from(snap.processing),
            u8::from(snap.asr_disabled),
            u8::from(snap.thread_alive),
        ),
        format!(
            "audio q={}/{} noise={:.4} thr={:.4}",
            snap.audio_queue_depth,
            snap.audio_queue_max,
            snap.noise_floor_rms,
            snap.speech_rms_threshold,
        ),
        format!(
            "asr backend={} step={} chunk={} raw={}",
            snap.asr_backend,
            snap.asr_step,
            snap.native_chunk_samples,
            u8::from(snap.wants_raw_audio),
        ),
        format!(
            "metrics chunks={} partials={} commits={} resets={}",
            snap.chunks_processed, snap.partial_updates, snap.final_commits, snap.recovery_resets,
        ),
        format!(
            "utt avg={:.2}s recording_for={:.2}s queue_avg={:.2}",
            snap.utterance_avg_sec, snap.recording_duration_sec, snap.queue_avg
        ),
    ];

    if let (Some(buf), Some(speech), Some(peak), Some(gain), Some(unchanged)) = (
        snap.utt_buf,
        snap.speech_samples,
        snap.peak_rms,
        snap.gain,
        snap.unchanged_steps,
    ) {
        lines.push(format!(
            "utt buf={buf} speech_samples={speech} peak={peak:.4} gain={gain:.2} unchanged={unchanged}"
        ));
    }

    if !snap.partial_transcript.is_empty() {
        lines.push(format!("partial: {}", snap.partial_transcript));
    }
    if !snap.final_transcript.is_empty() {
        lines.push(format!("final: {}", snap.final_transcript));
    }
    if !snap.log_lines.is_empty() {
        lines.push("logs:".into());
        lines.extend(snap.log_lines.iter().cloned());
    }

    lines.join("\n")
}

/// JSON-serializable debug_status payload (subset used by control socket).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugStatusPayload {
    pub app: DebugStatusApp,
    pub audio: DebugStatusAudio,
    pub asr: DebugStatusAsr,
    pub metrics: serde_json::Value,
    pub logs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugStatusApp {
    pub asr_backend: String,
    pub tts_backend: String,
    pub overlay_debug_mode: bool,
    pub recording: bool,
    pub processing: bool,
    pub asr_disabled: bool,
    pub asr_thread_alive: bool,
    pub model_load_failed: bool,
    pub control_socket: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugStatusAudio {
    pub queue_depth: usize,
    pub queue_max: Option<usize>,
    pub noise_floor_rms: f64,
    pub speech_rms_threshold: f64,
    pub speech_rms_multiplier: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugStatusAsr {
    pub debug_step_num: u64,
    pub native_chunk_samples: u64,
    pub wants_raw_audio: bool,
    pub consecutive_failures: u32,
    pub current_transcript: String,
    pub last_final_transcript: String,
}

pub fn debug_status_to_json(status: &DebugStatusPayload) -> String {
    serde_json::to_string(status).unwrap_or_else(|_| "{}".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recent_log_buffer_returns_tail() {
        let buf = RecentLogBuffer::new(10);
        buf.push("a");
        buf.push("audio queue high");
        let tail = buf.tail(1);
        assert_eq!(tail.len(), 1);
        assert!(tail[0].contains("audio queue high"));
    }

    #[test]
    fn format_debug_overlay_contains_expected_prefixes() {
        let text = format_debug_overlay_lines(&DebugSnapshot {
            recording: false,
            processing: false,
            asr_disabled: false,
            thread_alive: true,
            audio_queue_depth: 3,
            audio_queue_max: 200,
            noise_floor_rms: 0.001,
            speech_rms_threshold: 0.01,
            asr_backend: "sherpa".into(),
            asr_step: 9,
            native_chunk_samples: 1600,
            wants_raw_audio: false,
            chunks_processed: 1,
            partial_updates: 0,
            final_commits: 0,
            recovery_resets: 0,
            utterance_avg_sec: 0.0,
            recording_duration_sec: 0.0,
            queue_avg: 0.0,
            utt_buf: Some(3200),
            speech_samples: Some(1600),
            peak_rms: Some(0.021),
            gain: Some(1.4),
            unchanged_steps: Some(2),
            partial_transcript: "working partial".into(),
            final_transcript: "finished final".into(),
            log_lines: vec!["line".into()],
        });
        assert!(text.contains("state rec=0 proc=0 asr_disabled=0 thread_alive=1"));
        assert!(text.contains("audio q=3/200"));
        assert!(text.contains("asr backend=sherpa step=9 chunk=1600 raw=0"));
        assert!(
            text.contains("utt buf=3200 speech_samples=1600 peak=0.0210 gain=1.40 unchanged=2")
        );
        assert!(text.contains("partial: working partial"));
        assert!(text.contains("final: finished final"));
        assert!(text.contains("logs:"));
    }
}
