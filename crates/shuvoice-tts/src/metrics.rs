//! TTS metrics event hooks.

use std::sync::Arc;

use parking_lot::Mutex;

/// Observer interface for TTS lifecycle metrics.
pub trait TtsMetrics: Send + Sync {
    fn observe_tts_speak(&self) {}
    fn observe_tts_interrupt(&self) {}
    fn observe_tts_synth_failure(&self) {}
    fn observe_tts_playback_completion(&self) {}
    fn observe_tts_pause(&self) {}
    fn observe_tts_selection_failure(&self) {}
    fn observe_tts_speed_change(&self) {}
    fn observe_tts_speed_restart(&self) {}
    fn observe_tts_speed_unsupported(&self) {}
    fn observe_tts_speed_apply_failure(&self) {}
    fn observe_tts_synth_latency(&self, _seconds: f64) {}
    fn observe_tts_playback_duration(&self, _seconds: f64) {}
    fn observe_tts_queue_overflow(&self) {}
    fn observe_tts_join_timeout(&self) {}
    fn observe_tts_spawn_failure(&self) {}
}

/// No-op metrics sink.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopMetrics;

impl TtsMetrics for NoopMetrics {}

/// In-memory counters for tests and early integration.
#[derive(Debug, Default, Clone)]
pub struct CountingMetrics {
    inner: Arc<Mutex<Counts>>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Counts {
    pub speak_count: u64,
    pub interrupt_count: u64,
    pub synth_failures: u64,
    pub playback_completions: u64,
    pub pause_count: u64,
    pub selection_failures: u64,
    pub speed_change_count: u64,
    pub speed_restart_count: u64,
    pub speed_unsupported_count: u64,
    pub speed_apply_failure_count: u64,
    pub queue_overflow_count: u64,
    pub join_timeout_count: u64,
    pub spawn_failure_count: u64,
    pub synth_latency_samples: Vec<f64>,
    pub playback_duration_samples: Vec<f64>,
}

impl CountingMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> Counts {
        self.inner.lock().clone()
    }
}

impl TtsMetrics for CountingMetrics {
    fn observe_tts_speak(&self) {
        self.inner.lock().speak_count += 1;
    }
    fn observe_tts_interrupt(&self) {
        self.inner.lock().interrupt_count += 1;
    }
    fn observe_tts_synth_failure(&self) {
        self.inner.lock().synth_failures += 1;
    }
    fn observe_tts_playback_completion(&self) {
        self.inner.lock().playback_completions += 1;
    }
    fn observe_tts_pause(&self) {
        self.inner.lock().pause_count += 1;
    }
    fn observe_tts_selection_failure(&self) {
        self.inner.lock().selection_failures += 1;
    }
    fn observe_tts_speed_change(&self) {
        self.inner.lock().speed_change_count += 1;
    }
    fn observe_tts_speed_restart(&self) {
        self.inner.lock().speed_restart_count += 1;
    }
    fn observe_tts_speed_unsupported(&self) {
        self.inner.lock().speed_unsupported_count += 1;
    }
    fn observe_tts_speed_apply_failure(&self) {
        self.inner.lock().speed_apply_failure_count += 1;
    }
    fn observe_tts_synth_latency(&self, seconds: f64) {
        self.inner.lock().synth_latency_samples.push(seconds);
    }
    fn observe_tts_playback_duration(&self, seconds: f64) {
        self.inner.lock().playback_duration_samples.push(seconds);
    }
    fn observe_tts_queue_overflow(&self) {
        self.inner.lock().queue_overflow_count += 1;
    }
    fn observe_tts_join_timeout(&self) {
        self.inner.lock().join_timeout_count += 1;
    }
    fn observe_tts_spawn_failure(&self) {
        self.inner.lock().spawn_failure_count += 1;
    }
}
