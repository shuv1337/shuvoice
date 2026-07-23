//! Pure runtime policy helpers shared by the session actor.

use crate::audio_math::{
    compute_utterance_rms_threshold, observe_recording_chunk, select_preroll_chunks,
};
use crate::circuit_breaker::should_ignore_start_during_rearm;
use crate::types::RecordingStatus;
use crate::utterance::UtteranceState;

pub use crate::circuit_breaker::{
    ASR_CIRCUIT_COOLDOWN, ASR_MAX_FAILURES, BreakerAction, CircuitBreaker, ERROR_TOAST_SECONDS,
    MIN_SPLASH_VISIBLE, PTT_REARM_GRACE, remaining_splash_ms,
};
pub use crate::flush::{
    FLUSH_NOISE_ESCALATION, FLUSH_NOISE_MAX_RMS, FLUSH_NOISE_MIN_RMS, FLUSH_TAIL_MAX_STEPS,
    FLUSH_TAIL_STABLE_REQUIRED, TailFlushDecision, evaluate_tail_flush_step,
    flush_noise_escalation, make_flush_noise,
};
pub use crate::streaming_health::should_trigger_stall_flush;

/// Inputs required to begin an utterance (threshold + optional preroll).
#[derive(Debug, Clone)]
pub struct BeginUtteranceParams<'a> {
    pub noise_floor_rms: f32,
    pub speech_rms_threshold: f32,
    pub speech_rms_multiplier: f32,
    pub preroll_chunks: &'a [Vec<f32>],
    pub wants_raw_audio: bool,
    pub auto_gain_settle_chunks: usize,
    pub auto_gain_target_peak: f32,
    pub auto_gain_max: f32,
}

/// Begin a new utterance: compute threshold, reset state, prepend preroll.
pub fn begin_utterance(state: &mut UtteranceState, params: BeginUtteranceParams<'_>) {
    let threshold = compute_utterance_rms_threshold(
        params.noise_floor_rms,
        params.speech_rms_threshold,
        params.speech_rms_multiplier,
    );
    state.reset(threshold);
    for chunk in params.preroll_chunks {
        observe_recording_chunk(
            state,
            chunk,
            params.wants_raw_audio,
            params.auto_gain_settle_chunks,
            params.auto_gain_target_peak,
            params.auto_gain_max,
        );
    }
}

/// Whether silent utterance should be discarded.
pub fn is_silent_utterance(state: &UtteranceState, min_speech_samples: usize) -> bool {
    if min_speech_samples == 0 {
        return false;
    }
    state.speech_samples < min_speech_samples
}

/// Recording start gate combining thread/circuit/rearm checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartGate {
    Allow,
    AlreadyRecording,
    AsrThreadDead,
    RearmGrace,
    AsrDisabledNeedsRecovery,
}

pub fn evaluate_start_gate(
    recording: bool,
    asr_thread_alive: bool,
    processing: bool,
    since_stop: std::time::Duration,
    asr_disabled: bool,
) -> StartGate {
    if recording {
        return StartGate::AlreadyRecording;
    }
    if !asr_thread_alive {
        return StartGate::AsrThreadDead;
    }
    if should_ignore_start_during_rearm(processing, since_stop) {
        return StartGate::RearmGrace;
    }
    if asr_disabled {
        return StartGate::AsrDisabledNeedsRecovery;
    }
    StartGate::Allow
}

/// Convenience wrapper around [`RecordingStatus::from_flags`].
pub fn recording_status(
    asr_disabled: bool,
    asr_thread_alive: bool,
    recording: bool,
    processing: bool,
) -> RecordingStatus {
    RecordingStatus::from_flags(asr_disabled, asr_thread_alive, recording, processing)
}

/// Build preroll from drained + retained chunks.
pub fn capture_preroll(chunks: &[Vec<f32>], max_samples: usize) -> Vec<Vec<f32>> {
    select_preroll_chunks(chunks, max_samples)
}

/// Stop-side drain grace timeout used by the Python runtime.
pub const STOP_TAIL_GRACE: std::time::Duration = std::time::Duration::from_millis(120);

/// Audio worker poll timeout used by the ASR loop.
pub const ASR_LOOP_POLL_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(50);

/// Metrics log period.
pub const METRICS_LOG_PERIOD: std::time::Duration = std::time::Duration::from_secs(10);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_math::compute_utterance_rms_threshold;

    #[test]
    fn begin_utterance_prepends_preroll_and_sets_threshold() {
        let preroll = vec![vec![0.02_f32; 160]];
        let mut state = UtteranceState::new();
        begin_utterance(
            &mut state,
            BeginUtteranceParams {
                noise_floor_rms: 0.001,
                speech_rms_threshold: 0.008,
                speech_rms_multiplier: 1.8,
                preroll_chunks: &preroll,
                wants_raw_audio: true,
                auto_gain_settle_chunks: 2,
                auto_gain_target_peak: 0.15,
                auto_gain_max: 10.0,
            },
        );
        assert_eq!(state.total, 160);
        assert_eq!(state.speech_samples, 160);
        let expected = compute_utterance_rms_threshold(0.001, 0.008, 1.8);
        assert!((state.utterance_rms_threshold - expected).abs() < 1e-6);
    }

    #[test]
    fn start_gate_rearm() {
        assert_eq!(
            evaluate_start_gate(
                false,
                true,
                true,
                std::time::Duration::from_millis(100),
                false
            ),
            StartGate::RearmGrace
        );
        assert_eq!(
            evaluate_start_gate(
                false,
                true,
                true,
                std::time::Duration::from_millis(400),
                false
            ),
            StartGate::Allow
        );
    }

    #[test]
    fn recording_status_strings() {
        assert_eq!(
            recording_status(true, true, false, false).as_str(),
            "error:asr_disabled"
        );
        assert_eq!(
            recording_status(false, false, false, false).as_str(),
            "error:asr_thread_dead"
        );
        assert_eq!(
            recording_status(false, true, true, false).as_str(),
            "recording"
        );
        assert_eq!(
            recording_status(false, true, false, true).as_str(),
            "processing"
        );
        assert_eq!(recording_status(false, true, false, false).as_str(), "idle");
    }
}
