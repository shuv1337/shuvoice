//! Audio feedback tone helpers.
//!
//! PCM synthesis is owned by `shuvoice-core`; this module adds UI/config wiring.

use shuvoice_core::config::Config;
pub use shuvoice_core::generate_tone;

/// Default start/stop feedback parameters (mirrors `Config` feedback fields).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeedbackConfig {
    pub enabled: bool,
    pub start_freq: u32,
    pub stop_freq: u32,
    pub duration_ms: u32,
    pub volume: f64,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        let cfg = Config::default();
        Self::from_config(&cfg)
    }
}

impl FeedbackConfig {
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            enabled: cfg.audio_feedback,
            start_freq: cfg.feedback_start_freq,
            stop_freq: cfg.feedback_stop_freq,
            duration_ms: cfg.feedback_duration_ms,
            volume: cfg.feedback_volume,
        }
    }

    pub fn tone_for_start(&self, sample_rate: u32) -> Option<Vec<f32>> {
        if !self.enabled {
            return None;
        }
        Some(generate_tone(
            f64::from(self.start_freq),
            self.duration_ms,
            self.volume,
            sample_rate,
        ))
    }

    pub fn tone_for_stop(&self, sample_rate: u32) -> Option<Vec<f32>> {
        if !self.enabled {
            return None;
        }
        Some(generate_tone(
            f64::from(self.stop_freq),
            self.duration_ms,
            self.volume,
            sample_rate,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_tone_length_and_amplitude() {
        // Match core integer sample-count policy: sr * ms / 1000.
        let tone = generate_tone(440.0, 10, 0.25, 10_000);
        assert_eq!(tone.len(), 100);
        let peak = tone.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        assert!(peak <= 0.25 + 1e-5);
    }

    #[test]
    fn generate_tone_respects_sample_rate() {
        let tone = generate_tone(880.0, 50, 0.08, 8000);
        assert_eq!(tone.len(), 400);
    }

    #[test]
    fn config_defaults_match_core() {
        let fb = FeedbackConfig::default();
        assert!(fb.enabled);
        assert_eq!(fb.start_freq, 880);
        assert_eq!(fb.stop_freq, 660);
        assert_eq!(fb.duration_ms, 70);
    }
}
