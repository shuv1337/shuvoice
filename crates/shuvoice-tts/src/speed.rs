//! Shared TTS playback speed helpers.
//!
//! Thin facade over [`shuvoice_core::tts_speed`] so providers/player keep a
//! stable import path even if core evolves.

pub use shuvoice_core::tts_speed::{
    TTS_PLAYBACK_SPEED_DEFAULT, TTS_PLAYBACK_SPEED_MAX, TTS_PLAYBACK_SPEED_MIN,
    TTS_PLAYBACK_SPEED_STEP,
};

use crate::error::TtsError;

/// Validate speed is inside the inclusive 0.5–2.0 range (config load path).
pub fn validate_tts_playback_speed(speed: f64) -> Result<f64, TtsError> {
    shuvoice_core::tts_speed::validate_tts_playback_speed(speed)
        .map_err(|err| TtsError::config(err.to_string()))
}

/// Clamp speed into the supported range (runtime path).
pub fn normalize_tts_playback_speed(speed: f64) -> f64 {
    shuvoice_core::tts_speed::normalize_tts_playback_speed(speed)
}

/// Step speed by `steps * 0.1`, clamped.
pub fn step_tts_playback_speed(speed: f64, steps: i32) -> f64 {
    shuvoice_core::tts_speed::step_tts_playback_speed(speed, steps)
}

/// Human-readable multiplier text, e.g. `1.25×`.
pub fn format_tts_playback_speed(speed: f64) -> String {
    shuvoice_core::tts_speed::format_tts_playback_speed(speed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_in_range() {
        assert_eq!(validate_tts_playback_speed(0.5).unwrap(), 0.5);
        assert_eq!(validate_tts_playback_speed(1.0).unwrap(), 1.0);
        assert_eq!(validate_tts_playback_speed(1.25).unwrap(), 1.25);
        assert_eq!(validate_tts_playback_speed(2.0).unwrap(), 2.0);
    }

    #[test]
    fn validate_rejects_invalid() {
        assert!(validate_tts_playback_speed(0.49).is_err());
        assert!(validate_tts_playback_speed(2.01).is_err());
        assert!(validate_tts_playback_speed(f64::NAN).is_err());
        assert!(validate_tts_playback_speed(f64::INFINITY).is_err());
    }

    #[test]
    fn normalize_clamps() {
        assert_eq!(normalize_tts_playback_speed(0.1), 0.5);
        assert_eq!(normalize_tts_playback_speed(9.0), 2.0);
        assert_eq!(normalize_tts_playback_speed(f64::NAN), 1.0);
    }

    #[test]
    fn step_moves_by_fixed_increment() {
        assert_eq!(step_tts_playback_speed(1.0, 1), 1.1);
        assert_eq!(step_tts_playback_speed(1.0, -1), 0.9);
        assert_eq!(step_tts_playback_speed(0.5, -1), 0.5);
        assert_eq!(step_tts_playback_speed(2.0, 1), 2.0);
    }

    #[test]
    fn format_readable() {
        assert_eq!(format_tts_playback_speed(1.0), "1.0×");
        assert_eq!(format_tts_playback_speed(1.25), "1.25×");
        assert_eq!(format_tts_playback_speed(0.50), "0.5×");
    }
}
