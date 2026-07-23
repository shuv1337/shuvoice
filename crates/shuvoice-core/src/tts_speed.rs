//! TTS playback speed helpers.

use crate::error::{CoreError, CoreResult};

pub const TTS_PLAYBACK_SPEED_MIN: f64 = 0.5;
pub const TTS_PLAYBACK_SPEED_MAX: f64 = 2.0;
pub const TTS_PLAYBACK_SPEED_STEP: f64 = 0.1;
pub const TTS_PLAYBACK_SPEED_DEFAULT: f64 = 1.0;

fn ensure_finite(value: f64) -> CoreResult<f64> {
    if !value.is_finite() {
        return Err(CoreError::validation(
            "tts_playback_speed must be a finite number".to_string(),
        ));
    }
    Ok(value)
}

fn parse_speed_str(speed: &str) -> CoreResult<f64> {
    let text = speed.trim();
    let value: f64 = text
        .parse()
        .map_err(|_| CoreError::validation("tts_playback_speed must be a number".to_string()))?;
    ensure_finite(value)
}

/// Validate speed is inside the inclusive 0.5–2.0 range and round to 2 decimals.
pub fn validate_tts_playback_speed(speed: f64) -> CoreResult<f64> {
    let value = ensure_finite(speed)?;
    if !(TTS_PLAYBACK_SPEED_MIN..=TTS_PLAYBACK_SPEED_MAX).contains(&value) {
        return Err(CoreError::validation(format!(
            "tts_playback_speed must be between {TTS_PLAYBACK_SPEED_MIN:.1} and {TTS_PLAYBACK_SPEED_MAX:.1}"
        )));
    }
    Ok((value * 100.0).round() / 100.0)
}

/// Validate from string input (config/CLI).
pub fn validate_tts_playback_speed_str(speed: &str) -> CoreResult<f64> {
    validate_tts_playback_speed(parse_speed_str(speed)?)
}

/// Clamp speed into range and round to 2 decimals.
pub fn normalize_tts_playback_speed(speed: f64) -> f64 {
    if !speed.is_finite() {
        return TTS_PLAYBACK_SPEED_DEFAULT;
    }
    let clamped = speed.clamp(TTS_PLAYBACK_SPEED_MIN, TTS_PLAYBACK_SPEED_MAX);
    (clamped * 100.0).round() / 100.0
}

/// Alias kept for call-sites that want an explicit lossy name.
pub fn normalize_tts_playback_speed_lossy(speed: f64) -> f64 {
    normalize_tts_playback_speed(speed)
}

/// Step speed by fixed increments.
pub fn step_tts_playback_speed(speed: f64, steps: i32) -> f64 {
    let current = normalize_tts_playback_speed(speed);
    normalize_tts_playback_speed(current + f64::from(steps) * TTS_PLAYBACK_SPEED_STEP)
}

/// Human-readable multiplier label, e.g. `1.25×`.
pub fn format_tts_playback_speed(speed: f64) -> String {
    let value = normalize_tts_playback_speed(speed);
    let mut text = format!("{value:.2}");
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    if !text.contains('.') {
        text.push_str(".0");
    }
    format!("{text}×")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_in_range() {
        assert_eq!(validate_tts_playback_speed(1.0).unwrap(), 1.0);
        assert_eq!(validate_tts_playback_speed_str("1.25").unwrap(), 1.25);
    }

    #[test]
    fn validate_rejects_invalid() {
        assert!(validate_tts_playback_speed(0.49).is_err());
        assert!(validate_tts_playback_speed(2.01).is_err());
        assert!(validate_tts_playback_speed_str("fast").is_err());
        assert!(validate_tts_playback_speed(f64::INFINITY).is_err());
    }

    #[test]
    fn normalize_clamps() {
        assert_eq!(normalize_tts_playback_speed(0.1), TTS_PLAYBACK_SPEED_MIN);
        assert_eq!(normalize_tts_playback_speed(9.9), TTS_PLAYBACK_SPEED_MAX);
    }

    #[test]
    fn step_moves_by_increment() {
        assert_eq!(step_tts_playback_speed(1.0, 1), 1.1);
        assert_eq!(step_tts_playback_speed(1.0, -1), 0.9);
    }

    #[test]
    fn format_readable() {
        assert_eq!(format_tts_playback_speed(1.0), "1.0×");
        assert_eq!(format_tts_playback_speed(1.25), "1.25×");
    }
}
