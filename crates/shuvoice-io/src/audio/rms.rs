//! RMS helper.

/// Return RMS for a mono audio chunk.
#[must_use]
pub fn audio_rms(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = audio.iter().map(|x| x * x).sum();
    (sum_sq / audio.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_edge_cases() {
        assert_eq!(audio_rms(&[]), 0.0);
        assert_eq!(audio_rms(&[0.0, 0.0, 0.0, 0.0]), 0.0);
        let known = [1.0f32, -1.0, 1.0, -1.0];
        assert!((audio_rms(&known) - 1.0).abs() < 1e-6);
    }
}
