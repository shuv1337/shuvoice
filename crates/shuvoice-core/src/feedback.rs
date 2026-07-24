//! Audio feedback tone synthesis (samples only; no device I/O).

/// Generate a mono sine-wave tone with a tiny edge fade.
pub fn generate_tone(freq: f64, duration_ms: u32, volume: f64, sample_rate: u32) -> Vec<f32> {
    let duration_ms = duration_ms.max(1);
    let sample_rate = sample_rate.max(1);
    let volume = volume.max(0.0);

    // Truncate fractional samples so duration never exceeds the requested bound.
    let sample_count = (sample_rate as usize * duration_ms as usize / 1000).max(1);

    let mut tone = Vec::with_capacity(sample_count);
    for i in 0..sample_count {
        let t = i as f64 / f64::from(sample_rate);
        let sample = (2.0 * std::f64::consts::PI * freq * t).sin() as f32;
        tone.push(sample);
    }

    let fade = 32usize.min(sample_count / 2);
    if fade > 0 {
        for i in 0..fade {
            let ramp = i as f32 / fade as f32;
            tone[i] *= ramp;
            tone[sample_count - 1 - i] *= ramp;
        }
    }

    for sample in &mut tone {
        *sample = (*sample * volume as f32).clamp(-1.0, 1.0);
    }
    tone
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_tone_length_and_amplitude() {
        let tone = generate_tone(440.0, 100, 0.25, 1000);
        assert_eq!(tone.len(), 100);
        assert!(tone.iter().all(|s| s.abs() <= 0.25 + 1e-5));
    }
}
