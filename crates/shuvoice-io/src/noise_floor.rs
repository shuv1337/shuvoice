//! Idle noise-floor EMA used for speech gating / gain thresholds.

/// Exponential moving average noise floor tracker.
#[derive(Debug, Clone, Default)]
pub struct NoiseFloor {
    pub rms: f32,
}

impl NoiseFloor {
    pub fn update(&mut self, chunk_rms: f32) {
        if chunk_rms <= 0.0 {
            return;
        }
        if self.rms <= 0.0 {
            self.rms = chunk_rms;
        } else {
            self.rms = 0.98 * self.rms + 0.02 * chunk_rms;
        }
    }

    /// Compute utterance RMS threshold (Python `begin_utterance` parity).
    #[must_use]
    pub fn utterance_threshold(
        &self,
        silence_rms_threshold: f32,
        silence_rms_multiplier: f32,
    ) -> f32 {
        let mut dynamic = self.rms * silence_rms_multiplier;
        if silence_rms_threshold > 0.0 {
            dynamic = dynamic.min(silence_rms_threshold * 3.0);
        }
        silence_rms_threshold.max(dynamic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_and_threshold() {
        let mut n = NoiseFloor::default();
        n.update(0.01);
        assert!((n.rms - 0.01).abs() < 1e-6);
        n.update(0.0); // ignored
        assert!((n.rms - 0.01).abs() < 1e-6);
        let thr = n.utterance_threshold(0.008, 1.8);
        assert!(thr >= 0.008);
    }
}
