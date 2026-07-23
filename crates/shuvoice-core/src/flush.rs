//! Streaming tail flush / noise policy helpers.

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Minimum RMS for tail-flush noise (pre-gain).
pub const FLUSH_NOISE_MIN_RMS: f32 = 0.005;
/// Escalation factor applied per stalled flush step.
pub const FLUSH_NOISE_ESCALATION: f32 = 1.4;
/// Maximum RMS for tail-flush noise.
pub const FLUSH_NOISE_MAX_RMS: f32 = 0.08;
/// Maximum silence flush steps after stop.
pub const FLUSH_TAIL_MAX_STEPS: usize = 20;
/// Stable steps required once text has been seen.
pub const FLUSH_TAIL_STABLE_REQUIRED: usize = 5;

/// Generate low-amplitude noise for flushing streaming transducers.
pub fn make_flush_noise<R: Rng + ?Sized>(
    n_samples: usize,
    noise_floor_rms: f32,
    escalation: f32,
    rng: &mut R,
) -> Vec<f32> {
    let base_rms = noise_floor_rms.max(FLUSH_NOISE_MIN_RMS);
    let rms = (base_rms * escalation).min(FLUSH_NOISE_MAX_RMS);
    let normal =
        Normal::new(0.0, f64::from(rms)).unwrap_or_else(|_| Normal::new(0.0, 0.005).unwrap());
    let mut out = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let sample = normal.sample(rng) as f32;
        out.push(sample.clamp(-1.0, 1.0));
    }
    out
}

/// Compute escalation multiplier for consecutive stalled flush steps.
pub fn flush_noise_escalation(stalled_consecutive: usize) -> f32 {
    FLUSH_NOISE_ESCALATION.powi(stalled_consecutive as i32)
}

/// Policy decision for whether tail flush should continue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailFlushDecision {
    Continue,
    StopStable,
    AbortNewRecording,
}

/// Evaluate tail-flush loop control for one step outcome.
pub fn evaluate_tail_flush_step(
    new_recording_started: bool,
    text_changed: bool,
    ever_had_text: bool,
    stable_steps: usize,
    step_index: usize,
) -> (
    TailFlushDecision,
    usize, /* next_stable */
    bool,  /* next_ever_had_text */
) {
    if new_recording_started {
        return (
            TailFlushDecision::AbortNewRecording,
            stable_steps,
            ever_had_text,
        );
    }
    if step_index >= FLUSH_TAIL_MAX_STEPS {
        return (TailFlushDecision::StopStable, stable_steps, ever_had_text);
    }
    if text_changed {
        return (TailFlushDecision::Continue, 0, true);
    }
    let next_stable = stable_steps + 1;
    let needed = if ever_had_text {
        FLUSH_TAIL_STABLE_REQUIRED
    } else {
        5
    };
    if next_stable >= needed {
        (TailFlushDecision::StopStable, next_stable, ever_had_text)
    } else {
        (TailFlushDecision::Continue, next_stable, ever_had_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn make_flush_noise_respects_bounds() {
        let mut rng = StdRng::seed_from_u64(1);
        let noise = make_flush_noise(4096, 0.01, 1.0, &mut rng);
        assert_eq!(noise.len(), 4096);
        assert!(noise.iter().all(|s| s.abs() <= 1.0));
    }

    #[test]
    fn escalation_grows() {
        assert!((flush_noise_escalation(0) - 1.0).abs() < 1e-6);
        assert!((flush_noise_escalation(1) - 1.4).abs() < 1e-6);
    }

    #[test]
    fn tail_flush_aborts_on_new_recording() {
        let (decision, _, _) = evaluate_tail_flush_step(true, false, true, 2, 1);
        assert_eq!(decision, TailFlushDecision::AbortNewRecording);
    }
}
