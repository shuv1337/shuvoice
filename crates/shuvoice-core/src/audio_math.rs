//! Audio math and gain / noise-floor policy helpers.

use crate::utterance::UtteranceState;

/// Return RMS for a mono audio chunk.
pub fn audio_rms(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = audio.iter().map(|s| s * s).sum();
    (sum_sq / audio.len() as f32).sqrt()
}

/// Apply utterance gain with clip. No-op (returns clone of input identity via Cow-like choice)
/// when gain <= 1.05 or audio empty — callers that need pointer identity can check gain first.
pub fn apply_utterance_gain(audio: &[f32], gain: f32) -> Vec<f32> {
    if gain <= 1.05 || audio.is_empty() {
        return audio.to_vec();
    }
    audio.iter().map(|s| (s * gain).clamp(-1.0, 1.0)).collect()
}

/// True when gain application is a no-op under Python rules.
pub fn utterance_gain_is_noop(gain: f32) -> bool {
    gain <= 1.05
}

/// EMA noise-floor update: `0.98 * floor + 0.02 * chunk_rms` (seed on first positive sample).
pub fn update_noise_floor(noise_floor_rms: f32, chunk_rms: f32) -> f32 {
    if chunk_rms <= 0.0 {
        return noise_floor_rms;
    }
    if noise_floor_rms <= 0.0 {
        chunk_rms
    } else {
        0.98 * noise_floor_rms + 0.02 * chunk_rms
    }
}

/// Compute per-utterance RMS threshold used at recording start.
///
/// `threshold = max(speech_floor, min(noise*mult, speech_floor*3))` when speech_floor > 0,
/// else `max(speech_floor, noise*mult)`.
pub fn compute_utterance_rms_threshold(
    noise_floor_rms: f32,
    speech_rms_threshold: f32,
    speech_rms_multiplier: f32,
) -> f32 {
    let mut dynamic_threshold = noise_floor_rms * speech_rms_multiplier;
    if speech_rms_threshold > 0.0 {
        dynamic_threshold = dynamic_threshold.min(speech_rms_threshold * 3.0);
    }
    speech_rms_threshold.max(dynamic_threshold)
}

/// Update peak / speech counters and optional auto-gain on a newly appended chunk.
pub fn observe_recording_chunk(
    state: &mut UtteranceState,
    chunk: &[f32],
    wants_raw_audio: bool,
    auto_gain_settle_chunks: usize,
    auto_gain_target_peak: f32,
    auto_gain_max: f32,
) {
    state.add_chunk(chunk);
    let chunk_rms = audio_rms(chunk);
    state.last_chunk_rms = chunk_rms;
    state.peak_rms = state.peak_rms.max(chunk_rms);

    if chunk_rms >= state.utterance_rms_threshold {
        state.speech_samples += chunk.len();
        state.speech_chunks_seen += 1;
    }

    if wants_raw_audio {
        return;
    }
    if state.speech_chunks_seen < auto_gain_settle_chunks.max(1) {
        return;
    }
    if state.peak_rms > 0.003 {
        state.utterance_gain = (auto_gain_target_peak / state.peak_rms).min(auto_gain_max);
    }
}

/// Select trailing preroll chunks totaling at most `max_samples`.
pub fn select_preroll_chunks(chunks: &[Vec<f32>], max_samples: usize) -> Vec<Vec<f32>> {
    if max_samples == 0 || chunks.is_empty() {
        return Vec::new();
    }
    let mut preroll: Vec<Vec<f32>> = Vec::new();
    let mut remaining = max_samples;
    for chunk in chunks.iter().rev() {
        if remaining == 0 {
            break;
        }
        if chunk.len() <= remaining {
            preroll.push(chunk.clone());
            remaining -= chunk.len();
        } else {
            let start = chunk.len() - remaining;
            preroll.push(chunk[start..].to_vec());
            remaining = 0;
        }
    }
    preroll.reverse();
    preroll
}

/// Convert milliseconds to sample count at `sample_rate`.
pub fn ms_to_samples(sample_rate: u32, ms: u32) -> usize {
    sample_rate as usize * ms as usize / 1000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_utterance_gain_scales_and_clips() {
        let audio = [0.2_f32, -0.5, 0.9];
        let out = apply_utterance_gain(&audio, 2.0);
        assert!((out[0] - 0.4).abs() < 1e-6);
        assert!((out[1] + 1.0).abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn apply_utterance_gain_noop_for_small_gain() {
        let audio = [0.2_f32, -0.5];
        let out = apply_utterance_gain(&audio, 1.01);
        assert_eq!(out, audio);
        assert!(utterance_gain_is_noop(1.01));
    }

    #[test]
    fn begin_threshold_caps_inflated_dynamic_noise_gate() {
        let thr = compute_utterance_rms_threshold(0.150, 0.008, 1.8);
        assert!((thr - 0.024).abs() < 1e-6);
    }

    #[test]
    fn preroll_keeps_trailing_window() {
        let chunks = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0], vec![6.0]];
        let selected = select_preroll_chunks(&chunks, 4);
        assert_eq!(selected, vec![vec![3.0], vec![4.0, 5.0], vec![6.0]]);
    }

    #[test]
    fn rms_of_known_vector() {
        let audio = [0.0_f32, 1.0, 0.0, -1.0];
        let rms = audio_rms(&audio);
        assert!((rms - (0.5_f32).sqrt()).abs() < 1e-6);
    }
}
