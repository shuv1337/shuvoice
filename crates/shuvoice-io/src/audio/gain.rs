//! App-side utterance auto-gain.

/// Apply utterance gain when meaningfully above unity.
#[must_use]
pub fn apply_utterance_gain(audio: &[f32], gain: f32) -> Vec<f32> {
    if gain <= 1.05 || audio.is_empty() {
        return audio.to_vec();
    }
    audio.iter().map(|s| (s * gain).clamp(-1.0, 1.0)).collect()
}

/// Auto-gain configuration.
#[derive(Debug, Clone, Copy)]
pub struct UtteranceGainConfig {
    pub target_peak: f32,
    pub max_gain: f32,
    pub settle_chunks: u32,
    /// Peak RMS must exceed this before gain updates (Python: 0.003).
    pub peak_floor: f32,
}

impl Default for UtteranceGainConfig {
    fn default() -> Self {
        Self {
            target_peak: 0.15,
            max_gain: 10.0,
            settle_chunks: 2,
            peak_floor: 0.003,
        }
    }
}

/// Tracks per-utterance gain from speech-level chunks.
#[derive(Debug, Clone)]
pub struct UtteranceGainTracker {
    cfg: UtteranceGainConfig,
    pub peak_rms: f32,
    pub speech_chunks_seen: u32,
    pub speech_samples: usize,
    pub utterance_gain: f32,
    pub wants_raw_audio: bool,
}

impl UtteranceGainTracker {
    #[must_use]
    pub fn new(cfg: UtteranceGainConfig, wants_raw_audio: bool) -> Self {
        Self {
            cfg,
            peak_rms: 0.0,
            speech_chunks_seen: 0,
            speech_samples: 0,
            utterance_gain: 1.0,
            wants_raw_audio,
        }
    }

    pub fn reset(&mut self) {
        self.peak_rms = 0.0;
        self.speech_chunks_seen = 0;
        self.speech_samples = 0;
        self.utterance_gain = 1.0;
    }

    /// Observe a chunk. `is_speech` when chunk_rms >= utterance threshold.
    pub fn observe_chunk(&mut self, chunk_len: usize, chunk_rms: f32, is_speech: bool) {
        self.peak_rms = self.peak_rms.max(chunk_rms);
        if is_speech {
            self.speech_samples += chunk_len;
            self.speech_chunks_seen += 1;
        }
        if self.wants_raw_audio {
            return;
        }
        if self.speech_chunks_seen < self.cfg.settle_chunks {
            return;
        }
        if self.peak_rms > self.cfg.peak_floor {
            self.utterance_gain = (self.cfg.target_peak / self.peak_rms).min(self.cfg.max_gain);
        }
    }

    /// Apply current gain when appropriate.
    #[must_use]
    pub fn apply<'a>(&self, audio: &'a [f32]) -> std::borrow::Cow<'a, [f32]> {
        if self.wants_raw_audio {
            return std::borrow::Cow::Borrowed(audio);
        }
        if self.utterance_gain <= 1.05 {
            return std::borrow::Cow::Borrowed(audio);
        }
        std::borrow::Cow::Owned(apply_utterance_gain(audio, self.utterance_gain))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_and_clips() {
        let out = apply_utterance_gain(&[0.6, -0.6], 2.0);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn noop_for_small_gain() {
        let input = [0.25f32, -0.25];
        let out = apply_utterance_gain(&input, 1.01);
        assert_eq!(out, input);
    }

    #[test]
    fn settles_then_updates() {
        let mut t = UtteranceGainTracker::new(UtteranceGainConfig::default(), false);
        t.observe_chunk(1600, 0.02, true);
        assert_eq!(t.utterance_gain, 1.0); // not settled
        t.observe_chunk(1600, 0.02, true);
        assert!((t.utterance_gain - (0.15 / 0.02)).abs() < 1e-5);
    }

    #[test]
    fn bypasses_when_raw() {
        let mut t = UtteranceGainTracker::new(UtteranceGainConfig::default(), true);
        t.observe_chunk(1600, 0.02, true);
        t.observe_chunk(1600, 0.02, true);
        assert_eq!(t.utterance_gain, 1.0);
    }
}
