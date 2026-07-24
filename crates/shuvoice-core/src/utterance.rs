//! Pure utterance buffering state for ASR loop orchestration.

use crate::error::{CoreError, CoreResult};

/// Mutable utterance state container.
#[derive(Debug, Clone)]
pub struct UtteranceState {
    pub buffer: Vec<Vec<f32>>,
    pub total: usize,
    pub last_text: String,
    pub speech_samples: usize,
    pub speech_chunks_seen: usize,
    pub peak_rms: f32,
    pub utterance_gain: f32,
    pub utterance_rms_threshold: f32,
    pub unchanged_steps: usize,
    pub last_chunk_rms: f32,
}

impl Default for UtteranceState {
    fn default() -> Self {
        Self::new()
    }
}

impl UtteranceState {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            total: 0,
            last_text: String::new(),
            speech_samples: 0,
            speech_chunks_seen: 0,
            peak_rms: 0.0,
            utterance_gain: 1.0,
            utterance_rms_threshold: 0.0,
            unchanged_steps: 0,
            last_chunk_rms: 0.0,
        }
    }

    pub fn reset(&mut self, rms_threshold: f32) {
        self.buffer.clear();
        self.total = 0;
        self.last_text.clear();
        self.speech_samples = 0;
        self.speech_chunks_seen = 0;
        self.peak_rms = 0.0;
        self.utterance_gain = 1.0;
        self.utterance_rms_threshold = rms_threshold;
        self.unchanged_steps = 0;
        self.last_chunk_rms = 0.0;
    }

    pub fn add_chunk(&mut self, chunk: &[f32]) {
        if chunk.is_empty() {
            return;
        }
        self.buffer.push(chunk.to_vec());
        self.total += chunk.len();
    }

    /// Consume one native-sized chunk from the front of the buffer.
    ///
    /// Returns `(samples, has_more)` where `has_more` is true when remainder
    /// still contains at least one full native chunk.
    pub fn consume_native_chunk(&mut self, native: usize) -> CoreResult<(Vec<f32>, bool)> {
        if native == 0 {
            return Err(CoreError::validation("native must be >= 1".to_string()));
        }
        if self.buffer.is_empty() {
            return Ok((Vec::new(), false));
        }

        let audio_data = if self.buffer.len() == 1 && self.buffer[0].len() >= native {
            std::mem::take(&mut self.buffer[0])
        } else {
            let mut concat = Vec::with_capacity(self.total);
            for chunk in self.buffer.drain(..) {
                concat.extend_from_slice(&chunk);
            }
            concat
        };

        // Match Python: to_process = audio_data[:native]; remainder = audio_data[native:]
        // Callers only invoke when total >= native, so len >= native holds.
        let (to_process, remainder) = if audio_data.len() >= native {
            (audio_data[..native].to_vec(), audio_data[native..].to_vec())
        } else {
            (audio_data, Vec::new())
        };

        self.buffer = if remainder.is_empty() {
            Vec::new()
        } else {
            vec![remainder]
        };
        self.total = self.buffer.first().map(Vec::len).unwrap_or(0);
        let has_more = self.total >= native;
        Ok((to_process, has_more))
    }

    /// Concatenate all buffered samples without consuming.
    pub fn concatenated(&self) -> Vec<f32> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        if self.buffer.len() == 1 {
            return self.buffer[0].clone();
        }
        let mut out = Vec::with_capacity(self.total);
        for chunk in &self.buffer {
            out.extend_from_slice(chunk);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_clears_all_fields() {
        let mut state = UtteranceState {
            buffer: vec![vec![1.0, 2.0]],
            total: 2,
            last_text: "hello".into(),
            speech_samples: 5,
            speech_chunks_seen: 3,
            peak_rms: 0.5,
            utterance_gain: 2.0,
            utterance_rms_threshold: 0.1,
            unchanged_steps: 3,
            last_chunk_rms: 0.2,
        };
        state.reset(0.33);
        assert!(state.buffer.is_empty());
        assert_eq!(state.total, 0);
        assert_eq!(state.last_text, "");
        assert_eq!(state.speech_samples, 0);
        assert_eq!(state.speech_chunks_seen, 0);
        assert_eq!(state.peak_rms, 0.0);
        assert_eq!(state.utterance_gain, 1.0);
        assert_eq!(state.utterance_rms_threshold, 0.33);
        assert_eq!(state.unchanged_steps, 0);
        assert_eq!(state.last_chunk_rms, 0.0);
    }

    #[test]
    fn add_chunk_increments_total() {
        let mut state = UtteranceState::new();
        state.add_chunk(&[1.0, 2.0, 3.0]);
        state.add_chunk(&[4.0]);
        assert_eq!(state.total, 4);
        assert_eq!(state.buffer.len(), 2);
    }

    #[test]
    fn consume_native_chunk_returns_chunk_remainder_and_has_more_false() {
        let mut state = UtteranceState::new();
        state.buffer = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        state.total = 9;
        let (to_process, has_more) = state.consume_native_chunk(6).unwrap();
        assert_eq!(to_process, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(!has_more);
        assert_eq!(state.total, 3);
        assert_eq!(state.buffer.len(), 1);
        assert_eq!(state.buffer[0], vec![6.0, 7.0, 8.0]);
    }

    #[test]
    fn consume_native_chunk_returns_has_more_true_when_enough_remainder() {
        let mut state = UtteranceState::new();
        state.buffer = vec![(0..10).map(|v| v as f32).collect()];
        state.total = 10;
        let (to_process, has_more) = state.consume_native_chunk(4).unwrap();
        assert_eq!(to_process, vec![0.0, 1.0, 2.0, 3.0]);
        assert!(has_more);
        assert_eq!(state.total, 6);
    }
}
