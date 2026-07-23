//! Audio capture helpers and optional cpal backend.

mod gain;
mod preroll;
mod resample;
mod rms;

pub use gain::{UtteranceGainConfig, UtteranceGainTracker, apply_utterance_gain};
pub use preroll::PrerollBuffer;
pub use resample::{IntegerDecimator, integer_ratio};
pub use rms::audio_rms;

/// Device preference helper shared by host backends.
#[must_use]
pub fn prefer_pulse_pipewire_index<'a, I>(devices: I) -> Option<usize>
where
    I: IntoIterator<Item = (usize, &'a str, u16)>,
{
    let mut pipewire_idx = None;
    for (idx, name, max_input_channels) in devices {
        if max_input_channels == 0 {
            continue;
        }
        let lower = name.to_ascii_lowercase();
        if lower == "pulse" || lower.starts_with("pulse ") {
            return Some(idx);
        }
        if pipewire_idx.is_none() && lower.contains("pipewire") {
            pipewire_idx = Some(idx);
        }
    }
    pipewire_idx
}

/// Capture configuration.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub chunk_samples: usize,
    pub fallback_sample_rate: u32,
    pub input_gain: f32,
    pub queue_max_size: usize,
    /// Host-specific device name/id hint (optional).
    pub device_name: Option<String>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            chunk_samples: 1_600,
            fallback_sample_rate: 48_000,
            input_gain: 1.0,
            queue_max_size: 200,
            device_name: None,
        }
    }
}

/// Bounded drop-oldest audio queue (pure, host-agnostic).
#[derive(Debug)]
pub struct AudioQueue {
    max_size: usize,
    chunks: std::collections::VecDeque<Vec<f32>>,
    pub dropped_chunks: u64,
}

impl AudioQueue {
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size: max_size.max(1),
            chunks: std::collections::VecDeque::new(),
            dropped_chunks: 0,
        }
    }

    pub fn push(&mut self, chunk: Vec<f32>) {
        if self.chunks.len() >= self.max_size {
            let _ = self.chunks.pop_front();
            self.dropped_chunks += 1;
        }
        self.chunks.push_back(chunk);
    }

    pub fn pop(&mut self) -> Option<Vec<f32>> {
        self.chunks.pop_front()
    }

    pub fn drain(&mut self) -> Vec<Vec<f32>> {
        self.chunks.drain(..).collect()
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    #[must_use]
    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

/// Apply stream-side input gain with clip (in-place).
pub fn apply_input_gain(samples: &mut [f32], gain: f32) {
    if (gain - 1.0).abs() < f32::EPSILON {
        return;
    }
    for s in samples.iter_mut() {
        *s = (*s * gain).clamp(-1.0, 1.0);
    }
}

#[cfg(feature = "audio")]
mod cpal_source;

#[cfg(feature = "audio")]
pub use cpal_source::CpalAudioCapture;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_overflow_drops_oldest() {
        let mut q = AudioQueue::new(2);
        q.push(vec![0.1]);
        q.push(vec![0.2]);
        q.push(vec![0.3]);
        assert_eq!(q.dropped_chunks, 1);
        let drained = q.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0], vec![0.2]);
        assert_eq!(drained[1], vec![0.3]);
    }

    #[test]
    fn prefers_pulse_over_pipewire() {
        let devices = [(0, "hw:0", 2u16), (1, "pulse", 2u16), (2, "pipewire", 2u16)];
        assert_eq!(prefer_pulse_pipewire_index(devices), Some(1));
    }

    #[test]
    fn falls_back_to_pipewire() {
        let devices = [(0, "hw:0", 2u16), (1, "pipewire", 2u16)];
        assert_eq!(prefer_pulse_pipewire_index(devices), Some(1));
    }
}
