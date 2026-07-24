//! Streaming health heuristics for ASR loop stability.

/// Return true when the stream appears stalled despite active speech.
pub fn should_trigger_stall_flush(
    unchanged_steps: usize,
    chunk_rms: f32,
    utterance_threshold: f32,
    stall_chunks: usize,
    stall_rms_ratio: f32,
) -> bool {
    if unchanged_steps < stall_chunks.max(1) {
        return false;
    }
    let rms_ratio = stall_rms_ratio.max(0.0);
    if utterance_threshold <= 0.0 {
        return chunk_rms > 0.0;
    }
    chunk_rms >= (utterance_threshold * rms_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn false_before_chunk_limit() {
        assert!(!should_trigger_stall_flush(2, 0.12, 0.05, 4, 0.7));
    }

    #[test]
    fn false_when_rms_low() {
        assert!(!should_trigger_stall_flush(5, 0.02, 0.05, 4, 0.7));
    }

    #[test]
    fn true_when_stalled_and_active_speech() {
        assert!(should_trigger_stall_flush(5, 0.05, 0.05, 4, 0.7));
    }

    #[test]
    fn threshold_zero_uses_positive_rms() {
        assert!(should_trigger_stall_flush(4, 0.001, 0.0, 4, 0.7));
    }
}
