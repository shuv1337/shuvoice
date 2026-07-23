//! Integer-ratio decimation with carry (48k → 16k etc.).

/// Downsample by an integer ratio using decimation, retaining a carry buffer.
#[derive(Debug, Default, Clone)]
pub struct IntegerDecimator {
    ratio: usize,
    carry: Vec<f32>,
}

impl IntegerDecimator {
    #[must_use]
    pub fn new(ratio: usize) -> Self {
        Self {
            ratio: ratio.max(1),
            carry: Vec::new(),
        }
    }

    #[must_use]
    pub fn ratio(&self) -> usize {
        self.ratio
    }

    pub fn clear(&mut self) {
        self.carry.clear();
    }

    /// Push samples and return decimated output (may be empty if not enough).
    pub fn push(&mut self, audio: &[f32]) -> Vec<f32> {
        if self.ratio <= 1 {
            return audio.to_vec();
        }

        let mut buf = std::mem::take(&mut self.carry);
        buf.extend_from_slice(audio);

        let usable = (buf.len() / self.ratio) * self.ratio;
        if usable == 0 {
            self.carry = buf;
            return Vec::new();
        }

        let out: Vec<f32> = buf[..usable].iter().step_by(self.ratio).copied().collect();
        self.carry = buf[usable..].to_vec();
        out
    }
}

/// Validate fallback/sample_rate relationship.
pub fn integer_ratio(fallback: u32, sample_rate: u32) -> Result<u32, crate::error::AudioError> {
    if sample_rate == 0 || !fallback.is_multiple_of(sample_rate) {
        return Err(crate::error::AudioError::NonIntegerResampleRatio {
            fallback,
            sample_rate,
        });
    }
    Ok(fallback / sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_samples_by_ratio() {
        let mut d = IntegerDecimator::new(3);
        let out = d.push(&[0.0, 1.0, 2.0, 9.0, 8.0, 7.0]);
        assert_eq!(out, vec![0.0, 9.0]);
    }

    #[test]
    fn keeps_carry_between_calls() {
        let mut d = IntegerDecimator::new(3);
        let out1 = d.push(&[1.0, 2.0]);
        assert!(out1.is_empty());
        let out2 = d.push(&[3.0, 5.0, 6.0, 7.0]);
        // carry [1,2] + [3,5,6,7] = [1,2,3,5,6,7] usable 6 → 1,5
        assert_eq!(out2, vec![1.0, 5.0]);
    }
}
