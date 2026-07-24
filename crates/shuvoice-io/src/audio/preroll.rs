//! Recording preroll buffer (trailing N samples).

/// Retains trailing audio around PTT start.
#[derive(Debug, Default, Clone)]
pub struct PrerollBuffer {
    chunks: Vec<Vec<f32>>,
}

impl PrerollBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge drained chunks and keep only the trailing `max_samples`.
    pub fn capture_from_drain(&mut self, drained: Vec<Vec<f32>>, max_samples: usize) {
        let mut chunks = std::mem::take(&mut self.chunks);
        chunks.extend(drained);

        if max_samples == 0 || chunks.is_empty() {
            self.chunks.clear();
            return;
        }

        let mut preroll = Vec::new();
        let mut remaining = max_samples;
        for chunk in chunks.into_iter().rev() {
            if remaining == 0 {
                break;
            }
            if chunk.len() <= remaining {
                remaining -= chunk.len();
                preroll.push(chunk);
            } else {
                let start = chunk.len() - remaining;
                preroll.push(chunk[start..].to_vec());
                remaining = 0;
            }
        }
        preroll.reverse();
        self.chunks = preroll;
    }

    /// Take and clear preroll chunks.
    pub fn take(&mut self) -> Vec<Vec<f32>> {
        std::mem::take(&mut self.chunks)
    }

    #[must_use]
    pub fn total_samples(&self) -> usize {
        self.chunks.iter().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_trailing_window() {
        let mut p = PrerollBuffer::new();
        p.capture_from_drain(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0, 7.0]], 5);
        assert_eq!(p.total_samples(), 5);
        let taken = p.take();
        let flat: Vec<f32> = taken.into_iter().flatten().collect();
        assert_eq!(flat, vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        assert!(p.take().is_empty());
    }
}
