//! PCM s16le helpers for the playback path.

/// Convert raw bytes + carry into i16 mono samples, returning leftover odd byte.
pub fn chunk_to_samples(raw_chunk: &[u8], carry: &[u8]) -> (Vec<i16>, Vec<u8>) {
    let mut chunk = Vec::with_capacity(carry.len() + raw_chunk.len());
    chunk.extend_from_slice(carry);
    chunk.extend_from_slice(raw_chunk);

    let usable_len = chunk.len() - (chunk.len() % 2);
    if usable_len == 0 {
        return (Vec::new(), chunk);
    }
    let (usable, next_carry) = chunk.split_at(usable_len);
    let mut samples = Vec::with_capacity(usable_len / 2);
    for pair in usable.chunks_exact(2) {
        samples.push(i16::from_le_bytes([pair[0], pair[1]]));
    }
    (samples, next_carry.to_vec())
}

/// Parse `pcm_24000` style format strings.
pub fn parse_sample_rate(output_format: &str) -> u32 {
    let text = output_format.trim().to_ascii_lowercase();
    if let Some(rest) = text.strip_prefix("pcm_")
        && let Ok(rate) = rest.parse::<u32>()
        && rate > 0
    {
        return rate;
    }
    24_000
}

/// Linear-interpolation resampler (mono i16). Runs on the playback OS thread,
/// never inside a CPAL realtime callback.
pub fn resample_linear_i16(input: &[i16], from_hz: u32, to_hz: u32) -> Vec<i16> {
    if input.is_empty() || from_hz == 0 || to_hz == 0 {
        return Vec::new();
    }
    if from_hz == to_hz {
        return input.to_vec();
    }
    let out_len = ((input.len() as u64) * u64::from(to_hz) / u64::from(from_hz)).max(1) as usize;
    let mut out = Vec::with_capacity(out_len);
    let scale = f64::from(from_hz) / f64::from(to_hz);
    for i in 0..out_len {
        let src = i as f64 * scale;
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(input.len().saturating_sub(1));
        let frac = src - i0 as f64;
        let s0 = f64::from(input[i0.min(input.len() - 1)]);
        let s1 = f64::from(input[i1]);
        let v = s0 + (s1 - s0) * frac;
        out.push(v.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_byte_carry() {
        let (samples, carry) = chunk_to_samples(&[0x01, 0x00, 0x02], &[]);
        assert_eq!(samples, vec![1]);
        assert_eq!(carry, vec![0x02]);
        let (samples2, carry2) = chunk_to_samples(&[0x00, 0x03, 0x00], &carry);
        assert_eq!(samples2, vec![2, 3]);
        assert!(carry2.is_empty());
    }

    #[test]
    fn valid_i16_minus_one_is_not_rejected() {
        // Historical bug: 0xFF 0xFF (i16 -1) looks like MPEG sync; backend encoding is authority.
        let (samples, carry) = chunk_to_samples(&(-1i16).to_le_bytes(), &[]);
        assert_eq!(samples, vec![-1]);
        assert!(carry.is_empty());
    }

    #[test]
    fn resample_identity() {
        let v = vec![1, 2, 3, 4];
        assert_eq!(resample_linear_i16(&v, 24_000, 24_000), v);
    }

    #[test]
    fn resample_doubles_length_approx() {
        let v = vec![0i16; 100];
        let out = resample_linear_i16(&v, 24_000, 48_000);
        assert!((out.len() as i32 - 200).abs() <= 1);
    }
}
