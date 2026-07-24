//! Shared PCM helpers.

use base64::Engine;

/// Encode float32 mono PCM as little-endian bytes, base64 (worker / tests).
pub fn encode_f32_le_b64(samples: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Decode base64 little-endian float32 mono PCM.
pub fn decode_f32_le_b64(b64: &str) -> Result<Vec<f32>, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| e.to_string())?;
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "PCM byte length {} is not a multiple of 4",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// OpenAI Realtime: clamp float32 → pcm16 LE base64.
pub fn encode_pcm16_le_b64(samples: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let v = (clamped * 32767.0) as i16;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Truncate to trailing window when over cap (Sherpa offline safety).
///
/// `max_seconds <= 0` disables truncation.
pub fn truncate_trailing(samples: &[f32], sample_rate: u32, max_seconds: f64) -> &[f32] {
    if max_seconds <= 0.0 || sample_rate == 0 {
        return samples;
    }
    let max_samples = ((sample_rate as f64) * max_seconds).floor() as usize;
    if max_samples == 0 || samples.len() <= max_samples {
        samples
    } else {
        &samples[samples.len() - max_samples..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_f32_b64() {
        let src = vec![0.0, 0.5, -0.25, 1.0];
        let b64 = encode_f32_le_b64(&src);
        let back = decode_f32_le_b64(&b64).unwrap();
        assert_eq!(src, back);
    }

    #[test]
    fn pcm16_clamps() {
        let b64 = encode_pcm16_le_b64(&[2.0, -2.0, 0.0]);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .unwrap();
        assert_eq!(bytes.len(), 6);
        let s0 = i16::from_le_bytes([bytes[0], bytes[1]]);
        let s1 = i16::from_le_bytes([bytes[2], bytes[3]]);
        assert_eq!(s0, 32767);
        assert_eq!(s1, -32767);
    }

    #[test]
    fn truncate_trailing_window() {
        let samples: Vec<f32> = (0..16_000 * 3).map(|i| i as f32).collect();
        let out = truncate_trailing(&samples, 16_000, 1.0);
        assert_eq!(out.len(), 16_000);
        assert_eq!(out[0], (16_000 * 2) as f32);
    }

    #[test]
    fn truncate_disabled_when_zero() {
        let samples = vec![1.0; 100];
        assert_eq!(truncate_trailing(&samples, 16_000, 0.0).len(), 100);
    }
}
