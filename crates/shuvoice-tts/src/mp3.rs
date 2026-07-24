//! MP3 → PCM s16le mono decode via symphonia.
//!
//! Production player path expects PCM. Backends that receive MP3 (e.g. Kokoro
//! with `tts_output_format = "mp3"`) must decode before yielding, or the player
//! will reject non-PCM payloads via [`crate::player::pcm`].

use std::io::Cursor;

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::sample::Sample;

use crate::error::TtsError;

/// Decoded PCM plus the sample rate reported by the container/codec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedPcm {
    pub sample_rate_hz: u32,
    pub samples: Vec<i16>,
}

/// Decode a complete MP3 blob into mono s16le PCM samples.
pub fn decode_mp3_to_pcm(data: &[u8]) -> Result<DecodedPcm, TtsError> {
    if data.is_empty() {
        return Err(TtsError::decode("MP3 payload is empty"));
    }

    let cursor = Cursor::new(data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("mp3");

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|err| TtsError::decode(format!("failed to probe MP3: {err}")))?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| TtsError::decode("no decodable audio track in MP3 payload"))?
        .clone();

    let sample_rate_hz = track
        .codec_params
        .sample_rate
        .ok_or_else(|| TtsError::decode("MP3 track missing sample rate"))?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|err| TtsError::decode(format!("failed to create MP3 decoder: {err}")))?;

    let track_id = track.id;
    let mut samples = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::IoError(err)) => {
                return Err(TtsError::decode(format!("MP3 read failed: {err}")));
            }
            Err(err) => {
                let msg = err.to_string();
                if msg.contains("end of stream") || msg.contains("eof") {
                    break;
                }
                return Err(TtsError::decode(format!("MP3 demux failed: {err}")));
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => return Err(TtsError::decode(format!("MP3 decode failed: {err}"))),
        };

        append_mono_i16(&decoded, &mut samples)?;
    }

    if samples.is_empty() {
        return Err(TtsError::decode("MP3 decoded to zero samples"));
    }

    Ok(DecodedPcm {
        sample_rate_hz,
        samples,
    })
}

/// Convert decoded PCM samples to little-endian byte chunks suitable for the player queue.
pub fn pcm_samples_to_le_bytes(samples: &[i16]) -> bytes::Bytes {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    bytes::Bytes::from(out)
}

fn append_mono_i16(buffer: &AudioBufferRef<'_>, out: &mut Vec<i16>) -> Result<(), TtsError> {
    match buffer {
        AudioBufferRef::U8(buf) => mix_planes(buf, out, |s| u_int_to_i16(u32::from(s), 8)),
        AudioBufferRef::U16(buf) => mix_planes(buf, out, |s| u_int_to_i16(u32::from(s), 16)),
        AudioBufferRef::U24(buf) => mix_planes(buf, out, |s| u_int_to_i16(s.inner(), 24)),
        AudioBufferRef::U32(buf) => mix_planes(buf, out, |s| u_int_to_i16(s, 32)),
        AudioBufferRef::S8(buf) => mix_planes(buf, out, |s| i_int_to_i16(i32::from(s), 8)),
        AudioBufferRef::S16(buf) => mix_planes(buf, out, |s| s),
        AudioBufferRef::S24(buf) => mix_planes(buf, out, |s| i_int_to_i16(s.inner(), 24)),
        AudioBufferRef::S32(buf) => mix_planes(buf, out, |s| i_int_to_i16(s, 32)),
        AudioBufferRef::F32(buf) => mix_planes(buf, out, float_to_i16),
        AudioBufferRef::F64(buf) => mix_planes(buf, out, |s| float_to_i16(s as f32)),
    }
}

fn mix_planes<S, F>(
    buf: &symphonia::core::audio::AudioBuffer<S>,
    out: &mut Vec<i16>,
    mut convert: F,
) -> Result<(), TtsError>
where
    S: Sample + Copy,
    F: FnMut(S) -> i16,
{
    let channels = buf.spec().channels.count().max(1);
    let frames = buf.frames();
    out.reserve(frames);
    for frame in 0..frames {
        // Average channels in i32 to avoid overflow, then clamp.
        let mut acc = 0i32;
        for ch in 0..channels {
            acc += i32::from(convert(buf.chan(ch)[frame]));
        }
        let mono = acc / channels as i32;
        out.push(mono.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16);
    }
    Ok(())
}

fn float_to_i16(sample: f32) -> i16 {
    let clamped = sample.clamp(-1.0, 1.0);
    if clamped < 0.0 {
        (clamped * 32768.0).round() as i16
    } else {
        (clamped * 32767.0).round() as i16
    }
}

fn i_int_to_i16(sample: i32, bits: u32) -> i16 {
    if bits <= 16 {
        return sample.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
    let shift = bits - 16;
    (sample >> shift) as i16
}

fn u_int_to_i16(sample: u32, bits: u32) -> i16 {
    let mid = 1u32 << (bits.saturating_sub(1).min(31));
    // Convert unsigned to centered signed, then scale to i16.
    let centered = sample as i64 - mid as i64;
    let scale = i64::from(i16::MAX);
    let denom = mid as i64;
    if denom == 0 {
        return 0;
    }
    ((centered * scale) / denom).clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
}

/// Reject MP3 as a player input format (strict mode).
pub fn reject_mp3_player_input() -> TtsError {
    TtsError::config(
        "MP3 playback is not accepted by the PCM player; decode to PCM s16le first \
         or set tts_output_format to pcm_24000",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mp3_rejected() {
        let err = decode_mp3_to_pcm(&[]).unwrap_err();
        assert!(matches!(err, TtsError::Decode(_)));
    }

    #[test]
    fn garbage_mp3_rejected() {
        let err = decode_mp3_to_pcm(b"not an mp3").unwrap_err();
        assert!(matches!(err, TtsError::Decode(_)));
    }

    #[test]
    fn reject_helper_message() {
        let err = reject_mp3_player_input();
        assert!(err.to_string().contains("MP3"));
    }

    #[test]
    fn float_conversion_endpoints() {
        assert_eq!(float_to_i16(0.0), 0);
        assert_eq!(float_to_i16(1.0), i16::MAX);
        assert_eq!(float_to_i16(-1.0), i16::MIN);
    }

    #[test]
    fn int_conversion_preserves_s16() {
        assert_eq!(i_int_to_i16(12_345, 16), 12_345);
        assert_eq!(i_int_to_i16(-12_345, 16), -12_345);
    }

    #[test]
    fn u16_midpoint_is_near_zero() {
        let mid = u_int_to_i16(32_768, 16);
        assert!(mid.abs() <= 1, "mid={mid}");
    }
}
