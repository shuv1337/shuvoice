//! TTS backend trait and provider implementations.

mod elevenlabs;
mod http_util;
mod kokoro;
mod melotts;
mod openai;
pub mod piper;

pub use elevenlabs::{ElevenLabsConfig, ElevenLabsTtsBackend};
pub use kokoro::{KokoroConfig, KokoroTtsBackend};
pub use melotts::{
    CHILD_ENV_ALLOWLIST, MeloTtsBackend, MeloTtsConfig, MeloWireMode, MeloWorkerSpawn,
    build_isolated_child_env,
};
pub use openai::{OpenAiConfig, OpenAiTtsBackend};
pub use piper::{PiperConfig, PiperTtsBackend, find_piper_binary, piper_sample_rate_from_sidecar};

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::Stream;

use crate::error::TtsError;
use crate::types::{AudioEncoding, BackendId, Capabilities, SynthesisRequest, VoiceInfo};

/// Stream of PCM s16le mono byte chunks.
pub type PcmStream = Pin<Box<dyn Stream<Item = Result<Bytes, TtsError>> + Send>>;

/// Backend synthesis result: authoritative encoding/rate + PCM byte stream.
///
/// The player trusts [`sample_rate_hz`] / [`encoding`] — it does **not** sniff
/// PCM payloads for container formats (avoids false positives on valid i16 data).
pub struct SynthesisStream {
    pub sample_rate_hz: u32,
    pub encoding: AudioEncoding,
    pub chunks: PcmStream,
}

/// Common runtime surface used by the ShuVoice TTS player.
#[async_trait]
pub trait TtsBackend: Send + Sync {
    fn id(&self) -> BackendId;

    fn capabilities(&self) -> Capabilities;

    /// Default / nominal PCM sample rate when not overridden per utterance.
    fn sample_rate_hz(&self) -> u32;

    /// Return missing dependency/runtime errors for this backend.
    fn dependency_errors(&self) -> Vec<String>;

    /// Start synthesis. Yields **PCM s16le mono** chunks with authoritative rate.
    ///
    /// If a provider returns MP3, decode before yielding and set
    /// [`SynthesisStream::sample_rate_hz`] to the **decoded** rate.
    async fn synthesize_stream(
        &self,
        request: SynthesisRequest,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<SynthesisStream, TtsError>;

    /// Return available voices for UI selectors.
    async fn list_voices(&self) -> Result<Vec<VoiceInfo>, TtsError>;
}

/// Shared backend handle.
pub type SharedBackend = Arc<dyn TtsBackend>;

/// Validate non-empty text and enforce max length in **Unicode scalar values**
/// (Rust `char`s), matching user-facing character limits.
pub(crate) fn ensure_text(text: &str, max_chars: usize) -> Result<String, TtsError> {
    let text_value = text.trim();
    if text_value.is_empty() {
        return Err(TtsError::EmptyText);
    }
    let char_count = text_value.chars().count();
    if char_count > max_chars {
        return Err(TtsError::TextTooLong {
            len: char_count,
            max: max_chars,
        });
    }
    Ok(text_value.to_string())
}

pub(crate) fn parse_pcm_sample_rate(output_format: &str, default: u32) -> u32 {
    let text = output_format.trim().to_ascii_lowercase();
    if let Some(rest) = text.strip_prefix("pcm_")
        && let Ok(rate) = rest.parse::<u32>()
        && rate > 0
    {
        return rate;
    }
    default
}

pub(crate) fn positive_finite_speed(speed: f64, provider: &str) -> Result<f64, TtsError> {
    if !speed.is_finite() || speed <= 0.0 {
        return Err(TtsError::speed_apply(format!(
            "{provider} speed must be a positive finite number"
        )));
    }
    Ok((speed * 100.0).round() / 100.0)
}

/// Redact URLs, absolute paths, and noisy stderr for user-visible event messages.
pub fn redact_for_ui(message: &str) -> String {
    let mut out = message.to_string();
    // URLs
    if let Ok(re) = regex_lite_url() {
        out = re.replace_all(&out, "[redacted-url]").into_owned();
    }
    // Absolute unix paths
    out = redact_paths(&out);
    // Collapse whitespace / cap length
    let collapsed = out.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX: usize = 240;
    if collapsed.chars().count() > MAX {
        let trimmed: String = collapsed.chars().take(MAX.saturating_sub(1)).collect();
        format!("{trimmed}…")
    } else {
        collapsed
    }
}

fn redact_paths(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '/' {
            // consume path-ish run
            out.push_str("[redacted-path]");
            while let Some(&n) = chars.peek() {
                if n.is_whitespace() || n == '\'' || n == '"' || n == ',' || n == ')' || n == ']' {
                    break;
                }
                chars.next();
            }
        } else if c == '~' && chars.peek() == Some(&'/') {
            out.push_str("[redacted-path]");
            chars.next(); // '/'
            while let Some(&n) = chars.peek() {
                if n.is_whitespace() || n == '\'' || n == '"' || n == ',' || n == ')' || n == ']' {
                    break;
                }
                chars.next();
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn regex_lite_url() -> Result<RegexUrl, ()> {
    Ok(RegexUrl)
}

/// Minimal URL redactor without pulling the `regex` crate (not in tts deps).
struct RegexUrl;
impl RegexUrl {
    fn replace_all<'a>(&self, hay: &'a str, rep: &str) -> std::borrow::Cow<'a, str> {
        let mut out = String::new();
        let bytes = hay.as_bytes();
        let mut i = 0;
        let mut last = 0;
        while i + 8 <= bytes.len() {
            let is_http = bytes[i..].starts_with(b"http://") || bytes[i..].starts_with(b"https://");
            if is_http {
                out.push_str(&hay[last..i]);
                out.push_str(rep);
                i += 7;
                while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                last = i;
            } else {
                i += 1;
            }
        }
        if last == 0 {
            return std::borrow::Cow::Borrowed(hay);
        }
        out.push_str(&hay[last..]);
        std::borrow::Cow::Owned(out)
    }
}

#[cfg(test)]
mod tests {
    use super::ensure_text;
    use crate::error::TtsError;

    #[test]
    fn ensure_text_counts_unicode_scalars_not_bytes() {
        // 3 scalars, 6 bytes UTF-8
        let s = "ééé";
        assert_eq!(s.len(), 6);
        assert_eq!(s.chars().count(), 3);
        assert!(ensure_text(s, 3).is_ok());
        let err = ensure_text(s, 2).unwrap_err();
        match err {
            TtsError::TextTooLong { len, max } => {
                assert_eq!(len, 3);
                assert_eq!(max, 2);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn redact_strips_url() {
        let r = super::redact_for_ui("err at https://api.example/v1/x path /tmp/a.onnx");
        assert!(!r.contains("api.example"));
        assert!(!r.contains("/tmp/"));
    }
}
