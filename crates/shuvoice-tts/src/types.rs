//! Shared TTS value types.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::speed::{TTS_PLAYBACK_SPEED_MAX, TTS_PLAYBACK_SPEED_MIN, normalize_tts_playback_speed};

/// Supported TTS backend identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendId {
    ElevenLabs,
    OpenAi,
    Local,
    MeloTts,
    Kokoro,
}

impl BackendId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ElevenLabs => "elevenlabs",
            Self::OpenAi => "openai",
            Self::Local => "local",
            Self::MeloTts => "melotts",
            Self::Kokoro => "kokoro",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "elevenlabs" => Some(Self::ElevenLabs),
            "openai" => Some(Self::OpenAi),
            "local" => Some(Self::Local),
            "melotts" => Some(Self::MeloTts),
            "kokoro" => Some(Self::Kokoro),
            _ => None,
        }
    }

    pub fn all() -> &'static [Self] {
        &[
            Self::ElevenLabs,
            Self::Kokoro,
            Self::Local,
            Self::MeloTts,
            Self::OpenAi,
        ]
    }
}

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Voice descriptor for UI selectors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
}

impl VoiceInfo {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
        }
    }

    pub fn with_description(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: description.into(),
        }
    }
}

/// Wire encoding of audio yielded by a backend after any required decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AudioEncoding {
    /// Little-endian signed 16-bit mono PCM.
    #[default]
    PcmS16Le,
}

/// Immutable per-utterance synthesis request.
///
/// Speed is captured here so the backend receives the exact speed chosen when
/// the utterance started. Playback must not rewrite PCM timing after synthesis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynthesisRequest {
    pub text: String,
    pub voice_id: String,
    pub model_id: String,
    pub playback_speed: f64,
}

impl SynthesisRequest {
    pub fn new(
        text: impl Into<String>,
        voice_id: impl Into<String>,
        model_id: impl Into<String>,
        playback_speed: f64,
    ) -> Self {
        Self {
            text: text.into(),
            voice_id: voice_id.into(),
            model_id: model_id.into(),
            playback_speed: normalize_tts_playback_speed(playback_speed),
        }
    }
}

/// Backend capability flags advertised to the player/UI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    pub supports_streaming: bool,
    pub supports_voice_list: bool,
    pub requires_api_key: bool,
    pub supports_speed_control: bool,
    pub speed_min: Option<f64>,
    pub speed_max: Option<f64>,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            supports_streaming: true,
            supports_voice_list: true,
            requires_api_key: false,
            supports_speed_control: false,
            speed_min: None,
            speed_max: None,
        }
    }
}

impl Capabilities {
    /// Effective speed bounds after intersecting with the global 0.5–2.0 range.
    pub fn speed_bounds(&self) -> Option<(f64, f64)> {
        if !self.supports_speed_control {
            return None;
        }
        let mut minimum = self
            .speed_min
            .map(|v| v.max(TTS_PLAYBACK_SPEED_MIN))
            .unwrap_or(TTS_PLAYBACK_SPEED_MIN);
        let mut maximum = self
            .speed_max
            .map(|v| v.min(TTS_PLAYBACK_SPEED_MAX))
            .unwrap_or(TTS_PLAYBACK_SPEED_MAX);
        if minimum > maximum {
            std::mem::swap(&mut minimum, &mut maximum);
        }
        Some((round2(minimum), round2(maximum)))
    }
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

/// Player transport states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PlayerState {
    #[default]
    Idle,
    Synthesizing,
    Playing,
    Paused,
    Error,
}

impl PlayerState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Synthesizing => "synthesizing",
            Self::Playing => "playing",
            Self::Paused => "paused",
            Self::Error => "error",
        }
    }

    pub fn is_active(self) -> bool {
        matches!(self, Self::Synthesizing | Self::Playing | Self::Paused)
    }
}

impl fmt::Display for PlayerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Wire-stable player status payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatusPayload {
    pub state: PlayerState,
    pub voice_id: String,
    pub model_id: String,
    pub text_len: usize,
    pub playback_speed: f64,
    pub selected_playback_speed: f64,
    pub active_request_speed: Option<f64>,
}

/// Event metadata attached to player state transitions.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EventInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_playback_speed: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synth_latency_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub playback_duration_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub speed_apply_failure: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_sample_rate_hz: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding: Option<AudioEncoding>,
}

/// Player state-change event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlayerEvent {
    pub state: PlayerState,
    pub info: EventInfo,
}

/// Default local voice and model identifiers.
pub const DEFAULT_LOCAL_TTS_VOICE_ID: &str = "default";
pub const DEFAULT_LOCAL_TTS_MODEL_ID: &str = "piper";
pub const DEFAULT_MELOTTS_VOICE_ID: &str = "EN-US";
pub const DEFAULT_MELOTTS_MODEL_ID: &str = "melotts";
pub const DEFAULT_KOKORO_TTS_VOICE_ID: &str = "af_heart";
pub const DEFAULT_KOKORO_TTS_MODEL_ID: &str = "kokoro";
pub const DEFAULT_KOKORO_TTS_BASE_URL: &str = "http://localhost:8880/v1";
pub const DEFAULT_ELEVENLABS_TTS_BASE_URL: &str = "https://api.elevenlabs.io/v1";
pub const DEFAULT_OPENAI_TTS_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_ELEVENLABS_TTS_VOICE_ID: &str = "zNsotODqUhvbJ5wMG7Ei";
pub const DEFAULT_ELEVENLABS_TTS_MODEL_ID: &str = "eleven_flash_v2_5";
pub const DEFAULT_OPENAI_TTS_VOICE_ID: &str = "onyx";
pub const DEFAULT_OPENAI_TTS_MODEL_ID: &str = "gpt-4o-mini-tts";

/// Voice cache TTL used by ElevenLabs and Kokoro (seconds).
pub const VOICE_CACHE_TTL_SECS: u64 = 300;
