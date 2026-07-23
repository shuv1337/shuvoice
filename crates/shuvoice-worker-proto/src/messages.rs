//! JSON control-plane messages.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::ProtocolError;
use crate::frame::{Frame, FrameKind};
use crate::limits::PROTOCOL_VERSION;
use crate::manifest::WorkerManifest;

/// Correlates a request with its responses and binary data frames.
pub type RequestId = Uuid;

/// All version-1 JSON control messages.
///
/// Unknown future variants deserialize as [`ControlMessage::Unknown`] so older
/// clients/workers can ignore additive messages when appropriate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlMessage {
    /// Client → worker handshake opener.
    Hello(Hello),
    /// Worker → client successful handshake.
    HelloOk(HelloOk),
    /// Worker → client failed handshake.
    HelloErr(HelloErr),

    // ── ASR requests ───────────────────────────────────────────────
    Load(LoadRequest),
    Reset(IdRequest),
    /// Metadata for a following PCM binary frame with the same `request_id`.
    ProcessChunk(ProcessAudioRequest),
    /// Metadata for following PCM binary frame(s) with the same `request_id`.
    ProcessUtterance(ProcessAudioRequest),
    Finish(FinishRequest),
    Cancel(IdRequest),
    Close(CloseRequest),

    // ── TTS requests ───────────────────────────────────────────────
    Synthesize(SynthesizeRequest),
    ListVoices(IdRequest),

    // ── Responses / events ─────────────────────────────────────────
    Ack(Ack),
    Error(ErrorResponse),
    PartialTranscript(TranscriptEvent),
    FinalTranscript(TranscriptEvent),
    Progress(ProgressEvent),
    Voices(VoicesResponse),
    /// Marks the start of streamed TTS PCM binary frames for `request_id`.
    AudioStart(AudioStreamEvent),
    /// Marks the end of streamed TTS/ASR binary audio for `request_id`.
    AudioEnd(AudioStreamEvent),
    /// Unsolicited worker lifecycle event.
    Event(WorkerEvent),

    /// Forward-compatible catch-all for unknown `type` values.
    #[serde(other)]
    Unknown,
}

impl ControlMessage {
    /// Serialize into a JSON frame.
    pub fn to_frame(&self) -> Result<Frame, ProtocolError> {
        let payload = serde_json::to_vec(self)?;
        Frame::json_bytes(payload)
    }

    /// Parse a JSON frame payload into a control message.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, ProtocolError> {
        Ok(serde_json::from_slice(bytes)?)
    }

    /// Parse from a decoded frame, requiring [`FrameKind::Json`].
    pub fn from_frame(frame: &Frame) -> Result<Self, ProtocolError> {
        if frame.kind != FrameKind::Json {
            return Err(ProtocolError::UnexpectedMessage(
                "expected JSON control frame",
            ));
        }
        Self::from_slice(&frame.payload)
    }

    /// Request id carried by this message, when present.
    #[must_use]
    pub fn request_id(&self) -> Option<RequestId> {
        match self {
            Self::Load(m) => Some(m.request_id),
            Self::Reset(m) | Self::Cancel(m) | Self::ListVoices(m) => Some(m.request_id),
            Self::ProcessChunk(m) | Self::ProcessUtterance(m) => Some(m.request_id),
            Self::Finish(m) => Some(m.request_id),
            Self::Close(m) => m.request_id,
            Self::Synthesize(m) => Some(m.request_id),
            Self::Ack(m) => Some(m.request_id),
            Self::Error(m) => m.request_id,
            Self::PartialTranscript(m) | Self::FinalTranscript(m) => Some(m.request_id),
            Self::Progress(m) => Some(m.request_id),
            Self::Voices(m) => Some(m.request_id),
            Self::AudioStart(m) | Self::AudioEnd(m) => Some(m.request_id),
            Self::Event(m) => m.request_id,
            Self::Hello(_) | Self::HelloOk(_) | Self::HelloErr(_) | Self::Unknown => None,
        }
    }
}

// ── Handshake ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hello {
    pub protocol_version: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_version: Option<String>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Hello {
    #[must_use]
    pub fn new(client_name: impl Into<String>) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            client_name: Some(client_name.into()),
            client_version: None,
            extra: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HelloOk {
    pub protocol_version: u16,
    pub manifest: WorkerManifest,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HelloErr {
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Remote protocol version when rejecting for version mismatch (optional, additive).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_version: Option<u16>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

// ── Shared request envelopes ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdRequest {
    pub request_id: RequestId,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl IdRequest {
    #[must_use]
    pub fn new(request_id: RequestId) -> Self {
        Self {
            request_id,
            extra: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadRequest {
    pub request_id: RequestId,
    /// Backend-specific configuration object (model paths, providers, …).
    #[serde(default)]
    pub config: serde_json::Value,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PcmEncoding {
    #[default]
    F32Le,
    I16Le,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessAudioRequest {
    pub request_id: RequestId,
    pub sample_rate_hz: u32,
    #[serde(default = "default_channels")]
    pub channels: u16,
    #[serde(default)]
    pub encoding: PcmEncoding,
    /// When true, this is the final audio blob for the request (no `audio_end` needed).
    #[serde(default)]
    pub end: bool,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

fn default_channels() -> u16 {
    1
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinishRequest {
    pub request_id: RequestId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CloseRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<RequestId>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynthesizeRequest {
    pub request_id: RequestId,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_encoding: Option<PcmEncoding>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

// ── Responses ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ack {
    pub request_id: RequestId,
    /// Optional free-form result payload (e.g. empty object after reset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<RequestId>,
    pub code: String,
    pub message: String,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl ErrorResponse {
    pub fn into_protocol_error(self) -> ProtocolError {
        ProtocolError::Worker {
            code: self.code,
            message: self.message,
            request_id: self.request_id.map(|id| id.to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptEvent {
    pub request_id: RequestId,
    pub text: String,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgressEvent {
    pub request_id: RequestId,
    /// Download/load fraction in `0.0..=1.0`, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fraction: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceInfo {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locale: Option<String>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoicesResponse {
    pub request_id: RequestId,
    pub voices: Vec<VoiceInfo>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioStreamEvent {
    pub request_id: RequestId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channels: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding: Option<PcmEncoding>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerEvent {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<RequestId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}
