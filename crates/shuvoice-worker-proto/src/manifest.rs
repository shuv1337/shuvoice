//! Worker identity and capability advertisement.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Whether the worker exposes ASR, TTS, or both faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Asr,
    Tts,
    /// Reserved for multi-role workers; clients should inspect both capability blocks.
    Dual,
}

/// ASR-facing capability flags advertised in the handshake manifest.
///
/// Missing flags deserialize as `false` (safe defaults — never assume support).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AsrCapabilities {
    /// Backend can download/resolve models on load.
    #[serde(default)]
    pub supports_model_download: bool,
    /// Backend wants raw (ungained) float audio from the host.
    #[serde(default)]
    pub wants_raw_audio: bool,
    /// Supports streaming `process_chunk` calls.
    #[serde(default)]
    pub supports_streaming: bool,
    /// Supports one-shot `process_utterance` offline decode.
    #[serde(default)]
    pub supports_offline_utterance: bool,
    /// Supports mid-utterance cancel.
    #[serde(default)]
    pub supports_cancel: bool,
    /// Preferred input sample rate in Hz, when fixed by the runtime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_sample_rate_hz: Option<u32>,
    /// Native chunk size in samples, when the runtime has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_chunk_samples: Option<u32>,
    /// Forward-compatible extension bag.
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// TTS-facing capability flags advertised in the handshake manifest.
///
/// Missing flags deserialize as `false` (safe defaults).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TtsCapabilities {
    #[serde(default)]
    pub requires_api_key: bool,
    #[serde(default)]
    pub supports_native_speed: bool,
    /// True only when PCM can be emitted before the full utterance is synthesized.
    #[serde(default)]
    pub supports_streaming_audio: bool,
    #[serde(default)]
    pub supports_list_voices: bool,
    #[serde(default)]
    pub supports_cancel: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_sample_rate_hz: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_chars: Option<u32>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Handshake manifest describing the connected worker process.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerManifest {
    /// Stable backend identifier, e.g. `"sherpa"`, `"kokoro"`.
    pub backend_id: String,
    pub kind: BackendKind,
    /// Human/runtime version string for diagnostics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_version: Option<String>,
    /// Model name or path currently configured (may be filled after load).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Execution provider hint (`cpu`, `cuda`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asr: Option<AsrCapabilities>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tts: Option<TtsCapabilities>,
    /// Forward-compatible extension bag at the manifest root.
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl WorkerManifest {
    #[must_use]
    pub fn asr(backend_id: impl Into<String>, capabilities: AsrCapabilities) -> Self {
        Self {
            backend_id: backend_id.into(),
            kind: BackendKind::Asr,
            runtime_version: None,
            model: None,
            provider: None,
            asr: Some(capabilities),
            tts: None,
            extra: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn tts(backend_id: impl Into<String>, capabilities: TtsCapabilities) -> Self {
        Self {
            backend_id: backend_id.into(),
            kind: BackendKind::Tts,
            runtime_version: None,
            model: None,
            provider: None,
            asr: None,
            tts: Some(capabilities),
            extra: BTreeMap::new(),
        }
    }
}
