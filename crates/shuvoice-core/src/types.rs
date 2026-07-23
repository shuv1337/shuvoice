//! Domain enums and capability types shared across crates.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, CoreResult};

/// ASR backend registry keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsrBackendKind {
    Sherpa,
    Nemo,
    Moonshine,
    #[serde(rename = "openai_realtime")]
    OpenaiRealtime,
}

impl AsrBackendKind {
    pub const ALL: [Self; 4] = [
        Self::Sherpa,
        Self::Nemo,
        Self::Moonshine,
        Self::OpenaiRealtime,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sherpa => "sherpa",
            Self::Nemo => "nemo",
            Self::Moonshine => "moonshine",
            Self::OpenaiRealtime => "openai_realtime",
        }
    }
}

impl fmt::Display for AsrBackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for AsrBackendKind {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "sherpa" => Ok(Self::Sherpa),
            "nemo" => Ok(Self::Nemo),
            "moonshine" => Ok(Self::Moonshine),
            "openai_realtime" => Ok(Self::OpenaiRealtime),
            other => Err(CoreError::validation(format!(
                "asr_backend must be one of: nemo, sherpa, moonshine, openai_realtime (got {other})"
            ))),
        }
    }
}

/// TTS backend registry keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsBackendKind {
    Elevenlabs,
    Openai,
    Local,
    Melotts,
    Kokoro,
}

impl TtsBackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Elevenlabs => "elevenlabs",
            Self::Openai => "openai",
            Self::Local => "local",
            Self::Melotts => "melotts",
            Self::Kokoro => "kokoro",
        }
    }
}

impl fmt::Display for TtsBackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for TtsBackendKind {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "elevenlabs" => Ok(Self::Elevenlabs),
            "openai" => Ok(Self::Openai),
            "local" => Ok(Self::Local),
            "melotts" => Ok(Self::Melotts),
            "kokoro" => Ok(Self::Kokoro),
            other => Err(CoreError::validation(format!(
                "tts_backend must be one of: elevenlabs, openai, local, melotts, kokoro (got {other})"
            ))),
        }
    }
}

/// Configured Sherpa decode mode before auto-resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SherpaDecodeMode {
    #[default]
    Auto,
    Streaming,
    OfflineInstant,
}

impl SherpaDecodeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Streaming => "streaming",
            Self::OfflineInstant => "offline_instant",
        }
    }
}

impl fmt::Display for SherpaDecodeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SherpaDecodeMode {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "streaming" => Ok(Self::Streaming),
            "offline_instant" => Ok(Self::OfflineInstant),
            other => Err(CoreError::validation(format!(
                "sherpa_decode_mode must be one of: auto, streaming, offline_instant (got {other})"
            ))),
        }
    }
}

/// Effective Sherpa decode path after auto resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedSherpaDecodeMode {
    Streaming,
    OfflineInstant,
}

impl ResolvedSherpaDecodeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Streaming => "streaming",
            Self::OfflineInstant => "offline_instant",
        }
    }
}

impl fmt::Display for ResolvedSherpaDecodeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Execution provider for ONNX-style backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ComputeProvider {
    #[default]
    Cpu,
    Cuda,
}

impl ComputeProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

impl fmt::Display for ComputeProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ComputeProvider {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            other => Err(CoreError::validation(format!(
                "provider must be one of: cpu, cuda (got {other})"
            ))),
        }
    }
}

/// MeloTTS device selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MeloTtsDevice {
    #[default]
    Auto,
    Cpu,
    Cuda,
}

impl MeloTtsDevice {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

impl FromStr for MeloTtsDevice {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            other => Err(CoreError::validation(format!(
                "tts_melotts_device must be one of: auto, cpu, cuda (got {other})"
            ))),
        }
    }
}

/// Final text injection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum InjectionMode {
    #[default]
    Auto,
    Clipboard,
    Direct,
}

impl InjectionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Clipboard => "clipboard",
            Self::Direct => "direct",
        }
    }
}

impl fmt::Display for InjectionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for InjectionMode {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "clipboard" => Ok(Self::Clipboard),
            "direct" => Ok(Self::Direct),
            other => Err(CoreError::validation(format!(
                "typing_final_injection_mode must be one of: auto, clipboard, direct (got {other})"
            ))),
        }
    }
}

/// Typing case transform for committed STT text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TypingTextCase {
    #[default]
    Default,
    Lowercase,
}

impl TypingTextCase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Lowercase => "lowercase",
        }
    }
}

impl FromStr for TypingTextCase {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "default" => Ok(Self::Default),
            "lowercase" => Ok(Self::Lowercase),
            other => Err(CoreError::validation(format!(
                "typing_text_case must be one of: default, lowercase (got {other})"
            ))),
        }
    }
}

/// Overlay / typing output mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    #[default]
    FinalOnly,
    StreamingPartial,
}

impl OutputMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FinalOnly => "final_only",
            Self::StreamingPartial => "streaming_partial",
        }
    }
}

impl FromStr for OutputMode {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "final_only" => Ok(Self::FinalOnly),
            "streaming_partial" => Ok(Self::StreamingPartial),
            other => Err(CoreError::validation(format!(
                "output_mode must be one of: final_only, streaming_partial (got {other})"
            ))),
        }
    }
}

/// OpenAI realtime turn detection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OpenaiTurnDetection {
    #[default]
    Manual,
    ServerVad,
    SemanticVad,
}

impl OpenaiTurnDetection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::ServerVad => "server_vad",
            Self::SemanticVad => "semantic_vad",
        }
    }
}

impl FromStr for OpenaiTurnDetection {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "manual" => Ok(Self::Manual),
            "server_vad" => Ok(Self::ServerVad),
            "semantic_vad" => Ok(Self::SemanticVad),
            other => Err(CoreError::validation(format!(
                "openai_realtime_turn_detection must be one of: manual, server_vad, semantic_vad (got {other})"
            ))),
        }
    }
}

/// OpenAI realtime VAD eagerness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OpenaiVadEagerness {
    #[default]
    Auto,
    Low,
    Medium,
    High,
}

impl OpenaiVadEagerness {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

impl FromStr for OpenaiVadEagerness {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            other => Err(CoreError::validation(format!(
                "openai_realtime_vad_eagerness must be one of: auto, low, medium, high (got {other})"
            ))),
        }
    }
}

/// Recording lifecycle status reported by `control status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordingStatus {
    Idle,
    Recording,
    Processing,
    AsrDisabled,
    AsrThreadDead,
}

impl RecordingStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Recording => "recording",
            Self::Processing => "processing",
            Self::AsrDisabled => "error:asr_disabled",
            Self::AsrThreadDead => "error:asr_thread_dead",
        }
    }

    /// Derive the public status value from runtime flags.
    pub fn from_flags(
        asr_disabled: bool,
        asr_thread_alive: bool,
        recording: bool,
        processing: bool,
    ) -> Self {
        if asr_disabled {
            return Self::AsrDisabled;
        }
        if !asr_thread_alive {
            return Self::AsrThreadDead;
        }
        if recording {
            return Self::Recording;
        }
        if processing {
            return Self::Processing;
        }
        Self::Idle
    }
}

impl fmt::Display for RecordingStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Caption overlay visual state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverlayState {
    Listening,
    Processing,
    Error,
}

impl OverlayState {
    pub const LISTENING: Self = Self::Listening;
    pub const PROCESSING: Self = Self::Processing;
    pub const ERROR: Self = Self::Error;

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Listening => "listening",
            Self::Processing => "processing",
            Self::Error => "error",
        }
    }

    pub fn css_class(self) -> &'static str {
        match self {
            Self::Listening => "state-listening",
            Self::Processing => "state-processing",
            Self::Error => "state-error",
        }
    }
}

impl FromStr for OverlayState {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "listening" => Ok(Self::Listening),
            "processing" => Ok(Self::Processing),
            "error" => Ok(Self::Error),
            other => Err(CoreError::OverlayState(format!(
                "Unknown overlay state '{other}'. Expected one of: error, listening, processing"
            ))),
        }
    }
}

/// Map overlay state name to CSS class (Python `overlay_state_class`).
pub fn overlay_state_class(state: &str) -> CoreResult<&'static str> {
    Ok(OverlayState::from_str(state)?.css_class())
}

/// How a backend expects audio chunking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedChunking {
    #[default]
    Streaming,
    Windowed,
}

/// How utterance finalization is owned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FinalizationMode {
    #[default]
    LocalStreaming,
    OfflineInstant,
    RemoteManualCommit,
}

/// ASR backend capability advertisement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrCapabilities {
    pub supports_gpu: bool,
    pub supports_model_download: bool,
    pub wants_raw_audio: bool,
    pub expected_chunking: ExpectedChunking,
    pub finalization_mode: FinalizationMode,
    pub preferred_sample_rate: Option<u32>,
    /// Additive seam: whether partials are meaningful during capture.
    pub emits_partials: bool,
    /// Additive seam: whether in-flight work can be cancelled cleanly.
    pub supports_cancel: bool,
}

impl Default for AsrCapabilities {
    fn default() -> Self {
        Self {
            supports_gpu: false,
            supports_model_download: false,
            wants_raw_audio: false,
            expected_chunking: ExpectedChunking::Streaming,
            finalization_mode: FinalizationMode::LocalStreaming,
            preferred_sample_rate: None,
            emits_partials: true,
            supports_cancel: false,
        }
    }
}

/// TTS backend capability advertisement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TtsCapabilities {
    pub supports_streaming: bool,
    pub supports_voice_list: bool,
    pub requires_api_key: bool,
    pub supports_speed_control: bool,
    pub speed_min: Option<f64>,
    pub speed_max: Option<f64>,
}

impl Default for TtsCapabilities {
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

impl TtsCapabilities {
    /// Effective speed bounds clamped to global 0.5–2.0 limits.
    pub fn speed_bounds(&self) -> Option<(f64, f64)> {
        if !self.supports_speed_control {
            return None;
        }
        let mut minimum = self
            .speed_min
            .map(|v| v.max(crate::tts_speed::TTS_PLAYBACK_SPEED_MIN))
            .unwrap_or(crate::tts_speed::TTS_PLAYBACK_SPEED_MIN);
        let mut maximum = self
            .speed_max
            .map(|v| v.min(crate::tts_speed::TTS_PLAYBACK_SPEED_MAX))
            .unwrap_or(crate::tts_speed::TTS_PLAYBACK_SPEED_MAX);
        if minimum > maximum {
            std::mem::swap(&mut minimum, &mut maximum);
        }
        Some((
            (minimum * 100.0).round() / 100.0,
            (maximum * 100.0).round() / 100.0,
        ))
    }
}

/// TTS voice list entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
}

/// Immutable per-utterance TTS synthesis request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TtsSynthesisRequest {
    pub text: String,
    pub voice_id: String,
    pub model_id: String,
    pub playback_speed: f64,
}

/// Optional audio / playback device reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DeviceRef {
    Index(i64),
    Name(String),
}

impl DeviceRef {
    pub fn from_toml_value(value: &toml::Value) -> CoreResult<Option<Self>> {
        match value {
            toml::Value::Integer(v) => Ok(Some(Self::Index(*v))),
            toml::Value::String(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    Ok(None)
                } else if trimmed.chars().all(|c| c.is_ascii_digit()) {
                    Ok(Some(Self::Index(trimmed.parse::<i64>().map_err(|_| {
                        CoreError::validation(format!("invalid device index '{trimmed}'"))
                    })?)))
                } else {
                    Ok(Some(Self::Name(trimmed.to_string())))
                }
            }
            toml::Value::Boolean(_)
            | toml::Value::Float(_)
            | toml::Value::Datetime(_)
            | toml::Value::Array(_)
            | toml::Value::Table(_) => Err(CoreError::validation(
                "device must be a string, integer, or null".to_string(),
            )),
        }
    }
}

/// CUDA / ORT OOM marker list (case-insensitive substring match).
pub const CUDA_OOM_ERROR_MARKERS: &[&str] = &[
    "cublas_status_alloc_failed",
    "cublascreate",
    "cudnn_status_internal_error",
    "cudnn_status_alloc_failed",
    "cudnncreate",
    "cuda error: out of memory",
    "cuda out of memory",
    "out of memory",
    "failed to allocate memory",
    "bfc_arena",
];

/// Heuristic: does `message` look like a CUDA/ORT allocation failure?
pub fn looks_like_cuda_oom_error(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    CUDA_OOM_ERROR_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
}

/// Model name helper used by Sherpa auto decode resolution.
pub fn is_parakeet_model(model_name: &str) -> bool {
    model_name.to_ascii_lowercase().contains("parakeet")
}

/// Wizard / packaging dependency failure exit code.
pub const DEPENDENCY_EXIT_CODE: u8 = 78;

/// Control-socket command allowlist (order stable for help text).
pub const CONTROL_COMMANDS: &[&str] = &[
    "start",
    "stop",
    "toggle",
    "status",
    "ping",
    "metrics",
    "debug_status",
    "tts_speak",
    "tts_speak_clipboard",
    "tts_pause",
    "tts_resume",
    "tts_toggle_pause",
    "tts_restart",
    "tts_stop",
    "tts_status",
];

/// Whether `command` is a known control command.
pub fn is_valid_control_command(command: &str) -> bool {
    let normalized = command.trim().to_ascii_lowercase();
    CONTROL_COMMANDS.iter().any(|c| *c == normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_status_matrix() {
        assert_eq!(
            RecordingStatus::from_flags(true, true, true, true).as_str(),
            "error:asr_disabled"
        );
        assert_eq!(
            RecordingStatus::from_flags(false, false, false, false).as_str(),
            "error:asr_thread_dead"
        );
        assert_eq!(
            RecordingStatus::from_flags(false, true, true, false).as_str(),
            "recording"
        );
        assert_eq!(
            RecordingStatus::from_flags(false, true, false, true).as_str(),
            "processing"
        );
        assert_eq!(
            RecordingStatus::from_flags(false, true, false, false).as_str(),
            "idle"
        );
    }

    #[test]
    fn overlay_state_class_valid_and_invalid() {
        assert_eq!(overlay_state_class("listening").unwrap(), "state-listening");
        assert_eq!(
            overlay_state_class("processing").unwrap(),
            "state-processing"
        );
        assert_eq!(overlay_state_class("error").unwrap(), "state-error");
        assert!(
            overlay_state_class("unknown")
                .unwrap_err()
                .to_string()
                .contains("Unknown overlay state")
        );
    }

    #[test]
    fn cuda_oom_markers_match_known_strings() {
        assert!(looks_like_cuda_oom_error("CUBLAS_STATUS_ALLOC_FAILED"));
        assert!(looks_like_cuda_oom_error(
            "bfc_arena.cc:359 Failed to allocate memory for requested buffer"
        ));
        assert!(!looks_like_cuda_oom_error("connection reset by peer"));
    }
}
