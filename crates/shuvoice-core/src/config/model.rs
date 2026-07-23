//! Config model, validation, load/save.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

use super::defaults::{
    CURRENT_CONFIG_VERSION, DEFAULT_ELEVENLABS_TTS_API_KEY_ENV, DEFAULT_ELEVENLABS_TTS_MODEL_ID,
    DEFAULT_ELEVENLABS_TTS_VOICE_ID, DEFAULT_KOKORO_TTS_BASE_URL, DEFAULT_KOKORO_TTS_MODEL_ID,
    DEFAULT_KOKORO_TTS_VOICE_ID, DEFAULT_LOCAL_TTS_MODEL_ID, DEFAULT_LOCAL_TTS_VOICE_ID,
    DEFAULT_MELOTTS_MODEL_ID, DEFAULT_MELOTTS_VOICE_ID, DEFAULT_OPENAI_TTS_API_KEY_ENV,
    DEFAULT_OPENAI_TTS_MODEL_ID, DEFAULT_OPENAI_TTS_VOICE_ID, DEFAULT_SHERPA_MODEL_NAME,
    DEFAULT_TEXT_REPLACEMENTS, config_section_fields,
};
use super::io::{load_raw, toml_dumps, write_atomic};
use super::migrate::migrate_to_latest;
use crate::error::{CoreError, CoreResult};
use crate::postprocess::{CompiledTextReplacements, compile_text_replacements};
use crate::tts_speed::{TTS_PLAYBACK_SPEED_DEFAULT, validate_tts_playback_speed};
use crate::types::{
    AsrBackendKind, ComputeProvider, DeviceRef, InjectionMode, MeloTtsDevice, OpenaiTurnDetection,
    OpenaiVadEagerness, OutputMode, ResolvedSherpaDecodeMode, SherpaDecodeMode, TtsBackendKind,
    TypingTextCase, is_parakeet_model,
};
use crate::xdg::{
    config_dir as xdg_config_dir, config_path as xdg_config_path, data_dir as xdg_data_dir,
};

static SAFE_ID_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[a-zA-Z0-9_\-\./:@]+$").expect("safe id regex"));
static FONT_FAMILY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[A-Za-z0-9 ._-]+$").expect("font family regex"));

const OPENAI_REALTIME_MODELS: &[&str] = &[
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
    "gpt-4o-transcribe-latest",
    "whisper-1",
];

/// Outcome metadata from loading a config file (migrations, legacy mapping, persist).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigLoadReport {
    pub migration: crate::config::MigrationReport,
    pub derived_mode_from_legacy: bool,
    pub ignored_keys: Vec<String>,
    pub persist_attempted: bool,
    pub persist_error: Option<String>,
}

/// Fully validated runtime configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub config_version: u32,

    // Audio
    pub sample_rate: u32,
    pub chunk_ms: u32,
    pub fallback_sample_rate: u32,
    pub audio_device: Option<DeviceRef>,
    pub input_gain: f64,
    pub audio_queue_max_size: u32,
    pub recording_preroll_ms: u32,
    pub silence_rms_threshold: f64,
    pub silence_rms_multiplier: f64,
    pub min_speech_ms: u32,
    pub auto_gain_target_peak: f64,
    pub auto_gain_max: f64,
    pub auto_gain_settle_chunks: u32,

    // ASR
    pub asr_backend: AsrBackendKind,
    pub instant_mode: bool,
    pub model_name: String,
    pub right_context: u32,
    pub device: String,
    pub use_cuda_graph_decoder: bool,
    pub sherpa_model_name: String,
    pub sherpa_model_dir: Option<String>,
    pub sherpa_decode_mode: SherpaDecodeMode,
    pub sherpa_enable_parakeet_streaming: bool,
    pub sherpa_provider: ComputeProvider,
    pub sherpa_num_threads: u32,
    pub sherpa_chunk_ms: u32,
    pub sherpa_offline_max_utterance_sec: f64,
    pub moonshine_model_name: String,
    pub moonshine_model_dir: Option<String>,
    pub moonshine_model_precision: String,
    pub moonshine_chunk_ms: u32,
    pub moonshine_max_window_sec: f64,
    pub moonshine_max_tokens: u32,
    pub moonshine_provider: ComputeProvider,
    pub moonshine_onnx_threads: u32,
    pub openai_realtime_model: String,
    pub openai_realtime_api_key_env: String,
    pub openai_realtime_language: String,
    pub openai_realtime_latency_target_sec: f64,
    pub openai_realtime_turn_detection: OpenaiTurnDetection,
    pub openai_realtime_vad_eagerness: OpenaiVadEagerness,
    pub openai_realtime_request_timeout_sec: f64,
    pub openai_realtime_commit_timeout_sec: f64,

    // Overlay
    pub font_size: u32,
    pub font_family: Option<String>,
    pub bg_opacity: f64,
    pub border_radius: u32,
    pub bottom_margin: u32,
    pub overlay_debug_mode: bool,
    pub overlay_debug_max_lines: u32,

    // Control
    pub control_socket: Option<String>,

    // TTS
    pub tts_enabled: bool,
    pub tts_backend: TtsBackendKind,
    pub tts_default_voice_id: String,
    pub tts_model_id: String,
    pub tts_api_key_env: String,
    pub tts_output_format: String,
    pub tts_max_chars: u32,
    pub tts_request_timeout_sec: f64,
    pub tts_playback_speed: f64,
    pub tts_playback_device: Option<DeviceRef>,
    pub tts_overlay_auto_hide_sec: f64,
    pub tts_local_model_path: Option<String>,
    pub tts_local_voice: Option<String>,
    pub tts_local_device: Option<DeviceRef>,
    pub tts_melotts_device: MeloTtsDevice,
    pub tts_melotts_venv_path: Option<String>,
    pub tts_kokoro_base_url: String,

    // Typing
    pub output_mode: OutputMode,
    pub typing_final_injection_mode: InjectionMode,
    pub typing_text_case: TypingTextCase,
    pub use_clipboard_for_final: bool,
    pub preserve_clipboard: bool,
    pub typing_clipboard_settle_delay_ms: u32,
    pub typing_retry_attempts: u32,
    pub typing_retry_delay_ms: u32,
    pub typing_subprocess_timeout: f64,
    pub auto_capitalize: bool,
    pub text_replacements: BTreeMap<String, String>,

    // Streaming
    pub streaming_stall_guard: bool,
    pub streaming_stall_chunks: u32,
    pub streaming_stall_rms_ratio: f64,
    pub streaming_stall_flush_chunks: u32,

    // Feedback
    pub audio_feedback: bool,
    pub feedback_start_freq: u32,
    pub feedback_stop_freq: u32,
    pub feedback_duration_ms: u32,
    pub feedback_volume: f64,

    /// Hot-path compiled replacements.
    pub compiled_text_replacements: CompiledTextReplacements,
    resolved_sherpa_decode_mode: Option<ResolvedSherpaDecodeMode>,
}

impl Default for Config {
    fn default() -> Self {
        let text_replacements = DEFAULT_TEXT_REPLACEMENTS.clone();
        let compiled = compile_text_replacements(&text_replacements);
        Self {
            config_version: CURRENT_CONFIG_VERSION,
            sample_rate: 16000,
            chunk_ms: 100,
            fallback_sample_rate: 48000,
            audio_device: None,
            input_gain: 1.0,
            audio_queue_max_size: 200,
            recording_preroll_ms: 200,
            silence_rms_threshold: 0.008,
            silence_rms_multiplier: 1.8,
            min_speech_ms: 80,
            auto_gain_target_peak: 0.15,
            auto_gain_max: 10.0,
            auto_gain_settle_chunks: 2,
            asr_backend: AsrBackendKind::Sherpa,
            instant_mode: false,
            model_name: "nvidia/nemotron-speech-streaming-en-0.6b".into(),
            right_context: 13,
            device: "cuda".into(),
            use_cuda_graph_decoder: false,
            sherpa_model_name: DEFAULT_SHERPA_MODEL_NAME.into(),
            sherpa_model_dir: None,
            sherpa_decode_mode: SherpaDecodeMode::Auto,
            sherpa_enable_parakeet_streaming: false,
            sherpa_provider: ComputeProvider::Cpu,
            sherpa_num_threads: 2,
            sherpa_chunk_ms: 100,
            sherpa_offline_max_utterance_sec: 60.0,
            moonshine_model_name: "moonshine/tiny".into(),
            moonshine_model_dir: None,
            moonshine_model_precision: "float".into(),
            moonshine_chunk_ms: 100,
            moonshine_max_window_sec: 5.0,
            moonshine_max_tokens: 64,
            moonshine_provider: ComputeProvider::Cpu,
            moonshine_onnx_threads: 0,
            openai_realtime_model: "gpt-4o-transcribe".into(),
            openai_realtime_api_key_env: "OPENAI_API_KEY".into(),
            openai_realtime_language: "en".into(),
            openai_realtime_latency_target_sec: 0.8,
            openai_realtime_turn_detection: OpenaiTurnDetection::Manual,
            openai_realtime_vad_eagerness: OpenaiVadEagerness::Auto,
            openai_realtime_request_timeout_sec: 10.0,
            openai_realtime_commit_timeout_sec: 5.0,
            font_size: 22,
            font_family: None,
            bg_opacity: 0.75,
            border_radius: 16,
            bottom_margin: 60,
            overlay_debug_mode: false,
            overlay_debug_max_lines: 12,
            control_socket: None,
            tts_enabled: true,
            tts_backend: TtsBackendKind::Elevenlabs,
            tts_default_voice_id: DEFAULT_ELEVENLABS_TTS_VOICE_ID.into(),
            tts_model_id: DEFAULT_ELEVENLABS_TTS_MODEL_ID.into(),
            tts_api_key_env: DEFAULT_ELEVENLABS_TTS_API_KEY_ENV.into(),
            tts_output_format: "pcm_24000".into(),
            tts_max_chars: 5000,
            tts_request_timeout_sec: 30.0,
            tts_playback_speed: TTS_PLAYBACK_SPEED_DEFAULT,
            tts_playback_device: None,
            tts_overlay_auto_hide_sec: 2.0,
            tts_local_model_path: None,
            tts_local_voice: None,
            tts_local_device: None,
            tts_melotts_device: MeloTtsDevice::Auto,
            tts_melotts_venv_path: None,
            tts_kokoro_base_url: DEFAULT_KOKORO_TTS_BASE_URL.into(),
            output_mode: OutputMode::FinalOnly,
            typing_final_injection_mode: InjectionMode::Auto,
            typing_text_case: TypingTextCase::Default,
            use_clipboard_for_final: true,
            preserve_clipboard: false,
            typing_clipboard_settle_delay_ms: 40,
            typing_retry_attempts: 2,
            typing_retry_delay_ms: 40,
            typing_subprocess_timeout: 5.0,
            auto_capitalize: true,
            text_replacements,
            streaming_stall_guard: true,
            streaming_stall_chunks: 4,
            streaming_stall_rms_ratio: 0.7,
            streaming_stall_flush_chunks: 1,
            audio_feedback: true,
            feedback_start_freq: 880,
            feedback_stop_freq: 660,
            feedback_duration_ms: 70,
            feedback_volume: 0.08,
            compiled_text_replacements: compiled,
            resolved_sherpa_decode_mode: None,
        }
    }
}

impl Config {
    pub fn config_dir() -> PathBuf {
        let d = xdg_config_dir();
        let _ = std::fs::create_dir_all(&d);
        d
    }

    pub fn config_path() -> PathBuf {
        xdg_config_path()
    }

    pub fn data_dir() -> PathBuf {
        let d = xdg_data_dir();
        let _ = std::fs::create_dir_all(&d);
        d
    }

    pub fn chunk_samples(&self) -> usize {
        self.sample_rate as usize * self.chunk_ms as usize / 1000
    }

    pub fn resolved_sherpa_decode_mode(&self) -> Option<ResolvedSherpaDecodeMode> {
        if self.asr_backend != AsrBackendKind::Sherpa {
            return None;
        }
        if let Some(cached) = self.resolved_sherpa_decode_mode {
            return Some(cached);
        }
        Some(self.resolve_sherpa_decode_mode())
    }

    fn resolve_sherpa_decode_mode(&self) -> ResolvedSherpaDecodeMode {
        match self.sherpa_decode_mode {
            SherpaDecodeMode::Streaming => ResolvedSherpaDecodeMode::Streaming,
            SherpaDecodeMode::OfflineInstant => ResolvedSherpaDecodeMode::OfflineInstant,
            SherpaDecodeMode::Auto => {
                if self.instant_mode && is_parakeet_model(&self.sherpa_model_name) {
                    ResolvedSherpaDecodeMode::OfflineInstant
                } else {
                    ResolvedSherpaDecodeMode::Streaming
                }
            }
        }
    }

    /// Validate and normalize in place (Python `__post_init__`).
    pub fn validate(&mut self) -> CoreResult<()> {
        if self.config_version < 1 {
            return Err(CoreError::validation("config_version must be >= 1"));
        }
        if self.config_version > CURRENT_CONFIG_VERSION {
            return Err(CoreError::validation(format!(
                "config_version is newer than this ShuVoice build supports (got {}, max {CURRENT_CONFIG_VERSION})",
                self.config_version
            )));
        }

        self.sherpa_model_name = self.sherpa_model_name.trim().to_string();
        if self.sherpa_model_name.is_empty() {
            return Err(CoreError::validation("sherpa_model_name must not be empty"));
        }
        if self.sherpa_chunk_ms == 0 {
            return Err(CoreError::validation("sherpa_chunk_ms must be > 0"));
        }
        if self.sherpa_num_threads < 1 {
            return Err(CoreError::validation("sherpa_num_threads must be >= 1"));
        }
        if self.sherpa_offline_max_utterance_sec < 0.0 {
            return Err(CoreError::validation(
                "sherpa_offline_max_utterance_sec must be >= 0",
            ));
        }

        if self.moonshine_chunk_ms == 0 {
            return Err(CoreError::validation("moonshine_chunk_ms must be > 0"));
        }
        if self.moonshine_max_window_sec <= 0.0 {
            return Err(CoreError::validation(
                "moonshine_max_window_sec must be > 0",
            ));
        }
        if self.moonshine_max_tokens < 1 {
            return Err(CoreError::validation("moonshine_max_tokens must be >= 1"));
        }
        self.moonshine_model_name = self.moonshine_model_name.trim().to_string();
        if self.moonshine_model_name.is_empty() {
            return Err(CoreError::validation(
                "moonshine_model_name must not be empty",
            ));
        }
        if (self.moonshine_onnx_threads as i32) < 0 {
            return Err(CoreError::validation(
                "moonshine_onnx_threads must be >= 0 (0 = auto)",
            ));
        }
        self.moonshine_model_precision = self.moonshine_model_precision.trim().to_ascii_lowercase();
        if self.moonshine_model_precision.is_empty() {
            return Err(CoreError::validation(
                "moonshine_model_precision must not be empty",
            ));
        }

        self.openai_realtime_model = self.openai_realtime_model.trim().to_string();
        if !OPENAI_REALTIME_MODELS
            .iter()
            .any(|m| *m == self.openai_realtime_model)
        {
            return Err(CoreError::validation(format!(
                "openai_realtime_model must be one of: {}",
                {
                    let mut v = OPENAI_REALTIME_MODELS.to_vec();
                    v.sort();
                    v.join(", ")
                }
            )));
        }
        self.openai_realtime_api_key_env = self.openai_realtime_api_key_env.trim().to_string();
        if self.openai_realtime_api_key_env.is_empty()
            || !SAFE_ID_RE.is_match(&self.openai_realtime_api_key_env)
        {
            return Err(CoreError::validation(
                "openai_realtime_api_key_env must be a non-empty safe env var name",
            ));
        }
        if self.openai_realtime_api_key_env.starts_with("sk_")
            || self.openai_realtime_api_key_env.starts_with("sk-")
        {
            return Err(CoreError::validation(
                "openai_realtime_api_key_env looks like a raw API key value, expected an environment variable name",
            ));
        }
        self.openai_realtime_language = self.openai_realtime_language.trim().to_string();
        if !self.openai_realtime_language.is_empty()
            && !SAFE_ID_RE.is_match(&self.openai_realtime_language)
        {
            return Err(CoreError::validation(
                "openai_realtime_language contains unsupported characters",
            ));
        }
        if self.openai_realtime_latency_target_sec <= 0.0 {
            return Err(CoreError::validation(
                "openai_realtime_latency_target_sec must be > 0",
            ));
        }
        if self.openai_realtime_request_timeout_sec <= 0.0 {
            return Err(CoreError::validation(
                "openai_realtime_request_timeout_sec must be > 0",
            ));
        }
        if self.openai_realtime_commit_timeout_sec <= 0.0 {
            return Err(CoreError::validation(
                "openai_realtime_commit_timeout_sec must be > 0",
            ));
        }

        if self.audio_queue_max_size < 1 {
            return Err(CoreError::validation("audio_queue_max_size must be >= 1"));
        }
        if self.auto_gain_target_peak <= 0.0 {
            return Err(CoreError::validation("auto_gain_target_peak must be > 0"));
        }
        if self.auto_gain_max < 1.0 {
            return Err(CoreError::validation("auto_gain_max must be >= 1"));
        }
        if self.auto_gain_settle_chunks < 1 {
            return Err(CoreError::validation(
                "auto_gain_settle_chunks must be >= 1",
            ));
        }
        if self.streaming_stall_chunks < 1 {
            return Err(CoreError::validation("streaming_stall_chunks must be >= 1"));
        }
        if self.streaming_stall_flush_chunks < 1 {
            return Err(CoreError::validation(
                "streaming_stall_flush_chunks must be >= 1",
            ));
        }
        if self.streaming_stall_rms_ratio <= 0.0 {
            return Err(CoreError::validation(
                "streaming_stall_rms_ratio must be > 0",
            ));
        }
        if self.sample_rate == 0 {
            return Err(CoreError::validation("sample_rate must be > 0"));
        }
        if self.chunk_ms == 0 {
            return Err(CoreError::validation("chunk_ms must be > 0"));
        }
        if self.fallback_sample_rate == 0 {
            return Err(CoreError::validation("fallback_sample_rate must be > 0"));
        }
        if self.input_gain <= 0.0 {
            return Err(CoreError::validation("input_gain must be > 0"));
        }
        if self.font_size == 0 {
            return Err(CoreError::validation("font_size must be > 0"));
        }
        if let Some(ff) = self.font_family.clone() {
            let normalized = ff.trim();
            if normalized.is_empty() {
                self.font_family = None;
            } else if !FONT_FAMILY_RE.is_match(normalized) {
                return Err(CoreError::validation(
                    "font_family contains unsupported characters (allowed: letters, numbers, spaces, '.', '_' and '-')",
                ));
            } else {
                self.font_family = Some(normalized.to_string());
            }
        }
        if !(0.0..=1.0).contains(&self.bg_opacity) {
            return Err(CoreError::validation(
                "bg_opacity must be between 0.0 and 1.0",
            ));
        }
        if self.overlay_debug_max_lines < 1 {
            return Err(CoreError::validation(
                "overlay_debug_max_lines must be >= 1",
            ));
        }

        // Normalize optional path/voice strings (empty -> None).
        self.tts_local_model_path = normalize_opt_string(self.tts_local_model_path.take());
        self.tts_local_voice = normalize_opt_string(self.tts_local_voice.take());
        self.tts_melotts_venv_path = normalize_opt_string(self.tts_melotts_venv_path.take());
        self.sherpa_model_dir = normalize_opt_string(self.sherpa_model_dir.take());
        self.moonshine_model_dir = normalize_opt_string(self.moonshine_model_dir.take());
        self.control_socket = normalize_opt_string(self.control_socket.take());

        // TTS backend default voice/model remaps
        match self.tts_backend {
            TtsBackendKind::Openai => {
                if self.tts_default_voice_id.trim() == DEFAULT_ELEVENLABS_TTS_VOICE_ID {
                    self.tts_default_voice_id = DEFAULT_OPENAI_TTS_VOICE_ID.into();
                }
                if self.tts_model_id.trim() == DEFAULT_ELEVENLABS_TTS_MODEL_ID {
                    self.tts_model_id = DEFAULT_OPENAI_TTS_MODEL_ID.into();
                }
                if self.tts_api_key_env.trim() == DEFAULT_ELEVENLABS_TTS_API_KEY_ENV {
                    self.tts_api_key_env = DEFAULT_OPENAI_TTS_API_KEY_ENV.into();
                }
            }
            TtsBackendKind::Local => {
                let voice = self.tts_default_voice_id.trim();
                if voice.is_empty()
                    || voice == DEFAULT_ELEVENLABS_TTS_VOICE_ID
                    || voice == DEFAULT_OPENAI_TTS_VOICE_ID
                {
                    self.tts_default_voice_id = self
                        .tts_local_voice
                        .clone()
                        .filter(|s| !s.trim().is_empty())
                        .unwrap_or_else(|| DEFAULT_LOCAL_TTS_VOICE_ID.into());
                }
                let model = self.tts_model_id.trim();
                if model.is_empty()
                    || model == DEFAULT_ELEVENLABS_TTS_MODEL_ID
                    || model == DEFAULT_OPENAI_TTS_MODEL_ID
                {
                    self.tts_model_id = DEFAULT_LOCAL_TTS_MODEL_ID.into();
                }
            }
            TtsBackendKind::Melotts => {
                let voice = self.tts_default_voice_id.trim();
                if voice.is_empty()
                    || voice == DEFAULT_ELEVENLABS_TTS_VOICE_ID
                    || voice == DEFAULT_OPENAI_TTS_VOICE_ID
                {
                    self.tts_default_voice_id = DEFAULT_MELOTTS_VOICE_ID.into();
                }
                let model = self.tts_model_id.trim();
                if model.is_empty()
                    || model == DEFAULT_ELEVENLABS_TTS_MODEL_ID
                    || model == DEFAULT_OPENAI_TTS_MODEL_ID
                {
                    self.tts_model_id = DEFAULT_MELOTTS_MODEL_ID.into();
                }
            }
            TtsBackendKind::Kokoro => {
                let voice = self.tts_default_voice_id.trim();
                if voice.is_empty()
                    || voice == DEFAULT_ELEVENLABS_TTS_VOICE_ID
                    || voice == DEFAULT_OPENAI_TTS_VOICE_ID
                {
                    self.tts_default_voice_id = DEFAULT_KOKORO_TTS_VOICE_ID.into();
                }
                let model = self.tts_model_id.trim();
                if model.is_empty()
                    || model == DEFAULT_ELEVENLABS_TTS_MODEL_ID
                    || model == DEFAULT_OPENAI_TTS_MODEL_ID
                {
                    self.tts_model_id = DEFAULT_KOKORO_TTS_MODEL_ID.into();
                }
            }
            TtsBackendKind::Elevenlabs => {}
        }

        self.tts_default_voice_id = self.tts_default_voice_id.trim().to_string();
        if self.tts_default_voice_id.is_empty() {
            return Err(CoreError::validation(
                "tts_default_voice_id must not be empty",
            ));
        }
        self.tts_model_id = self.tts_model_id.trim().to_string();
        if self.tts_model_id.is_empty() {
            return Err(CoreError::validation("tts_model_id must not be empty"));
        }
        if !SAFE_ID_RE.is_match(&self.tts_model_id) {
            return Err(CoreError::validation(format!(
                "tts_model_id contains invalid characters: {:?}",
                self.tts_model_id
            )));
        }
        if !SAFE_ID_RE.is_match(&self.tts_default_voice_id) {
            return Err(CoreError::validation(format!(
                "tts_default_voice_id contains invalid characters: {:?}",
                self.tts_default_voice_id
            )));
        }
        self.tts_api_key_env = self.tts_api_key_env.trim().to_string();
        if self.tts_api_key_env.is_empty() {
            return Err(CoreError::validation("tts_api_key_env must not be empty"));
        }
        self.tts_output_format = self.tts_output_format.trim().to_string();
        if self.tts_output_format.is_empty() {
            return Err(CoreError::validation("tts_output_format must not be empty"));
        }
        if self.tts_max_chars < 1 {
            return Err(CoreError::validation("tts_max_chars must be >= 1"));
        }
        if self.tts_request_timeout_sec <= 0.0 {
            return Err(CoreError::validation("tts_request_timeout_sec must be > 0"));
        }
        self.tts_playback_speed = validate_tts_playback_speed(self.tts_playback_speed)?;
        if self.tts_overlay_auto_hide_sec < 0.0 {
            return Err(CoreError::validation(
                "tts_overlay_auto_hide_sec must be >= 0",
            ));
        }

        let mut kokoro = self.tts_kokoro_base_url.trim().to_string();
        if kokoro.is_empty() {
            kokoro = DEFAULT_KOKORO_TTS_BASE_URL.into();
        }
        if self.tts_backend == TtsBackendKind::Kokoro {
            let ok = (kokoro.starts_with("http://") || kokoro.starts_with("https://"))
                && kokoro.split("://").nth(1).is_some_and(|rest| {
                    let host = rest.split('/').next().unwrap_or("");
                    !host.is_empty()
                });
            if !ok {
                return Err(CoreError::validation(
                    "tts_kokoro_base_url must be a valid http(s) URL",
                ));
            }
        }
        while kokoro.ends_with('/') {
            kokoro.pop();
        }
        if kokoro.is_empty() {
            kokoro = DEFAULT_KOKORO_TTS_BASE_URL.into();
        }
        self.tts_kokoro_base_url = kokoro;

        if self.typing_subprocess_timeout < 1.0 {
            return Err(CoreError::validation(
                "typing_subprocess_timeout must be >= 1.0",
            ));
        }

        // Normalize text replacements: builtins + user overrides.
        // Empty/whitespace-only optional path/voice fields are normalized to None
        // elsewhere in validate (runtime-equivalent to Python strip-or-None).
        // Replacement map keys are trimmed; compile order is longest-first then
        // lexicographic source (deterministic equal-length ties).
        let mut normalized = DEFAULT_TEXT_REPLACEMENTS.clone();
        for (key, value) in &self.text_replacements {
            let key_text = key.trim();
            if key_text.is_empty() {
                return Err(CoreError::validation(
                    "text_replacements keys must not be empty or whitespace-only",
                ));
            }
            normalized.insert(key_text.to_string(), value.trim().to_string());
        }
        self.text_replacements = normalized;

        self.apply_instant_mode_profile();
        self.compiled_text_replacements = compile_text_replacements(&self.text_replacements);
        self.resolved_sherpa_decode_mode = if self.asr_backend == AsrBackendKind::Sherpa {
            Some(self.resolve_sherpa_decode_mode())
        } else {
            None
        };
        Ok(())
    }

    fn apply_instant_mode_profile(&mut self) {
        if !self.instant_mode {
            return;
        }
        match self.asr_backend {
            AsrBackendKind::Nemo => {
                self.right_context = 0;
            }
            AsrBackendKind::Sherpa => {
                let resolved = self.resolve_sherpa_decode_mode();
                if resolved == ResolvedSherpaDecodeMode::Streaming {
                    self.sherpa_chunk_ms = self.sherpa_chunk_ms.min(80);
                }
            }
            AsrBackendKind::Moonshine => {
                if self.moonshine_model_name != "moonshine/tiny" {
                    self.moonshine_model_name = "moonshine/tiny".into();
                }
                self.moonshine_max_window_sec = self.moonshine_max_window_sec.min(3.0);
                self.moonshine_max_tokens = self.moonshine_max_tokens.min(48);
            }
            AsrBackendKind::OpenaiRealtime => {}
        }
    }

    /// Load config from the default XDG path.
    pub fn load() -> CoreResult<Self> {
        Self::load_from_path(Self::config_path())
    }

    /// Load config from an explicit path.
    pub fn load_from_path(path: impl AsRef<Path>) -> CoreResult<Self> {
        Ok(Self::load_from_path_with_report(path)?.0)
    }

    /// Load config and return migration/legacy/persist metadata.
    ///
    /// Persist failures are **non-fatal**: the validated config is still
    /// returned and the error is recorded on [`ConfigLoadReport`] plus emitted
    /// via `tracing::warn!`.
    pub fn load_from_path_with_report(
        path: impl AsRef<Path>,
    ) -> CoreResult<(Self, ConfigLoadReport)> {
        let path_buf = super::io::expand_user_path(path);
        let path = path_buf.as_path();
        let raw = load_raw(path)?;
        let (mut migrated, migration) = migrate_to_latest(&raw)?;

        let mut flat = flatten_raw(&migrated);
        let known = known_config_keys();
        let mut ignored_keys: Vec<String> = flat
            .keys()
            .filter(|k| *k != "config_version" && !known.contains(k.as_str()))
            .cloned()
            .collect();
        ignored_keys.sort();

        let mut derived_mode_from_legacy = false;
        let has_explicit_mode = flat.contains_key("typing_final_injection_mode");
        let has_legacy_flag = flat.contains_key("use_clipboard_for_final");
        if !has_explicit_mode
            && has_legacy_flag
            && let Some(Value::Bool(flag)) = flat.get("use_clipboard_for_final")
        {
            let derived = if *flag { "auto" } else { "direct" };
            flat.insert(
                "typing_final_injection_mode".into(),
                Value::String(derived.into()),
            );
            let typing = migrated
                .entry("typing".to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            if let Some(table) = typing.as_object_mut()
                && table
                    .get("typing_final_injection_mode")
                    .and_then(|v| v.as_str())
                    != Some(derived)
            {
                table.insert(
                    "typing_final_injection_mode".into(),
                    Value::String(derived.into()),
                );
                derived_mode_from_legacy = true;
            }
        }

        let mut cfg = Config::default();
        apply_flat_overrides(&mut cfg, &flat)?;
        cfg.validate()?;

        let should_persist =
            migration.to_version != migration.from_version || derived_mode_from_legacy;
        let mut persist_attempted = false;
        let mut persist_error = None;
        if path.exists() && should_persist {
            persist_attempted = true;
            migrated.insert("config_version".into(), Value::from(CURRENT_CONFIG_VERSION));
            match write_atomic(path, &migrated) {
                Ok(_) => {}
                Err(err) => {
                    let msg = err.to_string();
                    tracing::warn!(
                        error = %msg,
                        path = %path.display(),
                        "Failed to persist migrated config file"
                    );
                    persist_error = Some(msg);
                }
            }
        }

        Ok((
            cfg,
            ConfigLoadReport {
                migration,
                derived_mode_from_legacy,
                ignored_keys,
                persist_attempted,
                persist_error,
            },
        ))
    }

    /// Validate a default config after applying a mutation closure (test/helper ergonomics).
    pub fn try_with(mutate: impl FnOnce(&mut Self)) -> CoreResult<Self> {
        let mut cfg = Self::default();
        mutate(&mut cfg);
        cfg.validate()?;
        Ok(cfg)
    }

    /// Build nested JSON/TOML-compatible map for persistence.
    pub fn to_nested_map(&self, include_none: bool) -> Map<String, Value> {
        let mut data = Map::new();
        data.insert("config_version".into(), Value::from(self.config_version));

        for (section, fields) in config_section_fields() {
            let mut section_data = Map::new();
            for key in *fields {
                if let Some(value) = self.field_to_value(key) {
                    if value.is_null() && !include_none {
                        continue;
                    }
                    section_data.insert((*key).to_string(), value);
                }
            }
            if !section_data.is_empty() {
                data.insert((*section).to_string(), Value::Object(section_data));
            }
        }
        data
    }

    /// Serialize to TOML text using the project dumper.
    pub fn to_toml_string(&self) -> CoreResult<String> {
        toml_dumps(&self.to_nested_map(false))
    }

    /// Atomically write this config to path.
    pub fn save_to_path(&self, path: impl AsRef<Path>) -> CoreResult<Option<PathBuf>> {
        write_atomic(
            super::io::expand_user_path(path),
            &self.to_nested_map(false),
        )
    }

    fn field_to_value(&self, key: &str) -> Option<Value> {
        Some(match key {
            "sample_rate" => Value::from(self.sample_rate),
            "chunk_ms" => Value::from(self.chunk_ms),
            "fallback_sample_rate" => Value::from(self.fallback_sample_rate),
            "audio_device" => device_to_value(&self.audio_device),
            "input_gain" => json_f64(self.input_gain),
            "audio_queue_max_size" => Value::from(self.audio_queue_max_size),
            "recording_preroll_ms" => Value::from(self.recording_preroll_ms),
            "silence_rms_threshold" => json_f64(self.silence_rms_threshold),
            "silence_rms_multiplier" => json_f64(self.silence_rms_multiplier),
            "min_speech_ms" => Value::from(self.min_speech_ms),
            "auto_gain_target_peak" => json_f64(self.auto_gain_target_peak),
            "auto_gain_max" => json_f64(self.auto_gain_max),
            "auto_gain_settle_chunks" => Value::from(self.auto_gain_settle_chunks),
            "asr_backend" => Value::String(self.asr_backend.as_str().into()),
            "instant_mode" => Value::Bool(self.instant_mode),
            "model_name" => Value::String(self.model_name.clone()),
            "right_context" => Value::from(self.right_context),
            "device" => Value::String(self.device.clone()),
            "use_cuda_graph_decoder" => Value::Bool(self.use_cuda_graph_decoder),
            "sherpa_model_name" => Value::String(self.sherpa_model_name.clone()),
            "sherpa_model_dir" => opt_string(&self.sherpa_model_dir),
            "sherpa_decode_mode" => Value::String(self.sherpa_decode_mode.as_str().into()),
            "sherpa_enable_parakeet_streaming" => {
                Value::Bool(self.sherpa_enable_parakeet_streaming)
            }
            "sherpa_provider" => Value::String(self.sherpa_provider.as_str().into()),
            "sherpa_num_threads" => Value::from(self.sherpa_num_threads),
            "sherpa_chunk_ms" => Value::from(self.sherpa_chunk_ms),
            "sherpa_offline_max_utterance_sec" => json_f64(self.sherpa_offline_max_utterance_sec),
            "moonshine_model_name" => Value::String(self.moonshine_model_name.clone()),
            "moonshine_model_dir" => opt_string(&self.moonshine_model_dir),
            "moonshine_model_precision" => Value::String(self.moonshine_model_precision.clone()),
            "moonshine_chunk_ms" => Value::from(self.moonshine_chunk_ms),
            "moonshine_max_window_sec" => json_f64(self.moonshine_max_window_sec),
            "moonshine_max_tokens" => Value::from(self.moonshine_max_tokens),
            "moonshine_provider" => Value::String(self.moonshine_provider.as_str().into()),
            "moonshine_onnx_threads" => Value::from(self.moonshine_onnx_threads),
            "openai_realtime_model" => Value::String(self.openai_realtime_model.clone()),
            "openai_realtime_api_key_env" => {
                Value::String(self.openai_realtime_api_key_env.clone())
            }
            "openai_realtime_language" => Value::String(self.openai_realtime_language.clone()),
            "openai_realtime_latency_target_sec" => {
                json_f64(self.openai_realtime_latency_target_sec)
            }
            "openai_realtime_turn_detection" => {
                Value::String(self.openai_realtime_turn_detection.as_str().into())
            }
            "openai_realtime_vad_eagerness" => {
                Value::String(self.openai_realtime_vad_eagerness.as_str().into())
            }
            "openai_realtime_request_timeout_sec" => {
                json_f64(self.openai_realtime_request_timeout_sec)
            }
            "openai_realtime_commit_timeout_sec" => {
                json_f64(self.openai_realtime_commit_timeout_sec)
            }
            "font_size" => Value::from(self.font_size),
            "font_family" => opt_string(&self.font_family),
            "bg_opacity" => json_f64(self.bg_opacity),
            "border_radius" => Value::from(self.border_radius),
            "bottom_margin" => Value::from(self.bottom_margin),
            "overlay_debug_mode" => Value::Bool(self.overlay_debug_mode),
            "overlay_debug_max_lines" => Value::from(self.overlay_debug_max_lines),
            "control_socket" => opt_string(&self.control_socket),
            "tts_enabled" => Value::Bool(self.tts_enabled),
            "tts_backend" => Value::String(self.tts_backend.as_str().into()),
            "tts_default_voice_id" => Value::String(self.tts_default_voice_id.clone()),
            "tts_model_id" => Value::String(self.tts_model_id.clone()),
            "tts_api_key_env" => Value::String(self.tts_api_key_env.clone()),
            "tts_output_format" => Value::String(self.tts_output_format.clone()),
            "tts_max_chars" => Value::from(self.tts_max_chars),
            "tts_request_timeout_sec" => json_f64(self.tts_request_timeout_sec),
            "tts_playback_speed" => json_f64(self.tts_playback_speed),
            "tts_playback_device" => device_to_value(&self.tts_playback_device),
            "tts_overlay_auto_hide_sec" => json_f64(self.tts_overlay_auto_hide_sec),
            "tts_local_model_path" => opt_string(&self.tts_local_model_path),
            "tts_local_voice" => opt_string(&self.tts_local_voice),
            "tts_local_device" => device_to_value(&self.tts_local_device),
            "tts_melotts_device" => Value::String(self.tts_melotts_device.as_str().into()),
            "tts_melotts_venv_path" => opt_string(&self.tts_melotts_venv_path),
            "tts_kokoro_base_url" => Value::String(self.tts_kokoro_base_url.clone()),
            "output_mode" => Value::String(self.output_mode.as_str().into()),
            "typing_final_injection_mode" => {
                Value::String(self.typing_final_injection_mode.as_str().into())
            }
            "typing_text_case" => Value::String(self.typing_text_case.as_str().into()),
            "use_clipboard_for_final" => Value::Bool(self.use_clipboard_for_final),
            "preserve_clipboard" => Value::Bool(self.preserve_clipboard),
            "typing_clipboard_settle_delay_ms" => {
                Value::from(self.typing_clipboard_settle_delay_ms)
            }
            "typing_retry_attempts" => Value::from(self.typing_retry_attempts),
            "typing_retry_delay_ms" => Value::from(self.typing_retry_delay_ms),
            "typing_subprocess_timeout" => json_f64(self.typing_subprocess_timeout),
            "auto_capitalize" => Value::Bool(self.auto_capitalize),
            "text_replacements" => {
                let mut map = Map::new();
                for (k, v) in &self.text_replacements {
                    map.insert(k.clone(), Value::String(v.clone()));
                }
                Value::Object(map)
            }
            "streaming_stall_guard" => Value::Bool(self.streaming_stall_guard),
            "streaming_stall_chunks" => Value::from(self.streaming_stall_chunks),
            "streaming_stall_rms_ratio" => json_f64(self.streaming_stall_rms_ratio),
            "streaming_stall_flush_chunks" => Value::from(self.streaming_stall_flush_chunks),
            "audio_feedback" => Value::Bool(self.audio_feedback),
            "feedback_start_freq" => Value::from(self.feedback_start_freq),
            "feedback_stop_freq" => Value::from(self.feedback_stop_freq),
            "feedback_duration_ms" => Value::from(self.feedback_duration_ms),
            "feedback_volume" => json_f64(self.feedback_volume),
            _ => return None,
        })
    }
}

fn normalize_opt_string(value: Option<String>) -> Option<String> {
    value.and_then(|s| {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn known_config_keys() -> std::collections::HashSet<&'static str> {
    let mut set = HashSet::new();
    set.insert("config_version");
    for (_, fields) in config_section_fields() {
        for f in *fields {
            set.insert(*f);
        }
    }
    set
}

fn json_f64(v: f64) -> Value {
    serde_json::Number::from_f64(v)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn opt_string(v: &Option<String>) -> Value {
    match v {
        Some(s) => Value::String(s.clone()),
        None => Value::Null,
    }
}

fn device_to_value(device: &Option<DeviceRef>) -> Value {
    match device {
        None => Value::Null,
        Some(DeviceRef::Index(i)) => Value::from(*i),
        Some(DeviceRef::Name(s)) => Value::String(s.clone()),
    }
}

fn flatten_raw(raw: &Map<String, Value>) -> Map<String, Value> {
    let mut flat = Map::new();
    for (key, value) in raw {
        if key == "config_version" {
            flat.insert(key.clone(), value.clone());
            continue;
        }
        if let Some(table) = value.as_object() {
            for (k, v) in table {
                flat.insert(k.clone(), v.clone());
            }
        } else {
            flat.insert(key.clone(), value.clone());
        }
    }
    flat
}

fn apply_flat_overrides(cfg: &mut Config, flat: &Map<String, Value>) -> CoreResult<()> {
    for (key, value) in flat {
        match key.as_str() {
            "config_version" => cfg.config_version = as_u32(value, "config_version")?,
            "sample_rate" => cfg.sample_rate = as_u32(value, "sample_rate")?,
            "chunk_ms" => cfg.chunk_ms = as_u32(value, "chunk_ms")?,
            "fallback_sample_rate" => {
                cfg.fallback_sample_rate = as_u32(value, "fallback_sample_rate")?
            }
            "audio_device" => {
                cfg.audio_device = as_device(value, "audio_device", DeviceDigitPolicy::KeepAsName)?
            }
            "input_gain" => cfg.input_gain = as_f64(value, "input_gain")?,
            "audio_queue_max_size" => {
                cfg.audio_queue_max_size = as_u32(value, "audio_queue_max_size")?
            }
            "recording_preroll_ms" => {
                cfg.recording_preroll_ms = as_u32(value, "recording_preroll_ms")?
            }
            "silence_rms_threshold" => {
                cfg.silence_rms_threshold = as_f64(value, "silence_rms_threshold")?
            }
            "silence_rms_multiplier" => {
                cfg.silence_rms_multiplier = as_f64(value, "silence_rms_multiplier")?
            }
            "min_speech_ms" => cfg.min_speech_ms = as_u32(value, "min_speech_ms")?,
            "auto_gain_target_peak" => {
                cfg.auto_gain_target_peak = as_f64(value, "auto_gain_target_peak")?
            }
            "auto_gain_max" => cfg.auto_gain_max = as_f64(value, "auto_gain_max")?,
            "auto_gain_settle_chunks" => {
                cfg.auto_gain_settle_chunks = as_u32(value, "auto_gain_settle_chunks")?
            }
            "asr_backend" => {
                cfg.asr_backend = AsrBackendKind::from_str(&as_string(value, "asr_backend")?)?
            }
            "instant_mode" => cfg.instant_mode = as_bool(value, "instant_mode")?,
            "model_name" => cfg.model_name = as_string(value, "model_name")?,
            "right_context" => cfg.right_context = as_u32(value, "right_context")?,
            "device" => cfg.device = as_string(value, "device")?,
            "use_cuda_graph_decoder" => {
                cfg.use_cuda_graph_decoder = as_bool(value, "use_cuda_graph_decoder")?
            }
            "sherpa_model_name" => cfg.sherpa_model_name = as_string(value, "sherpa_model_name")?,
            "sherpa_model_dir" => cfg.sherpa_model_dir = as_opt_string(value)?,
            "sherpa_decode_mode" => {
                cfg.sherpa_decode_mode =
                    SherpaDecodeMode::from_str(&as_string(value, "sherpa_decode_mode")?)?
            }
            "sherpa_enable_parakeet_streaming" => {
                cfg.sherpa_enable_parakeet_streaming =
                    as_bool(value, "sherpa_enable_parakeet_streaming")?
            }
            "sherpa_provider" => {
                cfg.sherpa_provider =
                    ComputeProvider::from_str(&as_string(value, "sherpa_provider")?)?
            }
            "sherpa_num_threads" => cfg.sherpa_num_threads = as_u32(value, "sherpa_num_threads")?,
            "sherpa_chunk_ms" => cfg.sherpa_chunk_ms = as_u32(value, "sherpa_chunk_ms")?,
            "sherpa_offline_max_utterance_sec" => {
                cfg.sherpa_offline_max_utterance_sec =
                    as_f64(value, "sherpa_offline_max_utterance_sec")?
            }
            "moonshine_model_name" => {
                cfg.moonshine_model_name = as_string(value, "moonshine_model_name")?
            }
            "moonshine_model_dir" => cfg.moonshine_model_dir = as_opt_string(value)?,
            "moonshine_model_precision" => {
                cfg.moonshine_model_precision = as_string(value, "moonshine_model_precision")?
            }
            "moonshine_chunk_ms" => cfg.moonshine_chunk_ms = as_u32(value, "moonshine_chunk_ms")?,
            "moonshine_max_window_sec" => {
                cfg.moonshine_max_window_sec = as_f64(value, "moonshine_max_window_sec")?
            }
            "moonshine_max_tokens" => {
                cfg.moonshine_max_tokens = as_u32(value, "moonshine_max_tokens")?
            }
            "moonshine_provider" => {
                cfg.moonshine_provider =
                    ComputeProvider::from_str(&as_string(value, "moonshine_provider")?)?
            }
            "moonshine_onnx_threads" => {
                cfg.moonshine_onnx_threads = as_u32(value, "moonshine_onnx_threads")?
            }
            "openai_realtime_model" => {
                cfg.openai_realtime_model = as_string(value, "openai_realtime_model")?
            }
            "openai_realtime_api_key_env" => {
                cfg.openai_realtime_api_key_env = as_string(value, "openai_realtime_api_key_env")?
            }
            "openai_realtime_language" => {
                cfg.openai_realtime_language = as_string(value, "openai_realtime_language")?
            }
            "openai_realtime_latency_target_sec" => {
                cfg.openai_realtime_latency_target_sec =
                    as_f64(value, "openai_realtime_latency_target_sec")?
            }
            "openai_realtime_turn_detection" => {
                cfg.openai_realtime_turn_detection = OpenaiTurnDetection::from_str(&as_string(
                    value,
                    "openai_realtime_turn_detection",
                )?)?
            }
            "openai_realtime_vad_eagerness" => {
                cfg.openai_realtime_vad_eagerness = OpenaiVadEagerness::from_str(&as_string(
                    value,
                    "openai_realtime_vad_eagerness",
                )?)?
            }
            "openai_realtime_request_timeout_sec" => {
                cfg.openai_realtime_request_timeout_sec =
                    as_f64(value, "openai_realtime_request_timeout_sec")?
            }
            "openai_realtime_commit_timeout_sec" => {
                cfg.openai_realtime_commit_timeout_sec =
                    as_f64(value, "openai_realtime_commit_timeout_sec")?
            }
            "font_size" => cfg.font_size = as_u32(value, "font_size")?,
            "font_family" => cfg.font_family = as_opt_string(value)?,
            "bg_opacity" => cfg.bg_opacity = as_f64(value, "bg_opacity")?,
            "border_radius" => cfg.border_radius = as_u32(value, "border_radius")?,
            "bottom_margin" => cfg.bottom_margin = as_u32(value, "bottom_margin")?,
            "overlay_debug_mode" => cfg.overlay_debug_mode = as_bool(value, "overlay_debug_mode")?,
            "overlay_debug_max_lines" => {
                cfg.overlay_debug_max_lines = as_u32(value, "overlay_debug_max_lines")?
            }
            "control_socket" => cfg.control_socket = as_opt_string(value)?,
            "tts_enabled" => cfg.tts_enabled = as_bool(value, "tts_enabled")?,
            "tts_backend" => {
                cfg.tts_backend = TtsBackendKind::from_str(&as_string(value, "tts_backend")?)?
            }
            "tts_default_voice_id" => {
                cfg.tts_default_voice_id = as_string(value, "tts_default_voice_id")?
            }
            "tts_model_id" => cfg.tts_model_id = as_string(value, "tts_model_id")?,
            "tts_api_key_env" => cfg.tts_api_key_env = as_string(value, "tts_api_key_env")?,
            "tts_output_format" => cfg.tts_output_format = as_string(value, "tts_output_format")?,
            "tts_max_chars" => cfg.tts_max_chars = as_u32(value, "tts_max_chars")?,
            "tts_request_timeout_sec" => {
                cfg.tts_request_timeout_sec = as_f64(value, "tts_request_timeout_sec")?
            }
            "tts_playback_speed" => cfg.tts_playback_speed = as_f64(value, "tts_playback_speed")?,
            "tts_playback_device" => {
                cfg.tts_playback_device = as_device(
                    value,
                    "tts_playback_device",
                    DeviceDigitPolicy::ParseAsIndex,
                )?
            }
            "tts_overlay_auto_hide_sec" => {
                cfg.tts_overlay_auto_hide_sec = as_f64(value, "tts_overlay_auto_hide_sec")?
            }
            "tts_local_model_path" => cfg.tts_local_model_path = as_opt_string(value)?,
            "tts_local_voice" => cfg.tts_local_voice = as_opt_string(value)?,
            "tts_local_device" => {
                cfg.tts_local_device =
                    as_device(value, "tts_local_device", DeviceDigitPolicy::ParseAsIndex)?
            }
            "tts_melotts_device" => {
                cfg.tts_melotts_device =
                    MeloTtsDevice::from_str(&as_string(value, "tts_melotts_device")?)?
            }
            "tts_melotts_venv_path" => cfg.tts_melotts_venv_path = as_opt_string(value)?,
            "tts_kokoro_base_url" => {
                cfg.tts_kokoro_base_url = as_string(value, "tts_kokoro_base_url")?
            }
            "output_mode" => {
                cfg.output_mode = OutputMode::from_str(&as_string(value, "output_mode")?)?
            }
            "typing_final_injection_mode" => {
                cfg.typing_final_injection_mode =
                    InjectionMode::from_str(&as_string(value, "typing_final_injection_mode")?)?
            }
            "typing_text_case" => {
                cfg.typing_text_case =
                    TypingTextCase::from_str(&as_string(value, "typing_text_case")?)?
            }
            "use_clipboard_for_final" => {
                cfg.use_clipboard_for_final = as_bool(value, "use_clipboard_for_final")?
            }
            "preserve_clipboard" => cfg.preserve_clipboard = as_bool(value, "preserve_clipboard")?,
            "typing_clipboard_settle_delay_ms" => {
                cfg.typing_clipboard_settle_delay_ms =
                    as_u32(value, "typing_clipboard_settle_delay_ms")?
            }
            "typing_retry_attempts" => {
                cfg.typing_retry_attempts = as_u32(value, "typing_retry_attempts")?
            }
            "typing_retry_delay_ms" => {
                cfg.typing_retry_delay_ms = as_u32(value, "typing_retry_delay_ms")?
            }
            "typing_subprocess_timeout" => {
                cfg.typing_subprocess_timeout = as_f64(value, "typing_subprocess_timeout")?
            }
            "auto_capitalize" => cfg.auto_capitalize = as_bool(value, "auto_capitalize")?,
            "text_replacements" => {
                cfg.text_replacements = as_string_map(value, "text_replacements")?
            }
            "streaming_stall_guard" => {
                cfg.streaming_stall_guard = as_bool(value, "streaming_stall_guard")?
            }
            "streaming_stall_chunks" => {
                cfg.streaming_stall_chunks = as_u32(value, "streaming_stall_chunks")?
            }
            "streaming_stall_rms_ratio" => {
                cfg.streaming_stall_rms_ratio = as_f64(value, "streaming_stall_rms_ratio")?
            }
            "streaming_stall_flush_chunks" => {
                cfg.streaming_stall_flush_chunks = as_u32(value, "streaming_stall_flush_chunks")?
            }
            "audio_feedback" => cfg.audio_feedback = as_bool(value, "audio_feedback")?,
            "feedback_start_freq" => {
                cfg.feedback_start_freq = as_u32(value, "feedback_start_freq")?
            }
            "feedback_stop_freq" => cfg.feedback_stop_freq = as_u32(value, "feedback_stop_freq")?,
            "feedback_duration_ms" => {
                cfg.feedback_duration_ms = as_u32(value, "feedback_duration_ms")?
            }
            "feedback_volume" => cfg.feedback_volume = as_f64(value, "feedback_volume")?,
            // Unknown keys ignored (Python logs debug).
            _ => {}
        }
    }
    Ok(())
}

fn as_bool(value: &Value, field: &str) -> CoreResult<bool> {
    value
        .as_bool()
        .ok_or_else(|| CoreError::validation(format!("{field} must be true or false")))
}

fn as_u32(value: &Value, field: &str) -> CoreResult<u32> {
    if let Some(n) = value.as_u64() {
        return u32::try_from(n)
            .map_err(|_| CoreError::validation(format!("{field} out of range")));
    }
    if let Some(n) = value.as_i64() {
        if n < 0 {
            return Err(CoreError::validation(format!("{field} must be >= 0")));
        }
        return u32::try_from(n)
            .map_err(|_| CoreError::validation(format!("{field} out of range")));
    }
    // Permissive legacy TOML: accept finite integral floats in u32 range
    // (e.g. chunk_ms = 100.0). Non-integral / non-finite floats are rejected.
    if let Some(f) = value.as_f64() {
        if !f.is_finite() {
            return Err(CoreError::validation(format!(
                "{field} must be a finite integer"
            )));
        }
        if f < 0.0 || f.fract().abs() > f64::EPSILON {
            return Err(CoreError::validation(format!(
                "{field} must be an integer (got float {f})"
            )));
        }
        if f > f64::from(u32::MAX) {
            return Err(CoreError::validation(format!("{field} out of range")));
        }
        return Ok(f as u32);
    }
    if let Some(s) = value.as_str() {
        return s
            .parse::<u32>()
            .map_err(|_| CoreError::validation(format!("invalid {field}")));
    }
    Err(CoreError::validation(format!("invalid {field}")))
}

fn as_f64(value: &Value, field: &str) -> CoreResult<f64> {
    if let Some(n) = value.as_f64() {
        return Ok(n);
    }
    if let Some(n) = value.as_i64() {
        return Ok(n as f64);
    }
    if let Some(s) = value.as_str() {
        return s
            .parse::<f64>()
            .map_err(|_| CoreError::validation(format!("invalid {field}")));
    }
    Err(CoreError::validation(format!("invalid {field}")))
}

fn as_string(value: &Value, field: &str) -> CoreResult<String> {
    value
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| CoreError::validation(format!("{field} must be a string")))
}

fn as_opt_string(value: &Value) -> CoreResult<Option<String>> {
    if value.is_null() {
        return Ok(None);
    }
    let s = as_string(value, "string")?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(trimmed.to_string()))
    }
}

/// How digit-only string device values are interpreted.
///
/// Python diverges here:
/// - `audio_device` keeps digit strings as **names** (no `isdigit()` coercion).
/// - `tts_playback_device` / `tts_local_device` coerce digit strings to **indices**.
#[derive(Debug, Clone, Copy)]
enum DeviceDigitPolicy {
    KeepAsName,
    ParseAsIndex,
}

fn as_device(
    value: &Value,
    field: &str,
    digits: DeviceDigitPolicy,
) -> CoreResult<Option<DeviceRef>> {
    if value.is_null() {
        return Ok(None);
    }
    if let Some(n) = value.as_i64() {
        return Ok(Some(DeviceRef::Index(n)));
    }
    if let Some(n) = value.as_u64() {
        return Ok(Some(DeviceRef::Index(n as i64)));
    }
    // Finite integral floats → index (legacy TOML number forms).
    if let Some(f) = value.as_f64() {
        if f.is_finite() && f.fract().abs() <= f64::EPSILON {
            return Ok(Some(DeviceRef::Index(f as i64)));
        }
        return Err(CoreError::validation(format!(
            "{field} must be a string, integer, or null"
        )));
    }
    if let Some(s) = value.as_str() {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        match digits {
            DeviceDigitPolicy::ParseAsIndex
                if trimmed.chars().all(|c| c.is_ascii_digit()) && !trimmed.is_empty() =>
            {
                let idx: i64 = trimmed
                    .parse()
                    .map_err(|_| CoreError::validation(format!("invalid {field}")))?;
                return Ok(Some(DeviceRef::Index(idx)));
            }
            DeviceDigitPolicy::KeepAsName | DeviceDigitPolicy::ParseAsIndex => {
                return Ok(Some(DeviceRef::Name(trimmed.to_string())));
            }
        }
    }
    Err(CoreError::validation(format!(
        "{field} must be a string, integer, or null"
    )))
}

fn as_string_map(value: &Value, field: &str) -> CoreResult<BTreeMap<String, String>> {
    let obj = value.as_object().ok_or_else(|| {
        CoreError::validation(format!(
            "{field} must be a table/map of string keys to values"
        ))
    })?;
    let mut out = BTreeMap::new();
    for (k, v) in obj {
        let vs = v
            .as_str()
            .ok_or_else(|| CoreError::validation(format!("{field} values must be strings")))?;
        out.insert(k.clone(), vs.to_string());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::defaults::DEFAULT_TEXT_REPLACEMENTS;
    use std::sync::Mutex;
    use tempfile::tempdir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn load_defaults_when_config_missing() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = tempdir().unwrap();
        let cfg_home = dir.path().join("cfg");
        // SAFETY: ENV_LOCK serializes env mutation in this test binary.
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", &cfg_home);
        }
        let cfg = Config::load().unwrap();
        // SAFETY: paired cleanup under ENV_LOCK.
        unsafe {
            std::env::remove_var("XDG_CONFIG_HOME");
        }
        assert_eq!(cfg.sample_rate, 16000);
        assert_eq!(cfg.output_mode, OutputMode::FinalOnly);
        assert_eq!(cfg.typing_final_injection_mode, InjectionMode::Auto);
        assert_eq!(cfg.text_replacements, *DEFAULT_TEXT_REPLACEMENTS);
        assert_eq!(cfg.asr_backend, AsrBackendKind::Sherpa);
        assert_eq!(cfg.tts_backend, TtsBackendKind::Elevenlabs);
    }

    #[test]
    fn legacy_clipboard_flag_maps_to_mode() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "config_version = 1\n[typing]\nuse_clipboard_for_final = false\n",
        )
        .unwrap();
        let cfg = Config::load_from_path(&path).unwrap();
        assert_eq!(cfg.typing_final_injection_mode, InjectionMode::Direct);
    }

    #[test]
    fn instant_mode_parakeet_resolves_offline_instant() {
        let cfg = Config::try_with(|c| {
            c.instant_mode = true;
            c.sherpa_model_name = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8".into();
            c.sherpa_decode_mode = SherpaDecodeMode::Auto;
        })
        .unwrap();
        assert_eq!(
            cfg.resolved_sherpa_decode_mode(),
            Some(ResolvedSherpaDecodeMode::OfflineInstant)
        );
    }

    #[test]
    fn instant_mode_caps_sherpa_chunk_ms_in_streaming() {
        let cfg = Config::try_with(|c| {
            c.instant_mode = true;
            c.sherpa_chunk_ms = 100;
            c.sherpa_decode_mode = SherpaDecodeMode::Streaming;
        })
        .unwrap();
        assert_eq!(cfg.sherpa_chunk_ms, 80);
    }

    #[test]
    fn validation_rejects_bad_font_and_tts_backend() {
        assert!(
            Config::try_with(|c| c.font_size = 0)
                .unwrap_err()
                .to_string()
                .contains("font_size")
        );
        assert!(
            Config::try_with(|c| {
                c.font_family = Some("Sans\"; color: red;".into());
            })
            .unwrap_err()
            .to_string()
            .contains("font_family")
        );
        assert!(TtsBackendKind::from_str("azure").is_err());
    }

    #[test]
    fn openai_tts_defaults_remap() {
        let cfg = Config::try_with(|c| c.tts_backend = TtsBackendKind::Openai).unwrap();
        assert_eq!(cfg.tts_default_voice_id, DEFAULT_OPENAI_TTS_VOICE_ID);
        assert_eq!(cfg.tts_model_id, DEFAULT_OPENAI_TTS_MODEL_ID);
        assert_eq!(cfg.tts_api_key_env, DEFAULT_OPENAI_TTS_API_KEY_ENV);
    }
}
