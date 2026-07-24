//! First-run wizard state, defaults, summary, and validation (headless).
//!
//! Numeric/model defaults come from `shuvoice-core`. UI page copy and the
//! view-model remain here.

use serde::{Deserialize, Serialize};
use shuvoice_core::config::{
    DEFAULT_ELEVENLABS_TTS_VOICE_ID, DEFAULT_KOKORO_TTS_VOICE_ID, DEFAULT_LOCAL_TTS_VOICE_ID,
    DEFAULT_MELOTTS_VOICE_ID, DEFAULT_OPENAI_TTS_VOICE_ID,
    DEFAULT_SHERPA_MODEL_NAME as CORE_DEFAULT_SHERPA_MODEL_NAME,
    PARAKEET_TDT_V3_INT8_MODEL_NAME as CORE_PARAKEET_MODEL_NAME, wizard as core_wizard,
};
use shuvoice_core::{
    AsrBackendKind, InjectionMode, TtsBackendKind, TypingTextCase, format_tts_playback_speed,
    is_parakeet_model, normalize_tts_playback_speed, validate_tts_playback_speed,
};
use shuvoice_core::{config_path, wizard_done_path};

use crate::error::UiError;

/// Layer-shell namespace for the wizard window.
pub const WIZARD_NAMESPACE: &str = "shuvoice-wizard";

/// Wizard GTK application id.
pub const WIZARD_APPLICATION_ID: &str = "io.github.shuv1337.shuvoice.wizard";

pub const MARKER_FILE: &str = ".wizard-done";

/// Re-export core Zipformer default model name.
pub const DEFAULT_SHERPA_MODEL_NAME: &str = CORE_DEFAULT_SHERPA_MODEL_NAME;
/// Re-export core Parakeet TDT v3 int8 model name.
pub const PARAKEET_TDT_V3_INT8_MODEL_NAME: &str = CORE_PARAKEET_MODEL_NAME;
/// Wizard default Sherpa model (core wizard profile).
pub const DEFAULT_WIZARD_SHERPA_MODEL_NAME: &str = core_wizard::SHERPA_MODEL_NAME;
pub const DEFAULT_KEYBIND_ID: &str = "right_ctrl";
pub const DEFAULT_FINAL_INJECTION_MODE: &str = core_wizard::TYPING_FINAL_INJECTION_MODE;
pub const DEFAULT_TYPING_TEXT_CASE: &str = core_wizard::TYPING_TEXT_CASE;
pub const DEFAULT_TTS_BACKEND: &str = core_wizard::TTS_BACKEND;
pub const DEFAULT_TTS_PLAYBACK_SPEED: f64 = core_wizard::TTS_PLAYBACK_SPEED;
pub use shuvoice_core::config::DEFAULT_KOKORO_TTS_BASE_URL;
pub const DEFAULT_KOKORO_VOICE: &str = DEFAULT_KOKORO_TTS_VOICE_ID;
pub const DEFAULT_OPENAI_VOICE: &str = DEFAULT_OPENAI_TTS_VOICE_ID;
pub const DEFAULT_ELEVENLABS_VOICE: &str = DEFAULT_ELEVENLABS_TTS_VOICE_ID;
pub const DEFAULT_LOCAL_VOICE: &str = DEFAULT_LOCAL_TTS_VOICE_ID;
pub const DEFAULT_MELOTTS_VOICE: &str = DEFAULT_MELOTTS_VOICE_ID;

pub const KOKORO_PREVIEW_TEXT: &str = "Hello from ShuVoice. This is a Kokoro voice preview.";

/// Wizard stack page identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WizardPageId {
    Welcome,
    Asr,
    Keybind,
    Tts,
    Done,
}

impl WizardPageId {
    pub const ORDER: [Self; 5] = [
        Self::Welcome,
        Self::Asr,
        Self::Keybind,
        Self::Tts,
        Self::Done,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Welcome => "welcome",
            Self::Asr => "asr",
            Self::Keybind => "keybind",
            Self::Tts => "tts",
            Self::Done => "done",
        }
    }

    pub fn parse(value: &str) -> Result<Self, UiError> {
        match value {
            "welcome" => Ok(Self::Welcome),
            "asr" => Ok(Self::Asr),
            "keybind" => Ok(Self::Keybind),
            "tts" => Ok(Self::Tts),
            "done" => Ok(Self::Done),
            other => Err(UiError::invalid_state(
                "wizard_page",
                other,
                "asr, done, keybind, tts, welcome",
            )),
        }
    }

    pub fn next(self) -> Option<Self> {
        match self {
            Self::Welcome => Some(Self::Asr),
            Self::Asr => Some(Self::Keybind),
            Self::Keybind => Some(Self::Tts),
            Self::Tts => Some(Self::Done),
            Self::Done => None,
        }
    }

    pub fn back(self) -> Option<Self> {
        match self {
            Self::Welcome => None,
            Self::Asr => Some(Self::Welcome),
            Self::Keybind => Some(Self::Asr),
            Self::Tts => Some(Self::Keybind),
            Self::Done => Some(Self::Tts),
        }
    }
}

/// (id, label, description)
pub type OptionTriple = (&'static str, &'static str, &'static str);

pub const ASR_BACKENDS: &[OptionTriple] = &[
    (
        "sherpa",
        "Sherpa-ONNX",
        "Fast ONNX ASR with profiles for Streaming (Zipformer) or Instant (Parakeet).",
    ),
    (
        "nemo",
        "NeMo (NVIDIA)",
        "Highest accuracy streaming ASR.  Requires an NVIDIA GPU with CUDA.",
    ),
    (
        "moonshine",
        "Moonshine-ONNX",
        "Lightweight ONNX ASR with low resource usage.  CPU-friendly.",
    ),
    (
        "openai_realtime",
        "OpenAI Realtime Whisper",
        "Online low-latency Whisper transcription. Requires OPENAI_API_KEY.",
    ),
];

pub const FINAL_INJECTION_MODES: &[OptionTriple] = &[
    (
        "auto",
        "Auto (recommended)",
        "Uses clipboard paste by default. On XWayland apps it prefers xdotool-based paste, otherwise it falls back to direct typing when clipboard watchers are detected.",
    ),
    (
        "clipboard",
        "Clipboard paste (Ctrl+V)",
        "Copies final text to the clipboard and pastes with Ctrl+V. Best for apps that reject synthetic typing.",
    ),
    (
        "direct",
        "Direct typing (keystroke simulation)",
        "Types final text directly with wtype on Wayland or xdotool on XWayland, and avoids clipboard changes.",
    ),
];

pub const TYPING_TEXT_CASE_MODES: &[OptionTriple] = &[
    (
        "default",
        "Default",
        "Keeps ShuVoice's normal capitalization behavior for polished sentence-style output.",
    ),
    (
        "lowercase",
        "Lowercase",
        "Forces final committed STT output to lowercase for informal chat and casual conversation.",
    ),
];

/// (id, label, hypr_key_spec or None, description)
pub struct KeybindPreset {
    pub id: &'static str,
    pub label: &'static str,
    pub hypr_key_spec: Option<&'static str>,
    pub description: &'static str,
}

pub const KEYBIND_PRESETS: &[KeybindPreset] = &[
    KeybindPreset {
        id: "right_ctrl",
        label: "Right Control",
        hypr_key_spec: Some(", Control_R"),
        description: "Recommended default for hold-to-talk on most keyboards.",
    },
    KeybindPreset {
        id: "insert",
        label: "Insert",
        hypr_key_spec: Some(", Insert"),
        description: "Usually unused and easy to dedicate.",
    },
    KeybindPreset {
        id: "f9",
        label: "F9",
        hypr_key_spec: Some(", F9"),
        description: "Simple single-key push-to-talk.",
    },
    KeybindPreset {
        id: "super_v",
        label: "Super + V",
        hypr_key_spec: Some("SUPER, V"),
        description: "Modifier combo — mnemonic for Voice.",
    },
    KeybindPreset {
        id: "custom",
        label: "Custom",
        hypr_key_spec: None,
        description: "Set your own key in Hyprland config later.",
    },
];

pub const TTS_BACKENDS: &[OptionTriple] = &[
    (
        "elevenlabs",
        "ElevenLabs",
        "Cloud TTS with custom voice IDs. Keep the default voice or paste your own voice ID.",
    ),
    (
        "openai",
        "OpenAI",
        "Cloud TTS with built-in voice names like onyx, nova, and shimmer.",
    ),
    (
        "local",
        "Local Piper",
        "Use a local Piper .onnx model file or a directory of Piper voices already on disk.",
    ),
    (
        "melotts",
        "MeloTTS",
        "Local TTS using MeloTTS (MIT/MyShell). CPU real-time. 5 English voices.",
    ),
    (
        "kokoro",
        "Kokoro",
        "Local self-hosted TTS via an OpenAI-compatible API. Set the base URL for your Kokoro instance and choose a voice ID.",
    ),
];

pub const TTS_PLAYBACK_SPEED_PRESET_IDS: &[&str] = &["0.75", "1.0", "1.25", "1.5", "1.75", "2.0"];

pub const SHERPA_PROFILE_OPTIONS: &[OptionTriple] = &[
    (
        "streaming",
        "Streaming (Zipformer Kroko)",
        "Shows incremental transcript updates in the overlay while you hold push-to-talk. Final text is committed on key release.",
    ),
    (
        "instant",
        "Instant (Parakeet TDT v3 int8, recommended)",
        "Stable default profile. Emits one final transcript on key release and auto-enables instant_mode + sherpa_decode_mode=offline_instant.",
    ),
];

pub fn default_tts_voice_for_backend(backend: &str) -> &'static str {
    match backend.trim().to_ascii_lowercase().as_str() {
        "openai" => DEFAULT_OPENAI_VOICE,
        "local" => DEFAULT_LOCAL_VOICE,
        "melotts" => DEFAULT_MELOTTS_VOICE,
        "kokoro" => DEFAULT_KOKORO_VOICE,
        _ => DEFAULT_ELEVENLABS_VOICE,
    }
}

/// Human-readable voice label for wizard summaries and dropdowns.
pub fn tts_voice_label(backend: &str, voice_id: &str) -> String {
    let backend_id = backend.trim().to_ascii_lowercase();
    let mut value = voice_id.trim().to_string();
    if value.is_empty() {
        value = default_tts_voice_for_backend(&backend_id).to_string();
    }
    match backend_id.as_str() {
        "openai" => match value.to_ascii_lowercase().as_str() {
            "alloy" => "Alloy".into(),
            "ash" => "Ash".into(),
            "coral" => "Coral".into(),
            "echo" => "Echo".into(),
            "fable" => "Fable".into(),
            "nova" => "Nova".into(),
            "onyx" => "Onyx".into(),
            "sage" => "Sage".into(),
            "shimmer" => "Shimmer".into(),
            _ => value,
        },
        "melotts" => match value.as_str() {
            "EN-US" => "American English".into(),
            "EN-BR" => "British English".into(),
            "EN-INDIA" => "Indian English".into(),
            "EN-AU" => "Australian English".into(),
            "EN-Newest" => "Newest English".into(),
            _ => value,
        },
        "elevenlabs" if value == DEFAULT_ELEVENLABS_VOICE => format!("Default ({value})"),
        "kokoro" if value == DEFAULT_KOKORO_VOICE => format!("Default ({value})"),
        "local" if value.eq_ignore_ascii_case(DEFAULT_LOCAL_VOICE) => {
            "Auto (first discovered model)".into()
        }
        _ => value,
    }
}

pub fn tts_playback_speed_preset_id(speed: f64) -> &'static str {
    let value = if speed.is_finite() {
        speed
    } else {
        DEFAULT_TTS_PLAYBACK_SPEED
    };
    let mut best = "1.25";
    let mut best_diff = f64::INFINITY;
    for id in TTS_PLAYBACK_SPEED_PRESET_IDS {
        if let Ok(preset) = id.parse::<f64>() {
            let diff = (preset - value).abs();
            if diff < best_diff {
                best_diff = diff;
                best = id;
            }
        }
    }
    best
}

/// Whether the welcome wizard should run.
pub fn needs_wizard(marker_exists: bool, config_exists: bool) -> bool {
    if marker_exists {
        return false;
    }
    !config_exists
}

/// Filesystem-backed first-run check using core XDG paths.
pub fn needs_wizard_fs() -> bool {
    needs_wizard(wizard_done_path().is_file(), config_path().is_file())
}

/// Stable keybind setup status codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeybindSetupStatus {
    Added,
    AlreadyConfigured,
    Conflict,
    MissingConfig,
    SkippedCustom,
    NotAttempted,
    Error,
}

impl KeybindSetupStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::AlreadyConfigured => "already_configured",
            Self::Conflict => "conflict",
            Self::MissingConfig => "missing_config",
            Self::SkippedCustom => "skipped_custom",
            Self::NotAttempted => "not_attempted",
            Self::Error => "error",
        }
    }
}

pub fn finish_status_text(keybind_status: &str) -> &'static str {
    match keybind_status {
        "added" => "✓ Added push-to-talk keybind to Hyprland config.",
        "already_configured" => "✓ Push-to-talk keybind already configured.",
        "conflict" => "⚠ Selected key is already bound; Hyprland config unchanged.",
        "missing_config" => "⚠ Hyprland config not found; add keybind manually.",
        "skipped_custom" => "ℹ Custom keybind selected; configure it manually in Hyprland.",
        "not_attempted" => "ℹ Automatic keybind setup disabled.",
        "error" => "⚠ Could not update Hyprland config; check logs.",
        _ => "⚠ Keybind setup status unknown; check logs.",
    }
}

pub fn model_download_status_text(model_status: &str) -> &'static str {
    match model_status {
        "downloaded" => "✓ Model downloaded and ready.",
        "skipped" => "ℹ Model download skipped (backend downloads lazily).",
        "skipped_missing_deps" => {
            "⚠ Model not downloaded (missing dependencies). Run `shuvoice setup`."
        }
        "cancelled" => "ℹ Model download cancelled. You can run `shuvoice model download` later.",
        "incompatible_streaming" => {
            "⚠ Parakeet streaming is incompatible with this Sherpa runtime. Switched to Zipformer streaming profile."
        }
        "error" => "⚠ Model download failed. You can run `shuvoice model download` later.",
        _ => "",
    }
}

/// POSIX shell quoting for Hyprland `exec` payloads.
pub fn shell_quote(arg: &str) -> String {
    if arg.is_empty() {
        return "''".into();
    }
    if arg.chars().all(|c| {
        c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '-' | '+' | '=' | ',' | ':')
    }) {
        return arg.to_string();
    }
    format!("'{}'", arg.replace('\'', "'\"'\"'"))
}

/// Build `shuvoice control <cmd> --control-wait-sec 0` with a shell-quoted binary.
pub fn control_exec(command: &str, shuvoice_command: &str) -> String {
    format!(
        "{} control {command} --control-wait-sec 0",
        shell_quote(shuvoice_command)
    )
}

/// Format Hyprland bind/bindr lines for a PTT key.
pub fn format_hyprland_bind(hypr_key_spec: &str, shuvoice_command: &str) -> String {
    format!(
        "bind = {hypr_key_spec}, exec, {}\nbindr = {hypr_key_spec}, exec, {}",
        control_exec("start", shuvoice_command),
        control_exec("stop", shuvoice_command),
    )
}

pub fn format_hyprland_bind_for_keybind(
    keybind_id: &str,
    hypr_key_spec: &str,
    shuvoice_command: &str,
) -> String {
    let mut lines: Vec<String> = format_hyprland_bind(hypr_key_spec, shuvoice_command)
        .lines()
        .map(str::to_string)
        .collect();
    if keybind_id == "right_ctrl" {
        lines.push(format!(
            "bindr = CTRL, Control_R, exec, {}",
            control_exec("stop", shuvoice_command)
        ));
    }
    lines.push(format!(
        "bind = SUPER CTRL, S, exec, {}",
        control_exec("tts_speak", shuvoice_command)
    ));
    lines.push(format!(
        "bind = SUPER CTRL SHIFT, S, exec, {}",
        control_exec("tts_speak_clipboard", shuvoice_command)
    ));
    lines.join("\n")
}

/// Pure write-plan produced by the wizard (applied by app/io layer).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WizardWritePlan {
    pub asr_backend: String,
    pub sherpa_model_name: Option<String>,
    pub sherpa_enable_parakeet_streaming: bool,
    pub sherpa_provider: Option<String>,
    pub instant_mode: Option<bool>,
    pub sherpa_decode_mode: Option<String>,
    pub typing_final_injection_mode: String,
    pub typing_text_case: String,
    pub typing_output_mode: String,
    pub use_clipboard_for_final: bool,
    pub tts_backend: String,
    pub tts_default_voice_id: String,
    pub tts_playback_speed: f64,
    pub tts_local_model_path: Option<String>,
    pub tts_local_voice: Option<String>,
    pub tts_melotts_device: Option<String>,
    pub tts_kokoro_base_url: Option<String>,
    pub overwrite_existing: bool,
}

/// Headless wizard view-model / selection state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WizardVm {
    pub page: WizardPageId,
    pub force_reconfigure: bool,
    pub completed: bool,
    pub asr_backend: String,
    pub sherpa_model_name: String,
    pub sherpa_enable_parakeet_streaming: bool,
    pub sherpa_provider: String,
    pub typing_final_injection_mode: String,
    pub typing_text_case: String,
    pub tts_backend: String,
    pub tts_voice_id: String,
    pub tts_local_setup_mode: String,
    pub tts_local_model_path: String,
    pub tts_local_auto_voice_id: String,
    pub tts_melotts_device: String,
    pub tts_kokoro_base_url: String,
    pub tts_playback_speed: f64,
    pub keybind: String,
    pub auto_add_keybind: bool,
    pub auto_add_last_non_custom: bool,
    pub finish_in_progress: bool,
    pub finish_status: String,
    pub download_visible: bool,
    pub download_fraction: Option<f64>,
    pub download_text: String,
    pub download_note_visible: bool,
    pub cancel_download_visible: bool,
    pub launch_visible: bool,
    pub tts_config_error: Option<String>,
    pub summary: String,
}

impl WizardVm {
    pub fn new(force_reconfigure: bool) -> Self {
        let mut vm = Self {
            page: WizardPageId::Welcome,
            force_reconfigure,
            completed: false,
            asr_backend: core_wizard::ASR_BACKEND.into(),
            sherpa_model_name: DEFAULT_WIZARD_SHERPA_MODEL_NAME.into(),
            sherpa_enable_parakeet_streaming: false,
            sherpa_provider: core_wizard::SHERPA_PROVIDER.into(),
            typing_final_injection_mode: DEFAULT_FINAL_INJECTION_MODE.into(),
            typing_text_case: DEFAULT_TYPING_TEXT_CASE.into(),
            tts_backend: DEFAULT_TTS_BACKEND.into(),
            tts_voice_id: DEFAULT_KOKORO_VOICE.into(),
            tts_local_setup_mode: "automatic".into(),
            tts_local_model_path: String::new(),
            tts_local_auto_voice_id: "en_US-amy-medium".into(),
            tts_melotts_device: "auto".into(),
            tts_kokoro_base_url: core_wizard::TTS_KOKORO_BASE_URL.into(),
            tts_playback_speed: DEFAULT_TTS_PLAYBACK_SPEED,
            keybind: DEFAULT_KEYBIND_ID.into(),
            auto_add_keybind: true,
            auto_add_last_non_custom: true,
            finish_in_progress: false,
            finish_status: String::new(),
            download_visible: false,
            download_fraction: Some(0.0),
            download_text: String::new(),
            download_note_visible: false,
            cancel_download_visible: false,
            launch_visible: false,
            tts_config_error: None,
            summary: String::new(),
        };
        vm.refresh_summary();
        vm
    }

    pub fn navigate(&mut self, page: WizardPageId) {
        self.page = page;
        if page == WizardPageId::Done {
            self.refresh_summary();
        }
    }

    pub fn go_next(&mut self) -> Result<(), UiError> {
        if self.page == WizardPageId::Tts {
            self.validate_tts_selection()?;
        }
        if let Some(next) = self.page.next() {
            self.navigate(next);
        }
        Ok(())
    }

    pub fn go_back(&mut self) {
        if let Some(back) = self.page.back() {
            self.navigate(back);
        }
    }

    pub fn set_asr_backend(&mut self, backend: impl Into<String>) {
        self.asr_backend = backend.into();
    }

    pub fn set_sherpa_profile(&mut self, profile_id: &str) {
        if profile_id == "instant" {
            self.sherpa_model_name = PARAKEET_TDT_V3_INT8_MODEL_NAME.into();
            self.sherpa_enable_parakeet_streaming = false;
        } else {
            self.sherpa_model_name = DEFAULT_SHERPA_MODEL_NAME.into();
            self.sherpa_enable_parakeet_streaming = false;
        }
    }

    pub fn sherpa_profile_id(&self) -> &'static str {
        if self.sherpa_model_name == PARAKEET_TDT_V3_INT8_MODEL_NAME {
            "instant"
        } else {
            "streaming"
        }
    }

    pub fn set_keybind(&mut self, id: impl Into<String>) {
        self.keybind = id.into();
        self.sync_auto_add_keybind_state();
    }

    pub fn sync_auto_add_keybind_state(&mut self) {
        let hypr = KEYBIND_PRESETS
            .iter()
            .find(|p| p.id == self.keybind)
            .and_then(|p| p.hypr_key_spec);
        if hypr.is_none() {
            self.auto_add_last_non_custom = self.auto_add_keybind;
            self.auto_add_keybind = false;
        } else {
            self.auto_add_keybind = self.auto_add_last_non_custom;
        }
    }

    pub fn auto_add_enabled(&self) -> bool {
        self.auto_add_keybind
            && KEYBIND_PRESETS
                .iter()
                .find(|p| p.id == self.keybind)
                .and_then(|p| p.hypr_key_spec)
                .is_some()
    }

    pub fn set_tts_backend(&mut self, backend: impl Into<String>) {
        let backend = backend.into();
        self.tts_backend = backend.clone();
        self.tts_voice_id = default_tts_voice_for_backend(&backend).into();
        self.tts_config_error = None;
    }

    pub fn set_tts_playback_speed_preset(&mut self, preset_id: &str) {
        self.tts_playback_speed = preset_id
            .parse::<f64>()
            .map(normalize_tts_playback_speed)
            .unwrap_or(DEFAULT_TTS_PLAYBACK_SPEED);
        self.tts_config_error = None;
    }

    pub fn local_tts_auto_mode_enabled(&self) -> bool {
        self.tts_backend == "local" && self.tts_local_setup_mode == "automatic"
    }

    pub fn validate_tts_selection(&mut self) -> Result<(), UiError> {
        self.tts_config_error = None;
        if self.tts_backend == "kokoro" {
            let base = if self.tts_kokoro_base_url.trim().is_empty() {
                DEFAULT_KOKORO_TTS_BASE_URL
            } else {
                self.tts_kokoro_base_url.trim()
            };
            if !(base.starts_with("http://") || base.starts_with("https://"))
                || base == "http://"
                || base == "https://"
            {
                let msg = "Kokoro base URL must be a valid http(s) URL, for example http://localhost:8880/v1";
                self.tts_config_error = Some(msg.into());
                return Err(UiError::InvalidValue(msg.into()));
            }
            // require netloc-ish content after scheme
            let rest = base
                .trim_start_matches("https://")
                .trim_start_matches("http://");
            if rest.is_empty() || rest.starts_with('/') {
                let msg = "Kokoro base URL must be a valid http(s) URL, for example http://localhost:8880/v1";
                self.tts_config_error = Some(msg.into());
                return Err(UiError::InvalidValue(msg.into()));
            }
            self.tts_kokoro_base_url = base.trim_end_matches('/').to_string();
            return Ok(());
        }
        if self.tts_backend != "local" {
            return Ok(());
        }
        if self.local_tts_auto_mode_enabled() {
            if self.tts_local_auto_voice_id.trim().is_empty() {
                let msg = "Select a curated Local Piper voice for automatic setup.";
                self.tts_config_error = Some(msg.into());
                return Err(UiError::InvalidValue(msg.into()));
            }
            return Ok(());
        }
        if self.tts_local_model_path.trim().is_empty() {
            let msg = "Local Piper requires a .onnx model path or a directory of .onnx voices.";
            self.tts_config_error = Some(msg.into());
            return Err(UiError::InvalidValue(msg.into()));
        }
        Ok(())
    }

    pub fn refresh_summary(&mut self) {
        self.summary = format_summary(FormatSummaryArgs {
            asr_backend: &self.asr_backend,
            keybind_id: &self.keybind,
            auto_add_keybind: self.auto_add_enabled(),
            sherpa_model_name: Some(&self.sherpa_model_name),
            sherpa_enable_parakeet_streaming: self.sherpa_enable_parakeet_streaming,
            sherpa_provider: Some(&self.sherpa_provider),
            typing_final_injection_mode: &self.typing_final_injection_mode,
            typing_text_case: &self.typing_text_case,
            tts_backend: &self.tts_backend,
            tts_default_voice_id: Some(&self.tts_voice_id),
            tts_local_model_path: if self.tts_backend == "local" {
                Some(self.tts_local_model_path.as_str())
            } else {
                None
            },
            tts_kokoro_base_url: if self.tts_backend == "kokoro" {
                Some(self.tts_kokoro_base_url.as_str())
            } else {
                None
            },
            tts_playback_speed: Some(self.tts_playback_speed),
        });
    }

    pub fn build_write_plan(&self) -> Result<WizardWritePlan, UiError> {
        let injection = self.typing_final_injection_mode.trim().to_ascii_lowercase();
        let _injection_kind: InjectionMode = injection.parse().map_err(UiError::from)?;
        let text_case = self.typing_text_case.trim().to_ascii_lowercase();
        let _case_kind: TypingTextCase = text_case.parse().map_err(UiError::from)?;
        let tts_backend = self.tts_backend.trim().to_ascii_lowercase();
        let _tts_kind: TtsBackendKind = tts_backend.parse().map_err(UiError::from)?;
        let speed = validate_tts_playback_speed(self.tts_playback_speed).map_err(UiError::from)?;
        let _asr_kind: AsrBackendKind = self.asr_backend.parse().map_err(UiError::from)?;

        let mut plan = WizardWritePlan {
            asr_backend: self.asr_backend.clone(),
            sherpa_model_name: None,
            sherpa_enable_parakeet_streaming: false,
            sherpa_provider: None,
            instant_mode: None,
            sherpa_decode_mode: None,
            typing_final_injection_mode: injection.clone(),
            typing_text_case: text_case,
            typing_output_mode: "final_only".into(),
            use_clipboard_for_final: injection != "direct",
            tts_backend: tts_backend.clone(),
            tts_default_voice_id: if self.tts_voice_id.trim().is_empty() {
                default_tts_voice_for_backend(&tts_backend).into()
            } else {
                self.tts_voice_id.clone()
            },
            tts_playback_speed: speed,
            tts_local_model_path: None,
            tts_local_voice: None,
            tts_melotts_device: None,
            tts_kokoro_base_url: None,
            overwrite_existing: self.force_reconfigure,
        };

        if self.asr_backend == "sherpa" {
            let provider = self.sherpa_provider.trim().to_ascii_lowercase();
            if provider != "cpu" && provider != "cuda" {
                return Err(UiError::InvalidValue(
                    "sherpa_provider must be one of: cpu, cuda".into(),
                ));
            }
            let model = self.sherpa_model_name.clone();
            let is_parakeet = is_parakeet_model(&model);
            let enable_streaming = self.sherpa_enable_parakeet_streaming && is_parakeet;
            plan.sherpa_model_name = Some(model);
            plan.sherpa_enable_parakeet_streaming = enable_streaming;
            plan.sherpa_provider = Some(provider);
            if is_parakeet {
                if enable_streaming {
                    plan.instant_mode = Some(false);
                    plan.sherpa_decode_mode = Some("streaming".into());
                } else {
                    plan.instant_mode = Some(true);
                    plan.sherpa_decode_mode = Some("offline_instant".into());
                }
            } else {
                plan.instant_mode = Some(false);
                plan.sherpa_decode_mode = Some("auto".into());
            }
        }

        if tts_backend == "local" {
            if self.local_tts_auto_mode_enabled() {
                plan.tts_local_voice = Some(self.tts_local_auto_voice_id.clone());
                plan.tts_default_voice_id = self.tts_local_auto_voice_id.clone();
            } else if !self.tts_local_model_path.trim().is_empty() {
                plan.tts_local_model_path = Some(self.tts_local_model_path.clone());
            }
        } else if tts_backend == "melotts" {
            plan.tts_melotts_device = Some(self.tts_melotts_device.clone());
        } else if tts_backend == "kokoro" {
            plan.tts_kokoro_base_url = Some(
                self.tts_kokoro_base_url
                    .trim()
                    .trim_end_matches('/')
                    .to_string(),
            );
        }

        Ok(plan)
    }

    pub fn begin_finish(&mut self) {
        self.finish_in_progress = true;
        self.launch_visible = false;
        self.finish_status = "Applying settings…".into();
    }

    pub fn complete_finish(
        &mut self,
        keybind_status: &str,
        model_status: &str,
        model_message: &str,
    ) {
        let mut status = finish_status_text(keybind_status).to_string();
        let model_txt = model_download_status_text(model_status);
        if !model_txt.is_empty() {
            status = format!("{status}\n{model_txt}");
        }
        if !model_message.is_empty() {
            status = format!("{status}\n{model_message}");
        }
        self.finish_status = status;
        self.finish_in_progress = false;
        self.cancel_download_visible = false;
        self.download_note_visible = false;
        self.download_visible = true;
        self.download_fraction = Some(if model_status == "cancelled" {
            0.0
        } else {
            1.0
        });
        self.download_text = if model_status == "cancelled" {
            "Download cancelled — go Back to retry".into()
        } else {
            "Model setup finished".into()
        };
        self.launch_visible = true;
    }

    pub fn mark_launched(&mut self) {
        self.completed = true;
    }
}

/// Arguments for [`format_summary`].
pub struct FormatSummaryArgs<'a> {
    pub asr_backend: &'a str,
    pub keybind_id: &'a str,
    pub auto_add_keybind: bool,
    pub sherpa_model_name: Option<&'a str>,
    pub sherpa_enable_parakeet_streaming: bool,
    pub sherpa_provider: Option<&'a str>,
    pub typing_final_injection_mode: &'a str,
    pub typing_text_case: &'a str,
    pub tts_backend: &'a str,
    pub tts_default_voice_id: Option<&'a str>,
    pub tts_local_model_path: Option<&'a str>,
    pub tts_kokoro_base_url: Option<&'a str>,
    pub tts_playback_speed: Option<f64>,
}

pub fn format_summary(args: FormatSummaryArgs<'_>) -> String {
    let asr_name = ASR_BACKENDS
        .iter()
        .find(|(id, ..)| *id == args.asr_backend)
        .map(|(_, label, _)| *label)
        .unwrap_or(args.asr_backend);

    let (keybind_label, hypr_key) = KEYBIND_PRESETS
        .iter()
        .find(|p| p.id == args.keybind_id)
        .map(|p| (p.label, p.hypr_key_spec))
        .unwrap_or(("Custom", None));

    let injection_label = match args.typing_final_injection_mode {
        "auto" => "Auto (recommended)",
        "clipboard" => "Clipboard paste (Ctrl+V)",
        "direct" => "Direct typing (keystroke simulation)",
        other => other,
    };
    let text_case_label = match args.typing_text_case {
        "default" => "Default",
        "lowercase" => "Lowercase",
        other => other,
    };

    let tts_backend = if args.tts_backend.trim().is_empty() {
        DEFAULT_TTS_BACKEND
    } else {
        args.tts_backend
    };
    let tts_label = TTS_BACKENDS
        .iter()
        .find(|(id, ..)| *id == tts_backend)
        .map(|(_, l, _)| *l)
        .unwrap_or(tts_backend);
    let tts_voice = args
        .tts_default_voice_id
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default_tts_voice_for_backend(tts_backend));
    let speed = args
        .tts_playback_speed
        .and_then(|s| validate_tts_playback_speed(s).ok())
        .unwrap_or(DEFAULT_TTS_PLAYBACK_SPEED);
    let speed_label = format_tts_playback_speed(speed);

    let mut lines = vec![
        format!("ASR backend:      {asr_name}"),
        format!("Final injection:  {injection_label}"),
        format!("Text case:        {text_case_label}"),
        format!("TTS provider:     {tts_label}"),
        format!("TTS voice:        {tts_voice}"),
        format!("TTS speed:        {speed_label}"),
        format!("Push-to-talk:     {keybind_label}"),
    ];

    if tts_backend == "local" {
        let path = args.tts_local_model_path.unwrap_or("").trim();
        let path = if path.is_empty() {
            "Not configured"
        } else {
            path
        };
        lines.insert(5, format!("TTS model path:   {path}"));
    } else if tts_backend == "kokoro" {
        let url = args
            .tts_kokoro_base_url
            .unwrap_or(DEFAULT_KOKORO_TTS_BASE_URL)
            .trim_end_matches('/');
        lines.insert(5, format!("TTS base URL:     {url}"));
    }

    if args.asr_backend == "sherpa" {
        let chosen = args
            .sherpa_model_name
            .unwrap_or(DEFAULT_WIZARD_SHERPA_MODEL_NAME);
        let is_parakeet = is_parakeet_model(chosen);
        let parakeet_streaming = args.sherpa_enable_parakeet_streaming && is_parakeet;
        let model_label = if chosen == PARAKEET_TDT_V3_INT8_MODEL_NAME {
            "Parakeet TDT v3 (int8)"
        } else if chosen == DEFAULT_SHERPA_MODEL_NAME {
            "Zipformer Kroko (default)"
        } else {
            chosen
        };
        let (profile_label, decode_label) = if parakeet_streaming {
            ("Streaming (Parakeet)", "Streaming (explicit override)")
        } else if is_parakeet {
            ("Instant (Parakeet)", "Offline instant (auto-enabled)")
        } else {
            ("Streaming", "Streaming (auto)")
        };
        let provider_label = if args.sherpa_provider.unwrap_or("cpu") == "cuda" {
            "GPU (CUDA)"
        } else {
            "CPU"
        };
        lines.insert(1, format!("Sherpa profile: {profile_label}"));
        lines.insert(2, format!("Sherpa device:  {provider_label}"));
        lines.insert(3, format!("Sherpa model:   {model_label}"));
        lines.insert(4, format!("Sherpa decode:  {decode_label}"));
        lines.insert(5, "Output mode:    final_only".into());
    } else if args.asr_backend == "openai_realtime" {
        lines.insert(1, "ASR model:        gpt-4o-transcribe".into());
        lines.insert(2, "ASR language:     en".into());
        lines.insert(3, "ASR API key env:  OPENAI_API_KEY".into());
        lines.insert(4, "ASR turn detect:  manual".into());
    }

    if let Some(hypr) = hypr_key {
        let bind_lines = format_hyprland_bind_for_keybind(args.keybind_id, hypr, "shuvoice");
        let indented = bind_lines
            .lines()
            .map(|l| format!("  {l}"))
            .collect::<Vec<_>>()
            .join("\n");
        lines.push(String::new());
        if args.auto_add_keybind {
            lines.push("Wizard will try to add this to ~/.config/hypr/hyprland.conf".into());
            lines.push("(only if no conflicting bind already uses that key):".into());
        } else {
            lines.push("Add to ~/.config/hypr/hyprland.conf:".into());
        }
        lines.push(String::new());
        lines.push(indented);
    } else {
        lines.push(String::new());
        lines.push("Configure your keybind in ~/.config/hypr/hyprland.conf".into());
        lines.push("See README.md for bind/bindr examples.".into());
    }

    lines.join("\n")
}

/// Wizard window CSS.
pub fn wizard_css() -> &'static str {
    r#"window.wizard-window { background-color: rgba(15, 15, 20, 0.95); }
.wizard-page {
  padding: 48px 64px;
}
.wizard-title {
  color: white;
  font-size: 28px;
  font-weight: bold;
}
.wizard-subtitle {
  color: rgba(255, 255, 255, 0.7);
  font-size: 16px;
  margin-top: 4px;
  margin-bottom: 16px;
}
.wizard-desc {
  color: rgba(255, 255, 255, 0.55);
  font-size: 14px;
}
.wizard-radio label {
  color: white;
  font-size: 16px;
}
.wizard-radio-desc {
  color: rgba(255, 255, 255, 0.55);
  font-size: 13px;
  margin-left: 28px;
  margin-bottom: 8px;
}
.wizard-dropdown {
  min-width: 360px;
  margin-left: 4px;
  margin-top: 4px;
  margin-bottom: 4px;
}
.wizard-entry {
  min-width: 420px;
  margin-left: 28px;
  margin-top: 4px;
  margin-bottom: 4px;
}
.wizard-btn {
  padding: 8px 24px;
  font-size: 15px;
  border-radius: 8px;
  background-color: rgba(255, 255, 255, 0.12);
  color: white;
}
.wizard-btn:hover {
  background-color: rgba(255, 255, 255, 0.2);
}
.wizard-btn:focus-visible {
  outline: 2px solid rgba(255, 255, 255, 0.8);
  outline-offset: 2px;
}
.wizard-btn-primary {
  background-color: rgba(60, 120, 220, 0.85);
}
.wizard-btn-primary:hover {
  background-color: rgba(60, 120, 220, 1.0);
}
.wizard-summary {
  color: rgba(255, 255, 255, 0.8);
  font-size: 15px;
  font-family: monospace;
  background-color: rgba(255, 255, 255, 0.06);
  border-radius: 8px;
  padding: 16px 20px;
  margin-top: 12px;
}
"#
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn summary_for(vm: &WizardVm) -> String {
        format_summary(FormatSummaryArgs {
            asr_backend: &vm.asr_backend,
            keybind_id: &vm.keybind,
            auto_add_keybind: vm.auto_add_enabled(),
            sherpa_model_name: Some(&vm.sherpa_model_name),
            sherpa_enable_parakeet_streaming: vm.sherpa_enable_parakeet_streaming,
            sherpa_provider: Some(&vm.sherpa_provider),
            typing_final_injection_mode: &vm.typing_final_injection_mode,
            typing_text_case: &vm.typing_text_case,
            tts_backend: &vm.tts_backend,
            tts_default_voice_id: Some(&vm.tts_voice_id),
            tts_local_model_path: if vm.tts_backend == "local" {
                Some(vm.tts_local_model_path.as_str())
            } else {
                None
            },
            tts_kokoro_base_url: if vm.tts_backend == "kokoro" {
                Some(vm.tts_kokoro_base_url.as_str())
            } else {
                None
            },
            tts_playback_speed: Some(vm.tts_playback_speed),
        })
    }

    #[test]
    fn needs_wizard_true_on_fresh_install() {
        assert!(needs_wizard(false, false));
    }

    #[test]
    fn needs_wizard_false_after_marker_or_config() {
        assert!(!needs_wizard(true, false));
        assert!(!needs_wizard(false, true));
        assert!(!needs_wizard(true, true));
    }

    #[test]
    fn defaults_to_parakeet_instant_profile() {
        let vm = WizardVm::new(false);
        assert_eq!(vm.asr_backend, "sherpa");
        assert_eq!(vm.sherpa_model_name, PARAKEET_TDT_V3_INT8_MODEL_NAME);
        assert_eq!(vm.sherpa_provider, "cpu");
        assert_eq!(vm.tts_backend, "kokoro");
        assert!((vm.tts_playback_speed - 1.25).abs() < 1e-9);
        assert_eq!(vm.keybind, "right_ctrl");
        assert_eq!(vm.typing_final_injection_mode, DEFAULT_FINAL_INJECTION_MODE);
        assert_eq!(vm.typing_text_case, DEFAULT_TYPING_TEXT_CASE);
        assert_eq!(vm.page, WizardPageId::Welcome);
    }

    #[test]
    fn page_sequence() {
        let mut vm = WizardVm::new(false);
        assert_eq!(
            WizardPageId::ORDER.map(|p| p.as_str()),
            ["welcome", "asr", "keybind", "tts", "done"]
        );
        vm.go_next().unwrap();
        assert_eq!(vm.page, WizardPageId::Asr);
        vm.go_next().unwrap();
        assert_eq!(vm.page, WizardPageId::Keybind);
        vm.go_next().unwrap();
        assert_eq!(vm.page, WizardPageId::Tts);
        vm.go_next().unwrap();
        assert_eq!(vm.page, WizardPageId::Done);
    }

    #[test]
    fn catalog_tables_cover_public_surface() {
        let asr_ids: Vec<_> = ASR_BACKENDS.iter().map(|(id, ..)| *id).collect();
        for required in ["sherpa", "nemo", "moonshine", "openai_realtime"] {
            assert!(asr_ids.contains(&required), "missing asr {required}");
        }
        let tts_ids: Vec<_> = TTS_BACKENDS.iter().map(|(id, ..)| *id).collect();
        for required in ["elevenlabs", "openai", "local", "melotts", "kokoro"] {
            assert!(tts_ids.contains(&required), "missing tts {required}");
        }
        assert!(KEYBIND_PRESETS.len() >= 4);
        let kb_ids: Vec<_> = KEYBIND_PRESETS.iter().map(|p| p.id).collect();
        assert!(kb_ids.contains(&"insert"));
        assert!(kb_ids.contains(&"right_ctrl"));
        assert!(kb_ids.contains(&"custom"));
        let custom = KEYBIND_PRESETS.iter().find(|p| p.id == "custom").unwrap();
        assert!(custom.hypr_key_spec.is_none());
        assert_eq!(
            TTS_PLAYBACK_SPEED_PRESET_IDS,
            &["0.75", "1.0", "1.25", "1.5", "1.75", "2.0"]
        );
        for id in TTS_PLAYBACK_SPEED_PRESET_IDS {
            let v: f64 = id.parse().unwrap();
            assert!((0.5..=2.0).contains(&v));
        }
    }

    #[rstest]
    #[case::openai("openai", DEFAULT_OPENAI_VOICE)]
    #[case::local("local", DEFAULT_LOCAL_VOICE)]
    #[case::melotts("melotts", DEFAULT_MELOTTS_VOICE)]
    #[case::kokoro("kokoro", DEFAULT_KOKORO_VOICE)]
    #[case::elevenlabs("elevenlabs", DEFAULT_ELEVENLABS_VOICE)]
    fn default_tts_voice_matrix(#[case] backend: &str, #[case] expected: &str) {
        assert_eq!(default_tts_voice_for_backend(backend), expected);
    }

    #[rstest]
    #[case::melotts_us("melotts", "EN-US", "American English")]
    #[case::melotts_br("melotts", "EN-BR", "British English")]
    #[case::melotts_unknown("melotts", "xx", "xx")]
    #[case::kokoro_default("kokoro", "af_heart", "Default (af_heart)")]
    #[case::kokoro_custom("kokoro", "bm_george", "bm_george")]
    #[case::local_auto("local", "default", "Auto (first discovered model)")]
    #[case::openai_nova("openai", "nova", "Nova")]
    fn tts_voice_label_matrix(#[case] backend: &str, #[case] voice: &str, #[case] label: &str) {
        assert_eq!(tts_voice_label(backend, voice), label);
    }

    #[rstest]
    #[case::parakeet_instant(
        "sherpa",
        PARAKEET_TDT_V3_INT8_MODEL_NAME,
        false,
        "cpu",
        Some(true),
        Some("offline_instant")
    )]
    #[case::parakeet_streaming(
        "sherpa",
        PARAKEET_TDT_V3_INT8_MODEL_NAME,
        true,
        "cuda",
        Some(false),
        Some("streaming")
    )]
    #[case::zipformer_streaming(
        "sherpa",
        DEFAULT_SHERPA_MODEL_NAME,
        false,
        "cpu",
        Some(false),
        Some("auto")
    )]
    #[case::nemo("nemo", "", false, "", None, None)]
    #[case::moonshine("moonshine", "", false, "", None, None)]
    #[case::openai_rt("openai_realtime", "", false, "", None, None)]
    fn write_plan_asr_profile_matrix(
        #[case] asr: &str,
        #[case] model: &str,
        #[case] parakeet_streaming: bool,
        #[case] provider: &str,
        #[case] instant: Option<bool>,
        #[case] decode: Option<&str>,
    ) {
        let mut vm = WizardVm::new(false);
        vm.set_asr_backend(asr);
        if asr == "sherpa" {
            vm.sherpa_model_name = model.into();
            vm.sherpa_enable_parakeet_streaming = parakeet_streaming;
            vm.sherpa_provider = provider.into();
        }
        let plan = vm.build_write_plan().unwrap();
        assert_eq!(plan.asr_backend, asr);
        assert_eq!(plan.instant_mode, instant);
        assert_eq!(plan.sherpa_decode_mode.as_deref(), decode);
        if asr == "sherpa" {
            assert_eq!(plan.sherpa_model_name.as_deref(), Some(model));
            assert_eq!(plan.sherpa_provider.as_deref(), Some(provider));
            assert_eq!(plan.sherpa_enable_parakeet_streaming, parakeet_streaming);
        } else {
            assert!(plan.sherpa_model_name.is_none());
        }
        assert_eq!(plan.typing_output_mode, "final_only");
        assert_eq!(plan.typing_final_injection_mode, "auto");
        assert!(plan.use_clipboard_for_final);
        assert_eq!(plan.tts_backend, "kokoro");
        assert!((plan.tts_playback_speed - 1.25).abs() < 1e-9);
        assert!(!plan.overwrite_existing);
    }

    #[rstest]
    #[case::direct("direct", false)]
    #[case::clipboard("clipboard", true)]
    #[case::auto("auto", true)]
    fn write_plan_injection_mode_matrix(#[case] mode: &str, #[case] use_clipboard: bool) {
        let mut vm = WizardVm::new(false);
        vm.typing_final_injection_mode = mode.into();
        let plan = vm.build_write_plan().unwrap();
        assert_eq!(plan.typing_final_injection_mode, mode);
        assert_eq!(plan.use_clipboard_for_final, use_clipboard);
    }

    #[rstest]
    #[case::elevenlabs("elevenlabs", DEFAULT_ELEVENLABS_VOICE, None, None, None)]
    #[case::openai("openai", "nova", None, None, None)]
    #[case::local_auto("local", "en_US-amy-medium", None, Some("en_US-amy-medium"), None)]
    #[case::local_manual_path("local", "default", Some("/tmp/voice.onnx"), None, None)]
    #[case::melotts("melotts", "EN-BR", None, None, Some("cuda"))]
    #[case::kokoro("kokoro", "af_heart", None, None, None)]
    fn write_plan_tts_backend_matrix(
        #[case] backend: &str,
        #[case] voice: &str,
        #[case] local_path: Option<&str>,
        #[case] local_voice: Option<&str>,
        #[case] melo_device: Option<&str>,
    ) {
        let mut vm = WizardVm::new(false);
        vm.set_tts_backend(backend);
        vm.tts_voice_id = voice.into();
        if backend == "local" {
            if let Some(path) = local_path {
                vm.tts_local_setup_mode = "manual".into();
                vm.tts_local_model_path = path.into();
            } else {
                vm.tts_local_setup_mode = "automatic".into();
                vm.tts_local_auto_voice_id = local_voice.unwrap_or("en_US-amy-medium").into();
            }
        }
        if let Some(dev) = melo_device {
            vm.tts_melotts_device = dev.into();
        }
        if backend == "kokoro" {
            vm.tts_kokoro_base_url = "http://127.0.0.1:8880/v1/".into();
        }
        let plan = vm.build_write_plan().unwrap();
        assert_eq!(plan.tts_backend, backend);
        if backend == "local" && local_voice.is_some() {
            assert_eq!(plan.tts_default_voice_id, local_voice.unwrap());
            assert_eq!(plan.tts_local_voice.as_deref(), local_voice);
            assert!(plan.tts_local_model_path.is_none());
        } else if backend == "local" {
            assert_eq!(plan.tts_local_model_path.as_deref(), local_path);
        } else {
            assert_eq!(plan.tts_default_voice_id, voice);
        }
        if backend == "melotts" {
            assert_eq!(plan.tts_melotts_device.as_deref(), melo_device);
        }
        if backend == "kokoro" {
            assert_eq!(
                plan.tts_kokoro_base_url.as_deref(),
                Some("http://127.0.0.1:8880/v1")
            );
        }
    }

    #[rstest]
    #[case::ok_default(1.25, true)]
    #[case::ok_min(0.5, true)]
    #[case::ok_max(2.0, true)]
    #[case::bad_high(3.0, false)]
    #[case::bad_low(0.49, false)]
    #[case::bad_nan(f64::NAN, false)]
    fn write_plan_playback_speed_validation(#[case] speed: f64, #[case] ok: bool) {
        let mut vm = WizardVm::new(false);
        vm.tts_playback_speed = speed;
        let result = vm.build_write_plan();
        assert_eq!(result.is_ok(), ok, "speed={speed} result={result:?}");
        if ok {
            assert!((result.unwrap().tts_playback_speed - speed).abs() < 1e-9);
        }
    }

    #[rstest]
    #[case::bad_injection("typing_final_injection_mode", "invalid")]
    #[case::bad_case("typing_text_case", "titlecase")]
    #[case::bad_provider("sherpa_provider", "rocm")]
    #[case::bad_tts("tts_backend", "nonexistent")]
    fn write_plan_rejects_invalid_enums(#[case] field: &str, #[case] value: &str) {
        let mut vm = WizardVm::new(false);
        match field {
            "typing_final_injection_mode" => vm.typing_final_injection_mode = value.into(),
            "typing_text_case" => vm.typing_text_case = value.into(),
            "sherpa_provider" => vm.sherpa_provider = value.into(),
            "tts_backend" => vm.tts_backend = value.into(),
            _ => unreachable!(),
        }
        assert!(
            vm.build_write_plan().is_err(),
            "expected err for {field}={value}"
        );
    }

    #[rstest]
    #[case::bad_url("not-a-url")]
    #[case::bare_scheme("http://")]
    #[case::no_host("http:///v1")]
    fn validate_tts_rejects_bad_kokoro_url(#[case] url: &str) {
        let mut vm = WizardVm::new(false);
        vm.set_tts_backend("kokoro");
        vm.tts_kokoro_base_url = url.into();
        assert!(vm.validate_tts_selection().is_err());
    }

    #[test]
    fn validate_tts_local_requires_path_or_auto_voice() {
        let mut vm = WizardVm::new(false);
        vm.set_tts_backend("local");
        vm.tts_local_setup_mode = "manual".into();
        vm.tts_local_model_path.clear();
        assert!(vm.validate_tts_selection().is_err());
        vm.tts_local_setup_mode = "automatic".into();
        vm.tts_local_auto_voice_id.clear();
        assert!(vm.validate_tts_selection().is_err());
        vm.tts_local_auto_voice_id = "en_US-amy-medium".into();
        assert!(vm.validate_tts_selection().is_ok());
    }

    #[test]
    fn force_reconfigure_sets_overwrite_flag() {
        let vm = WizardVm::new(true);
        let plan = vm.build_write_plan().unwrap();
        assert!(plan.overwrite_existing);
    }

    #[test]
    fn sherpa_profile_selection() {
        let mut vm = WizardVm::new(false);
        assert_eq!(vm.sherpa_profile_id(), "instant");
        vm.set_sherpa_profile("streaming");
        assert_eq!(vm.sherpa_model_name, DEFAULT_SHERPA_MODEL_NAME);
        assert_eq!(vm.sherpa_profile_id(), "streaming");
        vm.set_sherpa_profile("instant");
        assert_eq!(vm.sherpa_model_name, PARAKEET_TDT_V3_INT8_MODEL_NAME);
    }

    #[test]
    fn custom_keybind_disables_auto_add() {
        let mut vm = WizardVm::new(false);
        assert!(vm.auto_add_enabled());
        vm.set_keybind("custom");
        assert!(!vm.auto_add_enabled());
        vm.set_keybind("insert");
        assert!(vm.auto_add_enabled());
    }

    #[derive(Clone, Copy)]
    struct SummaryCase {
        asr: &'static str,
        model: &'static str,
        parakeet_stream: bool,
        provider: &'static str,
        keybind: &'static str,
        auto_add: bool,
        injection: &'static str,
        text_case: &'static str,
        tts: &'static str,
        voice: &'static str,
        local_path: Option<&'static str>,
        kokoro_url: Option<&'static str>,
        speed: f64,
        must_contain: &'static [&'static str],
    }

    #[rstest]
    #[case::default_parakeet(SummaryCase {
        asr: "sherpa",
        model: PARAKEET_TDT_V3_INT8_MODEL_NAME,
        parakeet_stream: false,
        provider: "cpu",
        keybind: "right_ctrl",
        auto_add: true,
        injection: "auto",
        text_case: "default",
        tts: "kokoro",
        voice: "af_heart",
        local_path: None,
        kokoro_url: Some("http://localhost:8880/v1"),
        speed: 1.25,
        must_contain: &[
            "Sherpa-ONNX",
            "Right Control",
            "Instant (Parakeet)",
            "bind = , Control_R",
            "tts_speak",
            "--control-wait-sec 0",
            "1.25",
        ],
    })]
    #[case::zipformer_cuda(SummaryCase {
        asr: "sherpa",
        model: DEFAULT_SHERPA_MODEL_NAME,
        parakeet_stream: false,
        provider: "cuda",
        keybind: "insert",
        auto_add: true,
        injection: "clipboard",
        text_case: "lowercase",
        tts: "openai",
        voice: "nova",
        local_path: None,
        kokoro_url: None,
        speed: 1.5,
        must_contain: &[
            "Zipformer",
            "GPU (CUDA)",
            "Streaming (auto)",
            "Insert",
            "Clipboard paste",
            "Lowercase",
            "OpenAI",
            "nova",
            "1.5×",
        ],
    })]
    #[case::parakeet_streaming_label(SummaryCase {
        asr: "sherpa",
        model: PARAKEET_TDT_V3_INT8_MODEL_NAME,
        parakeet_stream: true,
        provider: "cpu",
        keybind: "f9",
        auto_add: true,
        injection: "direct",
        text_case: "default",
        tts: "melotts",
        voice: "EN-US",
        local_path: None,
        kokoro_url: None,
        speed: 1.0,
        must_contain: &[
            "Streaming (Parakeet)",
            "explicit override",
            "MeloTTS",
            "EN-US",
            "Direct typing",
        ],
    })]
    #[case::local_tts_path(SummaryCase {
        asr: "moonshine",
        model: "",
        parakeet_stream: false,
        provider: "",
        keybind: "super_v",
        auto_add: true,
        injection: "auto",
        text_case: "default",
        tts: "local",
        voice: "amy",
        local_path: Some("/models/piper"),
        kokoro_url: None,
        speed: 1.25,
        must_contain: &[
            "Moonshine",
            "Local Piper",
            "TTS model path:",
            "/models/piper",
            "Super + V",
        ],
    })]
    #[case::custom_keybind_readme(SummaryCase {
        asr: "nemo",
        model: "",
        parakeet_stream: false,
        provider: "",
        keybind: "custom",
        auto_add: false,
        injection: "auto",
        text_case: "default",
        tts: "elevenlabs",
        voice: "abc",
        local_path: None,
        kokoro_url: None,
        speed: 1.25,
        must_contain: &["NeMo", "Custom", "README.md", "hyprland.conf"],
    })]
    #[case::manual_copy_hint(SummaryCase {
        asr: "sherpa",
        model: DEFAULT_SHERPA_MODEL_NAME,
        parakeet_stream: false,
        provider: "cpu",
        keybind: "insert",
        auto_add: false,
        injection: "auto",
        text_case: "default",
        tts: "kokoro",
        voice: "af_heart",
        local_path: None,
        kokoro_url: Some("http://localhost:8880/v1"),
        speed: 1.25,
        must_contain: &["Add to ~/.config/hypr/hyprland.conf:", "bind = , Insert"],
    })]
    fn format_summary_matrix(#[case] c: SummaryCase) {
        let text = format_summary(FormatSummaryArgs {
            asr_backend: c.asr,
            keybind_id: c.keybind,
            auto_add_keybind: c.auto_add,
            sherpa_model_name: if c.model.is_empty() {
                None
            } else {
                Some(c.model)
            },
            sherpa_enable_parakeet_streaming: c.parakeet_stream,
            sherpa_provider: if c.provider.is_empty() {
                None
            } else {
                Some(c.provider)
            },
            typing_final_injection_mode: c.injection,
            typing_text_case: c.text_case,
            tts_backend: c.tts,
            tts_default_voice_id: Some(c.voice),
            tts_local_model_path: c.local_path,
            tts_kokoro_base_url: c.kokoro_url,
            tts_playback_speed: Some(c.speed),
        });
        for needle in c.must_contain {
            assert!(
                text.contains(needle),
                "summary missing {needle:?}\n---\n{text}\n---"
            );
        }
        if c.keybind == "right_ctrl" {
            assert!(text.contains("bindr = CTRL, Control_R"));
        }
    }

    #[test]
    fn shell_quote_and_control_exec() {
        assert_eq!(shell_quote("shuvoice"), "shuvoice");
        assert_eq!(
            shell_quote("/opt/shu voice/bin/shuvoice"),
            "'/opt/shu voice/bin/shuvoice'"
        );
        assert_eq!(
            control_exec("start", "/tmp/a b/shuvoice"),
            "'/tmp/a b/shuvoice' control start --control-wait-sec 0"
        );
        let quoted = format_hyprland_bind_for_keybind("insert", ", Insert", "/tmp/a b/shuvoice");
        assert!(quoted.contains("--control-wait-sec 0"));
        assert!(quoted.contains("'/tmp/a b/shuvoice'"));
    }

    #[test]
    fn format_hyprland_bind_variants() {
        let no_mod = format_hyprland_bind(", Insert", "shuvoice");
        assert!(
            no_mod.contains("bind = , Insert, exec, shuvoice control start --control-wait-sec 0")
        );
        assert!(
            no_mod.contains("bindr = , Insert, exec, shuvoice control stop --control-wait-sec 0")
        );

        let with_mod = format_hyprland_bind("SUPER, V", "/opt/shuvoice");
        assert!(
            with_mod.contains(
                "bind = SUPER, V, exec, /opt/shuvoice control start --control-wait-sec 0"
            )
        );

        let right = format_hyprland_bind_for_keybind("right_ctrl", ", Control_R", "shuvoice");
        assert!(
            right.contains(
                "bindr = CTRL, Control_R, exec, shuvoice control stop --control-wait-sec 0"
            )
        );
        assert!(right.contains("tts_speak"));
        assert!(right.contains("tts_speak_clipboard"));
    }

    #[test]
    fn finish_and_model_status_maps() {
        assert!(finish_status_text("added").contains("Added"));
        assert!(finish_status_text("already_configured").contains("already"));
        assert!(finish_status_text("conflict").contains("already bound"));
        assert!(finish_status_text("missing_config").contains("not found"));
        assert!(finish_status_text("skipped_custom").contains("Custom"));
        assert!(model_download_status_text("cancelled").contains("cancelled"));
        assert!(model_download_status_text("incompatible_streaming").contains("Zipformer"));
        assert!(model_download_status_text("downloaded").contains("ready"));
        assert!(model_download_status_text("skipped_missing_deps").contains("missing"));
    }

    #[test]
    fn complete_finish_deferred_does_not_claim_downloaded() {
        let mut vm = WizardVm::new(false);
        vm.complete_finish("added", "cancelled", "user stopped");
        assert!(vm.finish_status.contains("cancelled") || vm.download_text.contains("cancelled"));
        assert!(vm.launch_visible);
        assert!(!vm.finish_in_progress);
    }

    #[test]
    fn speed_preset_id_nearest() {
        assert_eq!(tts_playback_speed_preset_id(1.2), "1.25");
        assert_eq!(tts_playback_speed_preset_id(0.75), "0.75");
        assert_eq!(tts_playback_speed_preset_id(1.9), "2.0");
    }

    #[test]
    fn vm_summary_refresh_tracks_selection() {
        let mut vm = WizardVm::new(false);
        vm.set_tts_backend("melotts");
        vm.tts_voice_id = "EN-AU".into();
        vm.refresh_summary();
        let text = summary_for(&vm);
        assert!(text.contains("MeloTTS"));
        assert!(text.contains("EN-AU"));
    }
}
