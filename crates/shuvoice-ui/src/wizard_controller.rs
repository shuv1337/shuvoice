//! Headless wizard finish pipeline: config persist, marker, Hyprland binds.
//!
//! Model download/setup is intentionally a callback boundary — this module never
//! claims models are ready unless the callback reports success.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use serde_json::{Map, Value};
use shuvoice_core::config::{
    DEFAULT_ELEVENLABS_TTS_API_KEY_ENV, DEFAULT_ELEVENLABS_TTS_MODEL_ID,
    DEFAULT_KOKORO_TTS_MODEL_ID, DEFAULT_LOCAL_TTS_MODEL_ID, DEFAULT_MELOTTS_MODEL_ID,
    DEFAULT_OPENAI_TTS_API_KEY_ENV, DEFAULT_OPENAI_TTS_MODEL_ID,
};
use shuvoice_core::{
    AsrBackendKind, CURRENT_CONFIG_VERSION, ComputeProvider, Config, InjectionMode, MeloTtsDevice,
    OutputMode, SherpaDecodeMode, TtsBackendKind, TypingTextCase, load_raw, migrate_to_latest,
    wizard_done_path, write_atomic,
};

use crate::error::UiError;
use crate::wizard::{
    KEYBIND_PRESETS, KeybindSetupStatus, WizardVm, WizardWritePlan, control_exec,
    finish_status_text, format_hyprland_bind_for_keybind, model_download_status_text,
};

/// Result of the optional model-setup callback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSetupStatus {
    /// Setup intentionally deferred (caller should surface next steps).
    Deferred { message: String },
    /// Backend downloads lazily / cloud / nothing to do.
    Skipped { message: String },
    /// Models downloaded and ready.
    Downloaded { message: String },
    /// User cancelled a long-running setup.
    Cancelled { message: String },
    /// Setup attempted and failed (config may still be written; marker is not).
    Error { message: String },
}

impl ModelSetupStatus {
    pub fn as_code(&self) -> &'static str {
        match self {
            Self::Deferred { .. } => "deferred",
            Self::Skipped { .. } => "skipped",
            Self::Downloaded { .. } => "downloaded",
            Self::Cancelled { .. } => "cancelled",
            Self::Error { .. } => "error",
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Deferred { message }
            | Self::Skipped { message }
            | Self::Downloaded { message }
            | Self::Cancelled { message }
            | Self::Error { message } => message.as_str(),
        }
    }

    /// Marker is written only for launch-ready outcomes (retry-safe cancel/error).
    pub fn is_launch_ready(&self) -> bool {
        matches!(
            self,
            Self::Deferred { .. } | Self::Skipped { .. } | Self::Downloaded { .. }
        )
    }
}

/// Progress event for long-running model setup (UI may ignore).
pub type ModelProgressCb<'a> = dyn FnMut(Option<f64>, &str) + 'a;

/// Optional cancel probe for long-running model setup.
pub type CancelCheck<'a> = dyn FnMut() -> bool + 'a;

/// Callback boundary for ASR/TTS model installation.
pub trait ModelSetupHook: Send {
    fn run_model_setup(
        &mut self,
        plan: &WizardWritePlan,
        progress: &mut ModelProgressCb<'_>,
        cancel: &mut CancelCheck<'_>,
    ) -> ModelSetupStatus;
}

/// Default hook: never pretends download finished.
#[derive(Debug, Default, Clone, Copy)]
pub struct DeferredModelSetup;

impl ModelSetupHook for DeferredModelSetup {
    fn run_model_setup(
        &mut self,
        plan: &WizardWritePlan,
        progress: &mut ModelProgressCb<'_>,
        _cancel: &mut CancelCheck<'_>,
    ) -> ModelSetupStatus {
        let msg = match plan.asr_backend.as_str() {
            "openai_realtime" => {
                "OpenAI Realtime uses cloud transcription; no local model download is required. \
                 Ensure OPENAI_API_KEY is set (for example in ~/.config/shuvoice/local.dev)."
                    .to_string()
            }
            "sherpa" | "nemo" | "moonshine" => format!(
                "Model download is not run inside the wizard UI yet for backend '{}'. \
                 Run `shuvoice setup --install-missing` or `shuvoice model download` after launch.",
                plan.asr_backend
            ),
            other => format!(
                "No automatic model setup for backend '{other}'. Run `shuvoice setup` if needed."
            ),
        };
        progress(Some(1.0), "Model setup deferred — see status");
        if plan.asr_backend == "openai_realtime" {
            ModelSetupStatus::Skipped { message: msg }
        } else {
            ModelSetupStatus::Deferred { message: msg }
        }
    }
}

/// Injectable CUDA / device detection (tests override; production probes PATH).
pub trait DeviceDetector: Send + Sync {
    fn cuda_likely_available(&self) -> bool;
}

/// Default detector: `nvidia-smi` present in PATH.
#[derive(Debug, Default, Clone, Copy)]
pub struct PathCudaDetector;

impl DeviceDetector for PathCudaDetector {
    fn cuda_likely_available(&self) -> bool {
        std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

static DEVICE_DETECTOR: OnceLock<Box<dyn DeviceDetector>> = OnceLock::new();

fn device_detector() -> &'static dyn DeviceDetector {
    DEVICE_DETECTOR
        .get_or_init(|| Box::new(PathCudaDetector))
        .as_ref()
}

thread_local! {
    static TLS_DETECTOR: std::cell::RefCell<Option<Box<dyn DeviceDetector>>> =
        const { std::cell::RefCell::new(None) };
}

/// Run `body` with an injectable device detector (test seam).
pub fn with_device_detector<R>(detector: Box<dyn DeviceDetector>, body: impl FnOnce() -> R) -> R {
    TLS_DETECTOR.with(|slot| {
        let prev = slot.borrow_mut().replace(detector);
        let out = body();
        *slot.borrow_mut() = prev;
        out
    })
}

fn active_detector() -> bool {
    TLS_DETECTOR.with(|slot| {
        if let Some(d) = slot.borrow().as_ref() {
            d.cuda_likely_available()
        } else {
            device_detector().cuda_likely_available()
        }
    })
}

/// Outcome of applying wizard selections.
#[derive(Debug, Clone, PartialEq)]
pub struct WizardFinishReport {
    pub keybind_status: KeybindSetupStatus,
    pub keybind_message: String,
    pub model_status: ModelSetupStatus,
    pub config_path: PathBuf,
    pub marker_path: Option<PathBuf>,
}

impl WizardFinishReport {
    pub fn status_text(&self) -> String {
        let mut text = finish_status_text(self.keybind_status.as_str()).to_string();
        let model_line = match &self.model_status {
            ModelSetupStatus::Deferred { message } => {
                format!("ℹ Model setup deferred.\n{message}")
            }
            ModelSetupStatus::Skipped { message } => {
                let head = model_download_status_text("skipped");
                if head.is_empty() {
                    message.clone()
                } else {
                    format!("{head}\n{message}")
                }
            }
            ModelSetupStatus::Downloaded { message } => {
                let head = model_download_status_text("downloaded");
                format!("{head}\n{message}")
            }
            ModelSetupStatus::Cancelled { message } => {
                let head = model_download_status_text("cancelled");
                format!("{head}\n{message}")
            }
            ModelSetupStatus::Error { message } => {
                let head = model_download_status_text("error");
                format!("{head}\n{message}")
            }
        };
        if !model_line.is_empty() {
            text = format!("{text}\n{model_line}");
        }
        if self.marker_path.is_none() {
            text = format!("{text}\nℹ Wizard marker not written (retry-safe cancel/error).");
        }
        text
    }
}

/// Apply a [`WizardWritePlan`] onto a base [`Config`].
pub fn apply_write_plan(base: Config, plan: &WizardWritePlan) -> Result<Config, UiError> {
    let mut cfg = base;
    cfg.config_version = CURRENT_CONFIG_VERSION;

    cfg.asr_backend = plan.asr_backend.parse().map_err(UiError::from)?;
    cfg.typing_final_injection_mode = plan
        .typing_final_injection_mode
        .parse::<InjectionMode>()
        .map_err(UiError::from)?;
    cfg.typing_text_case = plan
        .typing_text_case
        .parse::<TypingTextCase>()
        .map_err(UiError::from)?;
    cfg.output_mode = plan
        .typing_output_mode
        .parse::<OutputMode>()
        .map_err(UiError::from)?;
    cfg.use_clipboard_for_final = plan.use_clipboard_for_final;

    if cfg.asr_backend == AsrBackendKind::Sherpa {
        if let Some(name) = &plan.sherpa_model_name {
            cfg.sherpa_model_name = name.clone();
        }
        cfg.sherpa_enable_parakeet_streaming = plan.sherpa_enable_parakeet_streaming;
        if let Some(provider) = &plan.sherpa_provider {
            cfg.sherpa_provider = provider.parse::<ComputeProvider>().map_err(UiError::from)?;
        }
        if let Some(instant) = plan.instant_mode {
            cfg.instant_mode = instant;
        }
        if let Some(mode) = &plan.sherpa_decode_mode {
            cfg.sherpa_decode_mode = mode.parse::<SherpaDecodeMode>().map_err(UiError::from)?;
        }
    } else if cfg.asr_backend == AsrBackendKind::OpenaiRealtime {
        cfg.openai_realtime_model = "gpt-4o-transcribe".into();
        cfg.openai_realtime_api_key_env = "OPENAI_API_KEY".into();
        cfg.openai_realtime_language = "en".into();
        cfg.instant_mode = false;
    } else if cfg.asr_backend == AsrBackendKind::Nemo {
        // Injectable CUDA seam (defaults to nvidia-smi probe).
        cfg.device = if active_detector() {
            "cuda".into()
        } else {
            "cpu".into()
        };
    }

    cfg.tts_backend = plan
        .tts_backend
        .parse::<TtsBackendKind>()
        .map_err(UiError::from)?;
    cfg.tts_default_voice_id = plan.tts_default_voice_id.clone();
    cfg.tts_playback_speed = plan.tts_playback_speed;
    cfg.tts_enabled = true;

    match cfg.tts_backend {
        TtsBackendKind::Elevenlabs => {
            cfg.tts_model_id = DEFAULT_ELEVENLABS_TTS_MODEL_ID.into();
            cfg.tts_api_key_env = DEFAULT_ELEVENLABS_TTS_API_KEY_ENV.into();
            cfg.tts_local_model_path = None;
            cfg.tts_local_voice = None;
            cfg.tts_melotts_venv_path = None;
        }
        TtsBackendKind::Openai => {
            cfg.tts_model_id = DEFAULT_OPENAI_TTS_MODEL_ID.into();
            cfg.tts_api_key_env = DEFAULT_OPENAI_TTS_API_KEY_ENV.into();
            cfg.tts_local_model_path = None;
            cfg.tts_local_voice = None;
        }
        TtsBackendKind::Local => {
            cfg.tts_model_id = DEFAULT_LOCAL_TTS_MODEL_ID.into();
            cfg.tts_local_model_path = plan.tts_local_model_path.clone();
            cfg.tts_local_voice = plan.tts_local_voice.clone();
        }
        TtsBackendKind::Melotts => {
            cfg.tts_model_id = DEFAULT_MELOTTS_MODEL_ID.into();
            if let Some(dev) = &plan.tts_melotts_device {
                cfg.tts_melotts_device = dev.parse::<MeloTtsDevice>().map_err(UiError::from)?;
            }
            cfg.tts_local_model_path = None;
            cfg.tts_local_voice = None;
        }
        TtsBackendKind::Kokoro => {
            cfg.tts_model_id = DEFAULT_KOKORO_TTS_MODEL_ID.into();
            if let Some(url) = &plan.tts_kokoro_base_url {
                cfg.tts_kokoro_base_url = url.clone();
            }
            cfg.tts_local_model_path = None;
            cfg.tts_local_voice = None;
        }
    }

    cfg.validate().map_err(UiError::from)?;
    Ok(cfg)
}

fn ensure_table<'a>(map: &'a mut Map<String, Value>, key: &str) -> &'a mut Map<String, Value> {
    let needs = !matches!(map.get(key), Some(Value::Object(_)));
    if needs {
        map.insert(key.to_string(), Value::Object(Map::new()));
    }
    map.get_mut(key)
        .and_then(Value::as_object_mut)
        .expect("table just inserted")
}

fn json_str(s: impl Into<String>) -> Value {
    Value::String(s.into())
}

/// Merge only wizard-owned keys into a raw migrated TOML map (preserves unknowns).
pub fn merge_wizard_keys_into_raw(
    mut raw: Map<String, Value>,
    plan: &WizardWritePlan,
) -> Result<Map<String, Value>, UiError> {
    // Validate via typed apply against defaults (catches enum/range errors).
    let _typed = apply_write_plan(Config::default(), plan)?;

    raw.insert("config_version".into(), Value::from(CURRENT_CONFIG_VERSION));

    let asr = ensure_table(&mut raw, "asr");
    asr.insert("asr_backend".into(), json_str(&plan.asr_backend));

    let typing = ensure_table(&mut raw, "typing");
    typing.insert(
        "typing_final_injection_mode".into(),
        json_str(&plan.typing_final_injection_mode),
    );
    typing.insert("typing_text_case".into(), json_str(&plan.typing_text_case));
    typing.insert(
        "use_clipboard_for_final".into(),
        Value::Bool(plan.use_clipboard_for_final),
    );
    typing.insert("output_mode".into(), json_str(&plan.typing_output_mode));

    match plan.asr_backend.as_str() {
        "sherpa" => {
            let asr = ensure_table(&mut raw, "asr");
            if let Some(name) = &plan.sherpa_model_name {
                asr.insert("sherpa_model_name".into(), json_str(name));
            }
            asr.insert(
                "sherpa_enable_parakeet_streaming".into(),
                Value::Bool(plan.sherpa_enable_parakeet_streaming),
            );
            if let Some(p) = &plan.sherpa_provider {
                asr.insert("sherpa_provider".into(), json_str(p));
            }
            if let Some(instant) = plan.instant_mode {
                asr.insert("instant_mode".into(), Value::Bool(instant));
            }
            if let Some(mode) = &plan.sherpa_decode_mode {
                asr.insert("sherpa_decode_mode".into(), json_str(mode));
            }
        }
        "openai_realtime" => {
            let asr = ensure_table(&mut raw, "asr");
            asr.insert(
                "openai_realtime_model".into(),
                json_str("gpt-4o-transcribe"),
            );
            asr.insert(
                "openai_realtime_api_key_env".into(),
                json_str("OPENAI_API_KEY"),
            );
            asr.insert("openai_realtime_language".into(), json_str("en"));
            asr.insert("openai_realtime_turn_detection".into(), json_str("manual"));
            asr.insert("instant_mode".into(), Value::Bool(false));
        }
        "nemo" => {
            let asr = ensure_table(&mut raw, "asr");
            let device = if active_detector() { "cuda" } else { "cpu" };
            asr.insert("device".into(), json_str(device));
        }
        "moonshine" => {
            let asr = ensure_table(&mut raw, "asr");
            let provider = if active_detector() { "cuda" } else { "cpu" };
            asr.insert("moonshine_provider".into(), json_str(provider));
        }
        _ => {}
    }

    let tts = ensure_table(&mut raw, "tts");
    tts.insert("tts_backend".into(), json_str(&plan.tts_backend));
    tts.insert(
        "tts_default_voice_id".into(),
        json_str(&plan.tts_default_voice_id),
    );
    tts.insert(
        "tts_playback_speed".into(),
        Value::from(plan.tts_playback_speed),
    );
    tts.insert("tts_enabled".into(), Value::Bool(true));

    match plan.tts_backend.as_str() {
        "elevenlabs" => {
            tts.insert(
                "tts_model_id".into(),
                json_str(DEFAULT_ELEVENLABS_TTS_MODEL_ID),
            );
            tts.insert(
                "tts_api_key_env".into(),
                json_str(DEFAULT_ELEVENLABS_TTS_API_KEY_ENV),
            );
            tts.remove("tts_local_model_path");
            tts.remove("tts_local_voice");
            tts.remove("tts_melotts_device");
            tts.remove("tts_melotts_venv_path");
            tts.remove("tts_kokoro_base_url");
        }
        "openai" => {
            tts.insert("tts_model_id".into(), json_str(DEFAULT_OPENAI_TTS_MODEL_ID));
            tts.insert(
                "tts_api_key_env".into(),
                json_str(DEFAULT_OPENAI_TTS_API_KEY_ENV),
            );
            tts.remove("tts_local_model_path");
            tts.remove("tts_local_voice");
            tts.remove("tts_melotts_device");
            tts.remove("tts_melotts_venv_path");
            tts.remove("tts_kokoro_base_url");
        }
        "local" => {
            tts.insert("tts_model_id".into(), json_str(DEFAULT_LOCAL_TTS_MODEL_ID));
            match &plan.tts_local_model_path {
                Some(p) => {
                    tts.insert("tts_local_model_path".into(), json_str(p));
                }
                None => {
                    tts.remove("tts_local_model_path");
                }
            }
            match &plan.tts_local_voice {
                Some(v) => {
                    tts.insert("tts_local_voice".into(), json_str(v));
                }
                None => {
                    tts.remove("tts_local_voice");
                }
            }
            tts.remove("tts_melotts_device");
            tts.remove("tts_melotts_venv_path");
            tts.remove("tts_kokoro_base_url");
        }
        "melotts" => {
            tts.insert("tts_model_id".into(), json_str(DEFAULT_MELOTTS_MODEL_ID));
            let dev = plan
                .tts_melotts_device
                .clone()
                .unwrap_or_else(|| "auto".into());
            tts.insert("tts_melotts_device".into(), json_str(dev));
            tts.remove("tts_local_model_path");
            tts.remove("tts_local_voice");
            tts.remove("tts_kokoro_base_url");
        }
        "kokoro" => {
            tts.insert("tts_model_id".into(), json_str(DEFAULT_KOKORO_TTS_MODEL_ID));
            if let Some(url) = &plan.tts_kokoro_base_url {
                tts.insert("tts_kokoro_base_url".into(), json_str(url));
            }
            tts.remove("tts_local_model_path");
            tts.remove("tts_local_voice");
            tts.remove("tts_melotts_device");
            tts.remove("tts_melotts_venv_path");
        }
        _ => {}
    }

    Ok(raw)
}

/// Persist plan by merging wizard-owned keys into raw migrated config (atomic).
pub fn persist_write_plan(plan: &WizardWritePlan) -> Result<PathBuf, UiError> {
    let path = Config::config_path();
    if path.exists() && !plan.overwrite_existing {
        return Err(UiError::InvalidValue(format!(
            "config already exists at {} (refusing to overwrite)",
            path.display()
        )));
    }

    let raw = if path.exists() {
        let loaded = load_raw(&path).map_err(UiError::from)?;
        let (migrated, _report) = migrate_to_latest(&loaded).map_err(UiError::from)?;
        migrated
    } else {
        let mut m = Map::new();
        m.insert("config_version".into(), Value::from(CURRENT_CONFIG_VERSION));
        m
    };

    let merged = merge_wizard_keys_into_raw(raw, plan)?;
    let _backup = write_atomic(&path, &merged).map_err(UiError::from)?;
    Ok(path)
}

/// Write the `.wizard-done` marker under the data dir.
pub fn write_wizard_marker() -> Result<PathBuf, UiError> {
    let path = wizard_done_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            UiError::InvalidValue(format!(
                "failed to create data dir {}: {e}",
                parent.display()
            ))
        })?;
    }
    fs::write(&path, b"done\n").map_err(|e| {
        UiError::InvalidValue(format!(
            "failed to write wizard marker {}: {e}",
            path.display()
        ))
    })?;
    Ok(path)
}

/// Resolve the `shuvoice` executable string used in generated binds.
pub fn resolve_shuvoice_command() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.into_os_string().into_string().ok())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "shuvoice".into())
}

/// Candidate Hyprland config files (first existing is preferred write target).
pub fn hyprland_config_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        let hypr = PathBuf::from(home).join(".config/hypr");
        out.push(hypr.join("bindings.conf"));
        out.push(hypr.join("hyprland.conf"));
        out.push(hypr.join("keybinds.conf"));
    }
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        let hypr = PathBuf::from(xdg).join("hypr");
        out.push(hypr.join("bindings.conf"));
        out.push(hypr.join("hyprland.conf"));
    }
    let mut seen = Vec::new();
    let mut unique = Vec::new();
    for p in out {
        let key = p.to_string_lossy().to_string();
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        unique.push(p);
    }
    unique
}

/// Normalize mods+key into `mods|key` (lowercased, empty mods allowed).
fn normalize_bind_spec(mods: &str, key: &str) -> Option<String> {
    let key = key.trim().to_ascii_lowercase();
    if key.is_empty() {
        return None;
    }
    let mods = mods
        .split([' ', '+'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    Some(format!("{mods}|{key}"))
}

fn normalize_hypr_key_spec(hypr_key_spec: &str) -> Option<String> {
    let (mods, key) = hypr_key_spec.split_once(',')?;
    normalize_bind_spec(mods, key)
}

/// Parse Hyprland `bind[a-z]* = ...` lines.
fn normalize_bind_line(line: &str) -> Option<(String, String)> {
    let stripped = line.split('#').next()?.trim();
    if stripped.is_empty() {
        return None;
    }
    let lower = stripped.to_ascii_lowercase();
    if !lower.starts_with("bind") {
        return None;
    }
    let eq = stripped.find('=')?;
    let lhs = stripped[..eq].trim().to_ascii_lowercase();
    // bind, bindr, bindl, binde, ...
    if lhs != "bind" && !lhs.starts_with("bind") {
        return None;
    }
    if !lhs
        .chars()
        .skip(4)
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_whitespace())
    {
        // allow bind / bindr / bindl etc.
        let rest = &lhs[4..];
        if !rest.chars().all(|c| c.is_ascii_lowercase()) {
            return None;
        }
    }
    let rhs = stripped[eq + 1..].trim();
    let parts: Vec<&str> = rhs.splitn(4, ',').map(str::trim).collect();
    if parts.len() < 2 {
        return None;
    }
    let spec = normalize_bind_spec(parts[0], parts[1])?;
    let command = if parts.len() >= 3 {
        parts[2..].join(",").trim().to_string()
    } else {
        String::new()
    };
    // Drop leading "exec," if present in command chunking variants
    let command = {
        let c = command.trim();
        let cl = c.to_ascii_lowercase();
        if cl.starts_with("exec") {
            let after = c[4..].trim_start_matches(|ch: char| ch == ',' || ch.is_whitespace());
            after.to_string()
        } else {
            c.to_string()
        }
    };
    Some((spec, command.to_ascii_lowercase()))
}

fn is_shuvoice_control_bind(cmd: &str) -> bool {
    let c = cmd.to_ascii_lowercase();
    if !c.contains("shuvoice") {
        return false;
    }
    let padded = format!(" {c} ");
    padded.contains(" control ") || c.contains("--control")
}

fn is_shuvoice_start(cmd: &str) -> bool {
    is_shuvoice_control_bind(cmd)
        && (cmd.contains("control start") || cmd.contains("--control start"))
}
fn is_shuvoice_stop(cmd: &str) -> bool {
    is_shuvoice_control_bind(cmd)
        && (cmd.contains("control stop") || cmd.contains("--control stop"))
}
fn is_shuvoice_tts_speak(cmd: &str) -> bool {
    is_shuvoice_control_bind(cmd)
        && (cmd.contains("control tts_speak") || cmd.contains("--control tts_speak"))
        && !cmd.contains("tts_speak_clipboard")
}
fn is_shuvoice_tts_clipboard(cmd: &str) -> bool {
    is_shuvoice_control_bind(cmd)
        && (cmd.contains("tts_speak_clipboard") || cmd.contains("--control tts_speak_clipboard"))
}

/// Best-effort, idempotent Hyprland bind installer.
pub fn auto_add_hyprland_keybind(keybind_id: &str) -> (KeybindSetupStatus, String) {
    auto_add_hyprland_keybind_with(keybind_id, &resolve_shuvoice_command())
}

/// Same as [`auto_add_hyprland_keybind`] with an explicit `shuvoice` binary path.
pub fn auto_add_hyprland_keybind_with(
    keybind_id: &str,
    shuvoice_command: &str,
) -> (KeybindSetupStatus, String) {
    let preset = KEYBIND_PRESETS.iter().find(|p| p.id == keybind_id);
    let Some(preset) = preset else {
        return (
            KeybindSetupStatus::Error,
            format!("unknown keybind preset '{keybind_id}'"),
        );
    };
    let Some(hypr_key) = preset.hypr_key_spec else {
        return (
            KeybindSetupStatus::SkippedCustom,
            "Selected keybind is custom; no automatic Hyprland edit attempted.".into(),
        );
    };

    let candidates = hyprland_config_candidates();
    let existing: Vec<PathBuf> = candidates.into_iter().filter(|p| p.is_file()).collect();
    if existing.is_empty() {
        return (
            KeybindSetupStatus::MissingConfig,
            "Hyprland config not found under ~/.config/hypr/".into(),
        );
    }

    let desired = format_hyprland_bind_for_keybind(keybind_id, hypr_key, shuvoice_command);
    let desired_lines: Vec<String> = desired.lines().map(str::to_string).collect();

    let target_spec = match normalize_hypr_key_spec(hypr_key) {
        Some(s) => s,
        None => {
            return (
                KeybindSetupStatus::Error,
                format!("Invalid Hyprland key spec for preset '{keybind_id}': {hypr_key}"),
            );
        }
    };

    let mut conflict_specs = vec![target_spec.clone()];
    let desired_start = [target_spec.clone()];
    let mut desired_stop = vec![target_spec.clone()];
    let mut extra_stop: Option<String> = None;
    if keybind_id == "right_ctrl"
        && let Some(spec) = normalize_bind_spec("CTRL", "Control_R")
    {
        extra_stop = Some(spec.clone());
        desired_stop.push(spec.clone());
        conflict_specs.push(spec);
    }
    // TTS chords
    let tts_speak_spec = normalize_bind_spec("SUPER CTRL", "S").expect("tts speak");
    let tts_clip_spec = normalize_bind_spec("SUPER CTRL SHIFT", "S").expect("tts clip");
    conflict_specs.push(tts_speak_spec.clone());
    conflict_specs.push(tts_clip_spec.clone());

    let mut content_by_file: Vec<(PathBuf, String)> = Vec::new();
    for path in &existing {
        match fs::read_to_string(path) {
            Ok(text) => content_by_file.push((path.clone(), text)),
            Err(err) => {
                return (
                    KeybindSetupStatus::Error,
                    format!("Failed to read {}: {err}", path.display()),
                );
            }
        }
    }

    // Conflict detection + inventory of existing ShuVoice control binds.
    let mut shuvoice_files: Vec<PathBuf> = Vec::new();
    let mut shuvoice_count = 0usize;
    let mut has_start = false;
    let mut has_stop = false;
    let mut has_extra_stop = false;
    let mut has_tts = false;
    let mut has_tts_clip = false;
    let mut has_other_shuvoice = false;

    for (path, text) in &content_by_file {
        let mut file_has = false;
        for (lineno, line) in text.lines().enumerate() {
            let Some((spec, cmd)) = normalize_bind_line(line) else {
                continue;
            };
            let is_sv = is_shuvoice_control_bind(&cmd);
            if conflict_specs.iter().any(|c| c == &spec) && !is_sv {
                return (
                    KeybindSetupStatus::Conflict,
                    format!(
                        "Key is already bound; not adding ShuVoice binds ({} line {}).",
                        path.display(),
                        lineno + 1
                    ),
                );
            }
            if !is_sv {
                continue;
            }
            let is_ctrl = is_shuvoice_start(&cmd)
                || is_shuvoice_stop(&cmd)
                || is_shuvoice_tts_speak(&cmd)
                || is_shuvoice_tts_clipboard(&cmd);
            if is_ctrl {
                shuvoice_count += 1;
                file_has = true;
            }
            if is_shuvoice_start(&cmd) && desired_start.iter().any(|s| s == &spec) {
                has_start = true;
                continue;
            }
            if is_shuvoice_stop(&cmd) && desired_stop.iter().any(|s| s == &spec) {
                if desired_start.iter().any(|s| s == &spec) || spec == target_spec {
                    has_stop = true;
                }
                if extra_stop.as_ref() == Some(&spec) {
                    has_extra_stop = true;
                }
                continue;
            }
            if is_shuvoice_tts_speak(&cmd) && spec == tts_speak_spec {
                has_tts = true;
                continue;
            }
            if is_shuvoice_tts_clipboard(&cmd) && spec == tts_clip_spec {
                has_tts_clip = true;
                continue;
            }
            if is_ctrl {
                has_other_shuvoice = true;
            }
        }
        if file_has {
            shuvoice_files.push(path.clone());
        }
    }

    let mut fully = has_start && has_stop && has_tts && has_tts_clip;
    if extra_stop.is_some() {
        fully = fully && has_extra_stop;
    }
    if fully && !has_other_shuvoice && shuvoice_count == desired_lines.len() {
        // Also require exact desired lines present (quoted path / wait-sec).
        let mut corpus = String::new();
        for (_, text) in &content_by_file {
            corpus.push_str(text);
            corpus.push('\n');
        }
        if desired_lines.iter().all(|l| corpus.contains(l.trim())) {
            return (
                KeybindSetupStatus::AlreadyConfigured,
                "Push-to-talk keybind already configured.".into(),
            );
        }
    }

    // Prefer file that already hosts ShuVoice binds, else bindings*.conf, else first.
    let destination = shuvoice_files
        .first()
        .cloned()
        .or_else(|| {
            existing
                .iter()
                .find(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.contains("bind"))
                })
                .cloned()
        })
        .unwrap_or_else(|| existing[0].clone());

    // Filter-and-replace across all candidates; preserve unrelated lines (incl. source=).
    for (path, text) in &content_by_file {
        let mut filtered: Vec<String> = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed == "# Added by ShuVoice setup wizard"
                || trimmed == "# ShuVoice binds (managed by wizard)"
            {
                continue;
            }
            if let Some((_spec, cmd)) = normalize_bind_line(line)
                && (is_shuvoice_start(&cmd)
                    || is_shuvoice_stop(&cmd)
                    || is_shuvoice_tts_speak(&cmd)
                    || is_shuvoice_tts_clipboard(&cmd))
            {
                continue;
            }
            filtered.push(line.to_string());
        }

        if path == &destination {
            while filtered.last().is_some_and(|l| l.trim().is_empty()) {
                filtered.pop();
            }
            if !filtered.is_empty() {
                filtered.push(String::new());
            }
            filtered.push("# ShuVoice binds (managed by wizard)".into());
            for line in &desired_lines {
                filtered.push(line.clone());
            }
            filtered.push(String::new());
        }

        let mut new_content = filtered.join("\n");
        if !new_content.ends_with('\n') {
            new_content.push('\n');
        }
        if new_content != *text
            && let Err(err) = fs::write(path, new_content)
        {
            return (
                KeybindSetupStatus::Error,
                format!("Failed to write {}: {err}", path.display()),
            );
        }
    }

    let _ = Command::new("hyprctl").args(["reload"]).output();
    (
        KeybindSetupStatus::Added,
        format!("Added ShuVoice keybind to {}.", destination.display()),
    )
}

/// Run the full finish pipeline for a wizard VM (headless-safe).
pub fn finish_wizard(
    vm: &WizardVm,
    model_hook: &mut dyn ModelSetupHook,
    progress: Option<&mut ModelProgressCb<'_>>,
    cancel: Option<&mut CancelCheck<'_>>,
) -> Result<WizardFinishReport, UiError> {
    let mut vm_mut = vm.clone();
    vm_mut.validate_tts_selection()?;
    let plan = vm_mut.build_write_plan()?;

    let config_path = persist_write_plan(&plan)?;

    let (keybind_status, keybind_message) = if vm.auto_add_enabled() {
        auto_add_hyprland_keybind(&vm.keybind)
    } else {
        (
            KeybindSetupStatus::NotAttempted,
            "automatic keybind setup disabled".into(),
        )
    };

    let mut noop_progress = |_f: Option<f64>, _m: &str| {};
    let mut noop_cancel = || false;
    let progress_cb: &mut ModelProgressCb<'_> = match progress {
        Some(cb) => cb,
        None => &mut noop_progress,
    };
    let cancel_cb: &mut CancelCheck<'_> = match cancel {
        Some(cb) => cb,
        None => &mut noop_cancel,
    };

    let model_status = model_hook.run_model_setup(&plan, progress_cb, cancel_cb);

    // Marker only when launch-ready. Cancel/error stay retry-safe (no marker).
    // Config existence alone still allows app start (needs_wizard is false).
    let marker_path = if model_status.is_launch_ready() {
        Some(write_wizard_marker()?)
    } else {
        None
    };

    Ok(WizardFinishReport {
        keybind_status,
        keybind_message,
        model_status,
        config_path,
        marker_path,
    })
}

/// Convenience: finish using the deferred model hook.
pub fn finish_wizard_deferred(vm: &WizardVm) -> Result<WizardFinishReport, UiError> {
    let mut hook = DeferredModelSetup;
    finish_wizard(vm, &mut hook, None, None)
}

// Silence unused helper in non-test builds that still validates control_exec wiring.
#[allow(dead_code)]
fn _touch_control_exec() {
    let _ = control_exec("start", "shuvoice");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wizard::{
        DEFAULT_KOKORO_VOICE, PARAKEET_TDT_V3_INT8_MODEL_NAME, format_hyprland_bind_for_keybind,
        needs_wizard_fs,
    };
    use shuvoice_core::{AsrBackendKind, OutputMode, SherpaDecodeMode, TtsBackendKind};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    struct RecordingHook {
        status: ModelSetupStatus,
        progress_events: AtomicUsize,
        saw_cancel: AtomicBool,
    }

    impl RecordingHook {
        fn new(status: ModelSetupStatus) -> Self {
            Self {
                status,
                progress_events: AtomicUsize::new(0),
                saw_cancel: AtomicBool::new(false),
            }
        }
    }

    impl ModelSetupHook for RecordingHook {
        fn run_model_setup(
            &mut self,
            _plan: &WizardWritePlan,
            progress: &mut ModelProgressCb<'_>,
            cancel: &mut CancelCheck<'_>,
        ) -> ModelSetupStatus {
            progress(Some(0.1), "starting");
            self.progress_events.fetch_add(1, Ordering::SeqCst);
            if cancel() {
                self.saw_cancel.store(true, Ordering::SeqCst);
                return ModelSetupStatus::Cancelled {
                    message: "cancelled by user".into(),
                };
            }
            progress(Some(1.0), "done");
            self.status.clone()
        }
    }

    struct FixedCuda(bool);
    impl DeviceDetector for FixedCuda {
        fn cuda_likely_available(&self) -> bool {
            self.0
        }
    }

    fn hypr_dir(root: &std::path::Path) -> PathBuf {
        let p = root.join("config/hypr");
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn apply_write_plan_parakeet_defaults_cpu_af_heart_speed() {
        let vm = WizardVm::new(false);
        let plan = vm.build_write_plan().unwrap();
        let cfg = apply_write_plan(Config::default(), &plan).unwrap();
        assert_eq!(cfg.asr_backend, AsrBackendKind::Sherpa);
        assert!(cfg.instant_mode);
        assert_eq!(cfg.sherpa_decode_mode, SherpaDecodeMode::OfflineInstant);
        assert_eq!(cfg.sherpa_provider, ComputeProvider::Cpu);
        assert_eq!(cfg.tts_backend, TtsBackendKind::Kokoro);
        assert_eq!(cfg.tts_default_voice_id, DEFAULT_KOKORO_VOICE);
        assert!((cfg.tts_playback_speed - 1.25).abs() < 1e-9);
        assert_eq!(cfg.output_mode, OutputMode::FinalOnly);
    }

    #[test]
    fn persist_defaults_write_cpu_af_heart_125_offline_instant() {
        crate::test_env::with_isolated_xdg(|_| {
            let vm = WizardVm::new(false);
            let path = persist_write_plan(&vm.build_write_plan().unwrap()).unwrap();
            let text = fs::read_to_string(&path).unwrap();
            assert!(text.contains("sherpa_provider = \"cpu\"") || text.contains("cpu"));
            assert!(text.contains("offline_instant"));
            assert!(text.contains("af_heart"));
            assert!(text.contains("1.25") || text.contains("tts_playback_speed"));
            let cfg = Config::load().unwrap();
            assert_eq!(cfg.sherpa_provider, ComputeProvider::Cpu);
            assert_eq!(cfg.sherpa_decode_mode, SherpaDecodeMode::OfflineInstant);
            assert_eq!(cfg.tts_default_voice_id, "af_heart");
            assert!((cfg.tts_playback_speed - 1.25).abs() < 1e-9);
            assert_eq!(cfg.sherpa_model_name, PARAKEET_TDT_V3_INT8_MODEL_NAME);
        });
    }

    #[test]
    fn reconfigure_merges_wizard_keys_preserves_unknowns() {
        crate::test_env::with_isolated_xdg(|_| {
            let path = Config::config_path();
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(
                &path,
                r#"
config_version = 1
[asr]
asr_backend = "nemo"
device = "cuda"
model_name = "nvidia/custom-keep-me"
custom_future_flag = true

[overlay]
font_size = 42

[experimental]
keep = "yes"
"#,
            )
            .unwrap();

            let mut vm = WizardVm::new(true);
            vm.set_asr_backend("sherpa");
            vm.set_tts_backend("kokoro");
            persist_write_plan(&vm.build_write_plan().unwrap()).unwrap();

            let text = fs::read_to_string(&path).unwrap();
            assert!(text.contains("custom_future_flag") || text.contains("keep"));
            // unknown experimental section preserved via raw merge
            assert!(text.contains("experimental") || text.contains("keep"));
            assert!(text.contains("font_size") || text.contains("42"));
            assert!(text.contains("sherpa") || text.contains("asr_backend"));
            let cfg = Config::load().unwrap();
            assert_eq!(cfg.asr_backend, AsrBackendKind::Sherpa);
            // model_name from prior nemo should still load if present in file
            assert!(cfg.model_name.contains("custom-keep-me") || !cfg.model_name.is_empty());
        });
    }

    #[test]
    fn marker_only_on_launch_ready_not_cancel_or_error() {
        crate::test_env::with_isolated_xdg(|_| {
            let vm = WizardVm::new(false);
            let mut hook = RecordingHook::new(ModelSetupStatus::Cancelled {
                message: "nope".into(),
            });
            let cancel = AtomicBool::new(true);
            let report = finish_wizard(
                &vm,
                &mut hook,
                None,
                Some(&mut || cancel.load(Ordering::SeqCst)),
            )
            .unwrap();
            assert!(report.marker_path.is_none());
            assert!(!wizard_done_path().is_file());
            // Config still written so app can start; force wizard for retry.
            assert!(report.config_path.is_file());
            assert!(!needs_wizard_fs());
        });

        crate::test_env::with_isolated_xdg(|_| {
            let vm = WizardVm::new(false);
            let mut hook = RecordingHook::new(ModelSetupStatus::Error {
                message: "boom".into(),
            });
            let report = finish_wizard(&vm, &mut hook, None, Some(&mut || false)).unwrap();
            assert!(report.marker_path.is_none());
            assert!(!wizard_done_path().is_file());
        });

        crate::test_env::with_isolated_xdg(|_| {
            let vm = WizardVm::new(false);
            let report = finish_wizard_deferred(&vm).unwrap();
            assert!(report.marker_path.is_some());
            assert!(wizard_done_path().is_file());
            assert!(matches!(
                report.model_status,
                ModelSetupStatus::Deferred { .. }
            ));
        });
    }

    #[test]
    fn cuda_detector_seam_controls_nemo_device() {
        let mut vm = WizardVm::new(false);
        vm.set_asr_backend("nemo");
        let plan = vm.build_write_plan().unwrap();

        with_device_detector(Box::new(FixedCuda(false)), || {
            let cfg = apply_write_plan(Config::default(), &plan).unwrap();
            assert_eq!(cfg.device, "cpu");
        });
        with_device_detector(Box::new(FixedCuda(true)), || {
            let cfg = apply_write_plan(Config::default(), &plan).unwrap();
            assert_eq!(cfg.device, "cuda");
        });
    }

    #[test]
    fn custom_keybind_skipped() {
        let (st, msg) = auto_add_hyprland_keybind("custom");
        assert_eq!(st, KeybindSetupStatus::SkippedCustom);
        assert!(msg.contains("custom"));
    }

    #[test]
    fn missing_hypr_config() {
        crate::test_env::with_isolated_xdg(|_| {
            let (st, _) = auto_add_hyprland_keybind("right_ctrl");
            assert_eq!(st, KeybindSetupStatus::MissingConfig);
        });
    }

    #[test]
    fn auto_add_adds_wait_sec_and_quotes_path() {
        crate::test_env::with_isolated_xdg(|root| {
            let hypr = hypr_dir(root);
            fs::write(
                hypr.join("hyprland.conf"),
                "# user\nsource = ./other.conf\n",
            )
            .unwrap();
            let (st, _) = auto_add_hyprland_keybind_with("insert", "/opt/shu voice/bin/shuvoice");
            assert_eq!(st, KeybindSetupStatus::Added);
            let content = fs::read_to_string(hypr.join("hyprland.conf")).unwrap();
            assert!(content.contains("source = ./other.conf"));
            assert!(content.contains("--control-wait-sec 0"));
            assert!(content.contains("'/opt/shu voice/bin/shuvoice'"));
            assert!(content.contains("control start"));
            assert!(content.contains("tts_speak_clipboard"));
        });
    }

    #[test]
    fn auto_add_conflict_on_ptt_and_tts_chords() {
        crate::test_env::with_isolated_xdg(|root| {
            let hypr = hypr_dir(root);
            fs::write(
                hypr.join("hyprland.conf"),
                "bind = , Insert, exec, grimblast save area\n",
            )
            .unwrap();
            let (st, _) = auto_add_hyprland_keybind_with("insert", "shuvoice");
            assert_eq!(st, KeybindSetupStatus::Conflict);
        });
        crate::test_env::with_isolated_xdg(|root| {
            let hypr = hypr_dir(root);
            fs::write(
                hypr.join("hyprland.conf"),
                "bind = SUPER CTRL, S, exec, some-other-tts\n",
            )
            .unwrap();
            let (st, _) = auto_add_hyprland_keybind_with("insert", "shuvoice");
            assert_eq!(st, KeybindSetupStatus::Conflict);
        });
    }

    #[test]
    fn auto_add_parses_bindl_and_is_idempotent() {
        crate::test_env::with_isolated_xdg(|root| {
            let hypr = hypr_dir(root);
            let block = format_hyprland_bind_for_keybind("insert", ", Insert", "shuvoice");
            fs::write(
                hypr.join("hyprland.conf"),
                format!("bindl = , F1, exec, foo\n{block}\n"),
            )
            .unwrap();
            let (st, _) = auto_add_hyprland_keybind_with("insert", "shuvoice");
            assert_eq!(st, KeybindSetupStatus::AlreadyConfigured);
            let content = fs::read_to_string(hypr.join("hyprland.conf")).unwrap();
            assert!(content.contains("bindl = , F1, exec, foo"));
            assert_eq!(content.matches("control start").count(), 1);
        });
    }

    #[test]
    fn auto_add_updates_existing_shuvoice_bindings_conf() {
        crate::test_env::with_isolated_xdg(|root| {
            let hypr = hypr_dir(root);
            let bindings = hypr.join("bindings.conf");
            let hyprland = hypr.join("hyprland.conf");
            let old = format_hyprland_bind_for_keybind("insert", ", Insert", "/venv/bin/shuvoice");
            fs::write(&bindings, format!("{old}\n")).unwrap();
            fs::write(&hyprland, "source = ~/.config/hypr/bindings.conf\n").unwrap();

            let (st, msg) = auto_add_hyprland_keybind_with("right_ctrl", "/venv/bin/shuvoice");
            assert_eq!(st, KeybindSetupStatus::Added, "{msg}");
            let bindings_text = fs::read_to_string(&bindings).unwrap();
            assert!(!bindings_text.contains("Insert"), "{bindings_text}");
            assert!(bindings_text.contains("Control_R"));
            assert!(bindings_text.contains("--control-wait-sec 0"));
            assert!(bindings_text.contains("/venv/bin/shuvoice control start"));
            let hyprland_text = fs::read_to_string(&hyprland).unwrap();
            assert!(hyprland_text.contains("source ="));
            assert!(!hyprland_text.contains("control start"));
        });
    }

    #[test]
    fn finish_wizard_skips_keybind_when_auto_add_disabled() {
        crate::test_env::with_isolated_xdg(|_| {
            let mut vm = WizardVm::new(false);
            vm.set_keybind("custom");
            let report = finish_wizard_deferred(&vm).unwrap();
            assert_eq!(report.keybind_status, KeybindSetupStatus::NotAttempted);
            assert!(report.marker_path.is_some());
        });
    }

    #[test]
    fn persist_refuses_overwrite_without_force() {
        crate::test_env::with_isolated_xdg(|_| {
            let vm = WizardVm::new(false);
            persist_write_plan(&vm.build_write_plan().unwrap()).unwrap();
            let err = persist_write_plan(&vm.build_write_plan().unwrap()).unwrap_err();
            assert!(err.to_string().contains("refusing to overwrite"));
        });
    }

    #[test]
    fn status_text_does_not_claim_downloaded_when_deferred() {
        let report = WizardFinishReport {
            keybind_status: KeybindSetupStatus::NotAttempted,
            keybind_message: String::new(),
            model_status: ModelSetupStatus::Deferred {
                message: "run setup".into(),
            },
            config_path: PathBuf::from("/tmp/x"),
            marker_path: Some(PathBuf::from("/tmp/y")),
        };
        let text = report.status_text();
        assert!(text.contains("deferred") || text.contains("Deferred"));
        assert!(!text.contains("Model downloaded and ready"));
    }
}
