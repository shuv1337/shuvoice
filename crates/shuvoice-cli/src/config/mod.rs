//! CLI config commands and runtime overrides over `shuvoice-core`.

use std::path::PathBuf;
use std::str::FromStr;

use serde_json::{Map, Value};
use shuvoice_core::{
    AsrBackendKind, CURRENT_CONFIG_VERSION, ComputeProvider, Config, DeviceRef, InjectionMode,
    OutputMode, TypingTextCase, config_path, data_dir, load_raw, migrate_to_latest,
    wizard_done_path, write_atomic,
};

use crate::error::{EXIT_FAILURE, EXIT_SUCCESS, ExitStatus};
use crate::parser::{ConfigSetKey, ConfigSetValue};

pub use shuvoice_core::Config as CoreConfig;

/// Runtime CLI overrides applied on top of loaded config.
#[derive(Debug, Clone, Default)]
pub struct RuntimeOverrides {
    pub asr_backend: Option<String>,
    pub device: Option<String>,
    pub right_context: Option<i64>,
    pub sherpa_model_dir: Option<String>,
    pub sherpa_model_name: Option<String>,
    pub sherpa_provider: Option<String>,
    pub sherpa_num_threads: Option<i64>,
    pub sherpa_chunk_ms: Option<i64>,
    pub moonshine_model_name: Option<String>,
    pub moonshine_model_dir: Option<String>,
    pub moonshine_model_precision: Option<String>,
    pub moonshine_chunk_ms: Option<i64>,
    pub moonshine_max_window_sec: Option<f64>,
    pub moonshine_max_tokens: Option<i64>,
    pub moonshine_provider: Option<String>,
    pub moonshine_onnx_threads: Option<i64>,
    pub audio_device: Option<String>,
    pub input_gain: Option<f64>,
    pub output_mode: Option<String>,
    pub control_socket: Option<String>,
}

pub fn load_config() -> Result<Config, String> {
    Config::load().map_err(|e| e.to_string())
}

pub fn apply_runtime_overrides(
    cfg: &mut Config,
    overrides: &RuntimeOverrides,
) -> Result<(), String> {
    if let Some(v) = &overrides.asr_backend {
        cfg.asr_backend = AsrBackendKind::from_str(v).map_err(|e| e.to_string())?;
    }
    if let Some(v) = &overrides.device {
        cfg.device = v.clone();
    }
    if let Some(v) = overrides.right_context {
        cfg.right_context =
            u32::try_from(v).map_err(|_| "right_context out of range".to_string())?;
    }
    if let Some(v) = &overrides.sherpa_model_dir {
        cfg.sherpa_model_dir = Some(v.clone());
    }
    if let Some(v) = &overrides.sherpa_model_name {
        cfg.sherpa_model_name = v.clone();
    }
    if let Some(v) = &overrides.sherpa_provider {
        cfg.sherpa_provider = ComputeProvider::from_str(v).map_err(|e| e.to_string())?;
    }
    if let Some(v) = overrides.sherpa_num_threads {
        cfg.sherpa_num_threads =
            u32::try_from(v).map_err(|_| "sherpa_num_threads out of range".to_string())?;
    }
    if let Some(v) = overrides.sherpa_chunk_ms {
        cfg.sherpa_chunk_ms =
            u32::try_from(v).map_err(|_| "sherpa_chunk_ms out of range".to_string())?;
    }
    if let Some(v) = &overrides.moonshine_model_name {
        cfg.moonshine_model_name = v.clone();
    }
    if let Some(v) = &overrides.moonshine_model_dir {
        cfg.moonshine_model_dir = Some(v.clone());
    }
    if let Some(v) = &overrides.moonshine_model_precision {
        cfg.moonshine_model_precision = v.clone();
    }
    if let Some(v) = overrides.moonshine_chunk_ms {
        cfg.moonshine_chunk_ms =
            u32::try_from(v).map_err(|_| "moonshine_chunk_ms out of range".to_string())?;
    }
    if let Some(v) = overrides.moonshine_max_window_sec {
        cfg.moonshine_max_window_sec = v;
    }
    if let Some(v) = overrides.moonshine_max_tokens {
        cfg.moonshine_max_tokens =
            u32::try_from(v).map_err(|_| "moonshine_max_tokens out of range".to_string())?;
    }
    if let Some(v) = &overrides.moonshine_provider {
        cfg.moonshine_provider = ComputeProvider::from_str(v).map_err(|e| e.to_string())?;
    }
    if let Some(v) = overrides.moonshine_onnx_threads {
        cfg.moonshine_onnx_threads =
            u32::try_from(v).map_err(|_| "moonshine_onnx_threads out of range".to_string())?;
    }
    if let Some(v) = &overrides.audio_device {
        cfg.audio_device = parse_device_ref(v)?;
    }
    if let Some(v) = overrides.input_gain {
        cfg.input_gain = v;
    }
    if let Some(v) = &overrides.output_mode {
        cfg.output_mode = OutputMode::from_str(v).map_err(|e| e.to_string())?;
    }
    if let Some(v) = &overrides.control_socket {
        cfg.control_socket = Some(v.clone());
    }
    cfg.validate().map_err(|e| e.to_string())
}

fn parse_device_ref(raw: &str) -> Result<Option<DeviceRef>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if trimmed.chars().all(|c| c.is_ascii_digit()) {
        let idx: i64 = trimmed
            .parse()
            .map_err(|_| format!("invalid device index '{trimmed}'"))?;
        Ok(Some(DeviceRef::Index(idx)))
    } else {
        Ok(Some(DeviceRef::Name(trimmed.to_string())))
    }
}

pub fn load_effective_config(overrides: &RuntimeOverrides) -> Result<Config, String> {
    let mut cfg = load_config()?;
    apply_runtime_overrides(&mut cfg, overrides)?;
    Ok(cfg)
}

/// First-run wizard gate: missing marker and missing config file.
pub fn needs_wizard() -> bool {
    if wizard_done_path().exists() {
        return false;
    }
    !config_path().exists()
}

pub fn cmd_path() -> ExitStatus {
    println!("{}", Config::config_path().display());
    ExitStatus::code(EXIT_SUCCESS)
}

pub fn cmd_validate() -> ExitStatus {
    let path = Config::config_path();
    match (|| -> Result<(Config, u32), String> {
        let raw = load_raw(&path).map_err(|e| e.to_string())?;
        let (_migrated, report) = migrate_to_latest(&raw).map_err(|e| e.to_string())?;
        let cfg = Config::load().map_err(|e| e.to_string())?;
        Ok((cfg, report.from_version))
    })() {
        Ok((cfg, from_version)) => {
            println!(
                "OK (schema={}, migrated_from={}, path={})",
                cfg.config_version,
                from_version,
                path.display()
            );
            ExitStatus::code(EXIT_SUCCESS)
        }
        Err(err) => {
            eprintln!("ERROR: {err}");
            ExitStatus::code(EXIT_FAILURE)
        }
    }
}

pub fn cmd_effective() -> ExitStatus {
    match Config::load() {
        Ok(cfg) => match cfg.to_toml_string() {
            Ok(text) => {
                print!("{text}");
                ExitStatus::code(EXIT_SUCCESS)
            }
            Err(err) => {
                eprintln!("ERROR: {err}");
                ExitStatus::code(EXIT_FAILURE)
            }
        },
        Err(err) => {
            eprintln!("ERROR: {err}");
            ExitStatus::code(EXIT_FAILURE)
        }
    }
}

pub fn cmd_set(key: ConfigSetKey, value: ConfigSetValue) -> ExitStatus {
    match set_config_key(key, value) {
        Ok(msg) => {
            println!("{msg}");
            ExitStatus::code(EXIT_SUCCESS)
        }
        Err(err) => {
            eprintln!("ERROR: {err}");
            ExitStatus::code(EXIT_FAILURE)
        }
    }
}

fn set_config_key(key: ConfigSetKey, value: ConfigSetValue) -> Result<String, String> {
    let key_norm = key.as_str();
    let value_norm = value.as_str().to_ascii_lowercase();

    match key {
        ConfigSetKey::TypingFinalInjectionMode => {
            let _ = InjectionMode::from_str(&value_norm).map_err(|e| e.to_string())?;
        }
        ConfigSetKey::TypingTextCase => {
            let _ = TypingTextCase::from_str(&value_norm).map_err(|e| e.to_string())?;
        }
        ConfigSetKey::OverlayDebugMode => {
            if !matches!(value_norm.as_str(), "true" | "false") {
                return Err("overlay_debug_mode must be one of: true, false".into());
            }
        }
    }

    let config_file = Config::config_path();
    let raw = load_raw(&config_file).map_err(|e| e.to_string())?;
    let (mut migrated, _report) = migrate_to_latest(&raw).map_err(|e| e.to_string())?;

    let section = match key {
        ConfigSetKey::TypingFinalInjectionMode | ConfigSetKey::TypingTextCase => "typing",
        ConfigSetKey::OverlayDebugMode => "overlay",
    };

    let table = migrated
        .entry(section.to_string())
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| format!("[{section}] must be a table"))?;

    match key {
        ConfigSetKey::OverlayDebugMode => {
            table.insert(key_norm.to_string(), Value::Bool(value_norm == "true"));
        }
        ConfigSetKey::TypingFinalInjectionMode => {
            table.insert(key_norm.to_string(), Value::String(value_norm.clone()));
            table.insert(
                "use_clipboard_for_final".into(),
                Value::Bool(value_norm != "direct"),
            );
        }
        ConfigSetKey::TypingTextCase => {
            table.insert(key_norm.to_string(), Value::String(value_norm.clone()));
        }
    }

    // Ensure version stamp present.
    migrated.insert("config_version".into(), Value::from(CURRENT_CONFIG_VERSION));

    // Persist then validate by reloading through core Config.
    let backup = write_atomic(&config_file, &migrated).map_err(|e| e.to_string())?;
    Config::load().map_err(|e| e.to_string())?;

    let extra = if matches!(key, ConfigSetKey::TypingFinalInjectionMode) {
        let legacy = if value_norm != "direct" {
            "true"
        } else {
            "false"
        };
        format!(
            " (use_clipboard_for_final={legacy}, path={}",
            config_file.display()
        )
    } else {
        format!(" (path={}", config_file.display())
    };

    let msg = if let Some(backup) = backup {
        format!(
            "OK set {key_norm}={value_norm}{extra}, backup={})",
            backup.display()
        )
    } else {
        format!("OK set {key_norm}={value_norm}{extra})")
    };
    Ok(msg)
}

/// Shared data-dir helper for setup/model paths.
pub fn models_data_dir() -> PathBuf {
    data_dir().join("models")
}
