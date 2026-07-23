//! Setup helpers: install plans, model download, Piper/MeloTTS automation.

pub mod http;
pub mod install;
pub mod layer_shell;
pub mod melotts;
pub mod piper;
pub mod sherpa_model;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use shuvoice_core::{Config, TtsBackendKind, expand_user_path};
use shuvoice_io::process::CommandRunner;

use self::http::{SharedDownloader, default_downloader};
use self::install::default_runner;

/// Injectable setup context.
pub struct SetupContext {
    pub runner: Arc<dyn CommandRunner>,
    pub downloader: SharedDownloader,
    /// Optional Sherpa archive URL override (tests).
    pub sherpa_archive_url_override: Option<String>,
}

impl Default for SetupContext {
    fn default() -> Self {
        Self {
            runner: default_runner(),
            downloader: default_downloader(),
            sherpa_archive_url_override: None,
        }
    }
}

/// Persist local Piper selection into config.toml after successful setup only.
pub fn persist_local_tts_selection(
    config: &mut Config,
    model_dir: &std::path::Path,
    voice_stem: &str,
) -> Result<(), String> {
    config.tts_backend = TtsBackendKind::Local;
    config.tts_default_voice_id = voice_stem.to_string();
    config.tts_model_id = "piper".into();
    config.tts_local_model_path = Some(path_for_config(model_dir));
    config.tts_local_voice = Some(voice_stem.to_string());
    config.validate().map_err(|e| e.to_string())?;
    config
        .save_to_path(Config::config_path())
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn path_for_config(path: &std::path::Path) -> String {
    let expanded = expand_user_path(path);
    if let Ok(home) = std::env::var("HOME") {
        let home = PathBuf::from(home);
        if let Ok(rel) = expanded.strip_prefix(&home) {
            return format!("~/{}", rel.display());
        }
    }
    expanded.display().to_string()
}

pub fn kokoro_base_url(config: &Config) -> String {
    config.tts_kokoro_base_url.trim_end_matches('/').to_string()
}

const KOKORO_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const KOKORO_OVERALL_TIMEOUT: Duration = Duration::from_secs(10);
const KOKORO_MAX_BODY_BYTES: u64 = 1_048_576; // 1 MiB

/// Preflight Kokoro voice list endpoint (no secrets).
///
/// Requires a successful response with at least one usable non-empty voice id.
/// Body is hard-capped; connect/overall timeouts are finite.
pub async fn check_kokoro_voices(base_url: &str) -> Result<usize, String> {
    let url = format!("{}/audio/voices", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .connect_timeout(KOKORO_CONNECT_TIMEOUT)
        .timeout(KOKORO_OVERALL_TIMEOUT)
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .get(&url)
        .header("Authorization", "Bearer sk-local")
        .send()
        .await
        .map_err(|e| format!("Kokoro endpoint unreachable: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Kokoro /audio/voices HTTP {}", resp.status()));
    }
    if let Some(len) = resp.content_length()
        && len > KOKORO_MAX_BODY_BYTES
    {
        return Err(format!(
            "Kokoro /audio/voices body Content-Length {len} exceeds cap {KOKORO_MAX_BODY_BYTES}"
        ));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("Kokoro body read failed: {e}"))?;
    if bytes.len() as u64 > KOKORO_MAX_BODY_BYTES {
        return Err(format!(
            "Kokoro /audio/voices body {} bytes exceeds cap {KOKORO_MAX_BODY_BYTES}",
            bytes.len()
        ));
    }
    let body: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("Kokoro JSON: {e}"))?;
    let voices = extract_usable_kokoro_voices(&body);
    if voices.is_empty() {
        return Err("Kokoro /audio/voices returned no usable voices".into());
    }
    Ok(voices.len())
}

fn extract_usable_kokoro_voices(body: &serde_json::Value) -> Vec<String> {
    let mut out = Vec::new();
    let push_str = |out: &mut Vec<String>, s: &str| {
        let t = s.trim();
        if !t.is_empty() {
            out.push(t.to_string());
        }
    };
    if let Some(arr) = body.get("voices").and_then(|v| v.as_array()) {
        for item in arr {
            if let Some(s) = item.as_str() {
                push_str(&mut out, s);
            } else if let Some(s) = item.get("id").and_then(|v| v.as_str()) {
                push_str(&mut out, s);
            } else if let Some(s) = item.get("name").and_then(|v| v.as_str()) {
                push_str(&mut out, s);
            } else if let Some(s) = item.get("voice_id").and_then(|v| v.as_str()) {
                push_str(&mut out, s);
            }
        }
    } else if let Some(arr) = body.as_array() {
        for item in arr {
            if let Some(s) = item.as_str() {
                push_str(&mut out, s);
            }
        }
    }
    out
}

pub fn tts_api_key_env_status(config: &Config) -> Result<String, String> {
    if !config.tts_enabled {
        return Ok("disabled".into());
    }
    let needs = matches!(
        config.tts_backend,
        TtsBackendKind::Elevenlabs | TtsBackendKind::Openai
    );
    if !needs {
        return Ok("not required".into());
    }
    let env_name = config.tts_api_key_env.trim();
    // Guardrail: never treat raw keys as env names; never log values.
    if env_name.starts_with("sk_") || env_name.starts_with("sk-") || env_name.starts_with("xi_") {
        return Err(
            "tts_api_key_env looks like a raw API key value, expected an environment variable name"
                .into(),
        );
    }
    match std::env::var(env_name) {
        Ok(v) if !v.trim().is_empty() => Ok(format!("{env_name} is set")),
        _ => Err(format!("Environment variable {env_name} is not set")),
    }
}

pub fn openai_asr_key_status(config: &Config) -> Result<String, String> {
    if config.asr_backend != shuvoice_core::AsrBackendKind::OpenaiRealtime {
        return Ok("n/a".into());
    }
    let env_name = config.openai_realtime_api_key_env.trim();
    if env_name.starts_with("sk-") || env_name.starts_with("sk_") {
        return Err(
            "openai_realtime_api_key_env looks like a raw API key value, expected an environment variable name"
                .into(),
        );
    }
    match std::env::var(env_name) {
        Ok(v) if !v.trim().is_empty() => Ok(format!("{env_name} is set")),
        _ => Err(format!("Environment variable {env_name} is not set")),
    }
}

/// Run a blocking setup step off the Tokio worker (installs, prompts, etc.).
pub async fn run_blocking_setup<F, T>(f: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| format!("setup blocking task join error: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kokoro_extracts_string_list_and_objects() {
        let body = serde_json::json!({"voices": ["af_heart", "", "af_sarah"]});
        assert_eq!(extract_usable_kokoro_voices(&body).len(), 2);
        let body = serde_json::json!([{"id": "a"}, {"name": "b"}, {"voice_id": "c"}]);
        // top-level array of objects is not used by Kokoro-FastAPI string path; only strings
        assert!(extract_usable_kokoro_voices(&body).is_empty());
        let body = serde_json::json!(["af_heart", "af_sarah"]);
        assert_eq!(extract_usable_kokoro_voices(&body).len(), 2);
    }

    #[test]
    fn path_for_config_expands_and_rehomes() {
        let old = std::env::var_os("HOME");
        // SAFETY: unit test mutates process-global HOME and restores it before return.
        unsafe {
            std::env::set_var("HOME", "/home/tester");
        }
        let s = path_for_config(std::path::Path::new(
            "/home/tester/.local/share/shuvoice/models/piper",
        ));
        assert_eq!(s, "~/.local/share/shuvoice/models/piper");
        let s = path_for_config(std::path::Path::new("~/models"));
        assert!(s.contains("models"), "{s}");
        // SAFETY: restore HOME to the value captured before this test mutated it.
        unsafe {
            match old {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
    }
}
