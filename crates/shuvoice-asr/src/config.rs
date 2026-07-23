//! ASR runtime configuration.
//!
//! Policy fields live on [`shuvoice_core::Config`]. This module adds only the
//! **connect/runtime seams** that do not belong in the on-disk TOML schema
//! (worker spawn, download URL override, WS URL override, optional checksum).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use shuvoice_core::{
    AsrBackendKind, ComputeProvider, Config, OpenaiTurnDetection, ResolvedSherpaDecodeMode,
    data_dir as core_data_dir, expand_user_path, is_parakeet_model,
};
use shuvoice_worker_proto::WorkerSpawnConfig;

use crate::error::{AsrError, AsrResult};

/// Re-export core config for callers that want the full product schema.
pub use shuvoice_core::Config as CoreConfig;
pub use shuvoice_core::{
    AsrBackendKind as BackendId, ComputeProvider as Provider, SherpaDecodeMode,
};

/// ASR-facing view: validated core config + optional runtime connect options.
#[derive(Debug, Clone)]
pub struct AsrConfig {
    pub core: Config,
    pub connect: AsrConnectOptions,
}

/// Runtime-only options (not persisted in `config.toml`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrConnectOptions {
    /// Explicit worker argv (`program` + args). Preferred over ad-hoc Vec when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_spawn: Option<WorkerSpawnConfigSerde>,
    /// Legacy: `["program", "arg1", …]` converted to [`WorkerSpawnConfig`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_command: Option<Vec<String>>,
    /// Connect to an already-running worker over a Unix socket (no spawn).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_socket_path: Option<PathBuf>,
    /// Override Sherpa model archive base URL (tests / mirrors).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sherpa_download_base_url: Option<String>,
    /// Optional SHA-256 (hex) of the model archive; verified when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sherpa_archive_sha256: Option<String>,
    /// Override OpenAI Realtime WebSocket URL (tests).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_realtime_ws_url: Option<String>,
    /// Soft cap on archive download size in bytes (default 4 GiB).
    #[serde(default = "default_max_download_bytes")]
    pub max_download_bytes: u64,
}

fn default_max_download_bytes() -> u64 {
    4 * 1024 * 1024 * 1024
}

impl Default for AsrConnectOptions {
    fn default() -> Self {
        Self {
            worker_spawn: None,
            worker_command: None,
            worker_socket_path: None,
            sherpa_download_base_url: None,
            sherpa_archive_sha256: None,
            openai_realtime_ws_url: None,
            max_download_bytes: default_max_download_bytes(),
        }
    }
}

/// Serde-friendly spawn config (paths as strings).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSpawnConfigSerde {
    pub program: PathBuf,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub client_name: Option<String>,
}

impl WorkerSpawnConfigSerde {
    pub fn into_spawn_config(self) -> WorkerSpawnConfig {
        let mut cfg = WorkerSpawnConfig::new(self.program).args(self.args);
        if let Some(name) = self.client_name {
            cfg = cfg.client_name(name);
        } else {
            cfg = cfg.client_name(format!("shuvoice-asr/{}", env!("CARGO_PKG_VERSION")));
        }
        cfg
    }
}

impl Default for AsrConfig {
    fn default() -> Self {
        let mut core = Config::default();
        // validate applies instant_mode + decode resolve caches
        let _ = core.validate();
        Self {
            core,
            connect: AsrConnectOptions::default(),
        }
    }
}

impl AsrConfig {
    /// Build from a validated core config.
    pub fn from_core(mut core: Config) -> AsrResult<Self> {
        core.validate()
            .map_err(|e| AsrError::startup(e.to_string()))?;
        Ok(Self {
            core,
            connect: AsrConnectOptions::default(),
        })
    }

    pub fn with_connect(mut self, connect: AsrConnectOptions) -> Self {
        self.connect = connect;
        self
    }

    pub fn backend(&self) -> AsrBackendKind {
        self.core.asr_backend
    }

    pub fn sample_rate(&self) -> u32 {
        self.core.sample_rate
    }

    pub fn chunk_ms(&self) -> u32 {
        self.core.chunk_ms
    }

    pub fn resolved_sherpa_decode_mode(&self) -> Option<ResolvedSherpaDecodeMode> {
        self.core.resolved_sherpa_decode_mode()
    }

    pub fn is_parakeet(&self) -> bool {
        is_parakeet_model(&self.core.sherpa_model_name)
            || self
                .core
                .sherpa_model_dir
                .as_deref()
                .is_some_and(is_parakeet_model)
    }

    pub fn sherpa_native_chunk_samples(&self) -> usize {
        self.core.sample_rate as usize * self.core.sherpa_chunk_ms as usize / 1000
    }

    pub fn moonshine_native_chunk_samples(&self) -> usize {
        self.core.sample_rate as usize * self.core.moonshine_chunk_ms as usize / 1000
    }

    pub fn openai_native_chunk_samples(&self) -> usize {
        24_000usize * self.core.chunk_ms as usize / 1000
    }

    /// NeMo right-context to native chunk-sample table.
    pub fn nemo_native_chunk_samples(&self) -> usize {
        match self.core.right_context {
            0 => 1280,
            1 => 2560,
            6 => 8960,
            _ => 17920,
        }
    }

    pub fn default_sherpa_model_dir(&self) -> PathBuf {
        core_data_dir()
            .join("models")
            .join("sherpa")
            .join(self.core.sherpa_model_name.trim())
    }

    pub fn sherpa_model_dir_resolved(&self) -> PathBuf {
        // Match CLI setup (`expand_user_path`) so `~/…` works at runtime and setup.
        self.core
            .sherpa_model_dir
            .as_ref()
            .map(expand_user_path)
            .unwrap_or_else(|| self.default_sherpa_model_dir())
    }

    pub fn sherpa_release_download_root(&self) -> String {
        self.connect
            .sherpa_download_base_url
            .clone()
            .unwrap_or_else(|| {
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models".into()
            })
    }

    /// Resolve a [`WorkerSpawnConfig`] from connect options, if any.
    pub fn worker_spawn_config(&self) -> Option<WorkerSpawnConfig> {
        if let Some(s) = self.connect.worker_spawn.clone() {
            return Some(s.into_spawn_config());
        }
        let cmd = self.connect.worker_command.as_ref()?;
        if cmd.is_empty() {
            return None;
        }
        let mut spawn = WorkerSpawnConfig::new(PathBuf::from(&cmd[0]));
        if cmd.len() > 1 {
            spawn = spawn.args(cmd[1..].to_vec());
        }
        spawn = spawn.client_name(format!("shuvoice-asr/{}", env!("CARGO_PKG_VERSION")));
        Some(spawn)
    }

    pub fn validate_openai_startup(&self) -> AsrResult<()> {
        let env_name = self.core.openai_realtime_api_key_env.trim();
        if env_name.is_empty() {
            return Err(AsrError::startup(
                "openai_realtime_api_key_env must not be empty",
            ));
        }
        if env_name.starts_with("sk_") || env_name.starts_with("sk-") {
            return Err(AsrError::startup(
                "openai_realtime_api_key_env looks like a raw API key value; \
                 set it to an environment variable name",
            ));
        }
        if std::env::var(env_name)
            .map(|v| v.trim().is_empty())
            .unwrap_or(true)
        {
            return Err(AsrError::startup(format!(
                "Missing OpenAI API key environment variable: {env_name}"
            )));
        }
        if self.core.openai_realtime_turn_detection != OpenaiTurnDetection::Manual {
            return Err(AsrError::startup(
                "OpenAI Realtime ASR currently supports only \
                 openai_realtime_turn_detection='manual'",
            ));
        }
        Ok(())
    }
}

/// `$XDG_DATA_HOME/shuvoice` (via core).
pub fn data_dir() -> PathBuf {
    core_data_dir()
}

/// Convenience: mutate core fields for tests without full validate of unrelated sections.
pub fn test_config(backend: AsrBackendKind) -> AsrConfig {
    let mut core = Config::default();
    core.asr_backend = backend;
    let _ = core.validate();
    AsrConfig {
        core,
        connect: AsrConnectOptions::default(),
    }
}

pub fn path_as_str(p: &Path) -> String {
    p.display().to_string()
}

// Silence unused import if ComputeProvider only used via re-export
#[allow(dead_code)]
fn _provider_cpu() -> ComputeProvider {
    ComputeProvider::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;
    use shuvoice_core::{PARAKEET_TDT_V3_INT8_MODEL_NAME, SherpaDecodeMode};

    #[test]
    fn auto_parakeet_instant_resolves_offline() {
        let mut cfg = test_config(AsrBackendKind::Sherpa);
        cfg.core.sherpa_model_name = PARAKEET_TDT_V3_INT8_MODEL_NAME.into();
        cfg.core.instant_mode = true;
        cfg.core.sherpa_decode_mode = SherpaDecodeMode::Auto;
        cfg.core.validate().unwrap();
        assert_eq!(
            cfg.resolved_sherpa_decode_mode(),
            Some(ResolvedSherpaDecodeMode::OfflineInstant)
        );
    }

    #[test]
    fn instant_mode_caps_sherpa_chunk() {
        let mut cfg = test_config(AsrBackendKind::Sherpa);
        cfg.core.instant_mode = true;
        cfg.core.sherpa_chunk_ms = 100;
        cfg.core.validate().unwrap();
        assert_eq!(cfg.core.sherpa_chunk_ms, 80);
    }

    #[test]
    fn nemo_right_context_zero_chunk() {
        let mut cfg = test_config(AsrBackendKind::Nemo);
        cfg.core.instant_mode = true;
        cfg.core.right_context = 13;
        cfg.core.validate().unwrap();
        assert_eq!(cfg.core.right_context, 0);
        assert_eq!(cfg.nemo_native_chunk_samples(), 1280);
    }

    #[test]
    fn worker_command_to_spawn() {
        let mut cfg = test_config(AsrBackendKind::Nemo);
        cfg.connect.worker_command = Some(vec!["python".into(), "worker.py".into()]);
        let spawn = cfg.worker_spawn_config().unwrap();
        assert_eq!(spawn.program(), Path::new("python"));
    }

    #[test]
    fn sherpa_model_dir_resolved_expands_tilde() {
        let mut cfg = test_config(AsrBackendKind::Sherpa);
        cfg.core.sherpa_model_dir = Some("~/models/sherpa-custom".into());
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        assert_eq!(
            cfg.sherpa_model_dir_resolved(),
            PathBuf::from(home).join("models/sherpa-custom")
        );
        // Absolute paths pass through unchanged.
        cfg.core.sherpa_model_dir = Some("/abs/sherpa".into());
        assert_eq!(
            cfg.sherpa_model_dir_resolved(),
            PathBuf::from("/abs/sherpa")
        );
    }
}
