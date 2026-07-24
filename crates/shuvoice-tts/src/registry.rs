//! TTS backend registry and factory.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::backend::{
    ElevenLabsConfig, ElevenLabsTtsBackend, KokoroConfig, KokoroTtsBackend, MeloWorkerSpawn,
    OpenAiConfig, OpenAiTtsBackend, PiperConfig, PiperTtsBackend, SharedBackend, TtsBackend,
};
#[cfg(feature = "worker-proto")]
use crate::backend::{MeloTtsBackend, MeloTtsConfig, MeloWireMode};
use crate::error::{TtsError, timeout_duration};
use crate::types::{
    BackendId, DEFAULT_ELEVENLABS_TTS_BASE_URL, DEFAULT_ELEVENLABS_TTS_MODEL_ID,
    DEFAULT_ELEVENLABS_TTS_VOICE_ID, DEFAULT_KOKORO_TTS_BASE_URL, DEFAULT_KOKORO_TTS_MODEL_ID,
    DEFAULT_KOKORO_TTS_VOICE_ID, DEFAULT_LOCAL_TTS_MODEL_ID, DEFAULT_LOCAL_TTS_VOICE_ID,
    DEFAULT_OPENAI_TTS_BASE_URL, DEFAULT_OPENAI_TTS_MODEL_ID, DEFAULT_OPENAI_TTS_VOICE_ID,
};

/// High-level configuration used to construct a backend.
///
/// # MeloTTS worker-runtime fields (CLI adapter contract)
///
/// Production Melo is **worker-proto only**. The CLI `tts_adapter` must populate:
///
/// | Field | Required | Meaning |
/// |---|---|---|
/// | [`melotts_venv_path`](Self::melotts_venv_path) | recommended | Isolated venv root (`…/melotts-venv`); python defaults to `{venv}/bin/python` |
/// | [`melotts_device`](Self::melotts_device) | yes (default `auto`) | `auto` \| `cpu` \| `cuda` |
/// | [`melotts_worker_root`](Self::melotts_worker_root) | **yes** (unless spawn override) | Path to the `workers/` tree containing `melotts/` + `shuvoice_worker_proto/` |
/// | [`melotts_python_binary`](Self::melotts_python_binary) | optional | Explicit interpreter; overrides `{venv}/bin/python` |
/// | [`melotts_worker_spawn`](Self::melotts_worker_spawn) | optional | Full typed program/args/env/current_dir override (tests/custom packs) |
/// | [`melotts_worker_command`](Self::melotts_worker_command) | optional | Simple `[program, arg…]` override when spawn is unset |
/// | [`melotts_worker_env`](Self::melotts_worker_env) | optional | Extra child env pairs merged into the derived spawn |
/// | [`melotts_helper_script`](Self::melotts_helper_script) | **must stay `None`** | Legacy only; ignored by the production WorkerProto path |
///
/// Derived production argv (when spawn/command overrides are unset):
/// `{python} -m melotts --device <device}` with
/// `current_dir = worker_root`,
/// `PYTHONPATH = worker_root`,
/// `PYTHONUNBUFFERED=1`,
/// `SHUVOICE_MELOTTS_DEVICE`,
/// `SHUVOICE_MELOTTS_VENV`.
#[derive(Debug, Clone)]
pub struct TtsBackendSettings {
    pub backend: BackendId,
    pub api_key_env: String,
    pub output_format: String,
    pub max_chars: usize,
    pub request_timeout_sec: f64,
    pub default_voice_id: String,
    pub model_id: String,
    pub local_model_path: Option<PathBuf>,
    pub local_voice: Option<String>,
    pub piper_binary: Option<PathBuf>,
    pub melotts_venv_path: Option<PathBuf>,
    pub melotts_device: String,
    /// Legacy helper path. Production WorkerProto **ignores** this field.
    pub melotts_helper_script: Option<PathBuf>,
    /// Root of the bundled `workers/` tree (`melotts/__main__.py` must exist).
    pub melotts_worker_root: Option<PathBuf>,
    /// Optional explicit Python interpreter for MeloTTS.
    pub melotts_python_binary: Option<PathBuf>,
    /// Full typed spawn override (program/args/env/current_dir).
    pub melotts_worker_spawn: Option<MeloWorkerSpawn>,
    /// Simple `[program, arg…]` override when [`Self::melotts_worker_spawn`] is unset.
    pub melotts_worker_command: Option<Vec<String>>,
    /// Extra env pairs merged into the derived WorkerProto spawn.
    pub melotts_worker_env: Vec<(String, String)>,
    /// Kokoro OpenAI-compatible base URL (default `http://localhost:8880/v1`).
    pub kokoro_base_url: String,
    /// ElevenLabs API base URL (default `https://api.elevenlabs.io/v1`).
    pub elevenlabs_base_url: String,
    /// OpenAI API base URL (default `https://api.openai.com/v1`).
    pub openai_base_url: String,
}

impl Default for TtsBackendSettings {
    fn default() -> Self {
        Self {
            backend: BackendId::ElevenLabs,
            api_key_env: "ELEVENLABS_API_KEY".into(),
            output_format: "pcm_24000".into(),
            max_chars: 5000,
            request_timeout_sec: 30.0,
            default_voice_id: DEFAULT_ELEVENLABS_TTS_VOICE_ID.into(),
            model_id: DEFAULT_ELEVENLABS_TTS_MODEL_ID.into(),
            local_model_path: None,
            local_voice: None,
            piper_binary: None,
            melotts_venv_path: None,
            melotts_device: "auto".into(),
            melotts_helper_script: None,
            melotts_worker_root: None,
            melotts_python_binary: None,
            melotts_worker_spawn: None,
            melotts_worker_command: None,
            melotts_worker_env: Vec::new(),
            kokoro_base_url: DEFAULT_KOKORO_TTS_BASE_URL.into(),
            elevenlabs_base_url: DEFAULT_ELEVENLABS_TTS_BASE_URL.into(),
            openai_base_url: DEFAULT_OPENAI_TTS_BASE_URL.into(),
        }
    }
}

/// Resolve a backend from its stable configuration name.
pub fn parse_backend_name(name: &str) -> Result<BackendId, TtsError> {
    BackendId::parse(name).ok_or_else(|| {
        let supported = BackendId::all()
            .iter()
            .map(|b| b.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        TtsError::UnknownBackend {
            name: name.to_string(),
            supported,
        }
    })
}

fn normalize_base_url(value: &str, default: &str) -> String {
    let trimmed = value.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        default.trim_end_matches('/').to_string()
    } else {
        trimmed.to_string()
    }
}

/// Construct a backend from validated settings.
pub fn create_tts_backend(settings: &TtsBackendSettings) -> Result<SharedBackend, TtsError> {
    let timeout = timeout_duration(settings.request_timeout_sec);
    match settings.backend {
        BackendId::ElevenLabs => {
            let cfg = ElevenLabsConfig {
                api_key_env: settings.api_key_env.clone(),
                output_format: settings.output_format.clone(),
                max_chars: settings.max_chars,
                request_timeout: timeout,
                default_voice_id: settings.default_voice_id.clone(),
                default_model_id: settings.model_id.clone(),
                base_url: normalize_base_url(
                    &settings.elevenlabs_base_url,
                    DEFAULT_ELEVENLABS_TTS_BASE_URL,
                ),
            };
            Ok(Arc::new(ElevenLabsTtsBackend::new(cfg)?))
        }
        BackendId::OpenAi => {
            let cfg = OpenAiConfig {
                api_key_env: settings.api_key_env.clone(),
                output_format: settings.output_format.clone(),
                max_chars: settings.max_chars,
                request_timeout: timeout,
                default_voice_id: if settings.default_voice_id.is_empty() {
                    DEFAULT_OPENAI_TTS_VOICE_ID.into()
                } else {
                    settings.default_voice_id.clone()
                },
                default_model_id: if settings.model_id.is_empty() {
                    DEFAULT_OPENAI_TTS_MODEL_ID.into()
                } else {
                    settings.model_id.clone()
                },
                base_url: normalize_base_url(
                    &settings.openai_base_url,
                    DEFAULT_OPENAI_TTS_BASE_URL,
                ),
            };
            Ok(Arc::new(OpenAiTtsBackend::new(cfg)?))
        }
        BackendId::Kokoro => {
            let cfg = KokoroConfig {
                base_url: normalize_base_url(
                    &settings.kokoro_base_url,
                    DEFAULT_KOKORO_TTS_BASE_URL,
                ),
                output_format: settings.output_format.clone(),
                max_chars: settings.max_chars,
                request_timeout: timeout,
                default_voice_id: if settings.default_voice_id.is_empty() {
                    DEFAULT_KOKORO_TTS_VOICE_ID.into()
                } else {
                    settings.default_voice_id.clone()
                },
                default_model_id: if settings.model_id.is_empty() {
                    DEFAULT_KOKORO_TTS_MODEL_ID.into()
                } else {
                    settings.model_id.clone()
                },
            };
            Ok(Arc::new(KokoroTtsBackend::new(cfg)?))
        }
        BackendId::Local => {
            let model_path = settings.local_model_path.clone().ok_or_else(|| {
                TtsError::config(
                    "Local TTS requires [tts].tts_local_model_path to point to a Piper model",
                )
            })?;
            let cfg = PiperConfig {
                model_path,
                default_voice_id: if settings.default_voice_id.is_empty() {
                    DEFAULT_LOCAL_TTS_VOICE_ID.into()
                } else {
                    settings.default_voice_id.clone()
                },
                local_voice: settings.local_voice.clone(),
                max_chars: settings.max_chars,
                request_timeout: timeout,
                piper_binary: settings.piper_binary.clone(),
            };
            let _ = DEFAULT_LOCAL_TTS_MODEL_ID;
            Ok(Arc::new(PiperTtsBackend::new(cfg)?))
        }
        BackendId::MeloTts => create_melotts_backend(settings, timeout),
    }
}

fn create_melotts_backend(
    settings: &TtsBackendSettings,
    timeout: Duration,
) -> Result<SharedBackend, TtsError> {
    // Fail closed when the crate is built without the worker-proto transport.
    // Never silently degrade to the legacy helper from the registry/factory.
    #[cfg(not(feature = "worker-proto"))]
    {
        let _ = (settings, timeout);
        Err(TtsError::config(
            "MeloTTS backend requires the `worker-proto` feature of shuvoice-tts",
        ))
    }

    #[cfg(feature = "worker-proto")]
    {
        use crate::types::DEFAULT_MELOTTS_VOICE_ID;
        let mut cfg = MeloTtsConfig {
            device: settings.melotts_device.clone(),
            max_chars: settings.max_chars,
            request_timeout: timeout,
            default_voice_id: if settings.default_voice_id.is_empty() {
                DEFAULT_MELOTTS_VOICE_ID.into()
            } else {
                settings.default_voice_id.clone()
            },
            // Production path: WorkerProto only. Legacy helper is unreachable.
            wire_mode: MeloWireMode::WorkerProto,
            helper_script: None,
            python_binary: settings.melotts_python_binary.clone(),
            worker_root: settings.melotts_worker_root.clone(),
            worker_spawn: settings.melotts_worker_spawn.clone(),
            worker_command: settings.melotts_worker_command.clone(),
            worker_env: settings.melotts_worker_env.clone(),
            ..MeloTtsConfig::default()
        };
        // Even if a caller stuffed melotts_helper_script, drop it — WorkerProto
        // must never consult the legacy helper.
        let _ignored_legacy_helper = &settings.melotts_helper_script;
        if let Some(venv) = &settings.melotts_venv_path {
            cfg.venv_path = venv.clone();
        }
        Ok(Arc::new(MeloTtsBackend::new(cfg)))
    }
}

/// Typed constructors for tests that need concrete backends.
pub fn create_elevenlabs_for_test(
    base_url: impl Into<String>,
    api_key_env: impl Into<String>,
) -> Result<ElevenLabsTtsBackend, TtsError> {
    let cfg = ElevenLabsConfig {
        base_url: base_url.into(),
        api_key_env: api_key_env.into(),
        request_timeout: Duration::from_secs(5),
        ..ElevenLabsConfig::default()
    };
    ElevenLabsTtsBackend::new(cfg)
}

pub fn create_openai_for_test(
    base_url: impl Into<String>,
    api_key_env: impl Into<String>,
) -> Result<OpenAiTtsBackend, TtsError> {
    let cfg = OpenAiConfig {
        base_url: base_url.into(),
        api_key_env: api_key_env.into(),
        request_timeout: Duration::from_secs(5),
        ..OpenAiConfig::default()
    };
    OpenAiTtsBackend::new(cfg)
}

pub fn create_kokoro_for_test(base_url: impl Into<String>) -> Result<KokoroTtsBackend, TtsError> {
    let cfg = KokoroConfig {
        base_url: base_url.into(),
        request_timeout: Duration::from_secs(5),
        ..KokoroConfig::default()
    };
    KokoroTtsBackend::new(cfg)
}

/// Downcast helper used only in unit tests.
pub fn backend_id_of(backend: &dyn TtsBackend) -> BackendId {
    backend.id()
}
