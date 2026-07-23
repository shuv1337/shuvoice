//! Production composition root for `shuvoice run`.
//!
//! Wires validated [`shuvoice_core::Config`] into session runtime, control
//! socket, capture/TTS/UI bridges, and ordered teardown. Leaf adapters live in
//! sibling modules (feature-gated where their crates are optional).
//!
//! # Feature surface
//!
//! | Feature        | Role                                              |
//! |----------------|---------------------------------------------------|
//! | `audio`        | CPAL capture bridge                               |
//! | `asr-sherpa`   | Native static Sherpa                              |
//! | `asr-openai`   | Native OpenAI Realtime (only when configured)     |
//! | `ui`           | GTK host + caption/TTS overlays                   |
//! | `tts`          | CPAL TTS player + feedback tones                  |
//! | `tts-worker`   | MeloTTS worker-proto only                         |
//! | `desktop`      | full packaged set (default)                       |
//!
//! Minimal (`--no-default-features`) builds compile the CLI surface; `run`
//! fails closed with exit 78 when the selected backend/feature set cannot
//! compose a real session.
//!
//! # GTK coordination
//!
//! Process entry (`run_blocking`) loads `local.dev` on a single thread, then
//! builds a multi-thread Tokio runtime and `block_on`s the main future.
//! After background tasks are spawned onto worker threads, the main future
//! **synchronously** calls [`ui_bridge::run_gtk_main_host`] so all GTK objects
//! stay on the main/GLib thread. Worker threads drive session/control/audio
//! pumps. After GTK returns, the main future awaits ordered teardown joins.
//!
//! # Teardown order
//!
//! 1. Stop accepting control connections
//! 2. Stop audio ingress
//! 3. Request session shutdown (bounded runtime shutdown)
//! 4. Explicit feedback tone worker shutdown (not only Drop)
//! 5. Drain/abort forwarder tasks (TTS bridge uses sticky ack drain)
//! 6. Request GTK quit (if still running)
//!
//! Detached OS threads after join timeouts are reported honestly
//! (`DetachedAfterTimeout`) and never claimed as clean release.

#![allow(clippy::too_many_arguments)]

#[cfg(feature = "audio")]
pub mod audio_bridge;
pub mod control_bridge;
#[cfg(feature = "tts")]
pub mod feedback;
pub mod io_adapters;
#[cfg(feature = "tts")]
pub mod tts_adapter;
#[cfg(feature = "ui")]
pub mod ui_bridge;
pub mod worker_runtime;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use shuvoice_app::traits::{FeedbackSink, OverlaySink, SystemClock, TtsEngine};
#[allow(unused_imports)] // SessionCommand used under feature = "tts"
use shuvoice_app::{
    EnqueueControlAdapter, SessionCommand, SessionEvent, SessionRuntime, TtsPlayerState,
    spawn_session_runtime,
};
use shuvoice_asr::DynAsrBackend;
use shuvoice_control::{ControlCommand, ControlServer};
use shuvoice_core::{AsrBackendKind, Config, OverlayState, TtsBackendKind};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::error::{EXIT_DEPENDENCY, EXIT_FAILURE, EXIT_SUCCESS, ExitStatus};
use crate::setup::layer_shell::layer_shell_present;

use self::control_bridge::ControlBridge;
use self::io_adapters::{IoSelection, IoTextInjector};
use self::worker_runtime::{WorkerRuntimeError, discover_asr_worker_runtime};

#[cfg(feature = "tts")]
use self::worker_runtime::{
    MELOTTS_WORKER_MODULE, PYTHONPATH_ENV, PYTHONUNBUFFERED_ENV, WorkersDiscoveryInputs,
    resolve_melotts_workers_root, resolve_melotts_workers_root_from_process,
};
#[cfg(any(feature = "asr-sherpa", feature = "asr-openai"))]
use shuvoice_asr::{AsrConfig, create_backend};
#[cfg(feature = "ui")]
use tokio::sync::oneshot;

// ── Errors ────────────────────────────────────────────────────────────────

#[derive(Debug)]
enum ComposeError {
    Dependency(String),
    #[allow(dead_code)]
    Runtime(String),
}

impl ComposeError {
    fn dep(msg: impl Into<String>) -> Self {
        Self::Dependency(msg.into())
    }
}

// ── Feature / dependency validation ───────────────────────────────────────

/// Validate that the selected ASR backend is compiled in and (when UI is
/// built) that gtk4-layer-shell is loadable.
///
/// Does **not** probe microphones, download models, or mutate services.
pub fn validate_composition_config(config: &Config) -> Result<(), String> {
    validate_asr_feature(config.asr_backend)?;
    validate_tts_feature(config)?;
    validate_audio_feedback_feature(config)?;
    validate_layer_shell_if_needed()?;
    Ok(())
}

fn validate_asr_feature(backend: AsrBackendKind) -> Result<(), String> {
    match backend {
        AsrBackendKind::Sherpa if !cfg!(feature = "asr-sherpa") => Err(
            "Sherpa ASR support not built into this binary (missing feature asr-sherpa). \
             Rebuild with: cargo build -p shuvoice-cli --features asr-sherpa \
             (or --features desktop)"
                .into(),
        ),
        AsrBackendKind::OpenaiRealtime if !cfg!(feature = "asr-openai") => Err(
            "OpenAI Realtime ASR support not built into this binary (missing feature asr-openai). \
             Rebuild with: cargo build -p shuvoice-cli --features asr-openai \
             (or --features desktop)"
                .into(),
        ),
        // NeMo / Moonshine: worker client is always linked; discovery at build time.
        _ => Ok(()),
    }
}

fn validate_tts_feature(config: &Config) -> Result<(), String> {
    if !config.tts_enabled {
        return Ok(());
    }
    if !cfg!(feature = "tts") {
        return Err(
            "TTS is enabled in config but this binary was built without the `tts` feature. \
             Rebuild with: cargo build -p shuvoice-cli --features tts \
             (or --features desktop)"
                .into(),
        );
    }
    if config.tts_backend == TtsBackendKind::Melotts && !cfg!(feature = "tts-worker") {
        return Err(
            "MeloTTS requires the CLI feature `tts-worker` (worker-proto only). \
             Rebuild with: cargo build -p shuvoice-cli --features tts-worker \
             (or --features desktop)"
                .into(),
        );
    }
    Ok(())
}

/// `audio_feedback=true` needs the tone player path (`tts` feature / CPAL output).
///
/// Without it the binary would silently install [`NullFeedback`] and lie that
/// tones are available. Fail closed with exit 78 instead.
fn validate_audio_feedback_feature(config: &Config) -> Result<(), String> {
    if !config.audio_feedback {
        return Ok(());
    }
    if cfg!(feature = "tts") {
        return Ok(());
    }
    Err(
        "audio_feedback=true requires the CLI feature `tts` (CPAL tone output). \
         Rebuild with: cargo build -p shuvoice-cli --features tts \
         (or --features desktop), or set audio_feedback=false in config"
            .into(),
    )
}

fn validate_layer_shell_if_needed() -> Result<(), String> {
    if !cfg!(feature = "ui") {
        return Ok(());
    }
    if layer_shell_present() {
        return Ok(());
    }
    Err(
        "libgtk4-layer-shell.so not found. Install it with: pacman -S gtk4-layer-shell \
         | apt install libgtk-4-layer-shell0 | dnf install gtk4-layer-shell"
            .into(),
    )
}

// ── ASR backend construction ──────────────────────────────────────────────

/// Build the ASR backend for production (or harness injection).
///
/// - Sherpa / OpenAI: native via [`AsrConfig`] + factory (feature-gated).
/// - NeMo / Moonshine: **only** via
///   [`discover_asr_worker_runtime`] + `WorkerAsrBackend::with_spawn`
///   (full spawn with `PYTHONPATH` / `current_dir`). Never the lossy factory path.
pub fn build_asr_backend(config: &Config) -> Result<Box<DynAsrBackend>, String> {
    validate_asr_feature(config.asr_backend)?;
    match config.asr_backend {
        AsrBackendKind::Sherpa => build_native_sherpa(config),
        AsrBackendKind::OpenaiRealtime => build_native_openai(config),
        AsrBackendKind::Nemo | AsrBackendKind::Moonshine => build_worker_asr(config),
    }
}

fn build_native_sherpa(config: &Config) -> Result<Box<DynAsrBackend>, String> {
    #[cfg(feature = "asr-sherpa")]
    {
        let asr = AsrConfig::from_core(config.clone()).map_err(|e| e.to_string())?;
        create_backend(asr).map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "asr-sherpa"))]
    {
        let _ = config;
        Err("Sherpa ASR support not built (missing feature asr-sherpa)".into())
    }
}

fn build_native_openai(config: &Config) -> Result<Box<DynAsrBackend>, String> {
    #[cfg(feature = "asr-openai")]
    {
        // Only constructed when config selects OpenAI. No network I/O until load.
        let asr = AsrConfig::from_core(config.clone()).map_err(|e| e.to_string())?;
        create_backend(asr).map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "asr-openai"))]
    {
        let _ = config;
        Err("OpenAI Realtime ASR support not built (missing feature asr-openai)".into())
    }
}

fn build_worker_asr(config: &Config) -> Result<Box<DynAsrBackend>, String> {
    use shuvoice_asr::worker::WorkerAsrBackend;

    let resolved = discover_asr_worker_runtime(config).map_err(format_worker_runtime_error)?;
    let asr = resolved
        .asr_config_lossy(config.clone())
        .map_err(|e| e.to_string())?;
    // REQUIRED: full spawn (env + current_dir). Do not use factory / lossy attach.
    let backend = WorkerAsrBackend::new(resolved.backend_kind(), asr).with_spawn(resolved.spawn);
    Ok(Box::new(backend))
}

/// Actionable exit-78 messaging for worker discovery failures (no path secrets).
#[must_use]
pub fn format_worker_runtime_error(err: WorkerRuntimeError) -> String {
    match err {
        WorkerRuntimeError::WorkersRootNotFound => {
            "ASR worker root not found. Install bundled workers \
             (/usr/lib/shuvoice/workers) or set SHUVOICE_WORKERS_DIR to a valid \
             workers tree, then run: shuvoice setup --install-missing"
                .into()
        }
        WorkerRuntimeError::PythonUnusable { label, detail } => {
            format!(
                "ASR worker Python unusable ({label}: {detail}). \
                 Run: shuvoice setup --install-missing  (creates the isolated worker venv)"
            )
        }
        WorkerRuntimeError::MissingPackage { package } => {
            format!(
                "workers tree is missing required package '{package}'. \
                 Reinstall shuvoice workers or set SHUVOICE_WORKERS_DIR"
            )
        }
        WorkerRuntimeError::MissingBackendModule { module } => {
            format!(
                "workers tree is missing backend module '{module}'. \
                 Reinstall shuvoice workers or set SHUVOICE_WORKERS_DIR"
            )
        }
        WorkerRuntimeError::InvalidEnvWorkersDir
        | WorkerRuntimeError::InvalidEnvWorkersDirNonUnicode => {
            "SHUVOICE_WORKERS_DIR is set but invalid (need absolute UTF-8 path to a workers tree)"
                .into()
        }
        WorkerRuntimeError::NotAWorkerBackend(name) => {
            format!("ASR backend '{name}' does not use an external Python worker")
        }
        WorkerRuntimeError::Config(msg) => format!("invalid ASR config for worker: {msg}"),
    }
}

/// Preflight/setup helper: dependency errors for the configured ASR backend.
///
/// NeMo/Moonshine use the same discovery path as production composition and
/// feed lossy connect options into [`shuvoice_asr::dependency_errors_for`] so a
/// setup-created venv + bundled workers tree reports READY. **Never** persists
/// a worker command into config.toml.
pub fn asr_dependency_errors_for(config: &Config) -> Vec<String> {
    let backend = config.asr_backend;
    match backend {
        AsrBackendKind::Nemo | AsrBackendKind::Moonshine => {
            match discover_asr_worker_runtime(config) {
                Ok(resolved) => match resolved.asr_config_lossy(config.clone()) {
                    Ok(asr) => rewrite_asr_feature_hints(shuvoice_asr::dependency_errors_for(
                        backend,
                        Some(&asr),
                    )),
                    Err(e) => vec![e.to_string()],
                },
                Err(err) => vec![format_worker_runtime_error(err)],
            }
        }
        AsrBackendKind::Sherpa | AsrBackendKind::OpenaiRealtime => {
            rewrite_asr_feature_hints(shuvoice_asr::dependency_errors_for(backend, None))
        }
    }
}

fn rewrite_asr_feature_hints(errors: Vec<String>) -> Vec<String> {
    errors
        .into_iter()
        .map(|e| {
            e.replace(
                "cargo build -p shuvoice-asr --features sherpa",
                "cargo build -p shuvoice-cli --features asr-sherpa",
            )
            .replace(
                "cargo build -p shuvoice-asr --features openai",
                "cargo build -p shuvoice-cli --features asr-openai",
            )
            .replace("Missing sherpa feature", "Missing asr-sherpa feature")
            .replace("Missing openai feature", "Missing asr-openai feature")
        })
        .collect()
}

// ── Null sinks ────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct NullOverlay;

impl OverlaySink for NullOverlay {
    fn show(&mut self, _state: OverlayState, _text: &str) {}
    fn set_state(&mut self, _state: OverlayState) {}
    fn set_text(&mut self, _text: &str) {}
    fn hide(&mut self) {}
}

#[derive(Debug, Default)]
struct NullFeedback;

impl FeedbackSink for NullFeedback {}

/// Shared feedback handle so composition can shut the tone worker down
/// **explicitly** after session shutdown (step 4), not only via Drop.
#[cfg(feature = "tts")]
#[derive(Clone)]
struct SharedFeedback {
    inner: Arc<std::sync::Mutex<Option<feedback::ToneFeedbackSink>>>,
}

#[cfg(feature = "tts")]
impl SharedFeedback {
    fn new(sink: feedback::ToneFeedbackSink) -> Self {
        Self {
            inner: Arc::new(std::sync::Mutex::new(Some(sink))),
        }
    }

    /// Take the sink (if still present) and run bounded shutdown.
    fn shutdown(&self) -> feedback::ShutdownOutcome {
        let taken = self.inner.lock().ok().and_then(|mut g| g.take());
        match taken {
            Some(mut fb) => fb.shutdown(),
            None => feedback::ShutdownOutcome::AlreadyStopped,
        }
    }
}

#[cfg(feature = "tts")]
impl FeedbackSink for SharedFeedback {
    fn play_start(&mut self) {
        if let Ok(mut g) = self.inner.lock()
            && let Some(fb) = g.as_mut()
        {
            fb.play_start();
        }
    }

    fn play_stop(&mut self) {
        if let Ok(mut g) = self.inner.lock()
            && let Some(fb) = g.as_mut()
        {
            fb.play_stop();
        }
    }
}

#[derive(Debug, Default)]
struct NullTts;

impl TtsEngine for NullTts {
    fn state(&self) -> TtsPlayerState {
        TtsPlayerState::Idle
    }
    fn speak(&mut self, _: &str, _: &str, _: &str) -> Result<bool, String> {
        Err("tts not available".into())
    }
    fn pause(&mut self) -> bool {
        false
    }
    fn resume(&mut self) -> bool {
        false
    }
    fn toggle_pause(&mut self) -> bool {
        false
    }
    fn restart(&mut self) -> bool {
        false
    }
    fn stop(&mut self) -> bool {
        false
    }
    fn set_playback_speed(&mut self, speed: f64) -> f64 {
        speed
    }
}

// ── TTS + feedback construction ───────────────────────────────────────────

/// Env key set on Melo worker children for the selected device overlay.
#[cfg(feature = "tts")]
pub const MELOTTS_DEVICE_ENV: &str = "SHUVOICE_MELOTTS_DEVICE";

/// Env key set on Melo worker children for the isolated venv path overlay.
#[cfg(feature = "tts")]
pub const MELOTTS_VENV_ENV: &str = "SHUVOICE_MELOTTS_VENV";

#[cfg(feature = "tts")]
struct TtsBundle {
    engine: Option<tts_adapter::PlayerTtsEngine>,
    bridge: Option<tts_adapter::TtsPlayerUpdateBridge>,
    bridge_rx: Option<tokio::sync::mpsc::Receiver<SessionCommand>>,
    feedback: feedback::ToneFeedbackSink,
    caps: Option<shuvoice_core::TtsCapabilities>,
}

/// Build production [`shuvoice_tts::TtsBackendSettings`] from config.
///
/// Base mapping comes from [`tts_adapter::tts_backend_settings_from_config`]
/// (intentionally leaves Melo `worker_root` unset). For MeloTTS this then:
///
/// 1. [`resolve_melotts_workers_root_from_process`] — requires `melotts/` **and**
///    `shuvoice_worker_proto/` (proto-only trees are rejected)
/// 2. Sets `melotts_worker_root = Some(path)` plus venv/python
/// 3. Clears legacy helper path
/// 4. Installs a typed [`shuvoice_tts::MeloWorkerSpawn`] (`python -m melotts …`)
///    with `PYTHONPATH` / `PYTHONUNBUFFERED` / device / venv env overlays
///
/// Requires CLI feature `tts-worker` when `tts_backend = melotts`.
/// Ready-check is enforced by [`tts_adapter::PlayerTtsEngine::from_backend`]
/// via [`tts_adapter::ensure_backend_ready`].
#[cfg(feature = "tts")]
pub fn compose_tts_backend_settings(
    config: &Config,
) -> Result<shuvoice_tts::TtsBackendSettings, String> {
    use shuvoice_tts::BackendId;

    let mut settings = tts_adapter::tts_backend_settings_from_config(config);
    settings.melotts_helper_script = None;
    if settings.backend != BackendId::MeloTts {
        return Ok(settings);
    }
    // Dedicated Melo resolver (not generic workers root with None).
    let discovered =
        resolve_melotts_workers_root_from_process().map_err(format_worker_runtime_error)?;
    apply_melotts_runtime_fields(config, &mut settings, discovered.path)?;
    Ok(settings)
}

/// Injectable variant of [`compose_tts_backend_settings`] (tests supply temp roots).
#[cfg(feature = "tts")]
pub fn compose_tts_backend_settings_with_discovery(
    config: &Config,
    melotts_discovery: &WorkersDiscoveryInputs,
) -> Result<shuvoice_tts::TtsBackendSettings, String> {
    use shuvoice_tts::BackendId;

    let mut settings = tts_adapter::tts_backend_settings_from_config(config);
    // Belt-and-suspenders: never carry a legacy helper into production.
    settings.melotts_helper_script = None;

    if settings.backend != BackendId::MeloTts {
        return Ok(settings);
    }

    let discovered =
        resolve_melotts_workers_root(melotts_discovery).map_err(format_worker_runtime_error)?;
    apply_melotts_runtime_fields(config, &mut settings, discovered.path)?;
    Ok(settings)
}

/// Patch Melo fields given a validated workers root (`melotts/` + proto).
#[cfg(feature = "tts")]
fn apply_melotts_runtime_fields(
    config: &Config,
    settings: &mut shuvoice_tts::TtsBackendSettings,
    worker_root: std::path::PathBuf,
) -> Result<(), String> {
    if !cfg!(feature = "tts-worker") {
        return Err(
            "MeloTTS requires the CLI feature `tts-worker` (worker-proto only). \
             Rebuild with: cargo build -p shuvoice-cli --features tts-worker \
             (or --features desktop)"
                .into(),
        );
    }

    #[cfg(feature = "tts-worker")]
    {
        apply_melotts_runtime_fields_inner(config, settings, worker_root)
    }
    #[cfg(not(feature = "tts-worker"))]
    {
        let _ = (config, settings, worker_root);
        unreachable!("gated above");
    }
}

#[cfg(feature = "tts-worker")]
fn apply_melotts_runtime_fields_inner(
    config: &Config,
    settings: &mut shuvoice_tts::TtsBackendSettings,
    worker_root: std::path::PathBuf,
) -> Result<(), String> {
    use shuvoice_tts::MeloWorkerSpawn;

    // 1) worker_root already validated (melotts + proto) by resolve_melotts_*.
    // 2) Isolated venv + interpreter (setup-managed default when unset).
    let venv_path = settings
        .melotts_venv_path
        .clone()
        .unwrap_or_else(|| crate::setup::melotts::melotts_venv_dir(config));
    let python = settings
        .melotts_python_binary
        .clone()
        .unwrap_or_else(|| venv_path.join("bin").join("python"));

    let device = normalize_melotts_device(&settings.melotts_device);

    // 3) Typed spawn: `<venv>/bin/python -m melotts --device <device>`
    //    with PYTHONPATH/PYTHONUNBUFFERED + device/venv overlays and current_dir.
    let spawn = MeloWorkerSpawn::new(&python)
        .args([
            "-m".to_string(),
            MELOTTS_WORKER_MODULE.to_string(),
            "--device".to_string(),
            device.clone(),
        ])
        .current_dir(&worker_root)
        .env_pair(PYTHONPATH_ENV, worker_root.to_string_lossy().into_owned())
        .env_pair(PYTHONUNBUFFERED_ENV, "1")
        .env_pair(MELOTTS_DEVICE_ENV, device)
        .env_pair(MELOTTS_VENV_ENV, venv_path.to_string_lossy().into_owned());

    settings.melotts_helper_script = None;
    settings.melotts_worker_root = Some(worker_root);
    settings.melotts_venv_path = Some(venv_path);
    settings.melotts_python_binary = Some(python);
    settings.melotts_worker_spawn = Some(spawn);
    // Prefer typed spawn; clear stringy argv override so it cannot win.
    settings.melotts_worker_command = None;
    Ok(())
}

/// Normalize Melo device tokens the same way the TTS backend does.
#[cfg(feature = "tts")]
#[must_use]
pub fn normalize_melotts_device(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "cpu" => "cpu".into(),
        "cuda" | "gpu" => "cuda".into(),
        "auto" | "" => "auto".into(),
        other => other.to_string(),
    }
}

/// Resolve the typed Melo worker spawn from already-composed settings.
///
/// Uses the same construction path as [`tts_adapter::create_shared_backend`]
/// so tests can assert `python -m melotts` without opening a real device.
#[cfg(feature = "tts-worker")]
pub fn resolve_composed_melotts_spawn(
    settings: &shuvoice_tts::TtsBackendSettings,
) -> Result<shuvoice_tts::MeloWorkerSpawn, String> {
    use std::time::Duration;

    use shuvoice_tts::{
        BackendId, DEFAULT_MELOTTS_VOICE_ID, MeloTtsBackend, MeloTtsConfig, MeloWireMode,
    };

    if settings.backend != BackendId::MeloTts {
        return Err("resolve_composed_melotts_spawn requires backend = melotts".into());
    }
    if !tts_adapter::tts_worker_feature_enabled() {
        return Err("MeloTTS requires CLI feature `tts-worker`".into());
    }

    let timeout = if settings.request_timeout_sec.is_finite() && settings.request_timeout_sec > 0.0
    {
        Duration::from_secs_f64(settings.request_timeout_sec)
    } else {
        Duration::from_secs(30)
    };

    let mut cfg = MeloTtsConfig {
        device: settings.melotts_device.clone(),
        max_chars: settings.max_chars,
        request_timeout: timeout,
        default_voice_id: if settings.default_voice_id.is_empty() {
            DEFAULT_MELOTTS_VOICE_ID.into()
        } else {
            settings.default_voice_id.clone()
        },
        helper_script: None,
        python_binary: settings.melotts_python_binary.clone(),
        wire_mode: MeloWireMode::WorkerProto,
        worker_root: settings.melotts_worker_root.clone(),
        worker_spawn: settings.melotts_worker_spawn.clone(),
        worker_command: settings.melotts_worker_command.clone(),
        worker_env: settings.melotts_worker_env.clone(),
        ..MeloTtsConfig::default()
    };
    cfg.wire_mode = MeloWireMode::WorkerProto;
    cfg.helper_script = None;
    if let Some(venv) = &settings.melotts_venv_path {
        cfg.venv_path = venv.clone();
    }

    MeloTtsBackend::new(cfg)
        .resolve_spawn()
        .map_err(|e| e.to_string())
}

#[cfg(feature = "tts")]
fn build_tts_bundle(config: &Config) -> Result<TtsBundle, String> {
    use shuvoice_tts::AudioOutputFactory;
    use std::sync::Arc;

    let factory: Arc<dyn AudioOutputFactory> =
        Arc::new(tts_adapter::cpal_output_factory_from_config(config).map_err(|e| e.to_string())?);

    let feedback = if config.audio_feedback {
        feedback::ToneFeedbackSink::from_config(config, Arc::clone(&factory))
    } else {
        feedback::ToneFeedbackSink::disabled()
    };

    if !config.tts_enabled {
        return Ok(TtsBundle {
            engine: None,
            bridge: None,
            bridge_rx: None,
            feedback,
            caps: None,
        });
    }

    // Base settings leave Melo root None; composition patches before backend build.
    let settings = compose_tts_backend_settings(config)?;

    let (bridge, bridge_rx) =
        tts_adapter::TtsPlayerUpdateBridge::new(tts_adapter::DEFAULT_TTS_PLAYER_BRIDGE_CAPACITY);
    let backend = tts_adapter::create_shared_backend(&settings).map_err(|e| e.to_string())?;
    let caps = tts_adapter::map_capabilities(&backend.capabilities());
    let engine = tts_adapter::PlayerTtsEngine::from_backend(
        backend,
        factory,
        config.tts_playback_speed,
        &bridge,
        tokio::runtime::Handle::current(),
    )
    .map_err(|e| e.to_string())?;

    Ok(TtsBundle {
        engine: Some(engine),
        bridge: Some(bridge),
        bridge_rx: Some(bridge_rx),
        feedback,
        caps: Some(caps),
    })
}

// ── Production entry ──────────────────────────────────────────────────────

/// Compose and run the long-lived desktop/session process.
///
/// Startup config/dependency/model/audio/control failures → exit **78**.
/// Runtime decode errors stay inside the session circuit breaker (process
/// remains alive). No live systemd/service mutation.
pub async fn run_production(config: Config) -> ExitStatus {
    match compose_and_run(config).await {
        Ok(()) => ExitStatus::code(EXIT_SUCCESS),
        Err(ComposeError::Dependency(msg)) => {
            eprintln!("ERROR: {msg}");
            ExitStatus::code(EXIT_DEPENDENCY).with_message(msg)
        }
        Err(ComposeError::Runtime(msg)) => {
            eprintln!("ERROR: {msg}");
            ExitStatus::code(EXIT_FAILURE).with_message(msg)
        }
    }
}

/// Redact filesystem paths from startup/dependency errors before journal/stderr.
///
/// Labels stay; absolute/home path bodies become `[path]` so logs remain actionable
/// without leaking user directory layout.
#[must_use]
pub fn redact_startup_error(msg: &str) -> String {
    let mut out = String::with_capacity(msg.len());
    let bytes = msg.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // `~/...` home paths
        if bytes[i] == b'~' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            i += 2;
            while i < bytes.len() {
                let c = bytes[i];
                if c.is_ascii_whitespace() || matches!(c, b';' | b',' | b')' | b']' | b'"' | b'\'')
                {
                    break;
                }
                i += 1;
            }
            out.push_str("[path]");
            continue;
        }
        // Absolute paths
        if bytes[i] == b'/' {
            let start = i;
            i += 1;
            while i < bytes.len() {
                let c = bytes[i];
                if c.is_ascii_whitespace() || matches!(c, b';' | b',' | b')' | b']' | b'"' | b'\'')
                {
                    break;
                }
                i += 1;
            }
            let slice = &msg[start..i];
            if slice.len() > 1 {
                out.push_str("[path]");
            } else {
                out.push_str(slice);
            }
            continue;
        }
        out.push(msg[i..].chars().next().unwrap());
        i += msg[i..].chars().next().unwrap().len_utf8();
    }
    out
}

/// Ordered bounded cleanup after `spawn_session_runtime` succeeded but the
/// process never entered the steady-state run loop.
///
/// Order matches production teardown (minus UI forwarders):
/// 1. control stop (if started)
/// 2. audio stop (if started)
/// 3. `runtime.shutdown().await`
/// 4. explicit shared feedback shutdown (if present)
///
/// Every post-session startup error path **must** call this before returning.
/// No `?` may cross the live-runtime boundary without it.
async fn abort_live_startup(
    control: Option<ControlServer>,
    #[cfg(feature = "audio")] audio: Option<audio_bridge::AudioBridge>,
    runtime: SessionRuntime,
    #[cfg(feature = "tts")] feedback: Option<&SharedFeedback>,
) {
    if let Some(mut server) = control {
        server.stop();
    }

    #[cfg(feature = "audio")]
    if let Some(mut bridge) = audio {
        let outcome = tokio::task::spawn_blocking(move || bridge.stop())
            .await
            .unwrap_or_else(|_| {
                warn!("audio bridge stop task join failed during abort_live_startup");
                audio_bridge::StopOutcome::DetachedAfterTimeout
            });
        if outcome.detached() {
            warn!("audio bridge DetachedAfterTimeout during startup abort");
        }
    }

    if let Err(err) = runtime.shutdown().await {
        warn!(%err, "session runtime shutdown during startup abort");
    }

    #[cfg(feature = "tts")]
    if let Some(fb) = feedback {
        let outcome = fb.shutdown();
        if outcome.detached() {
            warn!("feedback tone worker DetachedAfterTimeout during startup abort");
        }
    }
}

async fn compose_and_run(config: Config) -> Result<(), ComposeError> {
    validate_composition_config(&config).map_err(ComposeError::dep)?;
    if !cfg!(feature = "audio") {
        return Err(ComposeError::dep(
            "audio capture support not built into this binary (missing feature audio). \
             Rebuild with: cargo build -p shuvoice-cli --features desktop",
        ));
    }

    let backend = build_asr_backend(&config).map_err(ComposeError::dep)?;
    let injector = Arc::new(IoTextInjector::from_config(&config));
    let selection = Arc::new(IoSelection::with_defaults());
    let clock = Arc::new(SystemClock);

    // ── UI bus + single caption driver ────────────────────────────────────
    #[cfg(feature = "ui")]
    let (cmd_tx, cmd_rx, event_tx, event_rx, overlay, mapper, caption_vm, mut tts_vm) = {
        use shuvoice_ui::{CaptionStyle, CaptionVm, TtsCapabilities, TtsVm, UiBus};

        let bus = UiBus::new();
        let (cmd_tx, cmd_rx, event_tx, event_rx) = bus.split();
        let paired = ui_bridge::paired_overlay_sink_path(cmd_tx.clone());
        let caption_vm = CaptionVm::new(CaptionStyle::from_config(&config));
        let tts_vm = if config.tts_enabled {
            Some(TtsVm::from_config(
                &config,
                TtsCapabilities {
                    supports_streaming: false,
                    supports_voice_list: true,
                    requires_api_key: false,
                    supports_speed_control: true,
                    speed_min: Some(0.5),
                    speed_max: Some(2.0),
                },
            ))
        } else {
            None
        };
        (
            cmd_tx,
            cmd_rx,
            event_tx,
            event_rx,
            paired.sink,
            paired.mapper,
            caption_vm,
            tts_vm,
        )
    };
    #[cfg(not(feature = "ui"))]
    let overlay = NullOverlay;

    // ── TTS bridge-before-player + shared feedback factory ────────────────
    #[cfg(feature = "tts")]
    let tts_bundle = build_tts_bundle(&config).map_err(ComposeError::dep)?;
    #[cfg(all(feature = "ui", feature = "tts"))]
    if let (Some(vm), Some(caps)) = (tts_vm.as_mut(), tts_bundle.caps.as_ref()) {
        *vm = shuvoice_ui::TtsVm::from_config(&config, caps.clone());
    }

    #[cfg(feature = "tts")]
    let (tts_engine, tts_bridge, mut tts_bridge_rx, feedback_sink) = (
        tts_bundle.engine,
        tts_bundle.bridge,
        tts_bundle.bridge_rx,
        tts_bundle.feedback,
    );
    // Shared handle: session gets a clone; composition keeps owner for step-4 shutdown.
    #[cfg(feature = "tts")]
    let feedback_owner = SharedFeedback::new(feedback_sink);
    #[cfg(feature = "tts")]
    let feedback_for_session = feedback_owner.clone();
    #[cfg(not(feature = "tts"))]
    let (tts_engine, feedback_for_session) = (None::<NullTts>, NullFeedback);

    // ── Session runtime ───────────────────────────────────────────────────
    // After this succeeds, every error path MUST call `abort_live_startup`
    // before returning (no bare `?` across the live-runtime boundary).
    let mut runtime = match spawn_session_runtime(
        config.clone(),
        backend,
        injector,
        selection,
        overlay,
        feedback_for_session,
        clock,
        tts_engine,
    )
    .await
    {
        Ok(rt) => rt,
        Err(e) => {
            return Err(ComposeError::dep(redact_startup_error(&format!(
                "session startup failed: {e}"
            ))));
        }
    };

    info!(
        effective_sample_rate = runtime.effective_sample_rate,
        audio_chunk_samples = runtime.audio_chunk_samples,
        asr = %config.asr_backend.as_str(),
        "session runtime ready"
    );

    // ── Audio bridge at effective ASR rate (start via spawn_blocking) ─────
    #[cfg(feature = "audio")]
    let mut audio_bridge = {
        use self::audio_bridge::{AudioBridge, AudioBridgeConfig};

        let audio_cfg = AudioBridgeConfig::from_app_config(
            &config,
            runtime.effective_sample_rate,
            runtime.audio_chunk_samples,
        );
        let ingress = runtime.audio.clone();
        let started =
            tokio::task::spawn_blocking(move || AudioBridge::start(ingress, audio_cfg)).await;
        match started {
            Ok(Ok(bridge)) => bridge,
            Ok(Err(e)) => {
                #[cfg(feature = "tts")]
                {
                    abort_live_startup(None, None, runtime, Some(&feedback_owner)).await;
                }
                #[cfg(not(feature = "tts"))]
                {
                    abort_live_startup(None, None, runtime).await;
                }
                return Err(ComposeError::dep(redact_startup_error(&format!(
                    "audio capture start failed: {e}"
                ))));
            }
            Err(e) => {
                #[cfg(feature = "tts")]
                {
                    abort_live_startup(None, None, runtime, Some(&feedback_owner)).await;
                }
                #[cfg(not(feature = "tts"))]
                {
                    abort_live_startup(None, None, runtime).await;
                }
                return Err(ComposeError::dep(redact_startup_error(&format!(
                    "audio bridge join error: {e}"
                ))));
            }
        }
    };

    // ── Control server ────────────────────────────────────────────────────
    let control_adapter = runtime.control.clone();
    let handlers = ControlBridge::arc(control_adapter.clone());
    let mut control_server = match ControlServer::new(config.control_socket.as_deref(), handlers) {
        Ok(s) => s,
        Err(e) => {
            #[cfg(all(feature = "audio", feature = "tts"))]
            {
                abort_live_startup(None, Some(audio_bridge), runtime, Some(&feedback_owner)).await;
            }
            #[cfg(all(feature = "audio", not(feature = "tts")))]
            {
                abort_live_startup(None, Some(audio_bridge), runtime).await;
            }
            #[cfg(all(not(feature = "audio"), feature = "tts"))]
            {
                abort_live_startup(None, runtime, Some(&feedback_owner)).await;
            }
            #[cfg(all(not(feature = "audio"), not(feature = "tts")))]
            {
                abort_live_startup(None, runtime).await;
            }
            return Err(ComposeError::dep(redact_startup_error(&format!(
                "control socket prepare failed: {e}"
            ))));
        }
    };
    if let Err(e) = control_server.start() {
        #[cfg(all(feature = "audio", feature = "tts"))]
        {
            abort_live_startup(
                Some(control_server),
                Some(audio_bridge),
                runtime,
                Some(&feedback_owner),
            )
            .await;
        }
        #[cfg(all(feature = "audio", not(feature = "tts")))]
        {
            abort_live_startup(Some(control_server), Some(audio_bridge), runtime).await;
        }
        #[cfg(all(not(feature = "audio"), feature = "tts"))]
        {
            abort_live_startup(Some(control_server), runtime, Some(&feedback_owner)).await;
        }
        #[cfg(all(not(feature = "audio"), not(feature = "tts")))]
        {
            abort_live_startup(Some(control_server), runtime).await;
        }
        return Err(ComposeError::dep(redact_startup_error(&format!(
            "control socket start failed: {e}"
        ))));
    }
    info!(path = %control_server.socket_path().display(), "control server listening");

    // ── Exactly-once shutdown coordination ────────────────────────────────
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let shutdown_notify = Arc::new(Notify::new());
    #[cfg(feature = "ui")]
    let (quit_tx, quit_rx) = ui_bridge::gtk_quit_channel();
    #[cfg(feature = "ui")]
    let (life_tx, life_rx) = ui_bridge::gtk_lifecycle_channel();

    let request_shutdown = {
        let flag = Arc::clone(&shutdown_flag);
        let notify = Arc::clone(&shutdown_notify);
        #[cfg(feature = "ui")]
        let quit_tx = quit_tx.clone();
        Arc::new(move || {
            if flag.swap(true, Ordering::SeqCst) {
                return;
            }
            #[cfg(feature = "ui")]
            {
                let _ = quit_tx.try_request_quit();
            }
            notify.notify_waiters();
        }) as Arc<dyn Fn() + Send + Sync>
    };

    let mut forwarders: Vec<JoinHandle<()>> = Vec::new();
    let (essential_rx, partial_rx) = take_event_receivers(&mut runtime);

    // Session events → UI (essential-first) or silent drain.
    #[cfg(feature = "ui")]
    {
        let (signal_tx, signal_rx) = oneshot::channel::<ui_bridge::UiBridgeSignal>();
        let cmd_tx_m = cmd_tx.clone();
        forwarders.push(tokio::spawn(async move {
            ui_bridge::run_essential_first_ui_merge(
                essential_rx,
                partial_rx,
                cmd_tx_m,
                mapper,
                Some(signal_tx),
            )
            .await;
        }));
        let req = Arc::clone(&request_shutdown);
        forwarders.push(tokio::spawn(async move {
            if signal_rx.await.is_ok() {
                req();
            }
        }));
    }
    #[cfg(not(feature = "ui"))]
    {
        let req = Arc::clone(&request_shutdown);
        forwarders.push(tokio::spawn(async move {
            drain_session_events(essential_rx, partial_rx, req).await;
        }));
    }

    // UI → session control pump (retry/coalesce; never log payloads).
    #[cfg(feature = "ui")]
    {
        let bridge = ui_bridge::UiToSessionCommandBridge::new();
        let handle = runtime.handle.clone();
        let notify = Arc::clone(&shutdown_notify);
        let flag = Arc::clone(&shutdown_flag);
        forwarders.push(tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_millis(16));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    _ = tick.tick() => {
                        if flag.load(Ordering::Relaxed) {
                            break;
                        }
                        let h = &handle;
                        ui_bridge::pump_and_drain_ui_events_to_session(
                            &event_rx,
                            &bridge,
                            |cmd| h.try_enqueue(cmd).is_ok(),
                        );
                    }
                    _ = notify.notified() => break,
                }
            }
        }));
    }

    // TTS player events → session (FIFO then sticky coalesce; never drop terminal).
    // Leaf contract: only `drain_with_try`; retry while pending_remaining/has_pending.
    #[cfg(feature = "tts")]
    if let (Some(mut rx), Some(bridge)) = (tts_bridge_rx.take(), tts_bridge) {
        let control = control_adapter.clone();
        let notify = Arc::clone(&shutdown_notify);
        forwarders.push(tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_millis(5));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    biased;
                    _ = notify.notified() => {
                        pump_tts_bridge_until_idle(&bridge, &mut rx, &control, 64).await;
                        if bridge.has_pending() {
                            warn!(
                                "TTS bridge still has pending updates after final drain                                  (terminal state may not have reached session)"
                            );
                        }
                        break;
                    }
                    _ = tick.tick() => {
                        pump_tts_bridge_until_idle(&bridge, &mut rx, &control, 16).await;
                    }
                }
            }
        }));
    }

    // OS signals → single shutdown.
    {
        let req = Arc::clone(&request_shutdown);
        forwarders.push(tokio::spawn(async move {
            if wait_for_os_shutdown_signal().await {
                req();
            }
        }));
    }

    // GTK lifecycle Exiting → single shutdown.
    #[cfg(feature = "ui")]
    {
        let req = Arc::clone(&request_shutdown);
        forwarders.push(tokio::spawn(async move {
            loop {
                match life_rx.try_recv() {
                    Ok(ui_bridge::GtkHostLifecycle::Exiting) => {
                        req();
                        break;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                }
            }
        }));
    }

    // ── Main thread: GTK host (or headless wait) ──────────────────────────
    #[cfg(feature = "ui")]
    {
        let boot = ui_bridge::GtkMainHostBootstrap {
            caption_vm,
            tts_vm,
            cmd_rx,
            event_tx,
            quit_rx,
            lifecycle_tx: Some(life_tx),
        };
        // Blocks this future on the main thread; worker threads keep pumps alive.
        let _gtk_code = ui_bridge::run_gtk_main_host(boot);
        request_shutdown();
    }
    #[cfg(not(feature = "ui"))]
    {
        shutdown_notify.notified().await;
    }

    // ── Ordered bounded teardown ──────────────────────────────────────────
    // 1) Stop accepting control.
    control_server.stop();

    // 2) Stop audio ingress.
    #[cfg(feature = "audio")]
    {
        let outcome = tokio::task::spawn_blocking(move || audio_bridge.stop())
            .await
            .unwrap_or_else(|_| {
                warn!("audio bridge stop task join failed");
                audio_bridge::StopOutcome::DetachedAfterTimeout
            });
        if outcome.detached() {
            warn!("audio bridge DetachedAfterTimeout (not cleanly released)");
        }
    }

    // 3) Session shutdown (real bounded runtime shutdown).
    //    Actor stops calling feedback before we tear the tone worker down.
    if let Err(err) = runtime.shutdown().await {
        warn!(%err, "session runtime shutdown returned error");
    }

    // 4) Explicit feedback tone worker shutdown (not only Drop).
    #[cfg(feature = "tts")]
    {
        let outcome = feedback_owner.shutdown();
        if outcome.detached() {
            warn!("feedback tone worker DetachedAfterTimeout (not cleanly released)");
        }
    }

    // 5) Drain/abort forwarders — never detach JoinHandles silently.
    //    Shutdown notify already asked the TTS sticky drain to finish; abort is backup.
    for join in forwarders.drain(..) {
        join.abort();
        match tokio::time::timeout(Duration::from_millis(500), join).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) if e.is_cancelled() => {}
            Ok(Err(e)) => warn!(%e, "forwarder join error"),
            Err(_) => warn!("forwarder abort await timed out"),
        }
    }

    // 6) GTK quit already requested via request_shutdown; ensure once more.
    #[cfg(feature = "ui")]
    {
        let _ = quit_tx.try_request_quit();
    }

    let _ = shutdown_flag;
    Ok(())
}

/// Drain TTS player→session bridge with sticky ack semantics.
///
/// Retries while `pending_remaining` / [`tts_adapter::TtsPlayerUpdateBridge::has_pending`]
/// so terminal Idle/Error is never lost to a full command queue.
#[cfg(feature = "tts")]
async fn pump_tts_bridge_until_idle(
    bridge: &tts_adapter::TtsPlayerUpdateBridge,
    rx: &mut tokio::sync::mpsc::Receiver<SessionCommand>,
    control: &EnqueueControlAdapter,
    max_rounds: usize,
) {
    for _ in 0..max_rounds.max(1) {
        let stats = bridge.drain_with_try(rx, |cmd| control.try_enqueue(cmd).is_ok());
        if !stats.pending_remaining && !bridge.has_pending() {
            return;
        }
        tokio::task::yield_now().await;
    }
}

fn take_event_receivers(
    runtime: &mut SessionRuntime,
) -> (
    tokio::sync::mpsc::Receiver<SessionEvent>,
    tokio::sync::mpsc::Receiver<SessionEvent>,
) {
    let (_, dummy_e_rx) = tokio::sync::mpsc::channel(1);
    let (_, dummy_p_rx) = tokio::sync::mpsc::channel(1);
    let essential = std::mem::replace(&mut runtime.essential_rx, dummy_e_rx);
    let partial = std::mem::replace(&mut runtime.partial_rx, dummy_p_rx);
    (essential, partial)
}

async fn drain_session_events(
    mut essential_rx: tokio::sync::mpsc::Receiver<SessionEvent>,
    mut partial_rx: tokio::sync::mpsc::Receiver<SessionEvent>,
    request_shutdown: Arc<dyn Fn() + Send + Sync>,
) {
    let mut essentials_open = true;
    let mut partials_open = true;
    while essentials_open || partials_open {
        tokio::select! {
            biased;
            ev = essential_rx.recv(), if essentials_open => {
                match ev {
                    Some(SessionEvent::ShutdownComplete) => {
                        request_shutdown();
                        return;
                    }
                    // Never log SessionEvent — may hold transcripts.
                    Some(_) => {}
                    None => essentials_open = false,
                }
            }
            ev = partial_rx.recv(), if partials_open => {
                match ev {
                    Some(_) => {}
                    None => partials_open = false,
                }
            }
        }
    }
}

async fn wait_for_os_shutdown_signal() -> bool {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
                return true;
            }
        };
        tokio::select! {
            r = tokio::signal::ctrl_c() => r.is_ok(),
            r = sigterm.recv() => r.is_some(),
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await.is_ok()
    }
}

// ── Headless injectable harness (tests / no mic/display/network) ──────────

/// Headless composition: fake ASR + fake sinks, real [`ControlServer`] on a
/// temp socket. Exercises start/status/stop/shutdown and the effective sample
/// rate contract without requiring a microphone, display, or network.
pub struct HeadlessHarness {
    pub runtime: SessionRuntime,
    pub control_server: ControlServer,
    pub socket_path: PathBuf,
    pub control: EnqueueControlAdapter,
    event_drain: JoinHandle<()>,
}

impl HeadlessHarness {
    /// Spawn a headless session with an injectable ASR backend.
    pub async fn spawn(
        config: Config,
        backend: Box<DynAsrBackend>,
        socket_path: PathBuf,
    ) -> Result<Self, String> {
        let injector = Arc::new(IoTextInjector::with_defaults());
        let selection = Arc::new(IoSelection::with_defaults());
        let clock = Arc::new(SystemClock);

        let mut runtime = spawn_session_runtime(
            config,
            backend,
            injector,
            selection,
            NullOverlay,
            NullFeedback,
            clock,
            None::<NullTts>,
        )
        .await
        .map_err(|e| redact_startup_error(&format!("harness session spawn: {e}")))?;

        let control = runtime.control.clone();
        let handlers = ControlBridge::arc(control.clone());
        let path_str = socket_path.to_string_lossy().to_string();
        let mut control_server = match ControlServer::new(Some(&path_str), handlers) {
            Ok(s) => s,
            Err(e) => {
                let _ = runtime.shutdown().await;
                return Err(redact_startup_error(&format!(
                    "harness control prepare: {e}"
                )));
            }
        };
        if let Err(e) = control_server.start() {
            control_server.stop();
            let _ = runtime.shutdown().await;
            return Err(redact_startup_error(&format!("harness control start: {e}")));
        }

        let (essential_rx, partial_rx) = take_event_receivers(&mut runtime);
        let flag = Arc::new(AtomicBool::new(false));
        let flag2 = Arc::clone(&flag);
        let request = Arc::new(move || {
            flag2.store(true, Ordering::SeqCst);
        }) as Arc<dyn Fn() + Send + Sync>;
        let event_drain = tokio::spawn(async move {
            drain_session_events(essential_rx, partial_rx, request).await;
        });

        Ok(Self {
            runtime,
            control_server,
            socket_path,
            control,
            event_drain,
        })
    }

    #[must_use]
    pub fn effective_sample_rate(&self) -> u32 {
        self.runtime.effective_sample_rate
    }

    #[must_use]
    pub fn audio_chunk_samples(&self) -> usize {
        self.runtime.audio_chunk_samples
    }

    pub fn send_control(&self, cmd: ControlCommand) -> Result<String, String> {
        shuvoice_control::send_control_command_to(&self.socket_path, cmd, Duration::from_secs(2))
            .map_err(|e| e.to_string())
    }

    pub async fn shutdown(mut self) -> Result<(), String> {
        self.control_server.stop();
        self.runtime
            .shutdown()
            .await
            .map_err(|e| format!("harness session shutdown: {e}"))?;
        self.event_drain.abort();
        let _ = tokio::time::timeout(Duration::from_millis(500), self.event_drain).await;
        Ok(())
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "audio")]
    use shuvoice_app::fakes::offline_asr;
    use shuvoice_app::fakes::remote_asr;
    use shuvoice_core::AsrBackendKind;

    #[test]
    fn redact_startup_error_strips_absolute_and_home_paths() {
        let raw = "audio capture start failed: device /home/user/.config/foo open at /tmp/x.sock";
        let red = redact_startup_error(raw);
        assert!(red.contains("[path]"), "{red}");
        assert!(!red.contains("/home/user"), "{red}");
        assert!(!red.contains("/tmp/x.sock"), "{red}");
        assert!(red.contains("audio capture start failed"), "{red}");
        let home = redact_startup_error("missing model ~/Models/sherpa/x");
        assert!(home.contains("[path]"), "{home}");
        assert!(!home.contains("~/Models"), "{home}");
    }

    #[test]
    fn audio_feedback_without_tts_feature_is_exit78_gate() {
        let mut cfg = Config::default();
        // Avoid unrelated ASR feature gates so this test isolates audio_feedback.
        cfg.asr_backend = AsrBackendKind::Nemo;
        cfg.audio_feedback = true;
        cfg.tts_enabled = false;
        let _ = cfg.validate();
        const HAS_TTS: bool = cfg!(feature = "tts");
        if !HAS_TTS {
            let err = validate_audio_feedback_feature(&cfg).expect_err("must fail closed");
            assert!(
                err.contains("audio_feedback") && err.contains("tts"),
                "{err}"
            );
            // Full composition validation must also surface the gate.
            let err2 = validate_composition_config(&cfg).expect_err("composition must fail");
            assert!(
                err2.contains("audio_feedback") && err2.contains("tts"),
                "{err2}"
            );
        } else if let Err(err) = validate_audio_feedback_feature(&cfg) {
            panic!("unexpected audio_feedback gate error with tts feature: {err}");
        }
    }

    /// ASR owner load succeeds, then audio open fails → runtime is shut down.
    #[cfg(feature = "audio")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn startup_audio_open_failure_aborts_live_runtime() {
        use std::time::Duration;

        use shuvoice_io::audio::AudioConfig;

        use crate::compose::audio_bridge::{
            AudioBridge, AudioBridgeConfig, CaptureBackend, CaptureOpener,
        };

        struct FailOpen;
        impl CaptureOpener for FailOpen {
            fn open(self, _cfg: AudioConfig) -> Result<Box<dyn CaptureBackend>, String> {
                Err("injectable audio open failure at /tmp/should-redact.pcm".into())
            }
        }

        #[cfg(feature = "tts")]
        let feedback_owner = SharedFeedback::new(feedback::ToneFeedbackSink::disabled());
        #[cfg(feature = "tts")]
        let feedback_for_session = feedback_owner.clone();
        #[cfg(not(feature = "tts"))]
        let feedback_for_session = NullFeedback;

        let mut cfg = Config::default();
        cfg.tts_enabled = false;
        cfg.audio_feedback = false;
        let _ = cfg.validate();

        let runtime = spawn_session_runtime(
            cfg.clone(),
            Box::new(offline_asr("ok")),
            Arc::new(IoTextInjector::with_defaults()),
            Arc::new(IoSelection::with_defaults()),
            NullOverlay,
            feedback_for_session,
            Arc::new(SystemClock),
            None::<NullTts>,
        )
        .await
        .expect("session must start so ASR owner is live");

        let audio_cfg = AudioBridgeConfig::from_app_config(
            &cfg,
            runtime.effective_sample_rate,
            runtime.audio_chunk_samples,
        );
        let ingress = runtime.audio.clone();
        let err = match AudioBridge::start_with(ingress, audio_cfg, FailOpen) {
            Ok(_) => panic!("audio must fail open"),
            Err(e) => e,
        };

        #[cfg(feature = "tts")]
        {
            abort_live_startup(None, None, runtime, Some(&feedback_owner)).await;
        }
        #[cfg(not(feature = "tts"))]
        {
            abort_live_startup(None, None, runtime).await;
        }

        let red = redact_startup_error(&err);
        assert!(red.contains("injectable audio open failure"), "{red}");
        assert!(!red.contains("/tmp/should-redact.pcm"), "{red}");

        #[cfg(feature = "tts")]
        {
            let again = feedback_owner.shutdown();
            assert!(
                again.joined_cleanly()
                    || matches!(again, feedback::ShutdownOutcome::AlreadyStopped)
            );
        }

        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    /// Control prepare/start failure after audio is up → stack cleaned; no our socket left.
    #[cfg(feature = "audio")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn startup_control_failure_after_audio_aborts_stack() {
        use std::time::Duration;

        use shuvoice_io::audio::AudioConfig;

        use crate::compose::audio_bridge::{
            AudioBridge, AudioBridgeConfig, CaptureBackend, CaptureOpener,
        };

        struct IdleOpen;
        struct IdleBackend;
        impl CaptureBackend for IdleBackend {
            fn get_chunk(&mut self, timeout: Duration) -> Option<Vec<f32>> {
                std::thread::sleep(timeout.min(Duration::from_millis(5)));
                None
            }
            fn stop(&mut self) {}
            fn backend_hold_queue_dropped(&self) -> u64 {
                0
            }
            fn backend_callback_lock_fails(&self) -> u64 {
                0
            }
            fn resolved_device_name(&self) -> Option<&str> {
                Some("idle-test")
            }
        }
        impl CaptureOpener for IdleOpen {
            fn open(self, _cfg: AudioConfig) -> Result<Box<dyn CaptureBackend>, String> {
                Ok(Box::new(IdleBackend))
            }
        }

        let tmp = tempfile::tempdir().unwrap();
        let runtime_dir = tmp.path().join("run");
        std::fs::create_dir_all(&runtime_dir).unwrap();
        let old_rt = std::env::var_os("XDG_RUNTIME_DIR");
        // SAFETY: test-only env mutation; restored before return.
        unsafe {
            std::env::set_var("XDG_RUNTIME_DIR", &runtime_dir);
        }

        #[cfg(feature = "tts")]
        let feedback_owner = SharedFeedback::new(feedback::ToneFeedbackSink::disabled());
        #[cfg(feature = "tts")]
        let feedback_for_session = feedback_owner.clone();
        #[cfg(not(feature = "tts"))]
        let feedback_for_session = NullFeedback;

        let mut cfg = Config::default();
        cfg.tts_enabled = false;
        cfg.audio_feedback = false;
        let _ = cfg.validate();

        let runtime = match spawn_session_runtime(
            cfg.clone(),
            Box::new(offline_asr("ok")),
            Arc::new(IoTextInjector::with_defaults()),
            Arc::new(IoSelection::with_defaults()),
            NullOverlay,
            feedback_for_session,
            Arc::new(SystemClock),
            None::<NullTts>,
        )
        .await
        {
            Ok(rt) => rt,
            Err(e) => {
                // SAFETY: test-only process env mutation; restored on all exit paths before return.
                unsafe {
                    match old_rt {
                        Some(v) => std::env::set_var("XDG_RUNTIME_DIR", v),
                        None => std::env::remove_var("XDG_RUNTIME_DIR"),
                    }
                }
                panic!("session spawn: {e}");
            }
        };

        let audio_cfg = AudioBridgeConfig::from_app_config(
            &cfg,
            runtime.effective_sample_rate,
            runtime.audio_chunk_samples,
        );
        let ingress = runtime.audio.clone();
        let audio_bridge = match AudioBridge::start_with(ingress, audio_cfg, IdleOpen) {
            Ok(b) => b,
            Err(e) => {
                #[cfg(feature = "tts")]
                {
                    abort_live_startup(None, None, runtime, Some(&feedback_owner)).await;
                }
                #[cfg(not(feature = "tts"))]
                {
                    abort_live_startup(None, None, runtime).await;
                }
                // SAFETY: test-only process env mutation; restored on all exit paths before return.
                unsafe {
                    match old_rt {
                        Some(v) => std::env::set_var("XDG_RUNTIME_DIR", v),
                        None => std::env::remove_var("XDG_RUNTIME_DIR"),
                    }
                }
                panic!("audio start: {e}");
            }
        };

        // Illegal control path → prepare fails (absolute but not allowed / not .sock).
        let handlers = ControlBridge::arc(runtime.control.clone());
        let bad = ControlServer::new(Some("/etc/passwd"), handlers);
        assert!(bad.is_err(), "control prepare must fail for illegal path");

        #[cfg(feature = "tts")]
        {
            abort_live_startup(None, Some(audio_bridge), runtime, Some(&feedback_owner)).await;
        }
        #[cfg(not(feature = "tts"))]
        {
            abort_live_startup(None, Some(audio_bridge), runtime).await;
        }

        let sock_dir = runtime_dir.join("shuvoice");
        if sock_dir.exists() {
            let leftovers: Vec<_> = std::fs::read_dir(&sock_dir)
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("sock"))
                        .map(|e| e.path())
                        .collect()
                })
                .unwrap_or_default();
            assert!(
                leftovers.is_empty(),
                "lingering sockets after abort: {leftovers:?}"
            );
        }

        // SAFETY: test-only process env mutation; restored on all exit paths before return.
        unsafe {
            match old_rt {
                Some(v) => std::env::set_var("XDG_RUNTIME_DIR", v),
                None => std::env::remove_var("XDG_RUNTIME_DIR"),
            }
        }
    }

    #[test]
    fn feature_matrix_default_desktop_surface() {
        // Compile-time feature matrix checked via const block (not runtime assert!).
        const DESKTOP: bool = cfg!(feature = "desktop");
        const AUDIO: bool = cfg!(feature = "audio");
        const SHERPA: bool = cfg!(feature = "asr-sherpa");
        const OPENAI: bool = cfg!(feature = "asr-openai");
        const UI: bool = cfg!(feature = "ui");
        const TTS: bool = cfg!(feature = "tts");
        const TTS_WORKER: bool = cfg!(feature = "tts-worker");
        const DESKTOP_OK: bool = !DESKTOP || (AUDIO && SHERPA && OPENAI && UI && TTS && TTS_WORKER);
        const {
            assert!(DESKTOP_OK);
        }
    }

    #[test]
    fn no_default_run_is_fail_closed_for_sherpa() {
        let cfg = Config::default();
        assert_eq!(cfg.asr_backend, AsrBackendKind::Sherpa);
        const HAS_SHERPA: bool = cfg!(feature = "asr-sherpa");
        if !HAS_SHERPA {
            assert!(validate_composition_config(&cfg).is_err());
        }
    }

    #[test]
    fn layer_shell_skipped_without_ui_feature() {
        const HAS_UI: bool = cfg!(feature = "ui");
        if !HAS_UI {
            assert!(validate_layer_shell_if_needed().is_ok());
        }
    }

    #[test]
    fn openai_backend_not_constructed_without_feature() {
        let mut cfg = Config::default();
        cfg.asr_backend = AsrBackendKind::OpenaiRealtime;
        let _ = cfg.validate();
        const HAS_OPENAI: bool = cfg!(feature = "asr-openai");
        if !HAS_OPENAI {
            assert!(build_asr_backend(&cfg).is_err());
        }
    }

    #[cfg(all(feature = "tts", feature = "tts-worker"))]
    #[test]
    fn composed_melotts_reaches_typed_m_melotts_spawn_not_missing_root() {
        use std::fs::{self, File};
        use std::io::Write;
        use std::path::Path;

        use shuvoice_core::{MeloTtsDevice, TtsBackendKind};
        use shuvoice_tts::BackendId;

        #[cfg(unix)]
        use std::os::unix::fs::OpenOptionsExt;

        fn write_pkg(root: &Path, name: &str, with_main: bool) {
            let pkg = root.join(name);
            fs::create_dir_all(&pkg).unwrap();
            File::create(pkg.join("__init__.py")).unwrap();
            if with_main {
                File::create(pkg.join("__main__.py")).unwrap();
            }
        }

        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data");
        let workers = tmp.path().join("workers");
        fs::create_dir_all(&workers).unwrap();
        write_pkg(&workers, worker_runtime::WORKER_PROTO_PACKAGE, false);
        write_pkg(&workers, MELOTTS_WORKER_MODULE, true);

        // Isolated venv interpreter (fake executable; not executed).
        let venv = data.join("melotts-venv");
        let bin = venv.join("bin");
        fs::create_dir_all(&bin).unwrap();
        let python = bin.join("python");
        #[cfg(unix)]
        {
            let mut f = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o755)
                .open(&python)
                .unwrap();
            writeln!(f, "#!/bin/sh\nexit 0").unwrap();
        }
        #[cfg(not(unix))]
        {
            File::create(&python).unwrap();
        }

        let mut inputs = WorkersDiscoveryInputs::for_tests(&data);
        inputs.extra_candidates = vec![(
            workers.clone(),
            worker_runtime::WorkersRootSource::InstalledLib,
        )];

        let mut cfg = Config::default();
        cfg.tts_enabled = true;
        cfg.tts_backend = TtsBackendKind::Melotts;
        cfg.tts_melotts_device = MeloTtsDevice::Cpu;
        cfg.tts_melotts_venv_path = Some(venv.to_string_lossy().into_owned());
        let _ = cfg.validate();

        // Missing root (no candidates / empty discovery) must fail closed.
        let empty = WorkersDiscoveryInputs::for_tests(tmp.path().join("empty-data"));
        let missing = compose_tts_backend_settings_with_discovery(&cfg, &empty).unwrap_err();
        assert!(
            missing.to_lowercase().contains("worker") || missing.to_lowercase().contains("workers"),
            "expected workers-root failure, got: {missing}"
        );

        // Valid discovery → patched settings + typed -m melotts spawn.
        let settings = compose_tts_backend_settings_with_discovery(&cfg, &inputs)
            .expect("compose melotts settings");
        assert_eq!(settings.backend, BackendId::MeloTts);
        assert!(settings.melotts_helper_script.is_none());
        assert!(settings.melotts_worker_command.is_none());
        assert_eq!(
            settings.melotts_worker_root.as_deref(),
            Some(workers.as_path())
        );
        assert_eq!(settings.melotts_venv_path.as_deref(), Some(venv.as_path()));
        assert_eq!(
            settings.melotts_python_binary.as_deref(),
            Some(python.as_path())
        );

        let typed = settings
            .melotts_worker_spawn
            .as_ref()
            .expect("composition must install typed MeloWorkerSpawn");
        assert_eq!(typed.program, python);
        assert_eq!(
            typed.args,
            vec![
                "-m".to_string(),
                "melotts".to_string(),
                "--device".to_string(),
                "cpu".to_string()
            ]
        );
        assert_eq!(typed.current_dir.as_deref(), Some(workers.as_path()));
        let env: std::collections::HashMap<_, _> = typed.env.iter().cloned().collect();
        assert_eq!(
            env.get(PYTHONPATH_ENV).map(String::as_str),
            Some(workers.to_string_lossy().as_ref())
        );
        assert_eq!(env.get(PYTHONUNBUFFERED_ENV).map(String::as_str), Some("1"));
        assert_eq!(env.get(MELOTTS_DEVICE_ENV).map(String::as_str), Some("cpu"));
        assert_eq!(
            env.get(MELOTTS_VENV_ENV).map(String::as_str),
            Some(venv.to_string_lossy().as_ref())
        );

        // Backend resolve_spawn must honor the composed typed spawn (not missing-root).
        let spawn = resolve_composed_melotts_spawn(&settings).expect("resolve spawn");
        assert_eq!(spawn.program, python);
        assert_eq!(
            spawn.args,
            vec![
                "-m".to_string(),
                "melotts".to_string(),
                "--device".to_string(),
                "cpu".to_string()
            ]
        );
        assert_eq!(spawn.current_dir.as_deref(), Some(workers.as_path()));
        let spawn_env: std::collections::HashMap<_, _> = spawn.env.iter().cloned().collect();
        assert_eq!(
            spawn_env.get(PYTHONPATH_ENV).map(String::as_str),
            Some(workers.to_string_lossy().as_ref())
        );
        assert_eq!(
            spawn_env.get(PYTHONUNBUFFERED_ENV).map(String::as_str),
            Some("1")
        );
        assert_eq!(
            spawn_env.get(MELOTTS_DEVICE_ENV).map(String::as_str),
            Some("cpu")
        );
        assert_eq!(
            spawn_env.get(MELOTTS_VENV_ENV).map(String::as_str),
            Some(venv.to_string_lossy().as_ref())
        );

        // create_shared_backend must accept the composed settings (no missing-root).
        let backend = tts_adapter::create_shared_backend(&settings).expect("shared backend");
        assert_eq!(backend.id(), BackendId::MeloTts);
        let deps = backend.dependency_errors();
        assert!(
            deps.iter()
                .all(|e| !e.to_lowercase().contains("worker_root")),
            "composed root must not surface missing worker_root: {deps:?}"
        );
        // Same readiness gate as PlayerTtsEngine::from_backend.
        tts_adapter::ensure_backend_ready(&backend).expect("composed melo must be ready");
    }

    /// Proto-only workers tree must not satisfy Melo discovery.
    #[cfg(all(feature = "tts", feature = "tts-worker"))]
    #[test]
    fn composed_melotts_rejects_proto_only_workers_tree() {
        use std::fs::{self, File};
        use std::path::Path;

        use shuvoice_core::TtsBackendKind;

        fn write_pkg(root: &Path, name: &str, with_main: bool) {
            let pkg = root.join(name);
            fs::create_dir_all(&pkg).unwrap();
            File::create(pkg.join("__init__.py")).unwrap();
            if with_main {
                File::create(pkg.join("__main__.py")).unwrap();
            }
        }

        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data");
        let proto_only = tmp.path().join("proto-only");
        fs::create_dir_all(&proto_only).unwrap();
        write_pkg(&proto_only, worker_runtime::WORKER_PROTO_PACKAGE, false);
        // deliberately NO melotts/

        let mut inputs = WorkersDiscoveryInputs::for_tests(&data);
        inputs.extra_candidates =
            vec![(proto_only, worker_runtime::WorkersRootSource::InstalledLib)];

        let mut cfg = Config::default();
        cfg.tts_enabled = true;
        cfg.tts_backend = TtsBackendKind::Melotts;
        let _ = cfg.validate();

        let err = compose_tts_backend_settings_with_discovery(&cfg, &inputs).unwrap_err();
        let low = err.to_lowercase();
        assert!(
            low.contains("melotts")
                || low.contains("worker")
                || low.contains("module")
                || low.contains("package"),
            "proto-only tree must be rejected, got: {err}"
        );
    }

    /// drain_with_try keeps terminal coalesce sticky until enqueue acks (central contract).
    #[cfg(feature = "tts")]
    #[tokio::test(flavor = "current_thread")]
    async fn tts_bridge_drain_with_try_retries_terminal_until_ack() {
        use shuvoice_app::{SessionCommand, TtsPlayerState};
        use shuvoice_tts::types::{PlayerEvent, PlayerState};

        let (bridge, mut rx) = tts_adapter::TtsPlayerUpdateBridge::new(1);
        // Fill FIFO so the next event coalesces.
        let _ = bridge.try_send_command(SessionCommand::TtsPlayerUpdate {
            state: TtsPlayerState::Playing,
            error_message: None,
        });
        let terminal = PlayerEvent {
            state: PlayerState::Error,
            info: Default::default(),
        };
        let _ = bridge.try_send_player_event(&terminal);

        // Reject all enqueues → sticky/undelivered must remain.
        let stats = bridge.drain_with_try(&mut rx, |_| false);
        assert!(
            stats.pending_remaining || bridge.has_pending(),
            "terminal/latest must stay pending when enqueue fails"
        );

        // Accepting drain clears pending and delivers terminal Error.
        let mut saw_error = false;
        for _ in 0..4 {
            let _ = bridge.drain_with_try(&mut rx, |cmd| {
                if matches!(
                    cmd,
                    SessionCommand::TtsPlayerUpdate {
                        state: TtsPlayerState::Error,
                        ..
                    }
                ) {
                    saw_error = true;
                }
                true
            });
            if !bridge.has_pending() {
                break;
            }
        }
        assert!(
            saw_error,
            "terminal Error must be delivered via drain_with_try"
        );
        assert!(
            !bridge.has_pending(),
            "pending must clear after successful ack"
        );
    }

    /// Composed Melo + ensure_backend_ready fails closed without python binary.
    #[cfg(all(feature = "tts", feature = "tts-worker"))]
    #[test]
    fn composed_melotts_ensure_ready_fails_without_python_binary() {
        use std::fs::{self, File};
        use std::path::Path;

        use shuvoice_core::TtsBackendKind;

        fn write_pkg(root: &Path, name: &str, with_main: bool) {
            let pkg = root.join(name);
            fs::create_dir_all(&pkg).unwrap();
            File::create(pkg.join("__init__.py")).unwrap();
            if with_main {
                File::create(pkg.join("__main__.py")).unwrap();
            }
        }

        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data");
        let workers = tmp.path().join("workers");
        fs::create_dir_all(&workers).unwrap();
        write_pkg(&workers, worker_runtime::WORKER_PROTO_PACKAGE, false);
        write_pkg(&workers, MELOTTS_WORKER_MODULE, true);

        let venv = data.join("melotts-venv");
        fs::create_dir_all(venv.join("bin")).unwrap(); // no python file

        let mut inputs = WorkersDiscoveryInputs::for_tests(&data);
        inputs.extra_candidates = vec![(workers, worker_runtime::WorkersRootSource::InstalledLib)];

        let mut cfg = Config::default();
        cfg.tts_enabled = true;
        cfg.tts_backend = TtsBackendKind::Melotts;
        cfg.tts_melotts_venv_path = Some(venv.to_string_lossy().into_owned());
        let _ = cfg.validate();

        let settings = compose_tts_backend_settings_with_discovery(&cfg, &inputs).unwrap();
        assert!(settings.melotts_worker_root.is_some());
        let backend = tts_adapter::create_shared_backend(&settings).unwrap();
        let err = tts_adapter::ensure_backend_ready(&backend).expect_err("missing python");
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("python") || msg.contains("venv"),
            "expected python/venv readiness error, got: {err}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn headless_harness_start_status_stop_shutdown_and_24k_contract() {
        let tmp = tempfile::tempdir().unwrap();
        let runtime_dir = tmp.path().join("run");
        std::fs::create_dir_all(&runtime_dir).unwrap();
        let old_rt = std::env::var_os("XDG_RUNTIME_DIR");
        // SAFETY: test-only process env mutation under serial-ish temp path;
        // restored on all exit paths below before the test returns.
        unsafe {
            std::env::set_var("XDG_RUNTIME_DIR", &runtime_dir);
        }
        let sock = runtime_dir.join("shuvoice").join("test-harness.sock");
        if let Some(parent) = sock.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let mut cfg = Config::default();
        cfg.asr_backend = AsrBackendKind::OpenaiRealtime;
        cfg.tts_enabled = false;
        cfg.audio_feedback = false;
        cfg.control_socket = Some(sock.to_string_lossy().into_owned());
        cfg.chunk_ms = 100;
        cfg.sample_rate = 16_000;
        let _ = cfg.validate();

        let backend = Box::new(remote_asr("hello"));
        let harness = match HeadlessHarness::spawn(cfg, backend, sock.clone()).await {
            Ok(h) => h,
            Err(e) => {
                // SAFETY: restore XDG_RUNTIME_DIR to the value captured before mutation.
                unsafe {
                    match old_rt {
                        Some(v) => std::env::set_var("XDG_RUNTIME_DIR", v),
                        None => std::env::remove_var("XDG_RUNTIME_DIR"),
                    }
                }
                panic!("spawn harness: {e}");
            }
        };

        // Effective 24 kHz contract for OpenAI-shaped remote ASR.
        assert_eq!(harness.effective_sample_rate(), 24_000);
        assert_eq!(harness.audio_chunk_samples(), 2_400);

        let started = harness.send_control(ControlCommand::Start).expect("start");
        assert!(
            started.contains("OK"),
            "start response should be OK-prefixed: {started}"
        );

        let status = harness
            .send_control(ControlCommand::Status)
            .expect("status");
        assert!(status.starts_with("OK "), "{status}");

        let stopped = harness.send_control(ControlCommand::Stop).expect("stop");
        assert!(stopped.contains("OK"), "{stopped}");

        harness.shutdown().await.expect("shutdown");

        // SAFETY: restore XDG_RUNTIME_DIR to the value captured before mutation.
        unsafe {
            match old_rt {
                Some(v) => std::env::set_var("XDG_RUNTIME_DIR", v),
                None => std::env::remove_var("XDG_RUNTIME_DIR"),
            }
        }
    }
}
