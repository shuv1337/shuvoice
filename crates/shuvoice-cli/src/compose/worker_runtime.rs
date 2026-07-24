//! Deterministic optional ASR worker discovery and spawn composition.
//!
//! Discovers the bundled/reference `workers/` tree, validates the required
//! Python packages, resolves conventional isolated venv interpreters, and
//! builds a no-shell spawn spec for:
//!
//! ```text
//! <venv>/bin/python -m nemo_asr|moonshine_asr
//! ```
//!
//! Device / provider are **not** CLI flags. NeMo (`config.device`) and
//! Moonshine (`moonshine_provider`) are delivered in the worker `load`
//! JSON by [`shuvoice_asr::worker::WorkerAsrBackend`] — the Python
//! entrypoints only parse `--fake` / `-v`.
//!
//! # Integration notes
//!
//! Expected crate deps / features (declared by the integration owner):
//! - `shuvoice-core` (`Config`, `AsrBackendKind`, `data_dir`, `expand_user_path`)
//! - `shuvoice-asr` (pulled by CLI features `asr-sherpa` and/or `asr-openai`):
//!   `AsrConfig`, `AsrConnectOptions`, `WorkerSpawnConfigSerde`,
//!   `worker::WorkerSpawnConfig`, `worker::WorkerAsrBackend`,
//!   `worker::WorkerBackendKind`
//!
//! Wire from `compose/mod.rs` once the root declares the module:
//!
//! ```ignore
//! pub mod worker_runtime;
//! ```
//!
//! ## Recommended host usage
//!
//! ```ignore
//! use shuvoice_asr::worker::{WorkerAsrBackend, WorkerBackendKind};
//! use shuvoice_core::Config;
//! use shuvoice_cli::compose::worker_runtime as wr;
//!
//! let core: Config = /* validated product config */;
//! let resolved = wr::discover_asr_worker_runtime(&core)?;
//! // Lossy connect options are for preflight only (argv presence).
//! let asr = resolved.asr_config_lossy(core)?;
//! // REQUIRED: full spawn (env + current_dir). Do not rely on
//! // AsrConfig::worker_spawn_config() / resolve_attach alone.
//! let backend = WorkerAsrBackend::new(resolved.backend_kind(), asr)
//!     .with_spawn(resolved.spawn);
//! ```
//!
//! ## Cross-crate API gap (do not weaken PYTHONPATH/current_dir)
//!
//! [`shuvoice_asr::WorkerSpawnConfigSerde`] / [`AsrConnectOptions::worker_spawn`]
//! only carry `program` + `args` (+ optional `client_name`). They **cannot**
//! express `env` (needed for `PYTHONPATH`) or `current_dir` (workers tree).
//!
//! [`AsrConfig::worker_spawn_config`] therefore drops those fields, and
//! [`WorkerAsrBackend`] attach-via-config (`resolve_attach`) is insufficient
//! for real NeMo/Moonshine process spawns.
//!
//! **Minimal fix (integration owner / asr crate):** extend the connect seam,
//! e.g. one of:
//! 1. Add optional `env: Vec<(String,String)>` + `current_dir: Option<PathBuf>`
//!    to `WorkerSpawnConfigSerde`, and honor them in `into_spawn_config`.
//! 2. Or add a non-serde runtime field on `AsrConnectOptions` /
//!    `AsrConfig` such as `worker_spawn_runtime: Option<WorkerSpawnConfig>`
//!    that `worker_spawn_config()` / `resolve_attach` prefer over the serde form.
//!
//! Until then, hosts **must** call [`WorkerAsrBackend::with_spawn`] (or
//! `with_attach(WorkerAttach::Spawn(...))`) with the full
//! [`WorkerSpawnConfig`] from this module. The lossy
//! [`AsrConnectOptions`] helpers below are only for preflight /
//! `dependency_errors_for` (argv presence), not for correct process spawn.
//!
//! # Safety / hygiene
//!
//! - No shell. Argv is always a program path + discrete args.
//! - Paths use [`shuvoice_core::data_dir`] / [`expand_user_path`] (no ad-hoc `$HOME`).
//! - Pure helpers never log path contents (callers may log non-secret labels only).
//! - Release builds do not probe compile-time `CARGO_MANIFEST_DIR` workspace
//!   paths unless `SHUVOICE_ALLOW_DEV_WORKERS=1` is set.

#![allow(clippy::collapsible_if)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::let_and_return)]
#![allow(clippy::double_must_use)]
#![allow(clippy::result_unit_err)]
#![allow(clippy::suspicious_open_options)]
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use shuvoice_asr::worker::WorkerSpawnConfig;
use shuvoice_asr::{AsrConfig, AsrConnectOptions, WorkerSpawnConfigSerde};
use shuvoice_core::{AsrBackendKind, Config, data_dir, expand_user_path};

/// Env override for the workers tree (highest priority when set and valid).
pub const SHUVOICE_WORKERS_DIR_ENV: &str = "SHUVOICE_WORKERS_DIR";

/// Opt-in to probe the compile-time dev workspace workers tree in **release**
/// builds (`1` / `true` / `yes`). Debug builds allow the probe by default.
pub const SHUVOICE_ALLOW_DEV_WORKERS_ENV: &str = "SHUVOICE_ALLOW_DEV_WORKERS";

/// Conventional isolated NeMo worker venv directory name under the data dir.
///
/// Byte-identical to `setup::install::NEMO_WORKER_VENV_NAME` / `workers/README.md`.
/// Not re-exported from setup here: compose must not depend on the setup
/// module graph; lockstep is enforced by unit test.
pub const NEMO_WORKER_VENV_NAME: &str = "workers-nemo-venv";

/// Conventional isolated Moonshine worker venv directory name under the data dir.
///
/// Byte-identical to `setup::install::MOONSHINE_WORKER_VENV_NAME`.
pub const MOONSHINE_WORKER_VENV_NAME: &str = "workers-moonshine-venv";

/// Shared framing package required in every workers tree.
pub const WORKER_PROTO_PACKAGE: &str = "shuvoice_worker_proto";

/// NeMo worker module (`python -m nemo_asr`).
pub const NEMO_WORKER_MODULE: &str = "nemo_asr";

/// Moonshine worker module (`python -m moonshine_asr`).
pub const MOONSHINE_WORKER_MODULE: &str = "moonshine_asr";

/// MeloTTS worker module (`python -m melotts`) under the shared workers tree.
///
/// Not an ASR backend — used by TTS compose to require both `melotts/` and
/// [`WORKER_PROTO_PACKAGE`] before injecting `melotts_worker_root`.
pub const MELOTTS_WORKER_MODULE: &str = "melotts";

/// Packaged install locations (checked in order after the env override).
pub const INSTALLED_WORKERS_DIRS: &[&str] =
    &["/usr/lib/shuvoice/workers", "/usr/libexec/shuvoice/workers"];

/// Env key applied so `python -m <module>` can import tree packages.
pub const PYTHONPATH_ENV: &str = "PYTHONPATH";

/// Force unbuffered stdio on the worker interpreter.
pub const PYTHONUNBUFFERED_ENV: &str = "PYTHONUNBUFFERED";

// ── Types ─────────────────────────────────────────────────────────────────

/// Which optional external ASR worker to compose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AsrWorkerKind {
    Nemo,
    Moonshine,
}

impl AsrWorkerKind {
    /// Map a product backend id to a worker kind (None for native/cloud).
    #[must_use]
    pub fn from_backend(backend: AsrBackendKind) -> Option<Self> {
        match backend {
            AsrBackendKind::Nemo => Some(Self::Nemo),
            AsrBackendKind::Moonshine => Some(Self::Moonshine),
            AsrBackendKind::Sherpa | AsrBackendKind::OpenaiRealtime => None,
        }
    }

    #[must_use]
    pub fn backend_id(self) -> AsrBackendKind {
        match self {
            Self::Nemo => AsrBackendKind::Nemo,
            Self::Moonshine => AsrBackendKind::Moonshine,
        }
    }

    #[must_use]
    pub fn module_name(self) -> &'static str {
        match self {
            Self::Nemo => NEMO_WORKER_MODULE,
            Self::Moonshine => MOONSHINE_WORKER_MODULE,
        }
    }

    #[must_use]
    pub fn venv_name(self) -> &'static str {
        match self {
            Self::Nemo => NEMO_WORKER_VENV_NAME,
            Self::Moonshine => MOONSHINE_WORKER_VENV_NAME,
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nemo => "nemo",
            Self::Moonshine => "moonshine",
        }
    }
}

/// Where a workers root was found (deterministic priority labels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkersRootSource {
    /// Validated `SHUVOICE_WORKERS_DIR`.
    EnvOverride,
    /// `/usr/lib/shuvoice/workers`.
    InstalledLib,
    /// `/usr/libexec/shuvoice/workers`.
    InstalledLibexec,
    /// `CARGO_MANIFEST_DIR/../../workers` (dev workspace; gated).
    DevWorkspace,
    /// `$XDG_DATA_HOME/shuvoice/workers` — only when intentionally enabled.
    XdgData,
}

/// A validated workers tree root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredWorkersRoot {
    pub path: PathBuf,
    pub source: WorkersRootSource,
}

/// Fully resolved runtime inputs for spawning one ASR worker.
#[derive(Debug, Clone)]
pub struct ResolvedWorkerRuntime {
    pub kind: AsrWorkerKind,
    pub workers_root: DiscoveredWorkersRoot,
    pub python: PathBuf,
    pub module: &'static str,
    /// Full spawn spec including `PYTHONPATH` + `current_dir`.
    ///
    /// **Must** be attached via `WorkerAsrBackend::with_spawn` (see module docs).
    pub spawn: WorkerSpawnConfig,
}

impl ResolvedWorkerRuntime {
    /// Lossy connect options suitable only for dependency preflight.
    ///
    /// **Does not** carry `env` / `current_dir` — see module docs.
    #[must_use]
    pub fn connect_options_lossy(&self) -> AsrConnectOptions {
        asr_connect_options_from_spawn_lossy(&self.spawn)
    }

    /// `AsrConfig` with lossy connect options filled (preflight / factory id).
    ///
    /// Prefer attaching [`Self::spawn`] via `WorkerAsrBackend::with_spawn`.
    pub fn asr_config_lossy(&self, core: Config) -> Result<AsrConfig, WorkerRuntimeError> {
        let mut asr =
            AsrConfig::from_core(core).map_err(|e| WorkerRuntimeError::Config(e.to_string()))?;
        apply_spawn_to_asr_config_lossy(&mut asr, &self.spawn);
        Ok(asr)
    }

    #[must_use]
    pub fn backend_kind(&self) -> shuvoice_asr::worker::WorkerBackendKind {
        match self.kind {
            AsrWorkerKind::Nemo => shuvoice_asr::worker::WorkerBackendKind::Nemo,
            AsrWorkerKind::Moonshine => shuvoice_asr::worker::WorkerBackendKind::Moonshine,
        }
    }
}

/// Why a python interpreter path was rejected (no path bytes in Display).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PythonPathError {
    #[error("path does not exist")]
    NotFound,
    #[error("not a regular file")]
    NotRegularFile,
    #[error("not executable")]
    NotExecutable,
}

/// Which python resolution path failed (labels only — no filesystem paths).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PythonPathKind {
    /// Conventional `{data_dir}/{venv_name}/bin/python`.
    ConventionalVenv { venv_name: &'static str },
    /// Caller-supplied interpreter override.
    Custom,
}

impl PythonPathKind {
    #[must_use]
    pub fn as_label(self) -> &'static str {
        match self {
            Self::ConventionalVenv { venv_name } => venv_name,
            Self::Custom => "custom python",
        }
    }
}

/// Discovery / composition failures (no path secrets; short labels only).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorkerRuntimeError {
    #[error("ASR backend '{0}' does not use an external Python worker")]
    NotAWorkerBackend(&'static str),

    #[error(
        "workers root not found (checked SHUVOICE_WORKERS_DIR, \
         /usr/lib/shuvoice/workers, /usr/libexec/shuvoice/workers\
         , optional xdg/dev when enabled)"
    )]
    WorkersRootNotFound,

    #[error("workers root is missing required package '{package}'")]
    MissingPackage { package: &'static str },

    #[error("workers root is missing backend module '{module}'")]
    MissingBackendModule { module: &'static str },

    #[error("worker python unusable ({label}): {detail}")]
    PythonUnusable {
        label: &'static str,
        detail: PythonPathError,
    },

    #[error("SHUVOICE_WORKERS_DIR is set but empty or invalid")]
    InvalidEnvWorkersDir,

    #[error("SHUVOICE_WORKERS_DIR is set but is not valid UTF-8")]
    InvalidEnvWorkersDirNonUnicode,

    #[error("invalid ASR config: {0}")]
    Config(String),
}

impl WorkerRuntimeError {
    #[must_use]
    pub fn python_unusable(kind: PythonPathKind, detail: PythonPathError) -> Self {
        Self::PythonUnusable {
            label: kind.as_label(),
            detail,
        }
    }
}

/// Parsed view of `SHUVOICE_WORKERS_DIR` (fail-closed on non-Unicode).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvWorkersDir {
    /// Variable unset.
    Unset,
    /// Set but not valid Unicode (`OsString` could not be decoded as UTF-8).
    NonUnicode,
    /// Set to a Unicode string (may still be empty/invalid as a path).
    Value(String),
}

impl EnvWorkersDir {
    /// Read from the process environment via [`std::env::var_os`] (fail-closed).
    #[must_use]
    pub fn from_process() -> Self {
        match std::env::var_os(SHUVOICE_WORKERS_DIR_ENV) {
            None => Self::Unset,
            Some(os) => Self::from_os_string(os),
        }
    }

    /// Decode an arbitrary [`OsString`] the same way as the process env.
    #[must_use]
    pub fn from_os_string(os: OsString) -> Self {
        match os.into_string() {
            Ok(s) => Self::Value(s),
            Err(_) => Self::NonUnicode,
        }
    }
}

/// Explicit inputs for pure discovery (tests inject temp roots / data dirs).
#[derive(Debug, Clone)]
pub struct WorkersDiscoveryInputs {
    /// Parsed `SHUVOICE_WORKERS_DIR`.
    pub env_workers_dir: EnvWorkersDir,
    /// ShuVoice data dir (`$XDG_DATA_HOME/shuvoice`); venvs live here.
    pub data_dir: PathBuf,
    /// Optional override for the compile-time manifest dir (tests / debug).
    pub cargo_manifest_dir: Option<PathBuf>,
    /// When true, also probe `$data_dir/workers` (off by default).
    pub include_xdg_data_workers: bool,
    /// When true, probe `{cargo_manifest_dir}/../../workers`.
    ///
    /// Production default: `true` under `debug_assertions`, else only when
    /// `SHUVOICE_ALLOW_DEV_WORKERS` is an explicit truthy opt-in.
    pub allow_dev_workers: bool,
    /// Optional interpreter override (must be an executable regular file).
    pub python_override: Option<PathBuf>,
    /// Extra candidate roots inserted after installed paths (tests).
    pub extra_candidates: Vec<(PathBuf, WorkersRootSource)>,
}

impl Default for WorkersDiscoveryInputs {
    fn default() -> Self {
        Self {
            env_workers_dir: EnvWorkersDir::from_process(),
            data_dir: data_dir(),
            cargo_manifest_dir: Some(PathBuf::from(env!("CARGO_MANIFEST_DIR"))),
            include_xdg_data_workers: false,
            allow_dev_workers: allow_dev_workers_from_process(),
            python_override: None,
            extra_candidates: Vec::new(),
        }
    }
}

impl WorkersDiscoveryInputs {
    /// Process environment + core data dir (production default).
    #[must_use]
    pub fn from_process() -> Self {
        Self::default()
    }

    /// Test-friendly builder with an explicit data dir and no env override.
    #[must_use]
    pub fn for_tests(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            env_workers_dir: EnvWorkersDir::Unset,
            data_dir: data_dir.into(),
            cargo_manifest_dir: None,
            include_xdg_data_workers: false,
            allow_dev_workers: false,
            python_override: None,
            extra_candidates: Vec::new(),
        }
    }
}

/// Truthy opt-in values for `SHUVOICE_ALLOW_DEV_WORKERS`.
#[must_use]
pub fn env_flag_is_truthy(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Default dev-workspace probe policy from the process environment.
///
/// - Debug builds (`cfg!(debug_assertions)`): allowed unless you override
///   inputs manually.
/// - Release builds: only when `SHUVOICE_ALLOW_DEV_WORKERS` is truthy.
#[must_use]
pub fn allow_dev_workers_from_process() -> bool {
    if cfg!(debug_assertions) {
        return true;
    }
    match std::env::var_os(SHUVOICE_ALLOW_DEV_WORKERS_ENV) {
        Some(os) => os.to_str().is_some_and(env_flag_is_truthy),
        None => false,
    }
}

// ── Workers tree validation ───────────────────────────────────────────────

/// True when `root` looks like a usable workers tree for optional `module`.
///
/// Always requires [`WORKER_PROTO_PACKAGE`]. When `module` is `Some`, also
/// requires that backend package (`__init__.py` + `__main__.py`).
#[must_use]
pub fn is_workers_tree(root: &Path, module: Option<&str>) -> bool {
    if !package_present(root, WORKER_PROTO_PACKAGE) {
        return false;
    }
    if let Some(module) = module {
        if !backend_module_present(root, module) {
            return false;
        }
    }
    true
}

/// Validate required packages; returns a structured error on failure.
pub fn validate_workers_root(root: &Path, kind: AsrWorkerKind) -> Result<(), WorkerRuntimeError> {
    validate_workers_root_for_module(root, kind.module_name())
}

/// Validate a workers root for an arbitrary backend/TTS module name.
pub fn validate_workers_root_for_module(
    root: &Path,
    module: &'static str,
) -> Result<(), WorkerRuntimeError> {
    if !package_present(root, WORKER_PROTO_PACKAGE) {
        return Err(WorkerRuntimeError::MissingPackage {
            package: WORKER_PROTO_PACKAGE,
        });
    }
    if !backend_module_present(root, module) {
        return Err(WorkerRuntimeError::MissingBackendModule { module });
    }
    Ok(())
}

/// True when `root` contains MeloTTS worker package **and** the framing proto.
///
/// Requires:
/// - `shuvoice_worker_proto/__init__.py`
/// - `melotts/__init__.py` + `melotts/__main__.py`
#[must_use]
pub fn is_melotts_workers_tree(root: &Path) -> bool {
    is_workers_tree(root, Some(MELOTTS_WORKER_MODULE))
}

/// Validate MeloTTS workers root (proto + melotts module).
pub fn validate_melotts_workers_root(root: &Path) -> Result<(), WorkerRuntimeError> {
    validate_workers_root_for_module(root, MELOTTS_WORKER_MODULE)
}

fn package_present(root: &Path, package: &str) -> bool {
    let pkg = root.join(package);
    pkg.is_dir() && pkg.join("__init__.py").is_file()
}

fn backend_module_present(root: &Path, module: &str) -> bool {
    let pkg = root.join(module);
    pkg.is_dir() && pkg.join("__init__.py").is_file() && pkg.join("__main__.py").is_file()
}

// ── Root discovery ────────────────────────────────────────────────────────

/// Resolve the bundled/install/dev workers root with deterministic priority.
///
/// Priority (first valid wins):
/// 1. Safe `SHUVOICE_WORKERS_DIR` env override (expanded, must be a workers tree)
/// 2. `/usr/lib/shuvoice/workers`
/// 3. `/usr/libexec/shuvoice/workers`
/// 4. `inputs.extra_candidates` (test hooks)
/// 5. Optional XDG data workers (`$data_dir/workers`) when intentionally enabled
/// 6. Dev workspace: `{cargo_manifest_dir}/../../workers` when
///    `inputs.allow_dev_workers` is true (debug default / release opt-in)
///
/// When `kind` is `Some`, the chosen root must contain that backend module.
pub fn resolve_workers_root(
    inputs: &WorkersDiscoveryInputs,
    kind: Option<AsrWorkerKind>,
) -> Result<DiscoveredWorkersRoot, WorkerRuntimeError> {
    let module = kind.map(AsrWorkerKind::module_name);
    resolve_workers_root_for_module(inputs, module)
}

/// Resolve workers root requiring the MeloTTS package (not proto-only).
///
/// Same priority / safety policy as [`resolve_workers_root`], but the chosen
/// tree must pass [`is_melotts_workers_tree`]. Central TTS wiring should call
/// this (or [`resolve_melotts_workers_root_from_process`]) before setting
/// `melotts_worker_root`.
pub fn resolve_melotts_workers_root(
    inputs: &WorkersDiscoveryInputs,
) -> Result<DiscoveredWorkersRoot, WorkerRuntimeError> {
    let root = resolve_workers_root_for_module(inputs, Some(MELOTTS_WORKER_MODULE))?;
    validate_melotts_workers_root(&root.path)?;
    Ok(root)
}

/// Production convenience: process env + core data dir + compile-time manifest.
pub fn resolve_melotts_workers_root_from_process()
-> Result<DiscoveredWorkersRoot, WorkerRuntimeError> {
    resolve_melotts_workers_root(&WorkersDiscoveryInputs::from_process())
}

/// Resolve workers root requiring an optional package module name.
pub fn resolve_workers_root_for_module(
    inputs: &WorkersDiscoveryInputs,
    module: Option<&str>,
) -> Result<DiscoveredWorkersRoot, WorkerRuntimeError> {
    // 1) Explicit env override — must be present *and* valid when set.
    match &inputs.env_workers_dir {
        EnvWorkersDir::Unset => {}
        EnvWorkersDir::NonUnicode => {
            return Err(WorkerRuntimeError::InvalidEnvWorkersDirNonUnicode);
        }
        EnvWorkersDir::Value(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Err(WorkerRuntimeError::InvalidEnvWorkersDir);
            }
            let path = expand_user_path(trimmed);
            if !path.is_absolute() {
                // Require absolute after expansion so relative cwd surprises cannot win.
                return Err(WorkerRuntimeError::InvalidEnvWorkersDir);
            }
            if is_workers_tree(&path, module) {
                return Ok(DiscoveredWorkersRoot {
                    path: normalize_existing_dir(&path).unwrap_or(path),
                    source: WorkersRootSource::EnvOverride,
                });
            }
            return Err(WorkerRuntimeError::InvalidEnvWorkersDir);
        }
    }

    // 2–3) Installed locations.
    for (idx, candidate) in INSTALLED_WORKERS_DIRS.iter().enumerate() {
        let path = PathBuf::from(candidate);
        if is_workers_tree(&path, module) {
            let source = if idx == 0 {
                WorkersRootSource::InstalledLib
            } else {
                WorkersRootSource::InstalledLibexec
            };
            return Ok(DiscoveredWorkersRoot {
                path: normalize_existing_dir(&path).unwrap_or(path),
                source,
            });
        }
    }

    // 4) Explicit extra candidates (tests).
    for (path, source) in &inputs.extra_candidates {
        if is_workers_tree(path, module) {
            return Ok(DiscoveredWorkersRoot {
                path: normalize_existing_dir(path).unwrap_or_else(|| path.clone()),
                source: *source,
            });
        }
    }

    // 5) Optional XDG data workers (off unless intentional).
    if inputs.include_xdg_data_workers {
        let path = inputs.data_dir.join("workers");
        if is_workers_tree(&path, module) {
            return Ok(DiscoveredWorkersRoot {
                path: normalize_existing_dir(&path).unwrap_or(path),
                source: WorkersRootSource::XdgData,
            });
        }
    }

    // 6) Dev workspace — gated so release binaries do not use builder paths.
    if inputs.allow_dev_workers {
        if let Some(manifest) = inputs.cargo_manifest_dir.as_ref() {
            let path = manifest.join("../../workers");
            if is_workers_tree(&path, module) {
                return Ok(DiscoveredWorkersRoot {
                    path: normalize_existing_dir(&path).unwrap_or(path),
                    source: WorkersRootSource::DevWorkspace,
                });
            }
        }
    }

    Err(WorkerRuntimeError::WorkersRootNotFound)
}

/// Production convenience: process env + core data dir + compile-time manifest.
pub fn resolve_workers_root_from_process(
    kind: Option<AsrWorkerKind>,
) -> Result<DiscoveredWorkersRoot, WorkerRuntimeError> {
    resolve_workers_root(&WorkersDiscoveryInputs::from_process(), kind)
}

fn normalize_existing_dir(path: &Path) -> Option<PathBuf> {
    fs::canonicalize(path).ok().filter(|p| p.is_dir())
}

// ── Venv / python resolution ──────────────────────────────────────────────

/// Conventional venv directory: `{data_dir}/{workers-*-venv}`.
#[must_use]
pub fn conventional_venv_dir(data_dir: &Path, kind: AsrWorkerKind) -> PathBuf {
    data_dir.join(kind.venv_name())
}

/// Conventional interpreter path: `{venv}/bin/python`.
#[must_use]
pub fn conventional_venv_python(data_dir: &Path, kind: AsrWorkerKind) -> PathBuf {
    conventional_venv_dir(data_dir, kind)
        .join("bin")
        .join("python")
}

/// Classify why `path` is not a usable interpreter (exists / file / exec).
pub fn classify_python_path(path: &Path) -> Result<(), PythonPathError> {
    let meta = match fs::metadata(path) {
        Ok(m) => m,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Err(PythonPathError::NotFound);
        }
        Err(_) => {
            // Permissions / I/O on the path: treat as not found for callers
            // (no path bytes leaked; existence is the actionable signal).
            return Err(PythonPathError::NotFound);
        }
    };
    if !meta.is_file() {
        return Err(PythonPathError::NotRegularFile);
    }
    #[cfg(unix)]
    {
        if meta.permissions().mode() & 0o111 == 0 {
            return Err(PythonPathError::NotExecutable);
        }
    }
    Ok(())
}

/// True when `path` is a regular file with at least one execute bit (Unix).
#[must_use]
pub fn is_executable_regular_file(path: &Path) -> bool {
    classify_python_path(path).is_ok()
}

/// Require an executable regular-file interpreter at `path`.
///
/// `kind` selects the error label (`custom python` vs conventional venv name).
/// On success returns a canonicalized path when available.
pub fn require_python_executable(
    path: &Path,
    kind: PythonPathKind,
) -> Result<PathBuf, WorkerRuntimeError> {
    classify_python_path(path)
        .map_err(|detail| WorkerRuntimeError::python_unusable(kind, detail))?;
    Ok(fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()))
}

/// Resolve and require the conventional venv interpreter for `kind`.
pub fn resolve_worker_python(
    data_dir: &Path,
    kind: AsrWorkerKind,
) -> Result<PathBuf, WorkerRuntimeError> {
    let python = conventional_venv_python(data_dir, kind);
    require_python_executable(
        &python,
        PythonPathKind::ConventionalVenv {
            venv_name: kind.venv_name(),
        },
    )
}

/// Resolve python from an optional override, else the conventional venv.
///
/// Custom override failures use the `custom python` label so hosts can tell
/// a bad `--python` / config override from a missing managed venv.
pub fn resolve_worker_python_with_override(
    data_dir: &Path,
    kind: AsrWorkerKind,
    python_override: Option<&Path>,
) -> Result<PathBuf, WorkerRuntimeError> {
    if let Some(path) = python_override {
        return require_python_executable(path, PythonPathKind::Custom);
    }
    resolve_worker_python(data_dir, kind)
}

// ── Argv + spawn ──────────────────────────────────────────────────────────

/// Discrete argv **after** the program: `["-m", module]`.
///
/// No `--device`: device/provider travel in the worker `load` JSON.
#[must_use]
pub fn worker_module_args(module: &str) -> Vec<String> {
    vec!["-m".to_string(), module.to_string()]
}

/// Lossy serde spawn (program + args only) — **no** env/current_dir.
#[must_use]
pub fn worker_spawn_config_serde(
    python: &Path,
    module: &str,
    client_name: Option<String>,
) -> WorkerSpawnConfigSerde {
    WorkerSpawnConfigSerde {
        program: python.to_path_buf(),
        args: worker_module_args(module),
        client_name,
    }
}

/// Full [`WorkerSpawnConfig`] with `current_dir` + `PYTHONPATH` + unbuffered I/O.
///
/// Never goes through a shell. Does not log paths.
#[must_use]
pub fn build_worker_spawn_config(
    python: &Path,
    workers_root: &Path,
    module: &str,
    client_name: impl Into<String>,
) -> WorkerSpawnConfig {
    WorkerSpawnConfig::new(python)
        .args(worker_module_args(module))
        .current_dir(workers_root)
        .env_pair(PYTHONPATH_ENV, workers_root.as_os_str())
        .env_pair(PYTHONUNBUFFERED_ENV, "1")
        .client_name(client_name)
}

/// Default client name label for composed ASR workers.
#[must_use]
pub fn default_worker_client_name(kind: AsrWorkerKind) -> String {
    format!(
        "shuvoice-cli/{}/{}",
        kind.as_str(),
        env!("CARGO_PKG_VERSION")
    )
}

// ── AsrConfig seam helpers ────────────────────────────────────────────────

/// Lossy `AsrConnectOptions` from a full spawn (argv + client_name only).
#[must_use]
pub fn asr_connect_options_from_spawn_lossy(spawn: &WorkerSpawnConfig) -> AsrConnectOptions {
    let mut connect = AsrConnectOptions::default();
    connect.worker_spawn = Some(WorkerSpawnConfigSerde {
        program: spawn.program.clone(),
        args: spawn
            .args
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect(),
        client_name: Some(spawn.client_name.clone()),
    });
    connect
}

/// Apply lossy spawn metadata onto `asr.connect` for preflight/factory checks.
///
/// **Gap:** this cannot preserve `env` / `current_dir`. Prefer
/// `WorkerAsrBackend::with_spawn(spawn)` for the real attach path.
pub fn apply_spawn_to_asr_config_lossy(asr: &mut AsrConfig, spawn: &WorkerSpawnConfig) {
    asr.connect = asr_connect_options_from_spawn_lossy(spawn);
}

/// Build a full spawn for `kind` using explicit roots (pure; no process env).
pub fn compose_worker_spawn(
    kind: AsrWorkerKind,
    workers_root: &Path,
    python: &Path,
) -> Result<WorkerSpawnConfig, WorkerRuntimeError> {
    validate_workers_root(workers_root, kind)?;
    let python = require_python_executable(python, PythonPathKind::Custom)?;
    Ok(build_worker_spawn_config(
        &python,
        workers_root,
        kind.module_name(),
        default_worker_client_name(kind),
    ))
}

/// Discover workers root + venv python and build the full runtime resolution.
pub fn discover_worker_runtime_with_inputs(
    kind: AsrWorkerKind,
    inputs: &WorkersDiscoveryInputs,
) -> Result<ResolvedWorkerRuntime, WorkerRuntimeError> {
    let root = resolve_workers_root(inputs, Some(kind))?;
    validate_workers_root(&root.path, kind)?;
    let python = resolve_worker_python_with_override(
        &inputs.data_dir,
        kind,
        inputs.python_override.as_deref(),
    )?;
    let spawn = build_worker_spawn_config(
        &python,
        &root.path,
        kind.module_name(),
        default_worker_client_name(kind),
    );
    Ok(ResolvedWorkerRuntime {
        kind,
        workers_root: root,
        python,
        module: kind.module_name(),
        spawn,
    })
}

/// Discover for the backend selected in `cfg` (errors if not NeMo/Moonshine).
pub fn discover_asr_worker_runtime(
    cfg: &Config,
) -> Result<ResolvedWorkerRuntime, WorkerRuntimeError> {
    discover_asr_worker_runtime_with_inputs(cfg, &WorkersDiscoveryInputs::from_process())
}

/// Injectable variant of [`discover_asr_worker_runtime`].
///
/// `cfg` selects the backend kind only (device/provider are load-JSON concerns).
pub fn discover_asr_worker_runtime_with_inputs(
    cfg: &Config,
    inputs: &WorkersDiscoveryInputs,
) -> Result<ResolvedWorkerRuntime, WorkerRuntimeError> {
    let kind = AsrWorkerKind::from_backend(cfg.asr_backend)
        .ok_or_else(|| WorkerRuntimeError::NotAWorkerBackend(cfg.asr_backend.as_str()))?;
    discover_worker_runtime_with_inputs(kind, inputs)
}

/// Best current AsrConfig seam: lossy connect options + return full spawn.
///
/// Returns `(asr_config_with_lossy_connect, full_spawn)`. Hosts **must** still
/// attach via `WorkerAsrBackend::with_spawn(full_spawn)` until the cross-crate
/// gap is closed.
pub fn compose_asr_config_with_worker(
    core: Config,
    inputs: &WorkersDiscoveryInputs,
) -> Result<(AsrConfig, WorkerSpawnConfig), WorkerRuntimeError> {
    let resolved = discover_asr_worker_runtime_with_inputs(&core, inputs)?;
    let spawn = resolved.spawn.clone();
    let asr = resolved.asr_config_lossy(core)?;
    Ok((asr, spawn))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;

    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;

    fn write_package(root: &Path, name: &str, with_main: bool) {
        let pkg = root.join(name);
        fs::create_dir_all(&pkg).unwrap();
        File::create(pkg.join("__init__.py")).unwrap();
        if with_main {
            File::create(pkg.join("__main__.py")).unwrap();
        }
    }

    fn make_workers_tree(root: &Path, modules: &[&str]) {
        fs::create_dir_all(root).unwrap();
        write_package(root, WORKER_PROTO_PACKAGE, false);
        for m in modules {
            write_package(root, m, true);
        }
    }

    fn make_venv_python(data: &Path, venv_name: &str) -> PathBuf {
        let bin = data.join(venv_name).join("bin");
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
        python
    }

    fn base_config(kind: AsrWorkerKind) -> Config {
        let mut cfg = Config::default();
        cfg.asr_backend = kind.backend_id();
        let _ = cfg.validate();
        cfg
    }

    #[test]
    fn venv_names_match_setup_install_convention() {
        // Lockstep with `crates/shuvoice-cli/src/setup/install.rs` public
        // constants (compose must not import setup; values must stay equal).
        assert_eq!(NEMO_WORKER_VENV_NAME, "workers-nemo-venv");
        assert_eq!(MOONSHINE_WORKER_VENV_NAME, "workers-moonshine-venv");
    }

    #[test]
    fn is_workers_tree_requires_proto_and_optional_module() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("workers");
        make_workers_tree(&root, &[NEMO_WORKER_MODULE]);

        assert!(is_workers_tree(&root, None));
        assert!(is_workers_tree(&root, Some(NEMO_WORKER_MODULE)));
        assert!(!is_workers_tree(&root, Some(MOONSHINE_WORKER_MODULE)));

        let bad = tmp.path().join("bad");
        fs::create_dir_all(&bad).unwrap();
        write_package(&bad, NEMO_WORKER_MODULE, true);
        assert!(!is_workers_tree(&bad, Some(NEMO_WORKER_MODULE)));
    }

    #[test]
    fn resolve_priority_env_over_extra_over_xdg_over_dev() {
        let tmp = tempfile::tempdir().unwrap();
        let env_root = tmp.path().join("env-workers");
        let extra_root = tmp.path().join("extra-workers");
        let xdg_data = tmp.path().join("xdg-data");
        let xdg_root = xdg_data.join("workers");
        let manifest = tmp.path().join("crates").join("shuvoice-cli");
        fs::create_dir_all(&manifest).unwrap();
        let dev_actual = manifest.join("../../workers");
        make_workers_tree(&env_root, &[NEMO_WORKER_MODULE, MOONSHINE_WORKER_MODULE]);
        make_workers_tree(&extra_root, &[NEMO_WORKER_MODULE, MOONSHINE_WORKER_MODULE]);
        make_workers_tree(&xdg_root, &[NEMO_WORKER_MODULE, MOONSHINE_WORKER_MODULE]);
        make_workers_tree(&dev_actual, &[NEMO_WORKER_MODULE, MOONSHINE_WORKER_MODULE]);

        let mut inputs = WorkersDiscoveryInputs::for_tests(&xdg_data);
        inputs.env_workers_dir = EnvWorkersDir::Value(env_root.to_string_lossy().into_owned());
        inputs.cargo_manifest_dir = Some(manifest.clone());
        inputs.include_xdg_data_workers = true;
        inputs.allow_dev_workers = true;
        inputs.extra_candidates = vec![(extra_root.clone(), WorkersRootSource::InstalledLib)];

        let found = resolve_workers_root(&inputs, Some(AsrWorkerKind::Nemo)).unwrap();
        assert_eq!(found.source, WorkersRootSource::EnvOverride);
        assert_eq!(
            fs::canonicalize(&found.path).unwrap(),
            fs::canonicalize(&env_root).unwrap()
        );

        inputs.env_workers_dir = EnvWorkersDir::Unset;
        let found = resolve_workers_root(&inputs, Some(AsrWorkerKind::Nemo)).unwrap();
        assert_eq!(found.source, WorkersRootSource::InstalledLib);
        assert_eq!(
            fs::canonicalize(&found.path).unwrap(),
            fs::canonicalize(&extra_root).unwrap()
        );

        inputs.extra_candidates.clear();
        let found = resolve_workers_root(&inputs, Some(AsrWorkerKind::Nemo)).unwrap();
        assert_eq!(found.source, WorkersRootSource::XdgData);

        inputs.include_xdg_data_workers = false;
        let found = resolve_workers_root(&inputs, Some(AsrWorkerKind::Nemo)).unwrap();
        assert_eq!(found.source, WorkersRootSource::DevWorkspace);
        assert_eq!(
            fs::canonicalize(&found.path).unwrap(),
            fs::canonicalize(&dev_actual).unwrap()
        );

        // Release-like: dev probe disabled → not found.
        inputs.allow_dev_workers = false;
        assert_eq!(
            resolve_workers_root(&inputs, Some(AsrWorkerKind::Nemo)).unwrap_err(),
            WorkerRuntimeError::WorkersRootNotFound
        );
    }

    #[test]
    fn env_override_non_unicode_fails_closed() {
        let mut inputs = WorkersDiscoveryInputs::for_tests("/tmp");
        inputs.env_workers_dir = EnvWorkersDir::NonUnicode;
        assert_eq!(
            resolve_workers_root(&inputs, None).unwrap_err(),
            WorkerRuntimeError::InvalidEnvWorkersDirNonUnicode
        );

        // from_os_string path used by from_process.
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let os = OsString::from_vec(vec![0xff, 0xfe, 0xfd]);
            assert_eq!(EnvWorkersDir::from_os_string(os), EnvWorkersDir::NonUnicode);
        }
    }

    #[test]
    fn env_override_must_be_absolute_and_valid_tree() {
        let tmp = tempfile::tempdir().unwrap();
        let mut inputs = WorkersDiscoveryInputs::for_tests(tmp.path());
        inputs.env_workers_dir = EnvWorkersDir::Value("   ".into());
        assert_eq!(
            resolve_workers_root(&inputs, None).unwrap_err(),
            WorkerRuntimeError::InvalidEnvWorkersDir
        );

        inputs.env_workers_dir = EnvWorkersDir::Value("relative/workers".into());
        assert_eq!(
            resolve_workers_root(&inputs, None).unwrap_err(),
            WorkerRuntimeError::InvalidEnvWorkersDir
        );

        let missing = tmp.path().join("nope");
        inputs.env_workers_dir = EnvWorkersDir::Value(missing.to_string_lossy().into_owned());
        assert_eq!(
            resolve_workers_root(&inputs, None).unwrap_err(),
            WorkerRuntimeError::InvalidEnvWorkersDir
        );
    }

    #[test]
    fn xdg_data_workers_skipped_unless_intentional() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data");
        let xdg = data.join("workers");
        make_workers_tree(&xdg, &[NEMO_WORKER_MODULE]);
        let mut inputs = WorkersDiscoveryInputs::for_tests(&data);
        inputs.include_xdg_data_workers = false;
        assert_eq!(
            resolve_workers_root(&inputs, Some(AsrWorkerKind::Nemo)).unwrap_err(),
            WorkerRuntimeError::WorkersRootNotFound
        );
        inputs.include_xdg_data_workers = true;
        let found = resolve_workers_root(&inputs, Some(AsrWorkerKind::Nemo)).unwrap();
        assert_eq!(found.source, WorkersRootSource::XdgData);
    }

    #[test]
    fn conventional_and_custom_python_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path();
        let nemo_py = conventional_venv_python(data, AsrWorkerKind::Nemo);
        assert!(nemo_py.ends_with("workers-nemo-venv/bin/python"));

        assert_eq!(
            resolve_worker_python(data, AsrWorkerKind::Nemo).unwrap_err(),
            WorkerRuntimeError::PythonUnusable {
                label: NEMO_WORKER_VENV_NAME,
                detail: PythonPathError::NotFound,
            }
        );

        let created = make_venv_python(data, NEMO_WORKER_VENV_NAME);
        assert!(is_executable_regular_file(&created));
        let resolved = resolve_worker_python(data, AsrWorkerKind::Nemo).unwrap();
        assert_eq!(
            fs::canonicalize(&resolved).unwrap(),
            fs::canonicalize(&created).unwrap()
        );

        // Custom override: missing.
        let missing = data.join("no-such-python");
        assert_eq!(
            resolve_worker_python_with_override(data, AsrWorkerKind::Nemo, Some(missing.as_path()))
                .unwrap_err(),
            WorkerRuntimeError::PythonUnusable {
                label: "custom python",
                detail: PythonPathError::NotFound,
            }
        );

        // Custom override: directory, not a file.
        let dir = data.join("pythondir");
        fs::create_dir_all(&dir).unwrap();
        assert_eq!(
            require_python_executable(&dir, PythonPathKind::Custom).unwrap_err(),
            WorkerRuntimeError::PythonUnusable {
                label: "custom python",
                detail: PythonPathError::NotRegularFile,
            }
        );

        #[cfg(unix)]
        {
            let moon_bin = data.join(MOONSHINE_WORKER_VENV_NAME).join("bin");
            fs::create_dir_all(&moon_bin).unwrap();
            let moon_py = moon_bin.join("python");
            let mut f = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .mode(0o644)
                .open(&moon_py)
                .unwrap();
            writeln!(f, "not exec").unwrap();
            assert_eq!(
                classify_python_path(&moon_py).unwrap_err(),
                PythonPathError::NotExecutable
            );
            assert_eq!(
                resolve_worker_python(data, AsrWorkerKind::Moonshine).unwrap_err(),
                WorkerRuntimeError::PythonUnusable {
                    label: MOONSHINE_WORKER_VENV_NAME,
                    detail: PythonPathError::NotExecutable,
                }
            );

            // Custom non-executable uses custom label.
            assert_eq!(
                require_python_executable(&moon_py, PythonPathKind::Custom).unwrap_err(),
                WorkerRuntimeError::PythonUnusable {
                    label: "custom python",
                    detail: PythonPathError::NotExecutable,
                }
            );
        }
    }

    #[test]
    fn build_spawn_exact_argv_env_and_current_dir_without_device_flag() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("workers");
        make_workers_tree(&root, &[NEMO_WORKER_MODULE]);
        let data = tmp.path().join("data");
        let python = make_venv_python(&data, NEMO_WORKER_VENV_NAME);
        let python = fs::canonicalize(&python).unwrap();
        let root = fs::canonicalize(&root).unwrap();

        let spawn = build_worker_spawn_config(&python, &root, NEMO_WORKER_MODULE, "test-client");

        assert_eq!(spawn.program, python);
        assert_eq!(
            spawn
                .args
                .iter()
                .map(|a| a.to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            vec!["-m", "nemo_asr"]
        );
        // No --device anywhere in argv.
        assert!(spawn.args.iter().all(|a| a.to_string_lossy() != "--device"));
        assert_eq!(spawn.current_dir.as_deref(), Some(root.as_path()));
        assert_eq!(spawn.client_name, "test-client");

        let mut env: Vec<(String, String)> = spawn
            .env
            .iter()
            .map(|(k, v)| (k.to_string_lossy().into(), v.to_string_lossy().into()))
            .collect();
        env.sort();
        assert_eq!(
            env,
            vec![
                (PYTHONPATH_ENV.into(), root.to_string_lossy().into()),
                (PYTHONUNBUFFERED_ENV.into(), "1".into()),
            ]
        );
    }

    #[test]
    fn discover_runtime_composes_full_spawn_and_lossy_connect() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("workers");
        make_workers_tree(&root, &[MOONSHINE_WORKER_MODULE]);
        let data = tmp.path().join("data");
        let python = make_venv_python(&data, MOONSHINE_WORKER_VENV_NAME);

        let mut inputs = WorkersDiscoveryInputs::for_tests(&data);
        inputs.extra_candidates = vec![(root.clone(), WorkersRootSource::DevWorkspace)];

        let cfg = base_config(AsrWorkerKind::Moonshine);
        let resolved =
            discover_worker_runtime_with_inputs(AsrWorkerKind::Moonshine, &inputs).unwrap();

        assert_eq!(resolved.kind, AsrWorkerKind::Moonshine);
        assert_eq!(resolved.module, MOONSHINE_WORKER_MODULE);
        assert_eq!(
            fs::canonicalize(&resolved.python).unwrap(),
            fs::canonicalize(&python).unwrap()
        );
        assert_eq!(
            resolved
                .spawn
                .args
                .iter()
                .map(|a| a.to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            vec!["-m", "moonshine_asr"]
        );
        assert_eq!(
            resolved
                .spawn
                .current_dir
                .as_ref()
                .map(|p| fs::canonicalize(p).unwrap()),
            Some(fs::canonicalize(&root).unwrap())
        );

        let connect = resolved.connect_options_lossy();
        let serde = connect.worker_spawn.unwrap();
        assert_eq!(
            fs::canonicalize(&serde.program).unwrap(),
            fs::canonicalize(&python).unwrap()
        );
        assert_eq!(serde.args, vec!["-m", "moonshine_asr"]);

        let (asr, full) = compose_asr_config_with_worker(cfg, &inputs).unwrap();
        assert!(asr.connect.worker_spawn.is_some());
        assert_eq!(full.current_dir, resolved.spawn.current_dir);
        assert!(!full.env.is_empty());
        // Prove the gap: round-tripping through AsrConfig drops env/current_dir.
        let roundtrip = asr.worker_spawn_config().unwrap();
        assert!(roundtrip.current_dir.is_none());
        assert!(roundtrip.env.is_empty());
    }

    #[test]
    fn discover_honors_custom_python_override() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("workers");
        make_workers_tree(&root, &[NEMO_WORKER_MODULE]);
        let data = tmp.path().join("data");
        // Conventional venv missing; custom python present.
        let custom = tmp.path().join("custom-python");
        #[cfg(unix)]
        {
            let mut f = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .mode(0o755)
                .open(&custom)
                .unwrap();
            writeln!(f, "#!/bin/sh\nexit 0").unwrap();
        }
        #[cfg(not(unix))]
        {
            File::create(&custom).unwrap();
        }

        let mut inputs = WorkersDiscoveryInputs::for_tests(&data);
        inputs.extra_candidates = vec![(root, WorkersRootSource::DevWorkspace)];
        inputs.python_override = Some(custom.clone());

        let resolved = discover_worker_runtime_with_inputs(AsrWorkerKind::Nemo, &inputs).unwrap();
        assert_eq!(
            fs::canonicalize(&resolved.python).unwrap(),
            fs::canonicalize(&custom).unwrap()
        );
        assert_eq!(resolved.spawn.program, resolved.python);
    }

    #[test]
    fn sherpa_is_not_a_worker_backend() {
        let cfg = Config::default();
        assert!(matches!(
            discover_asr_worker_runtime_with_inputs(
                &cfg,
                &WorkersDiscoveryInputs::for_tests("/tmp")
            ),
            Err(WorkerRuntimeError::NotAWorkerBackend("sherpa"))
        ));
    }

    #[test]
    fn validate_workers_root_errors_are_specific() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("w");
        fs::create_dir_all(&root).unwrap();
        assert!(matches!(
            validate_workers_root(&root, AsrWorkerKind::Nemo),
            Err(WorkerRuntimeError::MissingPackage {
                package: WORKER_PROTO_PACKAGE
            })
        ));
        write_package(&root, WORKER_PROTO_PACKAGE, false);
        assert!(matches!(
            validate_workers_root(&root, AsrWorkerKind::Nemo),
            Err(WorkerRuntimeError::MissingBackendModule {
                module: NEMO_WORKER_MODULE
            })
        ));
        write_package(&root, NEMO_WORKER_MODULE, true);
        validate_workers_root(&root, AsrWorkerKind::Nemo).unwrap();
    }

    #[test]
    fn env_flag_truthy_values() {
        assert!(env_flag_is_truthy("1"));
        assert!(env_flag_is_truthy("true"));
        assert!(env_flag_is_truthy("YES"));
        assert!(env_flag_is_truthy("on"));
        assert!(!env_flag_is_truthy("0"));
        assert!(!env_flag_is_truthy(""));
        assert!(!env_flag_is_truthy("no"));
    }

    #[test]
    fn melotts_tree_requires_proto_and_melotts_module() {
        let tmp = tempfile::tempdir().unwrap();
        let proto_only = tmp.path().join("proto-only");
        make_workers_tree(&proto_only, &[]); // proto only, no melotts
        assert!(is_workers_tree(&proto_only, None));
        assert!(!is_melotts_workers_tree(&proto_only));
        assert!(matches!(
            validate_melotts_workers_root(&proto_only),
            Err(WorkerRuntimeError::MissingBackendModule {
                module: MELOTTS_WORKER_MODULE
            })
        ));

        let full = tmp.path().join("full");
        make_workers_tree(&full, &[MELOTTS_WORKER_MODULE]);
        assert!(is_melotts_workers_tree(&full));
        validate_melotts_workers_root(&full).unwrap();
    }

    #[test]
    fn resolve_melotts_rejects_proto_only_tree() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data");
        let proto_only = tmp.path().join("workers");
        make_workers_tree(&proto_only, &[NEMO_WORKER_MODULE]); // ASR modules, no melotts

        let mut inputs = WorkersDiscoveryInputs::for_tests(&data);
        inputs.extra_candidates = vec![(proto_only.clone(), WorkersRootSource::DevWorkspace)];

        // Generic resolve with None still accepts proto+nemo.
        assert!(resolve_workers_root(&inputs, None).is_ok());

        // Melo-specific resolve must reject.
        let err = resolve_melotts_workers_root(&inputs).unwrap_err();
        assert_eq!(err, WorkerRuntimeError::WorkersRootNotFound);

        // With melotts present, succeeds.
        make_workers_tree(&proto_only, &[MELOTTS_WORKER_MODULE]);
        let found = resolve_melotts_workers_root(&inputs).unwrap();
        assert!(is_melotts_workers_tree(&found.path));
    }

    #[test]
    fn tilde_env_override_expands_via_core() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("workers");
        make_workers_tree(&root, &[NEMO_WORKER_MODULE]);
        let expanded = expand_user_path(root.to_string_lossy().as_ref());
        assert!(is_workers_tree(&expanded, Some(NEMO_WORKER_MODULE)));
    }
}
