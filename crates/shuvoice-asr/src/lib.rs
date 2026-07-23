//! ASR traits, orchestration-facing capabilities, and backend adapters.
//!
//! # Features
//!
//! - default: traits, worker client (NeMo/Moonshine via `shuvoice-worker-proto`),
//!   mocks, factories; policy types from `shuvoice-core`
//! - `openai`: native OpenAI Realtime WebSocket backend
//! - `sherpa`: official `sherpa-onnx` 1.13.4 offline Parakeet + streaming
//!
//! # Convergence boundary
//!
//! Caps, finalization, circuit breaker, CUDA-OOM markers, and TOML config policy
//! are owned by `shuvoice-core`. This crate adds only:
//! - `AsrBackend` trait + factories
//! - runtime connect seams ([`AsrConnectOptions`])
//! - worker/process attach
//! - native OpenAI / Sherpa engines
//!
//! # NeMo / Moonshine
//!
//! No native Rust ML runtime. Use [`WorkerAsrBackend`] with
//! [`WorkerAttach::Spawn`] / supervisor / duplex injection.
//!
//! # Sherpa FFI ownership
//!
//! Recognizers live only inside `SherpaBackend`, accessed via `&mut self`.

pub mod backend;
pub mod caps;
pub mod circuit;
pub mod config;
pub mod cuda_oom;
pub mod error;
pub mod events;
pub mod factory;
pub mod mock;
pub mod pcm;
pub mod worker;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "sherpa")]
pub mod sherpa;

pub use backend::{AsrBackend, DynAsrBackend, ProgressFn};
pub use caps::{
    moonshine_caps, nemo_caps, openai_realtime_caps, sherpa_offline_caps, sherpa_streaming_caps,
};
pub use circuit::{
    ASR_CIRCUIT_COOLDOWN, ASR_MAX_FAILURES, BreakerAction, CircuitBreaker, ERROR_TOAST_SECONDS,
};
pub use config::{
    AsrConfig, AsrConnectOptions, BackendId, CoreConfig, Provider, WorkerSpawnConfigSerde,
    data_dir, test_config,
};
pub use cuda_oom::{
    CUDA_OOM_ERROR_MARKERS, looks_like_cuda_oom_error, looks_like_cuda_oom_error_dyn,
    looks_like_cuda_oom_str,
};
pub use error::{AsrError, AsrErrorClass, AsrResult, FallbackOutcome};
pub use events::AsrEvent;
pub use factory::{
    BackendFactoryFn, BackendRegistry, create_backend, create_backend_with_registry,
    dependency_errors, dependency_errors_for,
};
pub use mock::MockAsrBackend;
pub use worker::{
    PROTOCOL_VERSION, WorkerAsrBackend, WorkerAttach, WorkerBackendKind, spawn_mock_worker,
};

// Core type re-exports for hosts that only depend on shuvoice-asr.
pub use shuvoice_core::{
    AsrBackendKind, AsrCapabilities, ExpectedChunking, FinalizationMode,
    PARAKEET_TDT_V3_INT8_MODEL_NAME, ResolvedSherpaDecodeMode, SherpaDecodeMode,
};

#[cfg(feature = "openai")]
pub use openai::OpenAiRealtimeBackend;

#[cfg(feature = "sherpa")]
pub use sherpa::{DEFAULT_SHERPA_MODEL_NAME, SherpaBackend, download_model, is_model_dir_complete};

/// Crate version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Exit status used for dependency/configuration failures that systemd must not restart.
pub const DEPENDENCY_EXIT_CODE: u8 = shuvoice_core::DEPENDENCY_EXIT_CODE;
