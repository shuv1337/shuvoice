//! Backend construction and mockable registry.

use std::sync::Arc;

use shuvoice_core::AsrBackendKind;

use crate::backend::AsrBackend;
use crate::config::AsrConfig;
use crate::error::{AsrError, AsrResult};
use crate::mock::MockAsrBackend;
use crate::worker::{WorkerAsrBackend, WorkerBackendKind};

/// Factory function type for injectable construction (tests / custom backends).
pub type BackendFactoryFn = Arc<dyn Fn(AsrConfig) -> AsrResult<Box<dyn AsrBackend>> + Send + Sync>;

/// Create a backend for `config.backend` using built-in constructors.
pub fn create_backend(config: AsrConfig) -> AsrResult<Box<dyn AsrBackend>> {
    create_backend_with_registry(&BackendRegistry::default(), config)
}

/// Registry of backend constructors. Override entries for tests.
#[derive(Clone)]
pub struct BackendRegistry {
    factories: std::collections::HashMap<AsrBackendKind, BackendFactoryFn>,
}

impl BackendRegistry {
    pub fn builtin() -> Self {
        let mut reg = Self {
            factories: std::collections::HashMap::new(),
        };
        reg.insert(
            AsrBackendKind::Nemo,
            Arc::new(|cfg| {
                Ok(Box::new(WorkerAsrBackend::new(
                    WorkerBackendKind::Nemo,
                    cfg,
                )))
            }),
        );
        reg.insert(
            AsrBackendKind::Moonshine,
            Arc::new(|cfg| {
                Ok(Box::new(WorkerAsrBackend::new(
                    WorkerBackendKind::Moonshine,
                    cfg,
                )))
            }),
        );
        #[cfg(feature = "openai")]
        reg.insert(
            AsrBackendKind::OpenaiRealtime,
            Arc::new(|cfg| Ok(Box::new(crate::openai::OpenAiRealtimeBackend::new(cfg)))),
        );
        #[cfg(not(feature = "openai"))]
        reg.insert(
            AsrBackendKind::OpenaiRealtime,
            Arc::new(|_cfg| {
                Err(AsrError::dependency(
                    "openai feature is not enabled; rebuild with `--features openai`",
                ))
            }),
        );
        #[cfg(feature = "sherpa")]
        reg.insert(
            AsrBackendKind::Sherpa,
            Arc::new(|cfg| Ok(Box::new(crate::sherpa::SherpaBackend::new(cfg)))),
        );
        #[cfg(not(feature = "sherpa"))]
        reg.insert(
            AsrBackendKind::Sherpa,
            Arc::new(|_cfg| {
                Err(AsrError::dependency(
                    "sherpa feature is not enabled; rebuild with `--features sherpa`",
                ))
            }),
        );
        reg
    }

    pub fn empty() -> Self {
        Self {
            factories: std::collections::HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: AsrBackendKind, factory: BackendFactoryFn) {
        self.factories.insert(id, factory);
    }

    pub fn insert_mock(
        &mut self,
        id: AsrBackendKind,
        mock: impl Fn() -> MockAsrBackend + Send + Sync + 'static,
    ) {
        let mock = Arc::new(mock);
        self.insert(
            id,
            Arc::new(move |_cfg| Ok(Box::new(mock()) as Box<dyn AsrBackend>)),
        );
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

pub fn create_backend_with_registry(
    registry: &BackendRegistry,
    config: AsrConfig,
) -> AsrResult<Box<dyn AsrBackend>> {
    let id = config.backend();
    let factory = registry.factories.get(&id).ok_or_else(|| {
        AsrError::dependency(format!(
            "Unknown ASR backend '{id}'. Supported: {}",
            AsrBackendKind::ALL
                .iter()
                .map(|b| b.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })?;
    factory(config)
}

/// Lightweight preflight: dependency / feature availability without loading models.
///
/// When `config` is provided and already specifies a worker attach path
/// (`worker_command` / `worker_spawn` / `worker_socket_path`), NeMo/Moonshine
/// report **no** missing-dependency errors (the worker supplies the runtime).
pub fn dependency_errors(id: AsrBackendKind) -> Vec<String> {
    dependency_errors_for(id, None)
}

/// Like [`dependency_errors`] but aware of connect options on `config`.
pub fn dependency_errors_for(id: AsrBackendKind, config: Option<&AsrConfig>) -> Vec<String> {
    let mut errors = Vec::new();
    let worker_configured = config.is_some_and(|c| {
        c.connect.worker_socket_path.is_some()
            || c.connect.worker_spawn.is_some()
            || c.connect
                .worker_command
                .as_ref()
                .is_some_and(|cmd| !cmd.is_empty())
            || c.worker_spawn_config().is_some()
    });
    match id {
        AsrBackendKind::Sherpa => {
            #[cfg(not(feature = "sherpa"))]
            errors.push(
                "Missing sherpa feature. Rebuild with: cargo build -p shuvoice-asr --features sherpa"
                    .into(),
            );
        }
        AsrBackendKind::OpenaiRealtime => {
            #[cfg(not(feature = "openai"))]
            errors.push(
                "Missing openai feature. Rebuild with: cargo build -p shuvoice-asr --features openai"
                    .into(),
            );
        }
        AsrBackendKind::Nemo => {
            if !worker_configured {
                errors.push(
                    "NeMo requires an external worker process (WorkerSpawnConfig / worker_command); there is no native Rust NeMo runtime"
                        .into(),
                );
            }
        }
        AsrBackendKind::Moonshine => {
            if !worker_configured {
                errors.push(
                    "Moonshine requires an external worker process (WorkerSpawnConfig / worker_command); useful-moonshine is not embedded in native Rust"
                        .into(),
                );
            }
        }
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use shuvoice_core::FinalizationMode;

    #[test]
    fn default_registry_has_all_ids() {
        let reg = BackendRegistry::default();
        for id in AsrBackendKind::ALL {
            assert!(reg.factories.contains_key(&id), "missing {id}");
        }
    }

    #[test]
    fn mock_injection_works() {
        let mut reg = BackendRegistry::empty();
        reg.insert_mock(AsrBackendKind::Sherpa, MockAsrBackend::sherpa_offline);
        let cfg = crate::config::test_config(AsrBackendKind::Sherpa);
        let backend = create_backend_with_registry(&reg, cfg).unwrap();
        assert_eq!(
            backend.capabilities().finalization_mode,
            FinalizationMode::OfflineInstant
        );
        assert_eq!(backend.backend_id(), AsrBackendKind::Sherpa);
    }

    #[test]
    fn dependency_errors_empty_when_worker_configured() {
        let mut cfg = crate::config::test_config(AsrBackendKind::Nemo);
        cfg.connect.worker_command = Some(vec!["python".into(), "w.py".into()]);
        assert!(dependency_errors_for(AsrBackendKind::Nemo, Some(&cfg)).is_empty());
        assert!(!dependency_errors_for(AsrBackendKind::Nemo, None).is_empty());
    }

    #[test]
    fn nemo_constructs_worker_client() {
        let cfg = crate::config::test_config(AsrBackendKind::Nemo);
        let backend = create_backend(cfg).unwrap();
        assert!(backend.capabilities().wants_raw_audio);
        assert_eq!(backend.backend_id(), AsrBackendKind::Nemo);
    }
}
