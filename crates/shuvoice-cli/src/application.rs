//! Long-running application composition entry for `shuvoice run`.

use shuvoice_core::Config;

use crate::compose::{run_production, validate_composition_config};
use crate::error::ExitStatus;

/// Compose runtime services for the overlay process.
pub struct Application {
    pub config: Config,
}

impl Application {
    /// Validate selected backend/feature availability and layer-shell (when UI
    /// is built). Does not open devices, download models, or mutate services.
    pub fn new(config: Config) -> Result<Self, String> {
        validate_composition_config(&config)?;
        Ok(Self { config })
    }

    /// Run the fully composed production session.
    ///
    /// Startup dependency/config/model/audio/control failures return exit 78.
    /// Runtime ASR decode errors stay inside the session circuit breaker.
    pub async fn run(self) -> ExitStatus {
        run_production(self.config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shuvoice_core::AsrBackendKind;

    #[test]
    fn new_rejects_sherpa_without_feature() {
        let cfg = Config::default();
        assert_eq!(cfg.asr_backend, AsrBackendKind::Sherpa);
        if !cfg!(feature = "asr-sherpa") {
            match Application::new(cfg) {
                Ok(_) => panic!("expected missing asr-sherpa error"),
                Err(err) => assert!(err.contains("asr-sherpa"), "{err}"),
            }
        }
    }
}
