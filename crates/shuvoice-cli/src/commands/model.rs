//! Model download command.

use shuvoice_core::{AsrBackendKind, Config};

use crate::error::{EXIT_DEPENDENCY, EXIT_FAILURE, EXIT_SUCCESS, ExitStatus};
use crate::setup::sherpa_model::{self, download_sherpa_model, is_complete_sherpa_dir};

pub async fn download_model(config: &Config) -> ExitStatus {
    match download_model_impl(config, None).await {
        Ok(msg) => {
            println!("{msg}");
            ExitStatus::code(EXIT_SUCCESS)
        }
        Err(err) => {
            eprintln!("ERROR: {err}");
            if err.contains("feature") || err.contains("worker") || err.contains("requires") {
                ExitStatus::code(EXIT_DEPENDENCY)
            } else {
                ExitStatus::code(EXIT_FAILURE)
            }
        }
    }
}

pub async fn download_model_impl(
    config: &Config,
    archive_url_override: Option<String>,
) -> Result<String, String> {
    match config.asr_backend {
        AsrBackendKind::OpenaiRealtime => {
            Ok("Model download: skipped (OpenAI Realtime uses cloud transcription).".into())
        }
        // NeMo and Moonshine are consistent: optional external workers; lazy/HF download.
        // Never error here (would turn a prior setup PASS into exit 78).
        AsrBackendKind::Moonshine | AsrBackendKind::Nemo => Ok(format!(
            "Model download: skipped ({} is an optional external worker; models download lazily via worker_command / HF cache).",
            config.asr_backend.as_str()
        )),
        AsrBackendKind::Sherpa => {
            let target = sherpa_model::sherpa_model_dir(config);
            if is_complete_sherpa_dir(&target) {
                return Ok(format!(
                    "Model downloaded successfully. (already present at {})",
                    target.display()
                ));
            }
            let dir = download_sherpa_model(config, archive_url_override, &mut |frac, msg| {
                if let Some(f) = frac {
                    println!("[{:3.0}%] {msg}", f * 100.0);
                } else {
                    println!("{msg}");
                }
            })
            .await?;
            Ok(format!(
                "Model downloaded successfully. ({})",
                dir.display()
            ))
        }
    }
}
