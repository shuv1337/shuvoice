//! Sherpa model paths, status, Parakeet gate, download via `shuvoice-asr`.

use std::path::PathBuf;

use shuvoice_core::{AsrBackendKind, ComputeProvider, Config, data_dir, expand_user_path};

/// Resolve target model directory (core XDG layout). Paths expand `~`.
pub fn sherpa_model_dir(config: &Config) -> PathBuf {
    config
        .sherpa_model_dir
        .as_ref()
        .map(expand_user_path)
        .unwrap_or_else(|| {
            data_dir()
                .join("models")
                .join("sherpa")
                .join(config.sherpa_model_name.trim())
        })
}

pub fn is_complete_sherpa_dir(dir: &std::path::Path) -> bool {
    #[cfg(feature = "asr-sherpa")]
    {
        shuvoice_asr::is_model_dir_complete(dir)
    }
    #[cfg(not(feature = "asr-sherpa"))]
    {
        if !dir.is_dir() || !dir.join("tokens.txt").is_file() {
            return false;
        }
        for stem in ["encoder", "decoder", "joiner"] {
            let ok = std::fs::read_dir(dir)
                .map(|rd| {
                    rd.filter_map(|e| e.ok()).any(|e| {
                        let name = e.file_name().to_string_lossy().to_string();
                        name.starts_with(stem) && name.ends_with(".onnx")
                    })
                })
                .unwrap_or(false);
            if !ok {
                return false;
            }
        }
        true
    }
}

pub fn model_status_line(config: &Config) -> String {
    match config.asr_backend {
        AsrBackendKind::Sherpa => {
            let dir = sherpa_model_dir(config);
            if is_complete_sherpa_dir(&dir) {
                format!("present ({})", dir.display())
            } else {
                format!(
                    "missing ({}); will download '{}' on setup/model download",
                    dir.display(),
                    config.sherpa_model_name
                )
            }
        }
        AsrBackendKind::Nemo => format!(
            "worker/HF cache on first load (model_name={})",
            config.model_name
        ),
        AsrBackendKind::Moonshine => {
            "worker/HF cache on first load (optional external worker)".into()
        }
        AsrBackendKind::OpenaiRealtime => "cloud backend; no local model download required".into(),
    }
}

/// Download Sherpa model using hardened `shuvoice-asr` downloader.
#[allow(clippy::needless_return)]
pub async fn download_sherpa_model(
    config: &Config,
    archive_url_override: Option<String>,
    progress: &mut (dyn FnMut(Option<f32>, &str) + Send),
) -> Result<PathBuf, String> {
    let target = sherpa_model_dir(config);
    if is_complete_sherpa_dir(&target) {
        progress(Some(1.0), "Sherpa model already available");
        return Ok(target);
    }

    #[cfg(feature = "asr-sherpa")]
    {
        use shuvoice_asr::AsrConfig;
        use shuvoice_asr::sherpa::{DownloadOptions, download_model};

        let asr_cfg = AsrConfig::from_core(config.clone()).map_err(|e| e.to_string())?;
        let opts = DownloadOptions {
            model_name: config.sherpa_model_name.clone(),
            target_dir: target.clone(),
            base_url: asr_cfg.sherpa_release_download_root(),
            archive_url_override,
            cancel: None,
            max_bytes: asr_cfg.connect.max_download_bytes,
            expected_sha256: asr_cfg.connect.sherpa_archive_sha256.clone(),
        };
        download_model(opts, progress)
            .await
            .map_err(|e| e.to_string())?;
        if !is_complete_sherpa_dir(&target) {
            return Err(format!(
                "Sherpa model download completed but artifacts incomplete: {}",
                target.display()
            ));
        }
        Ok(target)
    }

    #[cfg(not(feature = "asr-sherpa"))]
    {
        let _ = archive_url_override;
        let _ = progress;
        Err(format!(
            "Sherpa model download requires --features asr-sherpa (model={})",
            config.sherpa_model_name
        ))
    }
}

/// Native static Sherpa does **not** support CUDA EP.
///
/// Fail closed when `sherpa_provider = "cuda"` with actionable CPU guidance.
pub fn sherpa_cuda_provider_errors(config: &Config) -> Vec<String> {
    if config.asr_backend != AsrBackendKind::Sherpa {
        return Vec::new();
    }
    if config.sherpa_provider != ComputeProvider::Cuda {
        return Vec::new();
    }
    vec![
        "Native static Sherpa does not support CUDA (sherpa_provider=cuda is unsupported). Set sherpa_provider = 'cpu' in ~/.config/shuvoice/config.toml. Recommended profile: sherpa_provider='cpu', instant_mode=true, sherpa_decode_mode='offline_instant' (Parakeet int8 is CPU-oriented). There is no automatic GPU→CPU fallback lie in setup/preflight."
            .into(),
    ]
}

/// Honest provider report for setup/preflight logs.
///
/// - `cpu` → requested=cpu effective=cpu
/// - `cuda` → requested=cuda effective=<unsupported> (caller must fail closed)
pub fn format_sherpa_provider_line(config: &Config) -> String {
    match config.sherpa_provider {
        ComputeProvider::Cpu => {
            "[INFO] Sherpa provider: requested=cpu effective=cpu".into()
        }
        ComputeProvider::Cuda => {
            "[INFO] Sherpa provider: requested=cuda effective=unsupported (native static Sherpa is CPU-only; set sherpa_provider='cpu')"
                .into()
        }
    }
}

/// Combined Sherpa runtime gate errors (CUDA provider + Parakeet streaming).
pub fn sherpa_runtime_errors(config: &Config) -> Vec<String> {
    let mut errs = sherpa_cuda_provider_errors(config);
    errs.extend(parakeet_startup_errors(config));
    errs
}

/// Parakeet streaming safety gate via `shuvoice-asr` when available.
pub fn parakeet_startup_errors(config: &Config) -> Vec<String> {
    if config.asr_backend != AsrBackendKind::Sherpa {
        return Vec::new();
    }

    #[cfg(feature = "asr-sherpa")]
    {
        use shuvoice_asr::AsrConfig;
        use shuvoice_asr::sherpa::parakeet_streaming_startup_error;

        let Ok(asr_cfg) = AsrConfig::from_core(config.clone()) else {
            return Vec::new();
        };
        match parakeet_streaming_startup_error(&asr_cfg) {
            Ok(()) => Vec::new(),
            Err(e) => vec![e.to_string()],
        }
    }

    #[cfg(not(feature = "asr-sherpa"))]
    {
        // Without the ASR crate, still block obvious Parakeet+streaming misconfig.
        use shuvoice_core::{ResolvedSherpaDecodeMode, is_parakeet_model};
        if !is_parakeet_model(&config.sherpa_model_name) {
            return Vec::new();
        }
        let mode = config
            .resolved_sherpa_decode_mode()
            .unwrap_or(ResolvedSherpaDecodeMode::Streaming);
        if mode == ResolvedSherpaDecodeMode::OfflineInstant {
            return Vec::new();
        }
        if mode == ResolvedSherpaDecodeMode::Streaming && config.sherpa_enable_parakeet_streaming {
            return Vec::new();
        }
        vec![
            "Configured Sherpa model appears to be Parakeet TDT, but ShuVoice is \
             configured for streaming mode. Use offline instant mode \
             (instant_mode=true / sherpa_decode_mode='offline_instant'), or rebuild \
             with --features asr-sherpa for full runtime gating."
                .into(),
        ]
    }
}

/// Backend-only dependency probe (no config-aware worker discovery).
///
/// Prefer [`asr_dependency_errors_for`] when a full [`Config`] is available so
/// NeMo/Moonshine share production worker discovery + `dependency_errors_for`.
pub fn asr_dependency_errors(backend: AsrBackendKind) -> Vec<String> {
    let mut cfg = Config::default();
    cfg.asr_backend = backend;
    let _ = cfg.validate();
    asr_dependency_errors_for(&cfg)
}

/// Config-aware ASR dependency errors (shared with production composition).
///
/// Uses `compose::worker_runtime` discovery + `shuvoice_asr::dependency_errors_for`
/// so a setup-created venv + bundled workers tree becomes READY. Never persists
/// a worker command into config.
pub fn asr_dependency_errors_for(config: &Config) -> Vec<String> {
    crate::compose::asr_dependency_errors_for(config)
}
