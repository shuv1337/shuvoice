//! Sherpa offline + streaming backend.
//!
//! # Threading
//!
//! The official `sherpa-onnx` C API is invoked only inside
//! [`tokio::task::spawn_blocking`] (or a single-thread runtime's blocking pool)
//! so multi-threaded Tokio workers are never stuck in native decode. The
//! recognizer slot is moved into the blocking task and moved back — single
//! owner at all times.
//!
//! # CUDA honesty
//!
//! The static in-process binding does **not** surface decode-time CUDA faults as
//! Rust `Result`s (native code may abort). Therefore:
//! - `sherpa_provider = "cuda"` **fails closed at load** with an actionable error
//! - [`try_fallback_to_cpu`] always returns [`FallbackOutcome::NotApplicable`]
//! - `capabilities.supports_gpu` is `false` for this build
//!
//! GPU isolation requires a restartable worker/subprocess, not in-process EP.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use async_trait::async_trait;
use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OfflineStream, OfflineTransducerModelConfig,
    OnlineRecognizer, OnlineRecognizerConfig, OnlineStream, OnlineTransducerModelConfig,
};
use shuvoice_core::{AsrBackendKind, AsrCapabilities, ComputeProvider, ResolvedSherpaDecodeMode};

use crate::backend::{AsrBackend, ProgressFn};
use crate::caps::{sherpa_offline_caps, sherpa_streaming_caps};
use crate::config::AsrConfig;
use crate::error::{AsrError, AsrResult, FallbackOutcome};
use crate::pcm;
use crate::sherpa::download::{DEFAULT_MAX_DOWNLOAD_BYTES, DownloadOptions, download_model};
use crate::sherpa::model::{
    ModelFiles, collect_model_files, is_model_dir_complete, resolve_model_dir,
};
use crate::sherpa::parakeet::{looks_like_parakeet_config, parakeet_streaming_startup_error};

/// Required capture rate for all Sherpa paths in this crate.
pub const SHERPA_REQUIRED_SAMPLE_RATE_HZ: u32 = 16_000;

enum RecognizerSlot {
    Offline(OfflineRecognizer),
    Online {
        recognizer: OnlineRecognizer,
        stream: Option<OnlineStream>,
    },
}

// SAFETY: we never share a slot across threads concurrently; ownership moves
// through spawn_blocking one task at a time.
unsafe impl Send for RecognizerSlot {}

/// Sherpa-ONNX backend.
pub struct SherpaBackend {
    config: AsrConfig,
    caps: AsrCapabilities,
    files: Option<ModelFiles>,
    slot: Option<RecognizerSlot>,
    cancel_download: Option<Arc<AtomicBool>>,
    /// Bumped on reset/cancel/shutdown; blocking ops check after return.
    generation: Arc<AtomicU64>,
}

impl SherpaBackend {
    pub fn new(config: AsrConfig) -> Self {
        let caps = match config.resolved_sherpa_decode_mode() {
            Some(ResolvedSherpaDecodeMode::OfflineInstant) => sherpa_offline_caps(),
            _ => sherpa_streaming_caps(),
        };
        Self {
            config,
            caps,
            files: None,
            slot: None,
            cancel_download: None,
            generation: Arc::new(AtomicU64::new(1)),
        }
    }

    pub fn set_cancel_flag(&mut self, flag: Arc<AtomicBool>) {
        self.cancel_download = Some(flag);
    }

    pub fn is_offline_mode(&self) -> bool {
        self.config.resolved_sherpa_decode_mode() == Some(ResolvedSherpaDecodeMode::OfflineInstant)
    }

    fn bump_generation(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn current_generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    fn reject_cuda_provider(config: &AsrConfig) -> AsrResult<()> {
        if config.core.sherpa_provider == ComputeProvider::Cuda {
            return Err(AsrError::startup(
                "sherpa_provider='cuda' is not supported for the static in-process sherpa-onnx \
                 binding in shuvoice-asr: the C API does not return decode-time GPU faults as \
                 recoverable Results (CUDA OOM may abort the process). Set sherpa_provider='cpu', \
                 or run GPU ASR in a restartable worker/subprocess with crash isolation.",
            ));
        }
        Ok(())
    }

    async fn ensure_model(&mut self, progress: &mut ProgressFn<'_>) -> AsrResult<()> {
        if self.config.core.sample_rate != SHERPA_REQUIRED_SAMPLE_RATE_HZ {
            return Err(AsrError::startup(format!(
                "Sherpa backend requires sample_rate={SHERPA_REQUIRED_SAMPLE_RATE_HZ} \
                 (got {}). Hosts must capture at 16 kHz (see capabilities.preferred_sample_rate).",
                self.config.core.sample_rate
            )));
        }
        Self::reject_cuda_provider(&self.config)?;
        parakeet_streaming_startup_error(&self.config)?;

        let dir = resolve_model_dir(&self.config);
        if !is_model_dir_complete(&dir) {
            let opts = DownloadOptions {
                model_name: self.config.core.sherpa_model_name.clone(),
                target_dir: dir.clone(),
                base_url: self.config.sherpa_release_download_root(),
                archive_url_override: None,
                cancel: self.cancel_download.clone(),
                max_bytes: if self.config.connect.max_download_bytes == 0 {
                    DEFAULT_MAX_DOWNLOAD_BYTES
                } else {
                    self.config.connect.max_download_bytes
                },
                expected_sha256: self.config.connect.sherpa_archive_sha256.clone(),
            };
            download_model(opts, progress).await?;
        }
        if !is_model_dir_complete(&dir) {
            return Err(AsrError::internal(format!(
                "Sherpa model auto-download failed and no valid local model directory is available. \
                 Expected artifacts under: {}",
                dir.display()
            )));
        }
        self.config.core.sherpa_model_dir = Some(dir.display().to_string());
        self.files = Some(collect_model_files(&dir)?);
        Ok(())
    }

    fn build_offline(files: &ModelFiles, config: &AsrConfig) -> AsrResult<OfflineRecognizer> {
        let mut offline_cfg = OfflineRecognizerConfig::default();
        offline_cfg.feat_config.sample_rate = config.core.sample_rate as i32;
        offline_cfg.feat_config.feature_dim = 80;
        offline_cfg.decoding_method = Some("greedy_search".into());
        offline_cfg.model_config.transducer = OfflineTransducerModelConfig {
            encoder: Some(files.encoder.display().to_string()),
            decoder: Some(files.decoder.display().to_string()),
            joiner: Some(files.joiner.display().to_string()),
        };
        offline_cfg.model_config.tokens = Some(files.tokens.display().to_string());
        offline_cfg.model_config.num_threads = config.core.sherpa_num_threads as i32;
        offline_cfg.model_config.provider = Some(ComputeProvider::Cpu.as_str().into());
        // Policy: Parakeet offline always uses the NeMo transducer model type.
        // Non-Parakeet offline leaves model_type unset for auto-detect.
        if looks_like_parakeet_config(config) {
            offline_cfg.model_config.model_type = Some("nemo_transducer".into());
        } else {
            offline_cfg.model_config.model_type = None;
        }
        OfflineRecognizer::create(&offline_cfg).ok_or_else(|| {
            AsrError::dependency(
                "Failed to initialize Sherpa offline recognizer. \
                 Ensure sherpa_model_dir points to a supported transducer model and \
                 the sherpa-onnx native library is linked.",
            )
        })
    }

    fn build_online(files: &ModelFiles, config: &AsrConfig) -> AsrResult<OnlineRecognizer> {
        let use_nemo = looks_like_parakeet_config(config);
        let mut online_cfg = OnlineRecognizerConfig::default();
        online_cfg.feat_config.sample_rate = config.core.sample_rate as i32;
        online_cfg.feat_config.feature_dim = 80;
        online_cfg.decoding_method = Some("greedy_search".into());
        online_cfg.model_config.transducer = OnlineTransducerModelConfig {
            encoder: Some(files.encoder.display().to_string()),
            decoder: Some(files.decoder.display().to_string()),
            joiner: Some(files.joiner.display().to_string()),
        };
        online_cfg.model_config.tokens = Some(files.tokens.display().to_string());
        online_cfg.model_config.num_threads = config.core.sherpa_num_threads as i32;
        online_cfg.model_config.provider = Some(ComputeProvider::Cpu.as_str().into());
        // Policy: streaming Parakeet only when gate allows; then nemo_transducer.
        if use_nemo {
            online_cfg.model_config.model_type = Some("nemo_transducer".into());
        }
        OnlineRecognizer::create(&online_cfg).ok_or_else(|| {
            AsrError::dependency(
                "Failed to initialize Sherpa streaming recognizer. \
                 Ensure sherpa_model_dir points to a supported transducer model and \
                 the sherpa-onnx native library is linked.",
            )
        })
    }

    fn load_recognizers_sync(
        files: &ModelFiles,
        config: &AsrConfig,
        offline: bool,
    ) -> AsrResult<RecognizerSlot> {
        if offline {
            Ok(RecognizerSlot::Offline(Self::build_offline(files, config)?))
        } else {
            let rec = Self::build_online(files, config)?;
            let stream = rec.create_stream();
            Ok(RecognizerSlot::Online {
                recognizer: rec,
                stream: Some(stream),
            })
        }
    }

    fn offline_decode_sync(
        slot: &RecognizerSlot,
        samples: &[f32],
        sample_rate: u32,
        max_sec: f64,
    ) -> AsrResult<String> {
        let RecognizerSlot::Offline(rec) = slot else {
            return Err(AsrError::internal("offline recognizer not loaded"));
        };
        let capped = pcm::truncate_trailing(samples, sample_rate, max_sec);
        if capped.len() < samples.len() {
            tracing::warn!(
                original = samples.len(),
                capped = capped.len(),
                "Sherpa offline utterance truncated to trailing window"
            );
        }
        let stream: OfflineStream = rec.create_stream();
        stream.accept_waveform(sample_rate as i32, capped);
        rec.decode(&stream);
        Ok(stream
            .get_result()
            .map(|r| r.text.trim().to_owned())
            .unwrap_or_default())
    }

    fn online_process_sync(
        slot: &mut RecognizerSlot,
        samples: &[f32],
        sample_rate: u32,
    ) -> AsrResult<String> {
        let RecognizerSlot::Online { recognizer, stream } = slot else {
            return Err(AsrError::internal("online recognizer not loaded"));
        };
        let stream = stream
            .as_ref()
            .ok_or_else(|| AsrError::internal("online stream missing; call reset()"))?;
        stream.accept_waveform(sample_rate as i32, samples);
        while recognizer.is_ready(stream) {
            recognizer.decode(stream);
        }
        Ok(recognizer
            .get_result(stream)
            .map(|r| r.text.trim().to_owned())
            .unwrap_or_default())
    }

    fn online_reset_sync(slot: &mut RecognizerSlot) -> AsrResult<()> {
        match slot {
            RecognizerSlot::Offline(_) => Ok(()),
            RecognizerSlot::Online { recognizer, stream } => {
                *stream = Some(recognizer.create_stream());
                Ok(())
            }
        }
    }

    /// Run `op` on the recognizer slot inside `spawn_blocking`, preserving ownership.
    async fn with_slot_blocking<T, F>(&mut self, op: F) -> AsrResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Option<RecognizerSlot>) -> AsrResult<T> + Send + 'static,
    {
        let gen_before = self.current_generation();
        let mut slot = self.slot.take();
        let result = tokio::task::spawn_blocking(move || {
            let out = op(&mut slot);
            (out, slot)
        })
        .await
        .map_err(|e| AsrError::internal(format!("sherpa blocking task join failed: {e}")))?;

        let (out, slot) = result;
        // If generation advanced (cancel/reset/shutdown), drop returned slot.
        if self.current_generation() != gen_before {
            drop(slot);
            return Err(AsrError::Cancelled(
                "sherpa operation cancelled (generation advanced)".into(),
            ));
        }
        self.slot = slot;
        out
    }
}

#[async_trait]
impl AsrBackend for SherpaBackend {
    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }

    fn backend_id(&self) -> AsrBackendKind {
        AsrBackendKind::Sherpa
    }

    fn native_chunk_samples(&self) -> usize {
        self.config.sherpa_native_chunk_samples()
    }

    fn required_sample_rate_hz(&self) -> Option<u32> {
        Some(SHERPA_REQUIRED_SAMPLE_RATE_HZ)
    }

    fn cpu_fallback_applied(&self) -> bool {
        false
    }

    async fn load(&mut self, progress: &mut ProgressFn<'_>) -> AsrResult<()> {
        progress(None, "Validating Sherpa model…");
        self.ensure_model(progress).await?;
        progress(Some(0.98), "Initializing Sherpa recognizer…");

        let files = self
            .files
            .clone()
            .ok_or_else(|| AsrError::internal("model files missing"))?;
        // Clone only the fields build_* needs — AsrConfig is large but Send.
        let config = self.config.clone();
        let offline = self.is_offline_mode();
        let gen_before = self.current_generation();

        let slot = tokio::task::spawn_blocking(move || {
            Self::load_recognizers_sync(&files, &config, offline)
        })
        .await
        .map_err(|e| AsrError::internal(format!("sherpa load join failed: {e}")))??;

        if self.current_generation() != gen_before {
            return Err(AsrError::Cancelled("sherpa load cancelled".into()));
        }
        self.slot = Some(slot);
        self.reset().await?;
        progress(Some(1.0), "Sherpa model ready");
        Ok(())
    }

    async fn reset(&mut self) -> AsrResult<()> {
        self.bump_generation();
        if self.slot.is_none() {
            return Err(AsrError::internal(
                "ASR backend is not loaded. Call load() first.",
            ));
        }
        self.with_slot_blocking(|slot| {
            let Some(s) = slot.as_mut() else {
                return Err(AsrError::internal("ASR backend is not loaded"));
            };
            Self::online_reset_sync(s)
        })
        .await
    }

    async fn process_chunk(&mut self, pcm_mono_f32: &[f32]) -> AsrResult<String> {
        if self.is_offline_mode() {
            return Err(AsrError::unsupported(
                "process_chunk() is not supported in offline instant mode. \
                 Use process_utterance() instead.",
            ));
        }
        if pcm_mono_f32.iter().any(|s| !s.is_finite()) {
            return Err(AsrError::unsupported(
                "Sherpa rejects non-finite PCM samples",
            ));
        }
        let samples = pcm_mono_f32.to_vec();
        let sr = self.config.core.sample_rate;
        self.with_slot_blocking(move |slot| {
            let Some(s) = slot.as_mut() else {
                return Err(AsrError::internal("online recognizer not loaded"));
            };
            Self::online_process_sync(s, &samples, sr)
        })
        .await
    }

    async fn process_utterance(&mut self, pcm_mono_f32: &[f32]) -> AsrResult<String> {
        if !self.is_offline_mode() {
            return Err(AsrError::unsupported(
                "process_utterance() is only supported in offline instant mode. \
                 Use process_chunk() for streaming mode.",
            ));
        }
        if pcm_mono_f32.iter().any(|s| !s.is_finite()) {
            return Err(AsrError::unsupported(
                "Sherpa rejects non-finite PCM samples",
            ));
        }
        let samples = pcm_mono_f32.to_vec();
        let sr = self.config.core.sample_rate;
        let max_sec = self.config.core.sherpa_offline_max_utterance_sec;
        self.with_slot_blocking(move |slot| {
            let Some(s) = slot.as_ref() else {
                return Err(AsrError::internal("offline recognizer not loaded"));
            };
            Self::offline_decode_sync(s, &samples, sr, max_sec)
        })
        .await
    }

    async fn cancel(&mut self) -> AsrResult<()> {
        // Invalidate in-flight blocking ops; reset stream for next utterance.
        self.bump_generation();
        if self.slot.is_some() {
            let _ = self
                .with_slot_blocking(|slot| {
                    if let Some(s) = slot.as_mut() {
                        Self::online_reset_sync(s)?;
                    }
                    Ok(())
                })
                .await;
        }
        Ok(())
    }

    async fn try_fallback_to_cpu(&mut self) -> AsrResult<FallbackOutcome> {
        // Honest: in-process C API cannot report decode-time CUDA OOM as Result.
        Ok(FallbackOutcome::NotApplicable {
            detail: "in-process sherpa-onnx does not surface decode-time CUDA failures as \
                     recoverable errors (process may abort). Use sherpa_provider='cpu' or a \
                     restartable GPU worker/subprocess."
                .into(),
        })
    }

    async fn shutdown(&mut self) -> AsrResult<()> {
        self.bump_generation();
        // Drop slot on blocking pool so destructors don't run on a Tokio worker.
        if let Some(slot) = self.slot.take() {
            let _ = tokio::task::spawn_blocking(move || drop(slot)).await;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::test_config;
    use shuvoice_core::{AsrBackendKind, SherpaDecodeMode};
    use std::time::Duration;

    #[test]
    fn cuda_provider_rejected() {
        let mut cfg = test_config(AsrBackendKind::Sherpa);
        cfg.core.sherpa_provider = ComputeProvider::Cuda;
        cfg.core.validate().unwrap();
        let err = SherpaBackend::reject_cuda_provider(&cfg).unwrap_err();
        assert!(err.to_string().contains("cuda"));
        assert!(err.to_string().contains("cpu"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn spawn_blocking_does_not_starve_current_thread_runtime() {
        // Regression: native work must not block the single-thread scheduler.
        use std::sync::atomic::{AtomicBool, Ordering};
        let flag = Arc::new(AtomicBool::new(false));
        let flag2 = Arc::clone(&flag);

        let ticker = tokio::spawn(async move {
            for _ in 0..20 {
                tokio::task::yield_now().await;
            }
            flag2.store(true, Ordering::SeqCst);
        });

        // Simulate a blocking sherpa-sized chunk of work off-worker.
        let blocker = tokio::task::spawn_blocking(|| {
            std::thread::sleep(Duration::from_millis(50));
            42
        });

        let (b, ()) = tokio::join!(blocker, async {
            ticker.await.unwrap();
        });
        assert_eq!(b.unwrap(), 42);
        assert!(
            flag.load(Ordering::SeqCst),
            "current_thread runtime starved during blocking work"
        );
    }

    #[test]
    fn offline_parakeet_uses_nemo_transducer_policy() {
        let mut cfg = test_config(AsrBackendKind::Sherpa);
        cfg.core.sherpa_model_name = shuvoice_core::PARAKEET_TDT_V3_INT8_MODEL_NAME.into();
        cfg.core.sherpa_decode_mode = SherpaDecodeMode::OfflineInstant;
        cfg.core.validate().unwrap();
        assert!(looks_like_parakeet_config(&cfg));
    }
}
