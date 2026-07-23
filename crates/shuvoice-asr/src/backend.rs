//! Core `AsrBackend` trait.

use std::time::Duration;

use async_trait::async_trait;
use shuvoice_core::{AsrBackendKind, AsrCapabilities};

use crate::error::{AsrResult, FallbackOutcome};

/// Progress callback signature for model download / load.
pub type ProgressFn<'a> = dyn FnMut(Option<f32>, &str) + Send + 'a;

/// Common runtime surface used by the ShuVoice ASR worker loop.
///
/// Methods take `&mut self` so FFI recognizer ownership stays single-threaded
/// on the calling task (or a dedicated blocking worker). Do not share one
/// backend across tasks without external serialization.
#[async_trait]
pub trait AsrBackend: Send {
    fn capabilities(&self) -> &AsrCapabilities;

    /// Registry key for this backend instance.
    fn backend_id(&self) -> AsrBackendKind;

    fn native_chunk_samples(&self) -> usize;

    /// Required capture sample rate in Hz, when fixed by the backend.
    fn required_sample_rate_hz(&self) -> Option<u32> {
        self.capabilities().preferred_sample_rate
    }

    /// Optional step counter for diagnostics (NeMo/Moonshine).
    fn debug_step(&self) -> Option<u64> {
        None
    }

    async fn load(&mut self, progress: &mut ProgressFn<'_>) -> AsrResult<()>;

    async fn reset(&mut self) -> AsrResult<()>;

    /// Process one native chunk; return cumulative transcript text.
    async fn process_chunk(&mut self, pcm_mono_f32: &[f32]) -> AsrResult<String>;

    /// Optional finalization for remote/manual-commit backends.
    ///
    /// When `timeout` is `None`, backends use their config default.
    async fn finish_utterance(&mut self, timeout: Option<Duration>) -> AsrResult<String> {
        let _ = timeout;
        Ok(String::new())
    }

    /// One-shot offline utterance decode (Sherpa offline_instant).
    async fn process_utterance(&mut self, pcm_mono_f32: &[f32]) -> AsrResult<String> {
        let _ = pcm_mono_f32;
        Err(crate::error::AsrError::unsupported(
            "process_utterance is not supported by this backend",
        ))
    }

    /// Best-effort cancel of in-flight work. Default: unsupported.
    async fn cancel(&mut self) -> AsrResult<()> {
        Err(crate::error::AsrError::unsupported(
            "cancel is not supported by this backend",
        ))
    }

    /// Session-wide CUDA→CPU recovery. Default: not applicable.
    ///
    /// Only advertise [`FallbackOutcome::Applied`] when the runtime can observe
    /// a recoverable GPU fault without process abort.
    async fn try_fallback_to_cpu(&mut self) -> AsrResult<FallbackOutcome> {
        Ok(FallbackOutcome::NotApplicable {
            detail: "backend has no GPU fallback path".into(),
        })
    }

    /// True after a successful CPU fallback this session.
    fn cpu_fallback_applied(&self) -> bool {
        false
    }

    async fn shutdown(&mut self) -> AsrResult<()> {
        Ok(())
    }
}

/// Object-safe helper for dyn dispatch erase of concrete backends.
pub type DynAsrBackend = dyn AsrBackend;
