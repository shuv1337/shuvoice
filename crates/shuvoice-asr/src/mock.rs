//! In-process mock backend for orchestrator tests.

use std::time::Duration;

use async_trait::async_trait;
use shuvoice_core::{AsrBackendKind, AsrCapabilities};

use crate::backend::{AsrBackend, ProgressFn};
use crate::caps::{
    moonshine_caps, nemo_caps, openai_realtime_caps, sherpa_offline_caps, sherpa_streaming_caps,
};
use crate::error::{AsrError, AsrResult, FallbackOutcome};

/// Configurable mock used by unit tests and factory injection.
pub struct MockAsrBackend {
    id: AsrBackendKind,
    caps: AsrCapabilities,
    native_chunk_samples: usize,
    pub loaded: bool,
    pub chunks: Vec<Vec<f32>>,
    pub last_text: String,
    pub fail_next: Option<AsrError>,
    pub utterance_text: String,
    cpu_fallback_applied: bool,
    pub allow_cpu_fallback: bool,
    step: u64,
}

impl MockAsrBackend {
    pub fn new(id: AsrBackendKind, caps: AsrCapabilities, native_chunk_samples: usize) -> Self {
        Self {
            id,
            caps,
            native_chunk_samples,
            loaded: false,
            chunks: Vec::new(),
            last_text: String::new(),
            fail_next: None,
            utterance_text: "mock final".into(),
            cpu_fallback_applied: false,
            allow_cpu_fallback: true,
            step: 0,
        }
    }

    pub fn sherpa_offline() -> Self {
        Self::new(AsrBackendKind::Sherpa, sherpa_offline_caps(), 1600)
    }

    pub fn sherpa_streaming() -> Self {
        Self::new(AsrBackendKind::Sherpa, sherpa_streaming_caps(), 1600)
    }

    pub fn openai() -> Self {
        Self::new(AsrBackendKind::OpenaiRealtime, openai_realtime_caps(), 2400)
    }

    pub fn nemo() -> Self {
        Self::new(AsrBackendKind::Nemo, nemo_caps(), 1280)
    }

    pub fn moonshine() -> Self {
        Self::new(AsrBackendKind::Moonshine, moonshine_caps(), 1600)
    }
}

#[async_trait]
impl AsrBackend for MockAsrBackend {
    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }

    fn backend_id(&self) -> AsrBackendKind {
        self.id
    }

    fn native_chunk_samples(&self) -> usize {
        self.native_chunk_samples
    }

    fn debug_step(&self) -> Option<u64> {
        Some(self.step)
    }

    fn cpu_fallback_applied(&self) -> bool {
        self.cpu_fallback_applied
    }

    async fn load(&mut self, progress: &mut ProgressFn<'_>) -> AsrResult<()> {
        progress(Some(1.0), "mock ready");
        self.loaded = true;
        Ok(())
    }

    async fn reset(&mut self) -> AsrResult<()> {
        if !self.loaded {
            return Err(AsrError::internal("mock not loaded"));
        }
        self.chunks.clear();
        self.last_text.clear();
        self.step = 0;
        Ok(())
    }

    async fn process_chunk(&mut self, pcm_mono_f32: &[f32]) -> AsrResult<String> {
        if let Some(err) = self.fail_next.take() {
            return Err(err);
        }
        if !self.loaded {
            return Err(AsrError::internal("mock not loaded"));
        }
        self.chunks.push(pcm_mono_f32.to_vec());
        self.step += 1;
        if pcm_mono_f32.iter().any(|s| *s != 0.0) {
            if self.last_text.is_empty() {
                self.last_text = "hello".into();
            } else if !self.last_text.contains("world") {
                self.last_text.push_str(" world");
            }
        }
        Ok(self.last_text.clone())
    }

    async fn process_utterance(&mut self, pcm_mono_f32: &[f32]) -> AsrResult<String> {
        if let Some(err) = self.fail_next.take() {
            return Err(err);
        }
        if !self.loaded {
            return Err(AsrError::internal("mock not loaded"));
        }
        self.chunks.push(pcm_mono_f32.to_vec());
        Ok(self.utterance_text.clone())
    }

    async fn finish_utterance(&mut self, _timeout: Option<Duration>) -> AsrResult<String> {
        if let Some(err) = self.fail_next.take() {
            return Err(err);
        }
        Ok(if self.last_text.is_empty() {
            "committed".into()
        } else {
            self.last_text.clone()
        })
    }

    async fn cancel(&mut self) -> AsrResult<()> {
        self.chunks.clear();
        self.last_text.clear();
        Ok(())
    }

    async fn try_fallback_to_cpu(&mut self) -> AsrResult<FallbackOutcome> {
        if self.cpu_fallback_applied {
            return Ok(FallbackOutcome::NotApplicable {
                detail: "already applied".into(),
            });
        }
        if !self.allow_cpu_fallback {
            return Ok(FallbackOutcome::NotApplicable {
                detail: "disabled".into(),
            });
        }
        self.cpu_fallback_applied = true;
        Ok(FallbackOutcome::Applied {
            detail: "mock switched to cpu".into(),
        })
    }
}
