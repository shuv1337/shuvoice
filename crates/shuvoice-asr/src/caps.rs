//! Capability constructors built on [`shuvoice_core`] types.
//!
//! Core owns the capability schema (`emits_partials`, `supports_cancel`, …).
//! This module only supplies the per-backend defaults.
//!
//! `supports_cancel` is **true only** when [`crate::AsrBackend::cancel`] is
//! implemented for that backend. Do not advertise cancel optimistically.

use shuvoice_core::{AsrCapabilities, ExpectedChunking, FinalizationMode};

/// Sherpa streaming (zipformer etc.). Cancel = stream reset (supported).
#[must_use]
pub fn sherpa_streaming_caps() -> AsrCapabilities {
    AsrCapabilities {
        supports_gpu: false, // static in-process binding: CUDA not safely recoverable
        supports_model_download: true,
        wants_raw_audio: false,
        expected_chunking: ExpectedChunking::Streaming,
        finalization_mode: FinalizationMode::LocalStreaming,
        preferred_sample_rate: Some(16_000),
        emits_partials: true,
        supports_cancel: true,
    }
}

/// Sherpa offline_instant (Parakeet default path).
#[must_use]
pub fn sherpa_offline_caps() -> AsrCapabilities {
    AsrCapabilities {
        supports_gpu: false,
        supports_model_download: true,
        wants_raw_audio: false,
        expected_chunking: ExpectedChunking::Streaming,
        finalization_mode: FinalizationMode::OfflineInstant,
        preferred_sample_rate: Some(16_000),
        emits_partials: false,
        supports_cancel: true, // cancel aborts in-flight generation via gen counter
    }
}

/// NeMo streaming (worker-backed). Cancel only if worker advertises it.
#[must_use]
pub fn nemo_caps() -> AsrCapabilities {
    AsrCapabilities {
        supports_gpu: true,
        supports_model_download: true,
        wants_raw_audio: true,
        expected_chunking: ExpectedChunking::Streaming,
        finalization_mode: FinalizationMode::LocalStreaming,
        preferred_sample_rate: Some(16_000),
        emits_partials: true,
        supports_cancel: false, // until worker cancel is wired
    }
}

/// Moonshine windowed (worker-backed).
#[must_use]
pub fn moonshine_caps() -> AsrCapabilities {
    AsrCapabilities {
        supports_gpu: true,
        supports_model_download: false,
        wants_raw_audio: true,
        expected_chunking: ExpectedChunking::Windowed,
        finalization_mode: FinalizationMode::LocalStreaming,
        preferred_sample_rate: Some(16_000),
        emits_partials: true,
        supports_cancel: false,
    }
}

/// OpenAI Realtime manual-commit. No cancel API wired (clear buffer is reset).
#[must_use]
pub fn openai_realtime_caps() -> AsrCapabilities {
    AsrCapabilities {
        supports_gpu: false,
        supports_model_download: false,
        wants_raw_audio: true,
        expected_chunking: ExpectedChunking::Streaming,
        finalization_mode: FinalizationMode::RemoteManualCommit,
        preferred_sample_rate: Some(24_000),
        emits_partials: true,
        supports_cancel: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wants_raw_matrix() {
        assert!(!sherpa_streaming_caps().wants_raw_audio);
        assert!(!sherpa_offline_caps().wants_raw_audio);
        assert!(nemo_caps().wants_raw_audio);
        assert!(moonshine_caps().wants_raw_audio);
        assert!(openai_realtime_caps().wants_raw_audio);
    }

    #[test]
    fn finalization_modes() {
        assert_eq!(
            sherpa_offline_caps().finalization_mode,
            FinalizationMode::OfflineInstant
        );
        assert_eq!(
            openai_realtime_caps().finalization_mode,
            FinalizationMode::RemoteManualCommit
        );
        assert_eq!(
            moonshine_caps().expected_chunking,
            ExpectedChunking::Windowed
        );
        assert!(!sherpa_offline_caps().emits_partials);
    }

    #[test]
    fn cancel_flags_honest() {
        assert!(sherpa_streaming_caps().supports_cancel);
        assert!(!openai_realtime_caps().supports_cancel);
        assert!(!nemo_caps().supports_cancel);
    }

    #[test]
    fn sample_rates_explicit() {
        assert_eq!(sherpa_offline_caps().preferred_sample_rate, Some(16_000));
        assert_eq!(openai_realtime_caps().preferred_sample_rate, Some(24_000));
    }
}
