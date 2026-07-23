//! Parakeet TDT gating (Python startup_errors / window_size probe).

use std::io::Read;
use std::path::Path;

use shuvoice_core::{ResolvedSherpaDecodeMode, is_parakeet_model};

use crate::config::AsrConfig;
use crate::error::{AsrError, AsrResult};
use crate::sherpa::model::pick_model_onnx;

pub fn looks_like_parakeet_name(name: &str) -> bool {
    is_parakeet_model(name)
}

pub fn looks_like_parakeet_config(config: &AsrConfig) -> bool {
    config.is_parakeet()
}

/// Byte-scan encoder ONNX for `window_size` metadata token.
pub fn encoder_has_window_size(encoder: &Path) -> AsrResult<bool> {
    const TOKEN: &[u8] = b"window_size";
    let mut file = std::fs::File::open(encoder)
        .map_err(|e| AsrError::internal(format!("open encoder: {e}")))?;
    let mut buf = [0u8; 1024 * 1024];
    let mut tail = Vec::new();
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| AsrError::internal(format!("read encoder: {e}")))?;
        if n == 0 {
            return Ok(false);
        }
        let mut data = tail.clone();
        data.extend_from_slice(&buf[..n]);
        if data.windows(TOKEN.len()).any(|w| w == TOKEN) {
            return Ok(true);
        }
        let keep = TOKEN.len().saturating_sub(1);
        tail = data[data.len().saturating_sub(keep)..].to_vec();
    }
}

/// Returns an error when Parakeet streaming is blocked, else Ok.
pub fn parakeet_streaming_startup_error(config: &AsrConfig) -> AsrResult<()> {
    if !looks_like_parakeet_config(config) {
        return Ok(());
    }
    let mode = config
        .resolved_sherpa_decode_mode()
        .unwrap_or(ResolvedSherpaDecodeMode::Streaming);
    if mode == ResolvedSherpaDecodeMode::OfflineInstant {
        return Ok(());
    }
    if mode == ResolvedSherpaDecodeMode::Streaming && config.core.sherpa_enable_parakeet_streaming {
        let dir = crate::sherpa::model::resolve_model_dir(config);
        if !dir.is_dir() {
            return Ok(());
        }
        let encoder = match pick_model_onnx(&dir, "encoder") {
            Ok(p) => p,
            Err(e) => {
                return Err(AsrError::startup(format!(
                    "Configured Sherpa model appears to be Parakeet TDT with streaming \
                     override enabled, but the model/runtime combination looks incompatible \
                     with Sherpa online decoding ({e}). Use offline instant mode."
                )));
            }
        };
        match encoder_has_window_size(&encoder) {
            Ok(true) => Ok(()),
            Ok(false) => Err(AsrError::startup(format!(
                "Configured Sherpa model appears to be Parakeet TDT with streaming \
                 override enabled, but the model/runtime combination looks incompatible \
                 with Sherpa online decoding (encoder metadata appears to be missing \
                 'window_size' required by Sherpa online decoder ({}). \
                 Use offline instant mode (instant_mode=true with sherpa_decode_mode='auto', \
                 or sherpa_decode_mode='offline_instant').",
                encoder.display()
            ))),
            Err(e) => Err(AsrError::startup(format!(
                "failed to inspect encoder metadata ({e})"
            ))),
        }
    } else {
        Err(AsrError::startup(
            "Configured Sherpa model appears to be Parakeet TDT, but ShuVoice is \
             configured for streaming mode. By default, Parakeet remains blocked in \
             streaming mode to avoid startup/runtime instability with incompatible model \
             metadata. Use offline instant mode (instant_mode=true with sherpa_decode_mode='auto', \
             or sherpa_decode_mode='offline_instant') for the stable path. \
             To force streaming anyway, set sherpa_enable_parakeet_streaming=true and \
             sherpa_decode_mode='streaming'.",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::test_config;
    use shuvoice_core::{AsrBackendKind, PARAKEET_TDT_V3_INT8_MODEL_NAME, SherpaDecodeMode};
    use tempfile::tempdir;

    #[test]
    fn blocks_parakeet_streaming_by_default() {
        let mut cfg = test_config(AsrBackendKind::Sherpa);
        cfg.core.sherpa_model_name = PARAKEET_TDT_V3_INT8_MODEL_NAME.into();
        cfg.core.sherpa_decode_mode = SherpaDecodeMode::Streaming;
        cfg.core.validate().unwrap();
        assert!(parakeet_streaming_startup_error(&cfg).is_err());
    }

    #[test]
    fn allows_offline_instant() {
        let mut cfg = test_config(AsrBackendKind::Sherpa);
        cfg.core.sherpa_model_name = PARAKEET_TDT_V3_INT8_MODEL_NAME.into();
        cfg.core.sherpa_decode_mode = SherpaDecodeMode::OfflineInstant;
        cfg.core.validate().unwrap();
        assert!(parakeet_streaming_startup_error(&cfg).is_ok());
    }

    #[test]
    fn streaming_enabled_requires_window_size() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("tokens.txt"), "a\n").unwrap();
        std::fs::write(dir.path().join("encoder.onnx"), b"onnx").unwrap();
        std::fs::write(dir.path().join("decoder.onnx"), b"x").unwrap();
        std::fs::write(dir.path().join("joiner.onnx"), b"x").unwrap();
        let mut cfg = test_config(AsrBackendKind::Sherpa);
        cfg.core.sherpa_model_name = "parakeet-tdt".into();
        cfg.core.sherpa_model_dir = Some(dir.path().display().to_string());
        cfg.core.sherpa_decode_mode = SherpaDecodeMode::Streaming;
        cfg.core.sherpa_enable_parakeet_streaming = true;
        cfg.core.validate().unwrap();
        let err = parakeet_streaming_startup_error(&cfg).unwrap_err();
        assert!(err.to_string().contains("window_size"));
    }

    #[test]
    fn streaming_enabled_with_window_size_ok() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("tokens.txt"), "a\n").unwrap();
        std::fs::write(dir.path().join("encoder.onnx"), b"onnx-window_size").unwrap();
        std::fs::write(dir.path().join("decoder.onnx"), b"x").unwrap();
        std::fs::write(dir.path().join("joiner.onnx"), b"x").unwrap();
        let mut cfg = test_config(AsrBackendKind::Sherpa);
        cfg.core.sherpa_model_name = "parakeet-tdt".into();
        cfg.core.sherpa_model_dir = Some(dir.path().display().to_string());
        cfg.core.sherpa_decode_mode = SherpaDecodeMode::Streaming;
        cfg.core.sherpa_enable_parakeet_streaming = true;
        cfg.core.validate().unwrap();
        assert!(parakeet_streaming_startup_error(&cfg).is_ok());
    }
}
