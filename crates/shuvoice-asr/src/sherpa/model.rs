//! Model directory discovery and validation.

use std::path::{Path, PathBuf};

use crate::config::AsrConfig;
use crate::error::{AsrError, AsrResult};

/// Historical default streaming zipformer name (matches core DEFAULT_SHERPA_MODEL_NAME).
pub const DEFAULT_SHERPA_MODEL_NAME: &str = "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06";

#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub dir: PathBuf,
    pub tokens: PathBuf,
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub joiner: PathBuf,
}

pub fn is_model_dir_complete(model_dir: &Path) -> bool {
    if !model_dir.is_dir() {
        return false;
    }
    if !model_dir.join("tokens.txt").is_file() {
        return false;
    }
    for stem in ["encoder", "decoder", "joiner"] {
        if pick_model_onnx(model_dir, stem).is_err() {
            return false;
        }
    }
    true
}

pub fn pick_model_onnx(model_dir: &Path, name: &str) -> AsrResult<PathBuf> {
    let exact = model_dir.join(format!("{name}.onnx"));
    if exact.is_file() {
        return Ok(exact);
    }
    let mut matches: Vec<PathBuf> = std::fs::read_dir(model_dir)
        .map_err(|e| AsrError::internal(format!("read model dir: {e}")))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.starts_with(name) && s.ends_with(".onnx"))
                    .unwrap_or(false)
        })
        .collect();
    matches.sort();
    matches.into_iter().next().ok_or_else(|| {
        AsrError::startup(format!(
            "Sherpa model directory is missing required streaming transducer artifact: {name}*.onnx"
        ))
    })
}

pub fn collect_model_files(model_dir: &Path) -> AsrResult<ModelFiles> {
    let tokens = model_dir.join("tokens.txt");
    if !tokens.is_file() {
        return Err(AsrError::startup(
            "Sherpa model directory is missing required file: tokens.txt",
        ));
    }
    Ok(ModelFiles {
        dir: model_dir.to_path_buf(),
        tokens,
        encoder: pick_model_onnx(model_dir, "encoder")?,
        decoder: pick_model_onnx(model_dir, "decoder")?,
        joiner: pick_model_onnx(model_dir, "joiner")?,
    })
}

/// Resolve configured or default model dir; does not download.
pub fn resolve_model_dir(config: &AsrConfig) -> PathBuf {
    config.sherpa_model_dir_resolved()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn complete_dir_detection() {
        let dir = tempdir().unwrap();
        assert!(!is_model_dir_complete(dir.path()));
        std::fs::write(dir.path().join("tokens.txt"), "a\n").unwrap();
        std::fs::write(dir.path().join("encoder.onnx"), b"x").unwrap();
        std::fs::write(dir.path().join("decoder.onnx"), b"x").unwrap();
        std::fs::write(dir.path().join("joiner.onnx"), b"x").unwrap();
        assert!(is_model_dir_complete(dir.path()));
    }

    #[test]
    fn pick_glob_variant() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("encoder.int8.onnx"), b"x").unwrap();
        let p = pick_model_onnx(dir.path(), "encoder").unwrap();
        assert!(p.ends_with("encoder.int8.onnx"));
    }
}
