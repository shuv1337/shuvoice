//! Shared error types for headless core policy.

use std::path::PathBuf;

use thiserror::Error;

/// Errors produced by pure core validation and configuration I/O.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("{0}")]
    Validation(String),

    #[error("config I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid TOML in {path}: {source}")]
    TomlParse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("TOML serialization error: {0}")]
    TomlSerialize(String),

    #[error("{0}")]
    Migration(String),

    #[error("{0}")]
    OverlayState(String),
}

impl CoreError {
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }

    pub fn migration(msg: impl Into<String>) -> Self {
        Self::Migration(msg.into())
    }
}

/// Result alias for core operations.
pub type CoreResult<T> = Result<T, CoreError>;
