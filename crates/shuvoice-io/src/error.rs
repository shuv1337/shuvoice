//! Error types for platform I/O adapters.

use std::io;
use std::time::Duration;

use thiserror::Error;

/// Process invocation failures (secrets must never appear in Display).
#[derive(Debug, Error)]
pub enum ProcessError {
    #[error("{program} failed with exit code {code}")]
    ExitCode { program: String, code: i32 },

    #[error("{program} timed out after {timeout:?}")]
    Timeout { program: String, timeout: Duration },

    #[error("{program} I/O error: {source}")]
    Io {
        program: String,
        #[source]
        source: io::Error,
    },

    #[error("{program} not found on PATH")]
    NotFound { program: String },

    #[error("{program} output exceeded {limit} bytes")]
    OutputTooLarge { program: String, limit: usize },
}

/// Selection capture failures.
#[derive(Debug, Error)]
pub enum SelectionError {
    #[error("No clipboard text found")]
    EmptyClipboard,

    #[error("No selected text found (primary selection and clipboard were empty)")]
    EmptySelection,

    #[error(transparent)]
    Process(#[from] ProcessError),
}

/// Audio capture / helper failures.
#[derive(Debug, Error)]
pub enum AudioError {
    #[error(
        "fallback_sample_rate must be an integer multiple of sample_rate (got {fallback} and {sample_rate})"
    )]
    NonIntegerResampleRatio { fallback: u32, sample_rate: u32 },

    #[error("audio device error: {0}")]
    Device(String),

    #[error("audio stream error: {0}")]
    Stream(String),

    #[error("audio feature disabled (build without `audio`)")]
    FeatureDisabled,

    #[error(transparent)]
    Io(#[from] io::Error),
}

/// Generic I/O crate error.
#[derive(Debug, Error)]
pub enum IoError {
    #[error(transparent)]
    Process(#[from] ProcessError),

    #[error(transparent)]
    Selection(#[from] SelectionError),

    #[error(transparent)]
    Inject(#[from] crate::inject::InjectError),

    #[error(transparent)]
    Audio(#[from] AudioError),

    #[error(transparent)]
    Io(#[from] io::Error),

    #[error("{0}")]
    Other(String),
}
