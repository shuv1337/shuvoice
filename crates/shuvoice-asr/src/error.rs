//! Typed ASR errors and classification.

use std::time::Duration;

use thiserror::Error;

/// Stable error classes used by the circuit breaker and UI toasts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsrErrorClass {
    Dependency,
    StartupCompat,
    Decode,
    CudaOom,
    RemoteTimeout,
    Cancelled,
    Unsupported,
    Transport,
    Protocol,
    Internal,
}

impl AsrErrorClass {
    /// Whether a failure of this class should advance the consecutive-failure counter.
    ///
    /// CUDA OOM is handled specially by the orchestrator when CPU fallback succeeds;
    /// the class itself still reports `true` so unhandled OOM counts.
    pub const fn counts_for_breaker_by_default(self) -> bool {
        !matches!(self, Self::Cancelled | Self::Unsupported)
    }
}

/// Fallible ASR operations.
#[derive(Debug, Error)]
pub enum AsrError {
    #[error("missing dependency: {0}")]
    Dependency(String),

    #[error("startup compatibility: {0}")]
    StartupCompat(String),

    #[error("decode failed: {0}")]
    Decode(String),

    #[error("CUDA out-of-memory: {0}")]
    CudaOom(String),

    #[error("remote timeout after {0:?}: {1}")]
    RemoteTimeout(Duration, String),

    #[error("cancelled: {0}")]
    Cancelled(String),

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    #[error("worker transport error: {0}")]
    Transport(String),

    #[error("worker protocol error: {0}")]
    Protocol(String),

    #[error("internal ASR error: {0}")]
    Internal(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl AsrError {
    pub fn class(&self) -> AsrErrorClass {
        match self {
            Self::Dependency(_) => AsrErrorClass::Dependency,
            Self::StartupCompat(_) => AsrErrorClass::StartupCompat,
            Self::Decode(_) => AsrErrorClass::Decode,
            Self::CudaOom(_) => AsrErrorClass::CudaOom,
            Self::RemoteTimeout(_, _) => AsrErrorClass::RemoteTimeout,
            Self::Cancelled(_) => AsrErrorClass::Cancelled,
            Self::Unsupported(_) => AsrErrorClass::Unsupported,
            Self::Transport(_) => AsrErrorClass::Transport,
            Self::Protocol(_) => AsrErrorClass::Protocol,
            Self::Internal(_) | Self::Io(_) | Self::Json(_) => AsrErrorClass::Internal,
        }
    }

    pub fn counts_for_breaker(&self) -> bool {
        self.class().counts_for_breaker_by_default()
    }

    pub fn dependency(msg: impl Into<String>) -> Self {
        Self::Dependency(msg.into())
    }

    pub fn startup(msg: impl Into<String>) -> Self {
        Self::StartupCompat(msg.into())
    }

    pub fn decode(msg: impl Into<String>) -> Self {
        Self::Decode(msg.into())
    }

    pub fn cuda_oom(msg: impl Into<String>) -> Self {
        Self::CudaOom(msg.into())
    }

    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }

    pub fn transport(msg: impl Into<String>) -> Self {
        Self::Transport(msg.into())
    }

    pub fn protocol(msg: impl Into<String>) -> Self {
        Self::Protocol(msg.into())
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    /// Reclassify a foreign error string, detecting CUDA OOM markers when present.
    pub fn from_runtime_message(msg: impl AsRef<str>) -> Self {
        let text = msg.as_ref();
        if crate::cuda_oom::looks_like_cuda_oom_str(text) {
            Self::cuda_oom(text.to_owned())
        } else {
            Self::decode(text.to_owned())
        }
    }
}

/// Result of a session-wide GPU→CPU recovery attempt.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackOutcome {
    /// Recognizer reloaded on CPU; subsequent calls must not count the OOM.
    Applied { detail: String },
    /// Already on CPU, already fell back, or backend has no GPU path.
    NotApplicable { detail: String },
    /// Fallback was attempted but reload failed.
    Failed { detail: String },
}

impl FallbackOutcome {
    pub const fn applied(&self) -> bool {
        matches!(self, Self::Applied { .. })
    }

    pub fn detail(&self) -> &str {
        match self {
            Self::Applied { detail } | Self::NotApplicable { detail } | Self::Failed { detail } => {
                detail
            }
        }
    }
}

pub type AsrResult<T> = Result<T, AsrError>;
