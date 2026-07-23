//! Session runtime errors.

use thiserror::Error;

/// Errors produced by the headless session actor.
#[derive(Debug, Error)]
pub enum AppError {
    #[error("{0}")]
    Message(String),

    #[error("session shut down")]
    ShutDown,

    #[error("command channel full")]
    CommandQueueFull,

    #[error("event channel closed")]
    EventClosed,

    #[error("ASR disabled")]
    AsrDisabled,

    #[error("ASR thread dead")]
    AsrThreadDead,

    #[error("TTS not available")]
    TtsNotAvailable,

    #[error("TTS disabled")]
    TtsDisabled,

    #[error("timed out waiting for STT processing")]
    SttProcessingTimeout,

    #[error("selection capture failed: {0}")]
    Selection(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl AppError {
    pub fn message(msg: impl Into<String>) -> Self {
        Self::Message(msg.into())
    }
}

pub type AppResult<T> = Result<T, AppError>;
