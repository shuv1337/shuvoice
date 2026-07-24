//! TTS error types.

use std::time::Duration;

use thiserror::Error;

/// Errors produced by TTS backends, the player, or helpers.
#[derive(Debug, Error)]
pub enum TtsError {
    #[error("{0}")]
    Message(String),

    #[error("TTS text must not be empty")]
    EmptyText,

    #[error("Selected text is too long ({len} chars, max {max})")]
    TextTooLong { len: usize, max: usize },

    #[error("Unknown TTS backend '{name}'. Supported backends: {supported}")]
    UnknownBackend { name: String, supported: String },

    #[error("Missing API key environment variable: {0}")]
    MissingApiKey(String),

    #[error("{0}")]
    SpeedApply(String),

    #[error("{0}")]
    Http(String),

    #[error("{0}")]
    TimedOut(String),

    #[error("{0}")]
    Backend(String),

    #[error("{0}")]
    Process(String),

    #[error("{0}")]
    Audio(String),

    #[error("{0}")]
    Decode(String),

    #[error("{0}")]
    Io(String),

    #[error("operation cancelled")]
    Cancelled,

    #[error("invalid configuration: {0}")]
    Config(String),
}

impl TtsError {
    pub fn message(msg: impl Into<String>) -> Self {
        Self::Message(msg.into())
    }

    pub fn backend(msg: impl Into<String>) -> Self {
        Self::Backend(msg.into())
    }

    pub fn http(msg: impl Into<String>) -> Self {
        Self::Http(msg.into())
    }

    pub fn process(msg: impl Into<String>) -> Self {
        Self::Process(msg.into())
    }

    pub fn audio(msg: impl Into<String>) -> Self {
        Self::Audio(msg.into())
    }

    pub fn decode(msg: impl Into<String>) -> Self {
        Self::Decode(msg.into())
    }

    pub fn io(msg: impl Into<String>) -> Self {
        Self::Io(msg.into())
    }

    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub fn speed_apply(msg: impl Into<String>) -> Self {
        Self::SpeedApply(msg.into())
    }

    pub fn timed_out(msg: impl Into<String>) -> Self {
        Self::TimedOut(msg.into())
    }

    pub fn is_speed_apply_failure(&self) -> bool {
        matches!(self, Self::SpeedApply(_))
    }

    pub fn error_class(&self) -> &'static str {
        match self {
            Self::Message(_) => "Message",
            Self::EmptyText => "EmptyText",
            Self::TextTooLong { .. } => "TextTooLong",
            Self::UnknownBackend { .. } => "UnknownBackend",
            Self::MissingApiKey(_) => "MissingApiKey",
            Self::SpeedApply(_) => "TTSSpeedApplyError",
            Self::Http(_) => "Http",
            Self::TimedOut(_) => "TimedOut",
            Self::Backend(_) => "Backend",
            Self::Process(_) => "Process",
            Self::Audio(_) => "Audio",
            Self::Decode(_) => "Decode",
            Self::Io(_) => "Io",
            Self::Cancelled => "Cancelled",
            Self::Config(_) => "Config",
        }
    }
}

impl From<std::io::Error> for TtsError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<reqwest::Error> for TtsError {
    fn from(value: reqwest::Error) -> Self {
        if value.is_timeout() {
            return Self::TimedOut("request timed out".into());
        }
        Self::Http(value.to_string())
    }
}

impl From<serde_json::Error> for TtsError {
    fn from(value: serde_json::Error) -> Self {
        Self::Backend(format!("invalid JSON: {value}"))
    }
}

impl From<url::ParseError> for TtsError {
    fn from(value: url::ParseError) -> Self {
        Self::Config(format!("invalid URL: {value}"))
    }
}

/// Helper for building timeouts from config seconds.
pub fn timeout_duration(seconds: f64) -> Duration {
    let millis = if seconds.is_finite() && seconds > 0.0 {
        (seconds * 1000.0).round() as u64
    } else {
        30_000
    };
    Duration::from_millis(millis.max(1))
}
