//! Shared UI errors.

use shuvoice_core::CoreError;
use thiserror::Error;

/// Errors produced by headless UI helpers.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum UiError {
    /// Unknown overlay / TTS / wizard state identifier.
    #[error("{0}")]
    InvalidState(String),
    /// Invalid numeric or selection configuration.
    #[error("{0}")]
    InvalidValue(String),
}

impl UiError {
    pub(crate) fn invalid_state(
        kind: &'static str,
        value: impl Into<String>,
        allowed: &'static str,
    ) -> Self {
        let value = value.into();
        Self::InvalidState(format!(
            "{kind} state '{value}' is invalid; expected one of: {allowed}"
        ))
    }

    pub(crate) fn from_core(err: CoreError) -> Self {
        match err {
            CoreError::OverlayState(msg) | CoreError::Validation(msg) => Self::InvalidValue(msg),
            other => Self::InvalidValue(other.to_string()),
        }
    }
}

impl From<CoreError> for UiError {
    fn from(value: CoreError) -> Self {
        Self::from_core(value)
    }
}
