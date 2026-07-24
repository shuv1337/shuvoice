//! CLI exit-code carrying errors.

use std::fmt;

/// Successful process exit.
pub const EXIT_SUCCESS: i32 = 0;
/// Generic operational failure.
pub const EXIT_FAILURE: i32 = 1;
/// Usage / argparse failure (clap default).
pub const EXIT_USAGE: i32 = 2;
/// Dependency / config / startup guard failure (systemd RestartPreventExitStatus).
pub const EXIT_DEPENDENCY: i32 = shuvoice_core::DEPENDENCY_EXIT_CODE as i32;

/// Shared packaging constant (must remain 78).
pub use shuvoice_core::DEPENDENCY_EXIT_CODE;

/// Error that maps directly to a process exit code.
#[derive(Debug)]
pub struct ExitStatus {
    pub code: i32,
    pub message: Option<String>,
}

impl ExitStatus {
    pub fn success() -> Self {
        Self {
            code: EXIT_SUCCESS,
            message: None,
        }
    }

    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            code: EXIT_FAILURE,
            message: Some(message.into()),
        }
    }

    pub fn dependency(message: impl Into<String>) -> Self {
        Self {
            code: EXIT_DEPENDENCY,
            message: Some(message.into()),
        }
    }

    pub fn code(code: i32) -> Self {
        Self {
            code,
            message: None,
        }
    }

    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.message {
            Some(msg) => write!(f, "{msg}"),
            None => write!(f, "exit {}", self.code),
        }
    }
}

impl std::error::Error for ExitStatus {}

impl From<anyhow::Error> for ExitStatus {
    fn from(value: anyhow::Error) -> Self {
        Self::failure(format!("{value:#}"))
    }
}
