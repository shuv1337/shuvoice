//! Orchestration-facing ASR events.

use crate::error::AsrErrorClass;

#[derive(Debug, Clone, PartialEq)]
pub enum AsrEvent {
    Partial {
        text: String,
        step: Option<u64>,
    },
    Final {
        text: String,
    },
    Warning {
        message: String,
    },
    Error {
        class: AsrErrorClass,
        message: String,
        counts_for_breaker: bool,
    },
}
