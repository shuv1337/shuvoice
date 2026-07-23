//! Wayland/X11 text injection.

mod error;
mod sanitize;
mod typer;

pub use error::{CommitOutcome, InjectError};
pub use sanitize::sanitize_final_injection_text;
pub use typer::{
    FinalInjectionMode, MAX_ARGV_PAYLOAD_BYTES, RecordingSleeper, Sleeper, StdSleeper,
    StreamingTyper, TyperConfig,
};
