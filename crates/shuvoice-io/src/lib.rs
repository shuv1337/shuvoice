//! Linux audio and Wayland-focused platform adapters for ShuVoice.
//!
//! This crate is intentionally independent of ASR/TTS backends. It provides:
//! - process runner abstraction (timeout + redacted errors)
//! - `local.dev` env loader / XDG helpers
//! - selection/clipboard capture
//! - safe streaming text injection
//! - Hyprland bind matchers
//! - pure audio helpers (+ optional `cpal` capture behind `audio` feature)
//! - Waybar format/systemd helpers
//! - diagnostics formatting / log ring buffer

//!
//! ## Residual notes
//! - `wl-copy`/`wtype` bulk payloads prefer stdin; `xdotool type` has no stdin
//!   path and refuses oversized argv payloads instead of truncating silently.
//! - Process-group kill requires the child `setpgid(0,0)` path (Linux/Unix).
//! - Audio callback uses `try_lock` and may drop frames under lock contention
//!   rather than block the RT thread.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod audio;
pub mod diagnostics;
pub mod env_loader;
pub mod error;
pub mod hyprland;
pub mod inject;
pub mod noise_floor;
pub mod process;
pub mod selection;
pub mod waybar;
pub mod xdg;

pub use diagnostics::{RecentLogBuffer, debug_status_to_json, metrics_to_human, metrics_to_json};
pub use env_loader::{load_local_dev_env, local_dev_env_path};
pub use error::{AudioError, IoError, ProcessError, SelectionError};
pub use inject::{
    CommitOutcome, FinalInjectionMode, InjectError, StreamingTyper, TyperConfig,
    sanitize_final_injection_text,
};
pub use noise_floor::NoiseFloor;
pub use process::{
    CommandRunner, DEFAULT_MAX_OUTPUT_BYTES, RunOptions, RunOutput, ScriptedRunner,
    StdCommandRunner, argv,
};
pub use selection::{SelectionCapture, capture_clipboard, capture_selection};
pub use xdg::{config_home, data_home, runtime_dir, shuvoice_config_dir, shuvoice_data_dir};
