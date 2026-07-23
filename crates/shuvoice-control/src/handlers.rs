//! Application callbacks invoked by the control server.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::commands::ControlCommand;
use crate::protocol::{fixed, sanitize_response_line};

/// Callbacks the control server dispatches into.
///
/// Mirrors the Python `ControlServer` constructor hooks.
///
/// **Contract:** implementations must be panic-safe and preferably non-blocking.
/// The server isolates panics, but a blocked handler still stalls the accept loop.
pub trait ControlHandlers: Send + Sync + 'static {
    fn on_start(&self);
    fn on_stop(&self);
    fn on_toggle(&self);
    fn on_status(&self) -> String;

    fn on_metrics(&self) -> String {
        "metrics unavailable".to_string()
    }

    fn on_debug_status(&self) -> String {
        "debug unavailable".to_string()
    }

    /// Handle a `tts_*` command. Return a full response line (`OK …` / `ERROR …`).
    fn on_tts_command(&self, command: ControlCommand) -> String {
        let _ = command;
        fixed::TTS_NOT_AVAILABLE.to_string()
    }
}

/// Function-pointer based handlers for lightweight wiring / tests.
pub struct FnControlHandlers<S, T, G, St, M, D, Ts>
where
    S: Fn() + Send + Sync + 'static,
    T: Fn() + Send + Sync + 'static,
    G: Fn() + Send + Sync + 'static,
    St: Fn() -> String + Send + Sync + 'static,
    M: Fn() -> String + Send + Sync + 'static,
    D: Fn() -> String + Send + Sync + 'static,
    Ts: Fn(ControlCommand) -> String + Send + Sync + 'static,
{
    pub on_start: S,
    pub on_stop: T,
    pub on_toggle: G,
    pub on_status: St,
    pub on_metrics: M,
    pub on_debug_status: D,
    pub on_tts_command: Option<Ts>,
}

impl<S, T, G, St, M, D, Ts> ControlHandlers for FnControlHandlers<S, T, G, St, M, D, Ts>
where
    S: Fn() + Send + Sync + 'static,
    T: Fn() + Send + Sync + 'static,
    G: Fn() + Send + Sync + 'static,
    St: Fn() -> String + Send + Sync + 'static,
    M: Fn() -> String + Send + Sync + 'static,
    D: Fn() -> String + Send + Sync + 'static,
    Ts: Fn(ControlCommand) -> String + Send + Sync + 'static,
{
    fn on_start(&self) {
        (self.on_start)();
    }

    fn on_stop(&self) {
        (self.on_stop)();
    }

    fn on_toggle(&self) {
        (self.on_toggle)();
    }

    fn on_status(&self) -> String {
        (self.on_status)()
    }

    fn on_metrics(&self) -> String {
        (self.on_metrics)()
    }

    fn on_debug_status(&self) -> String {
        (self.on_debug_status)()
    }

    fn on_tts_command(&self, command: ControlCommand) -> String {
        match &self.on_tts_command {
            Some(cb) => cb(command),
            None => fixed::TTS_NOT_AVAILABLE.to_string(),
        }
    }
}

fn run_catch<F, R>(label: &str, f: F) -> Result<R, ()>
where
    F: FnOnce() -> R,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => Ok(v),
        Err(_) => {
            tracing::error!("control handler panicked: {label}");
            Err(())
        }
    }
}

/// Dispatch a parsed command through handlers into a full response line.
///
/// Panics in handlers are caught and converted to `ERROR internal error`.
/// Response bodies are sanitized (no newlines) and size-capped.
pub fn dispatch(handlers: &Arc<dyn ControlHandlers>, command: ControlCommand) -> String {
    let raw = match command {
        ControlCommand::Start => {
            if run_catch("on_start", || handlers.on_start()).is_err() {
                return fixed::INTERNAL.to_string();
            }
            fixed::STARTED.to_string()
        }
        ControlCommand::Stop => {
            if run_catch("on_stop", || handlers.on_stop()).is_err() {
                return fixed::INTERNAL.to_string();
            }
            fixed::STOPPED.to_string()
        }
        ControlCommand::Toggle => {
            if run_catch("on_toggle", || handlers.on_toggle()).is_err() {
                return fixed::INTERNAL.to_string();
            }
            fixed::TOGGLED.to_string()
        }
        ControlCommand::Status => match run_catch("on_status", || handlers.on_status()) {
            Ok(s) => fixed::ok_status(&s),
            Err(()) => fixed::INTERNAL.to_string(),
        },
        ControlCommand::Metrics => match run_catch("on_metrics", || handlers.on_metrics()) {
            Ok(s) => fixed::ok_status(&s),
            Err(()) => fixed::INTERNAL.to_string(),
        },
        ControlCommand::DebugStatus => {
            match run_catch("on_debug_status", || handlers.on_debug_status()) {
                Ok(s) => fixed::ok_status(&s),
                Err(()) => fixed::INTERNAL.to_string(),
            }
        }
        ControlCommand::Ping => fixed::PONG.to_string(),
        other if other.is_tts() => {
            match run_catch("on_tts_command", || handlers.on_tts_command(other)) {
                Ok(s) => s,
                Err(()) => fixed::INTERNAL.to_string(),
            }
        }
        other => fixed::unknown_command(other.as_str()),
    };
    sanitize_response_line(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn panic_in_handler_returns_internal_error() {
        let handlers: Arc<dyn ControlHandlers> = Arc::new(FnControlHandlers {
            on_start: || panic!("boom"),
            on_stop: || {},
            on_toggle: || {},
            on_status: || "idle".into(),
            on_metrics: || "m".into(),
            on_debug_status: || "d".into(),
            on_tts_command: None::<fn(ControlCommand) -> String>,
        });
        let resp = dispatch(&handlers, ControlCommand::Start);
        assert_eq!(resp, fixed::INTERNAL);
    }

    #[test]
    fn status_newlines_sanitized() {
        let handlers: Arc<dyn ControlHandlers> = Arc::new(FnControlHandlers {
            on_start: || {},
            on_stop: || {},
            on_toggle: || {},
            on_status: || "idle\nOK pwned".into(),
            on_metrics: || "m".into(),
            on_debug_status: || "d".into(),
            on_tts_command: None::<fn(ControlCommand) -> String>,
        });
        let resp = dispatch(&handlers, ControlCommand::Status);
        assert!(!resp.contains('\n'));
        assert!(resp.starts_with("OK "));
    }

    #[test]
    fn start_invokes_callback() {
        let n = Arc::new(AtomicUsize::new(0));
        let n2 = Arc::clone(&n);
        let handlers: Arc<dyn ControlHandlers> = Arc::new(FnControlHandlers {
            on_start: move || {
                n2.fetch_add(1, Ordering::SeqCst);
            },
            on_stop: || {},
            on_toggle: || {},
            on_status: || "idle".into(),
            on_metrics: || "m".into(),
            on_debug_status: || "d".into(),
            on_tts_command: None::<fn(ControlCommand) -> String>,
        });
        assert_eq!(dispatch(&handlers, ControlCommand::Start), fixed::STARTED);
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }
}
