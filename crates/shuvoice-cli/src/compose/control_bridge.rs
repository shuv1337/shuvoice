//! Bridge `shuvoice-control::ControlHandlers` onto the app enqueue surface.
//!
//! # Design
//!
//! [`ControlBridge`] adapts [`EnqueueControlAdapter`] (or any
//! [`ControlHandlerSurface`]) into the control-server callback trait.
//!
//! - All handlers are **non-blocking**: they only `try_send` / read cached
//!   snapshots. The control accept loop must never wait on session work.
//! - `tts_status` is **honest**: it returns the cached runtime status string
//!   (player state mirrored by the session) and does **not** enqueue a no-op
//!   command that could falsely report success under queue pressure.
//!
//! # Integration notes
//!
//! Expected crate deps (declared by the integration owner):
//! - `shuvoice-app` (`ControlHandlerSurface`, `EnqueueControlAdapter`)
//! - `shuvoice-control` (`ControlHandlers`, `ControlCommand`)
//!
//! No extra features required.

#![allow(clippy::collapsible_if)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::let_and_return)]
#![allow(clippy::double_must_use)]
#![allow(clippy::result_unit_err)]
use std::sync::Arc;

use shuvoice_app::{ControlHandlerSurface, EnqueueControlAdapter};
use shuvoice_control::{ControlCommand, ControlHandlers};

/// Control-server handlers backed by an enqueue-only app surface.
#[derive(Clone)]
pub struct ControlBridge<S = EnqueueControlAdapter> {
    surface: S,
}

impl ControlBridge<EnqueueControlAdapter> {
    /// Wrap the production enqueue adapter.
    #[must_use]
    pub fn new(adapter: EnqueueControlAdapter) -> Self {
        Self { surface: adapter }
    }

    /// Wrap and erase to `Arc<dyn ControlHandlers>` for [`ControlServer`].
    #[must_use]
    pub fn arc(adapter: EnqueueControlAdapter) -> Arc<dyn ControlHandlers> {
        Arc::new(Self::new(adapter))
    }
}

impl<S> ControlBridge<S> {
    /// Wrap any [`ControlHandlerSurface`] (tests / alternate adapters).
    #[must_use]
    pub fn from_surface(surface: S) -> Self {
        Self { surface }
    }

    /// Borrow the inner surface.
    #[must_use]
    pub fn surface(&self) -> &S {
        &self.surface
    }
}

impl<S> ControlBridge<S>
where
    S: ControlHandlerSurface + Clone + 'static,
{
    /// Type-erase a generic surface bridge.
    #[must_use]
    pub fn into_arc(self) -> Arc<dyn ControlHandlers> {
        Arc::new(self)
    }
}

impl<S> ControlHandlers for ControlBridge<S>
where
    S: ControlHandlerSurface + Send + Sync + 'static,
{
    fn on_start(&self) {
        self.surface.on_start();
    }

    fn on_stop(&self) {
        self.surface.on_stop();
    }

    fn on_toggle(&self) {
        self.surface.on_toggle();
    }

    fn on_status(&self) -> String {
        self.surface.on_status()
    }

    fn on_metrics(&self) -> String {
        self.surface.on_metrics()
    }

    fn on_debug_status(&self) -> String {
        self.surface.on_debug_status()
    }

    fn on_tts_command(&self, command: ControlCommand) -> String {
        // Delegate by wire token. EnqueueControlAdapter::on_tts_command is
        // non-blocking and returns an honest cached status for `tts_status`.
        self.surface.on_tts_command(command.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use shuvoice_control::{dispatch, fixed};

    #[derive(Clone, Default)]
    struct FakeSurface {
        starts: Arc<AtomicUsize>,
        stops: Arc<AtomicUsize>,
        toggles: Arc<AtomicUsize>,
        status: Arc<Mutex<String>>,
        metrics: Arc<Mutex<String>>,
        debug: Arc<Mutex<String>>,
        tts_calls: Arc<Mutex<Vec<String>>>,
        tts_enabled: bool,
        queue_full: bool,
    }

    impl ControlHandlerSurface for FakeSurface {
        fn on_start(&self) {
            self.starts.fetch_add(1, Ordering::SeqCst);
        }
        fn on_stop(&self) {
            self.stops.fetch_add(1, Ordering::SeqCst);
        }
        fn on_toggle(&self) {
            self.toggles.fetch_add(1, Ordering::SeqCst);
        }
        fn on_status(&self) -> String {
            self.status.lock().expect("status").clone()
        }
        fn on_metrics(&self) -> String {
            self.metrics.lock().expect("metrics").clone()
        }
        fn on_debug_status(&self) -> String {
            self.debug.lock().expect("debug").clone()
        }
        fn on_tts_command(&self, command: &str) -> String {
            self.tts_calls
                .lock()
                .expect("tts")
                .push(command.to_string());
            if !self.tts_enabled {
                return "ERROR tts disabled".into();
            }
            if command == "tts_status" {
                // Honest: read-only snapshot, never a fake OK on enqueue.
                return format!("OK {}", self.on_status());
            }
            if self.queue_full {
                return "ERROR control queue full".into();
            }
            match command {
                "tts_speak" | "tts_speak_clipboard" => "OK tts speaking".into(),
                "tts_pause" => "OK tts paused".into(),
                "tts_stop" => "OK tts stopped".into(),
                other => format!("ERROR unknown tts command: {other}"),
            }
        }
    }

    #[test]
    fn start_stop_toggle_are_nonblocking_callbacks() {
        let surface = FakeSurface {
            tts_enabled: true,
            ..FakeSurface::default()
        };
        let bridge = ControlBridge::from_surface(surface.clone());
        let handlers: Arc<dyn ControlHandlers> = Arc::new(bridge);

        assert_eq!(dispatch(&handlers, ControlCommand::Start), fixed::STARTED);
        assert_eq!(dispatch(&handlers, ControlCommand::Stop), fixed::STOPPED);
        assert_eq!(dispatch(&handlers, ControlCommand::Toggle), fixed::TOGGLED);
        assert_eq!(surface.starts.load(Ordering::SeqCst), 1);
        assert_eq!(surface.stops.load(Ordering::SeqCst), 1);
        assert_eq!(surface.toggles.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn status_metrics_debug_pass_through() {
        let surface = FakeSurface {
            status: Arc::new(Mutex::new("recording".into())),
            metrics: Arc::new(Mutex::new("{\"n\":1}".into())),
            debug: Arc::new(Mutex::new("{\"state\":\"recording\"}".into())),
            tts_enabled: true,
            ..FakeSurface::default()
        };
        let handlers = ControlBridge::from_surface(surface).into_arc();
        assert_eq!(
            dispatch(&handlers, ControlCommand::Status),
            fixed::ok_status("recording")
        );
        assert_eq!(
            dispatch(&handlers, ControlCommand::Metrics),
            fixed::ok_status("{\"n\":1}")
        );
        assert_eq!(
            dispatch(&handlers, ControlCommand::DebugStatus),
            fixed::ok_status("{\"state\":\"recording\"}")
        );
    }

    #[test]
    fn tts_status_is_honest_cached_snapshot() {
        let surface = FakeSurface {
            status: Arc::new(Mutex::new("idle".into())),
            tts_enabled: true,
            queue_full: true, // would break enqueue-based lies
            ..FakeSurface::default()
        };
        let surface_calls = Arc::clone(&surface.tts_calls);
        let handlers = ControlBridge::from_surface(surface).into_arc();
        let resp = dispatch(&handlers, ControlCommand::TtsStatus);
        assert_eq!(resp, "OK idle");
        assert_eq!(surface_calls.lock().unwrap().as_slice(), ["tts_status"]);
    }

    #[test]
    fn tts_speak_maps_wire_token_and_surfaces_queue_full() {
        let surface = FakeSurface {
            tts_enabled: true,
            queue_full: true,
            ..FakeSurface::default()
        };
        let handlers = ControlBridge::from_surface(surface.clone()).into_arc();
        let resp = dispatch(&handlers, ControlCommand::TtsSpeak);
        assert_eq!(resp, "ERROR control queue full");
        assert_eq!(surface.tts_calls.lock().unwrap().as_slice(), ["tts_speak"]);
    }

    #[test]
    fn tts_disabled_is_reported() {
        let surface = FakeSurface {
            tts_enabled: false,
            ..FakeSurface::default()
        };
        let handlers = ControlBridge::from_surface(surface).into_arc();
        let resp = dispatch(&handlers, ControlCommand::TtsPause);
        assert_eq!(resp, "ERROR tts disabled");
    }
}
