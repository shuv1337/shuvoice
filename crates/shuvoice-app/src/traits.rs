//! Effect seams (TTS / inject / selection / overlay).

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;

use crate::types::{OverlayState, TtsPlayerState};

/// Injection failures surface as `Err` so the session only latches commit on success.
#[async_trait]
pub trait TextInjector: Send + Sync {
    async fn update_partial(&self, text: &str) -> Result<(), String>;
    async fn commit_final(&self, text: &str) -> Result<(), String>;
    async fn reset(&self) -> Result<(), String>;
}

#[async_trait]
pub trait SelectionCapture: Send + Sync {
    async fn capture_selection(&self) -> Result<String, String>;
    async fn capture_clipboard(&self) -> Result<String, String>;
}

/// TTS player surface. State is returned owned (no borrowed interior).
pub trait TtsEngine: Send {
    fn is_active(&self) -> bool {
        self.state().is_active()
    }
    fn state(&self) -> TtsPlayerState;
    fn supports_speed_control(&self) -> bool {
        false
    }
    fn speed_bounds(&self) -> Option<(f64, f64)> {
        None
    }
    fn speak(&mut self, text: &str, voice_id: &str, model_id: &str) -> Result<bool, String>;
    fn pause(&mut self) -> bool;
    fn resume(&mut self) -> bool;
    fn toggle_pause(&mut self) -> bool;
    fn restart(&mut self) -> bool;
    fn stop(&mut self) -> bool;
    fn set_playback_speed(&mut self, speed: f64) -> f64;
}

pub trait OverlaySink: Send {
    fn show(&mut self, state: OverlayState, text: &str);
    fn set_state(&mut self, state: OverlayState);
    fn set_text(&mut self, text: &str);
    fn hide(&mut self);
    fn set_debug_text(&mut self, _text: &str) {}
}

pub trait FeedbackSink: Send {
    fn play_start(&mut self) {}
    fn play_stop(&mut self) {}
}

pub trait Clock: Send + Sync {
    fn now(&self) -> Instant;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// Test clock: `origin + offset`.
#[derive(Debug, Clone)]
pub struct FakeClock {
    origin: Instant,
    offset: Arc<std::sync::Mutex<Duration>>,
}

impl FakeClock {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
            offset: Arc::new(std::sync::Mutex::new(Duration::ZERO)),
        }
    }

    pub fn shared_handle(&self) -> Self {
        Self {
            origin: self.origin,
            offset: Arc::clone(&self.offset),
        }
    }

    pub fn set_ms(&self, ms: u64) {
        *self.offset.lock().expect("clock") = Duration::from_millis(ms);
    }

    pub fn advance_ms(&self, ms: u64) {
        *self.offset.lock().expect("clock") += Duration::from_millis(ms);
    }

    pub fn elapsed(&self) -> Duration {
        *self.offset.lock().expect("clock")
    }
}

impl Default for FakeClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for FakeClock {
    fn now(&self) -> Instant {
        self.origin + *self.offset.lock().expect("clock")
    }
}
