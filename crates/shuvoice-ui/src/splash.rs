//! Branded splash overlay view-model (headless).
//!
//! Minimum-visibility policy uses `shuvoice-core::remaining_splash_ms`.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use shuvoice_core::remaining_splash_ms;

/// Layer-shell namespace for the splash surface.
pub const SPLASH_NAMESPACE: &str = "shuvoice-splash";

/// Re-export core minimum splash visibility.
pub use shuvoice_core::MIN_SPLASH_VISIBLE;

/// Seconds form of [`MIN_SPLASH_VISIBLE`] for call-sites that want an `f64`.
pub const MIN_SPLASH_VISIBLE_SEC: f64 = 2.0;

/// Pulse interval for indeterminate progress (ms).
pub const SPLASH_PULSE_INTERVAL_MS: u32 = 120;

/// Headless splash view model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplashVm {
    pub visible: bool,
    pub destroyed: bool,
    pub status: String,
    pub progress_fraction: Option<f64>,
    pub progress_text: String,
    pub pulsing: bool,
    /// Wall-clock instant used with core splash hold helpers.
    #[serde(skip)]
    pub shown_at: Option<Instant>,
}

impl Default for SplashVm {
    fn default() -> Self {
        Self::new()
    }
}

impl SplashVm {
    pub fn new() -> Self {
        Self {
            visible: true,
            destroyed: false,
            status: "Loading model…".into(),
            progress_fraction: Some(0.0),
            progress_text: "Starting…".into(),
            pulsing: false,
            shown_at: None,
        }
    }

    pub fn on_realize(&mut self, now: Instant) {
        if self.shown_at.is_none() {
            self.shown_at = Some(now);
        }
    }

    pub fn set_status(&mut self, text: impl Into<String>) {
        if self.destroyed {
            return;
        }
        self.status = text.into();
    }

    pub fn set_progress(&mut self, fraction: Option<f64>, text: Option<&str>) {
        if self.destroyed {
            return;
        }
        if let Some(t) = text
            && !t.is_empty()
        {
            self.set_status(t);
        }
        match fraction {
            None => {
                self.pulsing = true;
                self.progress_fraction = None;
                if let Some(t) = text
                    && !t.is_empty()
                {
                    self.progress_text = t.into();
                }
            }
            Some(f) => {
                self.pulsing = false;
                let bounded = f.clamp(0.0, 1.0);
                self.progress_fraction = Some(bounded);
                if let Some(t) = text
                    && !t.is_empty()
                {
                    self.progress_text = t.into();
                } else {
                    self.progress_text =
                        format!("Loading model… {}%", (bounded * 100.0).round() as i32);
                }
            }
        }
    }

    pub fn dismiss(&mut self) {
        self.pulsing = false;
        self.visible = false;
        self.destroyed = true;
        self.status.clear();
        self.progress_text.clear();
        self.progress_fraction = None;
    }

    /// Remaining time to satisfy `min_visible`, via core policy.
    pub fn remaining_ms(shown_at: Option<Instant>, min_visible: Duration, now: Instant) -> u64 {
        remaining_splash_ms(shown_at, min_visible, now)
    }

    /// App post-load delay: max(remaining, min_visible_ms).
    ///
    /// Uses the greater of the remaining visible time and post-load minimum.
    pub fn post_load_hold_ms(
        shown_at: Option<Instant>,
        min_visible: Duration,
        now: Instant,
    ) -> u64 {
        let remaining = remaining_splash_ms(shown_at, min_visible, now);
        remaining.max(min_visible.as_millis() as u64)
    }
}

/// Splash overlay CSS.
pub fn splash_css() -> &'static str {
    r#"window.splash-window { background-color: transparent; }
.splash-box {
  background-color: rgba(15, 15, 20, 0.92);
  border-radius: 24px;
  padding: 48px 64px;
}
.splash-title {
  color: white;
  font-size: 32px;
  font-weight: bold;
}
.splash-status {
  color: rgba(255, 255, 255, 0.6);
  font-size: 16px;
  margin-top: 8px;
}
.splash-progress {
  min-width: 340px;
  margin-top: 10px;
}
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_status_updates_label() {
        let mut s = SplashVm::new();
        s.set_status("Downloading…");
        assert_eq!(s.status, "Downloading…");
    }

    #[test]
    fn set_status_noop_when_dismissed() {
        let mut s = SplashVm::new();
        s.dismiss();
        s.set_status("x");
        assert!(s.status.is_empty());
    }

    #[test]
    fn set_progress_updates_bar_and_status() {
        let mut s = SplashVm::new();
        s.set_progress(Some(0.4), Some("Almost"));
        assert_eq!(s.progress_fraction, Some(0.4));
        assert_eq!(s.status, "Almost");
        assert_eq!(s.progress_text, "Almost");
        assert!(!s.pulsing);
    }

    #[test]
    fn indeterminate_starts_pulsing() {
        let mut s = SplashVm::new();
        s.set_progress(None, Some("Working"));
        assert!(s.pulsing);
        assert_eq!(s.progress_fraction, None);
    }

    #[test]
    fn dismiss_is_idempotent_and_stops_pulse() {
        let mut s = SplashVm::new();
        s.set_progress(None, None);
        s.dismiss();
        s.dismiss();
        assert!(s.destroyed);
        assert!(!s.pulsing);
        assert!(!s.visible);
    }

    #[test]
    fn on_realize_keeps_first_timestamp() {
        let mut s = SplashVm::new();
        let t0 = Instant::now();
        s.on_realize(t0);
        s.on_realize(t0 + Duration::from_secs(10));
        assert_eq!(s.shown_at, Some(t0));
    }

    #[test]
    fn remaining_and_post_load_hold_use_core() {
        let shown = Instant::now();
        let now = shown + Duration::from_millis(500);
        assert_eq!(
            SplashVm::remaining_ms(Some(shown), MIN_SPLASH_VISIBLE, now),
            1500
        );
        // post-load holds at least the full min window
        assert_eq!(
            SplashVm::post_load_hold_ms(Some(shown), MIN_SPLASH_VISIBLE, now),
            2000
        );
        let later = shown + Duration::from_secs(3);
        assert_eq!(
            SplashVm::post_load_hold_ms(Some(shown), MIN_SPLASH_VISIBLE, later),
            2000
        );
    }
}
