//! Interactive TTS overlay view-model (headless).
//!
//! Speed math and voice/capability types come from `shuvoice-core`. Overlay
//! lifecycle states, preview truncation, and the view-model stay UI-owned.

use serde::{Deserialize, Serialize};
use shuvoice_core::config::Config;

use crate::error::UiError;

/// Layer-shell namespace for the TTS control surface.
pub const TTS_NAMESPACE: &str = "tts-overlay";

/// Extra bottom margin above the STT capsule.
pub const TTS_MARGIN_OFFSET_PX: i32 = 96;

pub use shuvoice_core::{
    TTS_PLAYBACK_SPEED_DEFAULT, TTS_PLAYBACK_SPEED_MAX, TTS_PLAYBACK_SPEED_MIN,
    TTS_PLAYBACK_SPEED_STEP, TtsCapabilities, VoiceInfo, format_tts_playback_speed,
    normalize_tts_playback_speed, step_tts_playback_speed, validate_tts_playback_speed,
};

/// Compatibility alias — core's capability struct is the source of truth.
pub type SpeedCapabilities = TtsCapabilities;

/// Default auto-hide after idle (seconds), from core `Config` default.
pub fn default_tts_overlay_auto_hide_sec() -> f64 {
    Config::default().tts_overlay_auto_hide_sec
}

/// Stable constant for call-sites that want a literal default.
pub const DEFAULT_TTS_OVERLAY_AUTO_HIDE_SEC: f64 = 2.0;

pub const SPEED_UNSUPPORTED_TOOLTIP: &str =
    "This TTS backend does not support provider-native speed control.";

/// TTS overlay lifecycle state (UI surface only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsOverlayState {
    Idle,
    Synthesizing,
    Playing,
    Paused,
    Error,
}

impl TtsOverlayState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Synthesizing => "synthesizing",
            Self::Playing => "playing",
            Self::Paused => "paused",
            Self::Error => "error",
        }
    }

    pub fn parse(value: &str) -> Result<Self, UiError> {
        match value {
            "idle" => Ok(Self::Idle),
            "synthesizing" => Ok(Self::Synthesizing),
            "playing" => Ok(Self::Playing),
            "paused" => Ok(Self::Paused),
            "error" => Ok(Self::Error),
            other => Err(UiError::invalid_state(
                "tts_overlay",
                other,
                "error, idle, paused, playing, synthesizing",
            )),
        }
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Truncate preview text without splitting Unicode scalar values.
pub fn summarize_preview(text: &str, max_chars: usize) -> String {
    let value = text.trim();
    if value.is_empty() {
        return String::new();
    }
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let keep = max_chars.saturating_sub(1).max(1);
    let mut out: String = value.chars().take(keep).collect();
    while out.ends_with(char::is_whitespace) {
        out.pop();
    }
    out.push('…');
    out
}

/// Human-readable status label for a TTS state.
pub fn status_label_for_state(state: TtsOverlayState, error_message: Option<&str>) -> String {
    match state {
        TtsOverlayState::Synthesizing => "🔊 Synthesizing…".into(),
        TtsOverlayState::Playing => "🔊 Speaking…".into(),
        TtsOverlayState::Paused => "⏸ Paused".into(),
        TtsOverlayState::Error => {
            let detail = error_message.unwrap_or("TTS error").trim();
            let detail = if detail.is_empty() {
                "TTS error"
            } else {
                detail
            };
            format!("⚠ {detail}")
        }
        TtsOverlayState::Idle => "🔈 Idle".into(),
    }
}

/// Validate via core and map errors into [`UiError`].
pub fn validate_tts_playback_speed_ui(speed: f64) -> Result<f64, UiError> {
    validate_tts_playback_speed(speed).map_err(UiError::from)
}

/// Headless TTS overlay view model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TtsVm {
    pub visible: bool,
    pub state: TtsOverlayState,
    pub preview_text: String,
    pub playback_speed: f64,
    pub speed_supported: bool,
    pub speed_min: f64,
    pub speed_max: f64,
    pub voices: Vec<VoiceInfo>,
    pub selected_voice_id: String,
    pub auto_hide_sec: f64,
    /// When true, a hide timer should be (re)armed for idle.
    pub pending_auto_hide: bool,
    pub bottom_margin: i32,
}

impl TtsVm {
    pub fn new(
        default_voice_id: impl Into<String>,
        initial_speed: f64,
        caps: TtsCapabilities,
        auto_hide_sec: f64,
        bottom_margin: i32,
    ) -> Self {
        let (speed_min, speed_max) = caps
            .speed_bounds()
            .unwrap_or((TTS_PLAYBACK_SPEED_MIN, TTS_PLAYBACK_SPEED_MAX));
        let speed = normalize_tts_playback_speed(initial_speed).clamp(speed_min, speed_max);
        Self {
            visible: false,
            state: TtsOverlayState::Idle,
            preview_text: String::new(),
            playback_speed: round2(speed),
            speed_supported: caps.supports_speed_control,
            speed_min,
            speed_max,
            voices: Vec::new(),
            selected_voice_id: default_voice_id.into(),
            auto_hide_sec,
            pending_auto_hide: false,
            bottom_margin,
        }
    }

    pub fn from_config(cfg: &Config, caps: TtsCapabilities) -> Self {
        Self::new(
            cfg.tts_default_voice_id.clone(),
            cfg.tts_playback_speed,
            caps,
            cfg.tts_overlay_auto_hide_sec,
            i32::try_from(cfg.bottom_margin).unwrap_or(i32::MAX),
        )
    }

    pub fn layer_bottom_margin(&self) -> i32 {
        self.bottom_margin.saturating_add(TTS_MARGIN_OFFSET_PX)
    }

    pub fn status_label(&self) -> String {
        let err = if self.state == TtsOverlayState::Error {
            Some(self.preview_text.as_str())
        } else {
            None
        };
        status_label_for_state(self.state, err)
    }

    pub fn pause_button_label(&self) -> &'static str {
        if self.state == TtsOverlayState::Paused {
            "▶ Resume"
        } else {
            "⏸ Pause"
        }
    }

    pub fn speed_label(&self) -> String {
        if self.speed_supported {
            format!("Speed {}", format_tts_playback_speed(self.playback_speed))
        } else {
            "Speed unavailable".into()
        }
    }

    pub fn slower_enabled(&self) -> bool {
        self.speed_supported && self.playback_speed > self.speed_min
    }

    pub fn faster_enabled(&self) -> bool {
        self.speed_supported && self.playback_speed < self.speed_max
    }

    pub fn show(&mut self) {
        self.pending_auto_hide = false;
        self.visible = true;
    }

    pub fn hide(&mut self) {
        self.pending_auto_hide = false;
        self.visible = false;
        if self.state != TtsOverlayState::Error {
            self.state = TtsOverlayState::Idle;
            self.preview_text.clear();
        }
    }

    pub fn set_state(
        &mut self,
        state: TtsOverlayState,
        preview_text: &str,
        error_message: Option<&str>,
    ) {
        self.state = state;
        if state == TtsOverlayState::Error {
            let msg = error_message.unwrap_or("TTS failed").trim();
            self.preview_text = if msg.is_empty() {
                "TTS failed".into()
            } else {
                msg.into()
            };
        } else {
            self.preview_text = summarize_preview(preview_text, 50);
        }

        if state == TtsOverlayState::Idle {
            self.pending_auto_hide = true;
            self.visible = true;
        } else {
            self.show();
        }
    }

    /// Apply speed. Returns `Some(new)` when changed and `emit` is requested.
    pub fn apply_speed(&mut self, speed: f64, emit: bool) -> Option<f64> {
        if !self.speed_supported {
            return None;
        }
        let normalized =
            round2(normalize_tts_playback_speed(speed).clamp(self.speed_min, self.speed_max));
        if (normalized - self.playback_speed).abs() < 1e-6 {
            return None;
        }
        self.playback_speed = normalized;
        if emit { Some(normalized) } else { None }
    }

    pub fn step_speed(&mut self, steps: i32) -> Option<f64> {
        let next = step_tts_playback_speed(self.playback_speed, steps);
        self.apply_speed(next, true)
    }

    pub fn set_voices(&mut self, voices: Vec<VoiceInfo>, selected_voice_id: Option<&str>) {
        self.voices = voices;
        let target = selected_voice_id
            .unwrap_or(self.selected_voice_id.as_str())
            .trim();
        if self.voices.is_empty() {
            return;
        }
        if let Some(v) = self.voices.iter().find(|v| v.id == target) {
            self.selected_voice_id = v.id.clone();
        } else {
            self.selected_voice_id = self.voices[0].id.clone();
        }
    }

    pub fn select_voice_index(&mut self, idx: usize) -> Option<String> {
        let voice = self.voices.get(idx)?;
        self.selected_voice_id = voice.id.clone();
        Some(voice.id.clone())
    }

    /// Auto-hide delay in milliseconds. `None` means hide immediately.
    pub fn auto_hide_delay_ms(&self) -> Option<u64> {
        if self.auto_hide_sec <= 0.0 {
            return None;
        }
        Some((self.auto_hide_sec * 1000.0).max(0.0) as u64)
    }

    pub fn display_preview(&self) -> String {
        if self.state == TtsOverlayState::Error {
            String::new()
        } else {
            self.preview_text.clone()
        }
    }

    pub fn voice_dropdown_labels(&self) -> Vec<String> {
        if self.voices.is_empty() {
            vec!["Voice: default".into()]
        } else {
            self.voices
                .iter()
                .map(|v| format!("Voice: {}", v.name))
                .collect()
        }
    }
}

/// CSS for the TTS overlay.
pub fn tts_overlay_css() -> &'static str {
    r#"window { background-color: transparent; }
.tts-overlay-box {
  background-color: rgba(0, 0, 0, 0.82);
  border-radius: 14px;
  padding: 14px 18px;
}
.tts-status-label {
  color: white;
  font-size: 16px;
  font-weight: 700;
}
.tts-preview-label {
  color: rgba(255, 255, 255, 0.88);
  font-size: 13px;
}
.tts-control-btn {
  font-weight: 600;
}
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_preview_truncates_long_text() {
        let rendered = summarize_preview("abcdefghijklmnopqrstuvwxyz", 20);
        assert!(rendered.ends_with('…'));
        assert_eq!(rendered.chars().count(), 20);
    }

    #[test]
    fn summarize_preview_keeps_short_text() {
        assert_eq!(summarize_preview("hello", 20), "hello");
    }

    #[test]
    fn status_label_for_error_includes_message() {
        let label = status_label_for_state(TtsOverlayState::Error, Some("Network timeout"));
        assert!(label.contains("Network timeout"));
    }

    #[test]
    fn status_label_for_playing() {
        assert_eq!(
            status_label_for_state(TtsOverlayState::Playing, None),
            "🔊 Speaking…"
        );
    }

    #[test]
    fn speed_step_and_format_use_core() {
        assert_eq!(step_tts_playback_speed(1.0, 1), 1.1);
        assert_eq!(format_tts_playback_speed(1.25), "1.25×");
        assert_eq!(format_tts_playback_speed(1.0), "1.0×");
    }

    #[test]
    fn idle_schedules_auto_hide() {
        let mut vm = TtsVm::new(
            "af_heart",
            1.25,
            TtsCapabilities {
                supports_speed_control: true,
                speed_min: Some(0.5),
                speed_max: Some(2.0),
                ..TtsCapabilities::default()
            },
            2.0,
            60,
        );
        vm.set_state(TtsOverlayState::Playing, "hello world", None);
        assert!(vm.visible);
        assert!(!vm.pending_auto_hide);
        vm.set_state(TtsOverlayState::Idle, "hello world", None);
        assert!(vm.pending_auto_hide);
        assert_eq!(vm.auto_hide_delay_ms(), Some(2000));
        assert_eq!(vm.layer_bottom_margin(), 156);
    }

    #[test]
    fn unsupported_speed_blocks_emit() {
        let mut vm = TtsVm::new("v", 1.0, TtsCapabilities::default(), 2.0, 60);
        assert!(vm.apply_speed(1.5, true).is_none());
        assert_eq!(vm.speed_label(), "Speed unavailable");
    }
}
