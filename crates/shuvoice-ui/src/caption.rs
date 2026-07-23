//! STT caption overlay view-model (headless).
//!
//! Domain state/CSS mapping come from `shuvoice-core`. This module owns the
//! caption view-model, visual presentation helpers, and CSS string.

use serde::{Deserialize, Serialize};
use shuvoice_core::config::Config;
use shuvoice_core::overlay_state_class as core_overlay_state_class;

use crate::error::UiError;

/// Layer-shell namespace for the STT caption surface.
pub const CAPTION_NAMESPACE: &str = "stt-overlay";

/// Re-export core toast duration (seconds).
pub use shuvoice_core::ERROR_TOAST_SECONDS;

/// Re-export core overlay state type (serde wire-compatible).
pub use shuvoice_core::OverlayState;

/// All caption states in display order.
pub const OVERLAY_STATES: [OverlayState; 3] = [
    OverlayState::Listening,
    OverlayState::Processing,
    OverlayState::Error,
];

/// UI-facing presentation helpers for [`OverlayState`].
pub trait OverlayStateExt {
    fn icon_name(self) -> &'static str;
    fn status_text(self) -> &'static str;
    fn parse_ui(value: &str) -> Result<OverlayState, UiError>;
}

impl OverlayStateExt for OverlayState {
    fn icon_name(self) -> &'static str {
        match self {
            Self::Listening => "microphone-sensitivity-high-symbolic",
            Self::Processing => "system-run-symbolic",
            Self::Error => "dialog-error-symbolic",
        }
    }

    fn status_text(self) -> &'static str {
        match self {
            Self::Listening => "Listening…",
            Self::Processing => "Processing…",
            Self::Error => "Error",
        }
    }

    fn parse_ui(value: &str) -> Result<OverlayState, UiError> {
        value.parse::<OverlayState>().map_err(UiError::from)
    }
}

/// Map state id → CSS class via core policy.
pub fn overlay_state_class(state: &str) -> Result<&'static str, UiError> {
    core_overlay_state_class(state).map_err(UiError::from)
}

/// Default visual style matching the core `Config` overlay section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptionStyle {
    pub font_size: u32,
    pub font_family: Option<String>,
    pub bg_opacity: f64,
    pub border_radius: u32,
    pub bottom_margin: i32,
    pub overlay_debug_mode: bool,
    pub overlay_debug_max_lines: usize,
}

impl Default for CaptionStyle {
    fn default() -> Self {
        Self::from_config(&Config::default())
    }
}

impl CaptionStyle {
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            font_size: cfg.font_size,
            font_family: cfg.font_family.clone(),
            bg_opacity: cfg.bg_opacity,
            border_radius: cfg.border_radius,
            bottom_margin: i32::try_from(cfg.bottom_margin).unwrap_or(i32::MAX),
            overlay_debug_mode: cfg.overlay_debug_mode,
            overlay_debug_max_lines: cfg.overlay_debug_max_lines as usize,
        }
    }
}

/// Headless caption overlay view model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptionVm {
    pub visible: bool,
    pub state: OverlayState,
    pub text: String,
    pub debug_enabled: bool,
    pub debug_text: String,
    /// Active flash token; newer flashes supersede older timers.
    pub flash_token: Option<u64>,
    pub style: CaptionStyle,
}

impl CaptionVm {
    pub fn new(style: CaptionStyle) -> Self {
        let debug_enabled = style.overlay_debug_mode;
        Self {
            visible: false,
            state: OverlayState::Listening,
            text: String::new(),
            debug_enabled,
            debug_text: String::new(),
            flash_token: None,
            style,
        }
    }

    pub fn from_config(cfg: &Config) -> Self {
        Self::new(CaptionStyle::from_config(cfg))
    }

    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
        if !self.text.is_empty() {
            self.visible = true;
        }
    }

    pub fn set_debug_text(&mut self, text: impl Into<String>) {
        if !self.debug_enabled {
            return;
        }
        self.debug_text = text.into();
        if !self.debug_text.is_empty() {
            self.visible = true;
        }
    }

    pub fn set_state(&mut self, state: OverlayState) {
        self.state = state;
    }

    pub fn show(&mut self) {
        self.visible = true;
    }

    pub fn hide(&mut self) {
        self.visible = false;
        self.state = OverlayState::Listening;
        self.text.clear();
        self.debug_text.clear();
        self.flash_token = None;
    }

    /// Show a transient error toast. Returns the token that should expire later.
    pub fn flash_error(&mut self, text: impl Into<String>, token: u64) -> u64 {
        self.show();
        self.set_state(OverlayState::Error);
        self.set_text(text);
        self.flash_token = Some(token);
        token
    }

    /// Clear a flash only if `token` is still current.
    ///
    /// Returns whether the overlay was hidden. Callers should skip hide when
    /// recording is active or the ASR circuit is open (app policy).
    pub fn clear_flash_if_token(&mut self, token: u64, allow_hide: bool) -> bool {
        if self.flash_token != Some(token) {
            return false;
        }
        if !allow_hide {
            return false;
        }
        self.hide();
        true
    }
}

/// CSS string for the caption overlay.
pub fn caption_css(style: &CaptionStyle) -> String {
    let font_family_css = style
        .font_family
        .as_ref()
        .map(|f| format!("  font-family: \"{f}\";\n"))
        .unwrap_or_default();
    let debug_size = (style.font_size as f64 * 0.56).floor() as i32;
    let debug_size = debug_size.max(11);
    let op = style.bg_opacity;
    let radius = style.border_radius;
    let font_size = style.font_size;

    format!(
        r#"window {{ background-color: transparent; }}
.caption-box {{
  background-color: rgba(0, 0, 0, 0.75);
  border-radius: {radius}px;
  padding: 16px 28px;
}}
.caption-box.state-listening {{ background-color: rgba(0, 0, 0, {op}); }}
.caption-box.state-processing {{ background-color: rgba(20, 45, 90, {op}); }}
.caption-box.state-error {{ background-color: rgba(120, 20, 20, {op}); }}
.caption-label {{
  color: white;
  font-size: {font_size}px;
{font_family_css}  font-weight: bold;
}}
.caption-debug-label {{
  color: rgba(255, 255, 255, 0.9);
  font-size: {debug_size}px;
{font_family_css}  font-weight: 500;
}}
.recording-icon {{
  color: white;
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_state_class_valid_states() {
        assert_eq!(overlay_state_class("listening").unwrap(), "state-listening");
        assert_eq!(
            overlay_state_class("processing").unwrap(),
            "state-processing"
        );
        assert_eq!(overlay_state_class("error").unwrap(), "state-error");
    }

    #[test]
    fn overlay_state_class_rejects_unknown() {
        assert!(overlay_state_class("unknown").is_err());
    }

    #[test]
    fn set_text_auto_shows() {
        let mut vm = CaptionVm::new(CaptionStyle::default());
        vm.set_text("hello");
        assert!(vm.visible);
        assert_eq!(vm.text, "hello");
    }

    #[test]
    fn hide_resets_state_and_text() {
        let mut vm = CaptionVm::new(CaptionStyle::default());
        vm.set_text("x");
        vm.set_state(OverlayState::Error);
        vm.hide();
        assert!(!vm.visible);
        assert_eq!(vm.state, OverlayState::Listening);
        assert!(vm.text.is_empty());
    }

    #[test]
    fn flash_token_supersession() {
        let mut vm = CaptionVm::new(CaptionStyle::default());
        vm.flash_error("a", 1);
        vm.flash_error("b", 2);
        assert!(!vm.clear_flash_if_token(1, true));
        assert!(vm.visible);
        assert!(vm.clear_flash_if_token(2, true));
        assert!(!vm.visible);
    }

    #[test]
    fn debug_text_noop_when_disabled() {
        let mut vm = CaptionVm::new(CaptionStyle::default());
        vm.set_debug_text("x");
        assert!(vm.debug_text.is_empty());
    }

    #[test]
    fn icons_and_status_text() {
        assert_eq!(
            OverlayState::Listening.icon_name(),
            "microphone-sensitivity-high-symbolic"
        );
        assert_eq!(OverlayState::Processing.status_text(), "Processing…");
        assert_eq!(OverlayState::Error.status_text(), "Error");
    }

    #[test]
    fn style_from_config_defaults() {
        let style = CaptionStyle::default();
        assert_eq!(style.font_size, 22);
        assert!((style.bg_opacity - 0.75).abs() < 1e-9);
        assert_eq!(style.bottom_margin, 60);
    }
}
