//! UI view models and optional GTK4/layer-shell adapters for ShuVoice.
//!
//! # Feature flags
//!
//! - default: pure headless models, CSS strings, formatting helpers, and tests
//! - `gtk`: GTK4 + gtk4-layer-shell surface implementations
//!
//! Domain policy (overlay states, speed math, feedback PCM, wizard defaults)
//! lives in [`shuvoice_core`]. This crate owns view-models, CSS, Wayland
//! surface adapters, and UI-only presentation helpers.
//!
//! All GTK object mutation must happen on the GLib main context. Worker threads
//! should send [`protocol::UiCmd`] values through [`channel::UiCmdSender`].

#![cfg_attr(not(test), deny(clippy::unwrap_used))]

pub mod branding;
pub mod caption;
pub mod channel;
pub mod debug_view;
pub mod error;
pub mod feedback;
pub mod protocol;
pub mod splash;
pub mod tts_overlay;
pub mod waybar;
pub mod wizard;
pub mod wizard_controller;

#[cfg(test)]
mod test_env;

#[cfg(feature = "gtk")]
pub mod gtk;

#[cfg(feature = "gtk")]
pub use gtk::{
    CaptionOverlay, SplashOverlay, TtsOverlay, UiHost, WelcomeWizard, apply_caption_layer_policy,
    apply_splash_layer_policy, apply_tts_layer_policy, apply_wizard_layer_policy, install_css,
    layer_shell_supported, make_click_through, run_welcome_wizard_gtk,
    run_welcome_wizard_gtk_deferred,
};

pub use branding::{LOGO_FILENAMES, find_logo, logo_candidates};
pub use caption::{
    CAPTION_NAMESPACE, CaptionStyle, CaptionVm, ERROR_TOAST_SECONDS, OVERLAY_STATES, OverlayState,
    OverlayStateExt, caption_css, overlay_state_class,
};
pub use channel::{UiBus, UiCmdReceiver, UiCmdSender, UiEventReceiver, UiEventSender};
pub use debug_view::{
    DebugSnapshot, DebugStatusPayload, RecentLogBuffer, debug_status_to_json,
    format_debug_overlay_lines,
};
pub use error::UiError;
pub use feedback::{FeedbackConfig, generate_tone};
pub use protocol::{UiCmd, UiEvent};
pub use splash::{
    MIN_SPLASH_VISIBLE, MIN_SPLASH_VISIBLE_SEC, SPLASH_NAMESPACE, SPLASH_PULSE_INTERVAL_MS,
    SplashVm, splash_css,
};
pub use tts_overlay::{
    DEFAULT_TTS_OVERLAY_AUTO_HIDE_SEC, SpeedCapabilities, TTS_NAMESPACE,
    TTS_PLAYBACK_SPEED_DEFAULT, TTS_PLAYBACK_SPEED_MAX, TTS_PLAYBACK_SPEED_MIN,
    TTS_PLAYBACK_SPEED_STEP, TtsCapabilities, TtsOverlayState, TtsVm, VoiceInfo,
    default_tts_overlay_auto_hide_sec, format_tts_playback_speed, normalize_tts_playback_speed,
    status_label_for_state, step_tts_playback_speed, summarize_preview, tts_overlay_css,
    validate_tts_playback_speed, validate_tts_playback_speed_ui,
};
pub use waybar::{
    WaybarConfigInfo, build_waybar_payload, config_info_lines, sanitize_class, tts_backend_label,
};
pub use wizard::{
    FormatSummaryArgs, KEYBIND_PRESETS, KeybindPreset, KeybindSetupStatus, WIZARD_APPLICATION_ID,
    WIZARD_NAMESPACE, WizardPageId, WizardVm, WizardWritePlan, control_exec,
    default_tts_voice_for_backend, finish_status_text, format_hyprland_bind,
    format_hyprland_bind_for_keybind, format_summary, model_download_status_text, needs_wizard,
    needs_wizard_fs, shell_quote, tts_playback_speed_preset_id, tts_voice_label, wizard_css,
};
pub use wizard_controller::{
    DeferredModelSetup, DeviceDetector, ModelProgressCb, ModelSetupHook, ModelSetupStatus,
    PathCudaDetector, WizardFinishReport, apply_write_plan, auto_add_hyprland_keybind,
    auto_add_hyprland_keybind_with, finish_wizard, finish_wizard_deferred,
    hyprland_config_candidates, merge_wizard_keys_into_raw, persist_write_plan,
    resolve_shuvoice_command, with_device_detector, write_wizard_marker,
};

/// Main service GTK application id.
pub const APP_APPLICATION_ID: &str = "io.github.shuv1337.shuvoice";
