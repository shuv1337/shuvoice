//! GTK4 + gtk4-layer-shell surface adapters.
//!
//! All constructors and methods here must run on the GTK main thread.

mod caption;
mod layer;
mod splash;
mod tts;
mod wizard;
mod wizard_helpers;
mod wizard_run;

pub use caption::CaptionOverlay;
pub use layer::{
    apply_caption_layer_policy, apply_splash_layer_policy, apply_tts_layer_policy,
    apply_wizard_layer_policy, install_css, layer_shell_supported, make_click_through,
};
pub use splash::SplashOverlay;
pub use tts::TtsOverlay;
pub use wizard::WelcomeWizard;
pub use wizard_helpers::{
    TtsControlVisibility, asr_backend_index, keybind_auto_add_sensitive, keybind_index,
    sherpa_controls_visible, sherpa_profile_index, should_show_done_back, speed_preset_index,
    tts_backend_index, tts_visibility_from_vm,
};
pub use wizard_run::{
    WizardUiLaunchError, run_welcome_wizard_gtk, run_welcome_wizard_gtk_deferred,
};

use std::time::Duration;

use crate::CaptionVm;
use crate::SplashVm;
use crate::TtsVm;
use crate::channel::{UiCmdReceiver, UiEventSender};
use crate::protocol::UiCmd;
use glib::ControlFlow;

/// Host that owns the long-lived STT/TTS overlays and drains [`UiCmd`]s.
pub struct UiHost {
    pub caption: CaptionOverlay,
    pub tts: Option<TtsOverlay>,
    pub splash: Option<SplashOverlay>,
    event_tx: UiEventSender,
}

impl UiHost {
    pub fn new(
        app: &gtk4::Application,
        caption_vm: CaptionVm,
        tts_vm: Option<TtsVm>,
        event_tx: UiEventSender,
    ) -> Self {
        let caption = CaptionOverlay::new(app, caption_vm);
        let tts = tts_vm.map(|vm| TtsOverlay::new(app, vm, event_tx.clone()));
        Self {
            caption,
            tts,
            splash: None,
            event_tx,
        }
    }

    pub fn show_splash(&mut self, app: &gtk4::Application, vm: SplashVm) {
        self.splash = Some(SplashOverlay::new(app, vm));
    }

    pub fn apply_cmd(&mut self, cmd: UiCmd) {
        match cmd {
            UiCmd::CaptionSetText { text } => self.caption.set_text(&text),
            UiCmd::CaptionSetDebug { text } => self.caption.set_debug_text(&text),
            UiCmd::CaptionSetState { state } => self.caption.set_state(state),
            UiCmd::CaptionShow => self.caption.show(),
            UiCmd::CaptionHide => self.caption.hide(),
            UiCmd::CaptionFlashError { text, token, secs } => {
                self.caption.flash_error(&text, token, secs);
            }
            UiCmd::CaptionClearFlash { token } => {
                self.caption.clear_flash_if_token(token, true);
            }
            UiCmd::TtsSetState {
                state,
                preview_text,
                error_message,
            } => {
                if let Some(tts) = &mut self.tts {
                    tts.set_state(state, &preview_text, error_message.as_deref());
                }
            }
            UiCmd::TtsSetSpeed { speed } => {
                if let Some(tts) = &mut self.tts {
                    tts.set_speed(speed);
                }
            }
            UiCmd::TtsSetVoices {
                voices,
                selected_voice_id,
            } => {
                if let Some(tts) = &mut self.tts {
                    tts.set_voices(voices, selected_voice_id.as_deref());
                }
            }
            UiCmd::TtsShow => {
                if let Some(tts) = &mut self.tts {
                    tts.show();
                }
            }
            UiCmd::TtsHide => {
                if let Some(tts) = &mut self.tts {
                    tts.hide();
                }
            }
            UiCmd::SplashSetStatus { text } => {
                if let Some(splash) = &mut self.splash {
                    splash.set_status(&text);
                }
            }
            UiCmd::SplashSetProgress { fraction, text } => {
                if let Some(splash) = &mut self.splash {
                    splash.set_progress(fraction, text.as_deref());
                }
            }
            UiCmd::SplashDismiss => {
                if let Some(splash) = self.splash.take() {
                    splash.dismiss();
                }
            }
            UiCmd::WizardNavigate { .. }
            | UiCmd::WizardSetStatus { .. }
            | UiCmd::WizardSetProgress { .. }
            | UiCmd::WizardDownloadFinished { .. }
            | UiCmd::WizardClose => {
                // Wizard binary owns its own host.
            }
        }
    }

    /// Poll `cmd_rx` on a glib timeout (non-blocking). Keeps GTK responsive.
    pub fn attach_cmd_pump(mut self, cmd_rx: UiCmdReceiver) -> glib::SourceId {
        glib::timeout_add_local(Duration::from_millis(16), move || {
            loop {
                match cmd_rx.try_recv() {
                    Ok(cmd) => self.apply_cmd(cmd),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        return ControlFlow::Break;
                    }
                }
            }
            ControlFlow::Continue
        })
    }

    pub fn event_sender(&self) -> UiEventSender {
        self.event_tx.clone()
    }
}
