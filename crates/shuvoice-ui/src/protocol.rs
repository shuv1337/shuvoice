//! Cross-thread UI command / event protocol.

use serde::{Deserialize, Serialize};

use crate::caption::OverlayState;
use crate::tts_overlay::TtsOverlayState;
use crate::wizard::WizardPageId;

/// Commands sent toward UI surfaces (any thread → main/UI).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UiCmd {
    CaptionSetText {
        text: String,
    },
    CaptionSetDebug {
        text: String,
    },
    CaptionSetState {
        state: OverlayState,
    },
    CaptionShow,
    CaptionHide,
    CaptionFlashError {
        text: String,
        token: u64,
        secs: u32,
    },
    CaptionClearFlash {
        token: u64,
    },

    TtsSetState {
        state: TtsOverlayState,
        preview_text: String,
        error_message: Option<String>,
    },
    TtsSetSpeed {
        speed: f64,
    },
    TtsSetVoices {
        voices: Vec<crate::tts_overlay::VoiceInfo>,
        selected_voice_id: Option<String>,
    },
    TtsShow,
    TtsHide,

    SplashSetStatus {
        text: String,
    },
    SplashSetProgress {
        fraction: Option<f64>,
        text: Option<String>,
    },
    SplashDismiss,

    WizardNavigate {
        page: WizardPageId,
    },
    WizardSetStatus {
        text: String,
    },
    WizardSetProgress {
        fraction: Option<f64>,
        text: String,
    },
    WizardDownloadFinished {
        status_text: String,
        show_launch: bool,
    },
    WizardClose,
}

/// Events emitted by interactive UI surfaces (main/UI → app).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UiEvent {
    TtsPause,
    TtsResume,
    TtsRestart,
    TtsStop,
    TtsVoiceSelected { voice_id: String },
    TtsSpeedChanged { speed: f64 },

    WizardPageChanged { page: WizardPageId },
    WizardBack,
    WizardNext,
    WizardFinishRequested,
    WizardLaunch,
    WizardCancelDownload,
    WizardClosed { completed: bool },
}
