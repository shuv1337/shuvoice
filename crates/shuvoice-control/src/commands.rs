//! Canonical control-command allowlist.

use std::fmt;
use std::str::FromStr;

use crate::error::ControlError;

/// Ordered allowlist of control-socket commands (wire + CLI source of truth).
pub const CONTROL_COMMANDS: &[&str] = &[
    "start",
    "stop",
    "toggle",
    "start_alt",
    "toggle_alt",
    "status",
    "ping",
    "metrics",
    "debug_status",
    "maintenance_reserve",
    "maintenance_release",
    "tts_speak",
    "tts_speak_clipboard",
    "tts_pause",
    "tts_resume",
    "tts_toggle_pause",
    "tts_restart",
    "tts_stop",
    "tts_status",
];

/// Parsed control command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlCommand {
    Start,
    Stop,
    Toggle,
    /// Start recording with the alternate text case (opposite of `typing_text_case`).
    StartAlt,
    /// Toggle recording; a start uses the alternate text case.
    ToggleAlt,
    Status,
    Ping,
    Metrics,
    DebugStatus,
    MaintenanceReserve,
    MaintenanceRelease(u64),
    TtsSpeak,
    TtsSpeakClipboard,
    TtsPause,
    TtsResume,
    TtsTogglePause,
    TtsRestart,
    TtsStop,
    TtsStatus,
}

impl ControlCommand {
    /// Wire token for this command.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Stop => "stop",
            Self::Toggle => "toggle",
            Self::StartAlt => "start_alt",
            Self::ToggleAlt => "toggle_alt",
            Self::Status => "status",
            Self::Ping => "ping",
            Self::Metrics => "metrics",
            Self::DebugStatus => "debug_status",
            Self::MaintenanceReserve => "maintenance_reserve",
            Self::MaintenanceRelease(_) => "maintenance_release",
            Self::TtsSpeak => "tts_speak",
            Self::TtsSpeakClipboard => "tts_speak_clipboard",
            Self::TtsPause => "tts_pause",
            Self::TtsResume => "tts_resume",
            Self::TtsTogglePause => "tts_toggle_pause",
            Self::TtsRestart => "tts_restart",
            Self::TtsStop => "tts_stop",
            Self::TtsStatus => "tts_status",
        }
    }

    /// True for any `tts_*` command.
    #[must_use]
    pub const fn is_tts(self) -> bool {
        matches!(
            self,
            Self::TtsSpeak
                | Self::TtsSpeakClipboard
                | Self::TtsPause
                | Self::TtsResume
                | Self::TtsTogglePause
                | Self::TtsRestart
                | Self::TtsStop
                | Self::TtsStatus
        )
    }

    /// Parse a wire token (already lowercased/stripped).
    pub fn parse_token(token: &str) -> Result<Self, ControlError> {
        match token {
            "start" => Ok(Self::Start),
            "stop" => Ok(Self::Stop),
            "toggle" => Ok(Self::Toggle),
            "start_alt" => Ok(Self::StartAlt),
            "toggle_alt" => Ok(Self::ToggleAlt),
            "status" => Ok(Self::Status),
            "ping" => Ok(Self::Ping),
            "metrics" => Ok(Self::Metrics),
            "debug_status" => Ok(Self::DebugStatus),
            "maintenance_reserve" => Ok(Self::MaintenanceReserve),
            "maintenance_release" => Ok(Self::MaintenanceRelease(0)),
            "tts_speak" => Ok(Self::TtsSpeak),
            "tts_speak_clipboard" => Ok(Self::TtsSpeakClipboard),
            "tts_pause" => Ok(Self::TtsPause),
            "tts_resume" => Ok(Self::TtsResume),
            "tts_toggle_pause" => Ok(Self::TtsTogglePause),
            "tts_restart" => Ok(Self::TtsRestart),
            "tts_stop" => Ok(Self::TtsStop),
            "tts_status" => Ok(Self::TtsStatus),
            other => Err(ControlError::InvalidCommand(other.to_string())),
        }
    }
}

impl FromStr for ControlCommand {
    type Err = ControlError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalized = s.trim().to_ascii_lowercase();
        Self::parse_token(&normalized)
    }
}

impl fmt::Display for ControlCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_has_nineteen_commands() {
        assert_eq!(CONTROL_COMMANDS.len(), 19);
        for token in CONTROL_COMMANDS {
            assert!(ControlCommand::parse_token(token).is_ok());
        }
    }

    #[test]
    fn alt_case_commands_round_trip() {
        assert_eq!(
            ControlCommand::parse_token("start_alt").unwrap(),
            ControlCommand::StartAlt
        );
        assert_eq!(
            ControlCommand::parse_token("toggle_alt").unwrap(),
            ControlCommand::ToggleAlt
        );
        assert_eq!(ControlCommand::StartAlt.as_str(), "start_alt");
        assert_eq!(ControlCommand::ToggleAlt.as_str(), "toggle_alt");
        assert!(!ControlCommand::StartAlt.is_tts());
    }

    #[test]
    fn parse_is_case_insensitive() {
        assert_eq!(
            "TTS_SPEAK".parse::<ControlCommand>().unwrap(),
            ControlCommand::TtsSpeak
        );
    }
}
