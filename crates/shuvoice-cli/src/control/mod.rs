//! Control-socket CLI adapter over `shuvoice-control`.

use std::thread;
use std::time::{Duration, Instant};

use clap::ValueEnum;
use shuvoice_control::{
    ControlCommand, ControlError, DEFAULT_CLIENT_TIMEOUT, send_control_command,
    send_control_command_str,
};

/// Clap-facing control command enum (wire tokens match `shuvoice-control`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ControlCmd {
    Start,
    Stop,
    Toggle,
    Status,
    Ping,
    Metrics,
    #[value(name = "debug_status")]
    DebugStatus,
    #[value(name = "tts_speak")]
    TtsSpeak,
    #[value(name = "tts_speak_clipboard")]
    TtsSpeakClipboard,
    #[value(name = "tts_pause")]
    TtsPause,
    #[value(name = "tts_resume")]
    TtsResume,
    #[value(name = "tts_toggle_pause")]
    TtsTogglePause,
    #[value(name = "tts_restart")]
    TtsRestart,
    #[value(name = "tts_stop")]
    TtsStop,
    #[value(name = "tts_status")]
    TtsStatus,
}

impl ControlCmd {
    pub fn as_str(self) -> &'static str {
        self.to_control_command().as_str()
    }

    pub fn to_control_command(self) -> ControlCommand {
        match self {
            Self::Start => ControlCommand::Start,
            Self::Stop => ControlCommand::Stop,
            Self::Toggle => ControlCommand::Toggle,
            Self::Status => ControlCommand::Status,
            Self::Ping => ControlCommand::Ping,
            Self::Metrics => ControlCommand::Metrics,
            Self::DebugStatus => ControlCommand::DebugStatus,
            Self::TtsSpeak => ControlCommand::TtsSpeak,
            Self::TtsSpeakClipboard => ControlCommand::TtsSpeakClipboard,
            Self::TtsPause => ControlCommand::TtsPause,
            Self::TtsResume => ControlCommand::TtsResume,
            Self::TtsTogglePause => ControlCommand::TtsTogglePause,
            Self::TtsRestart => ControlCommand::TtsRestart,
            Self::TtsStop => ControlCommand::TtsStop,
            Self::TtsStatus => ControlCommand::TtsStatus,
        }
    }
}

impl From<ControlCmd> for ControlCommand {
    fn from(value: ControlCmd) -> Self {
        value.to_control_command()
    }
}

fn format_control_error(err: ControlError) -> String {
    err.to_string()
}

/// Send one control command (blocking client from `shuvoice-control`).
pub fn send_cmd(
    command: ControlCmd,
    socket_path: Option<&str>,
    timeout: Option<Duration>,
) -> Result<String, String> {
    send_control_command(command.into(), socket_path, timeout).map_err(format_control_error)
}

/// Send a raw string command.
pub fn send_cmd_str(
    command: &str,
    socket_path: Option<&str>,
    timeout: Option<Duration>,
) -> Result<String, String> {
    send_control_command_str(command, socket_path, timeout).map_err(format_control_error)
}

/// Run a control command with optional post-stop/toggle processing wait.
pub fn run_control(
    command: ControlCmd,
    socket_path: Option<&str>,
    wait_sec: f64,
) -> Result<String, String> {
    let mut status_before = String::new();
    if command == ControlCmd::Toggle && wait_sec > 0.0 {
        status_before = send_cmd(
            ControlCmd::Status,
            socket_path,
            Some(DEFAULT_CLIENT_TIMEOUT),
        )
        .unwrap_or_default();
    }

    let response = send_cmd(command, socket_path, Some(DEFAULT_CLIENT_TIMEOUT))?;

    let mut should_wait = false;
    if wait_sec > 0.0 {
        if command == ControlCmd::Stop {
            should_wait = true;
        } else if command == ControlCmd::Toggle {
            should_wait = status_before.trim().ends_with("recording");
        }
    }

    if should_wait {
        let deadline = Instant::now() + Duration::from_secs_f64(wait_sec);
        while Instant::now() < deadline {
            if let Ok(status) = send_cmd(
                ControlCmd::Status,
                socket_path,
                Some(Duration::from_secs(1)),
            ) {
                let state = status.strip_prefix("OK ").unwrap_or(status.as_str()).trim();
                if state != "processing" {
                    break;
                }
            }
            thread::sleep(Duration::from_millis(50));
        }
    }

    Ok(response)
}
