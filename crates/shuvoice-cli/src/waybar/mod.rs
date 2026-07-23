//! Waybar helper binary: status JSON and click actions.

mod actions;
mod deps;

pub use actions::{
    ACTION_LAUNCH_WIZARD, ACTION_MENU, ACTION_SERVICE_RESTART, ACTION_SERVICE_START,
    ACTION_SERVICE_STOP, ACTION_SERVICE_TOGGLE, ACTION_START_RECORD, ACTION_STATUS,
    ACTION_STOP_RECORD, ACTION_TOGGLE_DEBUG_OVERLAY, ACTION_TOGGLE_RECORD, ERR_NO_MENU,
    ERR_SOCKET_AFTER_RESTART, ERR_SOCKET_AFTER_START, menu_options, perform_action,
    query_control_state, query_runtime_state, wait_for_control_socket,
};
pub use deps::{
    BinaryLookup, Clock, ConfigWriter, ControlClient, MENU_LAUNCHERS, MenuPrompt, ProcessLauncher,
    Sleeper, StdClock, StdConfigWriter, StdControlClient, StdMenuPrompt, StdProcessLauncher,
    StdSleeper, WaybarDeps, WhichLookup, resolve_shuvoice_bin,
};

use std::time::Duration;

use clap::{Parser, ValueEnum};
use shuvoice_core::{Config, format_tts_playback_speed};
use shuvoice_io::hyprland::detect_keybind;
use shuvoice_io::waybar::{WaybarConfigInfo, build_waybar_payload, config_info_lines};

use crate::error::{EXIT_FAILURE, EXIT_SUCCESS, ExitStatus};

const DEFAULT_SERVICE: &str = "shuvoice.service";

#[derive(Debug, Clone, Parser)]
#[command(
    name = "shuvoice-waybar",
    about = "Waybar status helper for ShuVoice",
    disable_version_flag = true
)]
pub struct WaybarCli {
    #[arg(value_enum, default_value_t = WaybarCommand::Status)]
    pub command: WaybarCommand,

    /// systemd --user service name
    #[arg(long = "service", default_value = DEFAULT_SERVICE, env = "SHUVOICE_SERVICE")]
    pub service: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum WaybarCommand {
    Status,
    Menu,
    #[value(name = "toggle-record")]
    ToggleRecord,
    #[value(name = "start-record")]
    StartRecord,
    #[value(name = "stop-record")]
    StopRecord,
    #[value(name = "launch-wizard")]
    LaunchWizard,
    #[value(name = "service-start")]
    ServiceStart,
    #[value(name = "service-stop")]
    ServiceStop,
    #[value(name = "service-restart")]
    ServiceRestart,
    #[value(name = "service-toggle")]
    ServiceToggle,
}

impl WaybarCommand {
    /// Wire-stable action token.
    pub fn as_action_str(self) -> &'static str {
        match self {
            Self::Status => ACTION_STATUS,
            Self::Menu => ACTION_MENU,
            Self::ToggleRecord => ACTION_TOGGLE_RECORD,
            Self::StartRecord => ACTION_START_RECORD,
            Self::StopRecord => ACTION_STOP_RECORD,
            Self::LaunchWizard => ACTION_LAUNCH_WIZARD,
            Self::ServiceStart => ACTION_SERVICE_START,
            Self::ServiceStop => ACTION_SERVICE_STOP,
            Self::ServiceRestart => ACTION_SERVICE_RESTART,
            Self::ServiceToggle => ACTION_SERVICE_TOGGLE,
        }
    }
}

/// Run the waybar helper with default (production) dependencies.
pub async fn run_waybar(cli: WaybarCli) -> ExitStatus {
    run_waybar_with_deps(cli, &WaybarDeps::default())
}

/// Run the waybar helper with injectable dependencies (tests).
pub fn run_waybar_with_deps(cli: WaybarCli, deps: &WaybarDeps) -> ExitStatus {
    let mut config = match Config::load() {
        Ok(c) => c,
        Err(err) => {
            // Secret-safe: only surface the error display, never env dumps.
            let msg = err.to_string();
            let payload = build_waybar_payload("error:config", None, None, None, Some(&msg));
            println!(
                "{}",
                serde_json::to_string(&payload).unwrap_or_else(|_| "{}".into())
            );
            eprintln!("ERROR: {msg}");
            return ExitStatus::code(EXIT_FAILURE);
        }
    };

    let mut action_error: Option<String> = None;
    let mut exit_code = EXIT_SUCCESS;

    if let Err(err) = perform_action(deps, cli.command.as_action_str(), &mut config, &cli.service) {
        action_error = Some(err);
        exit_code = EXIT_FAILURE;
    }

    // Reload config after debug toggle so tooltip reflects new state.
    if let Ok(fresh) = Config::load() {
        config = fresh;
    }

    let (state, service_state, control_error) = query_runtime_state(deps, &config, &cli.service);
    let mut info = config_info_lines(&waybar_config_info(&config));
    if let Some(bind) = detect_keybind("start", None, Duration::from_secs(2)) {
        info.push(format!("PTT Key:  {bind}"));
    }
    if let Some(bind) = detect_keybind("tts_speak", None, Duration::from_secs(2)) {
        info.push(format!("TTS Key:  {bind}"));
    }
    if let Some(bind) = detect_keybind("tts_speak_clipboard", None, Duration::from_secs(2)) {
        info.push(format!("TTS Clip: {bind}"));
    }

    if std::env::var("SHUVOICE_WAYBAR_DEBUG_METRICS")
        .map(|v| {
            let l = v.to_ascii_lowercase();
            matches!(l.as_str(), "1" | "true" | "yes")
        })
        .unwrap_or(false)
    {
        match deps.control.send(
            "metrics",
            config.control_socket.as_deref(),
            Duration::from_millis(300),
        ) {
            Ok(metrics) => {
                let body = metrics
                    .strip_prefix("OK ")
                    .unwrap_or(metrics.as_str())
                    .trim();
                info.push(format!("Metrics:  {body}"));
            }
            Err(_) => info.push("Metrics:  unavailable".into()),
        }
    }

    let payload = build_waybar_payload(
        &state,
        Some(&info),
        service_state.as_deref(),
        control_error.as_deref(),
        action_error.as_deref(),
    );
    println!(
        "{}",
        serde_json::to_string(&payload).unwrap_or_else(|_| "{}".into())
    );
    if let Some(err) = action_error {
        // Secret-safe: action errors are already plain strings without env dumps.
        eprintln!("ERROR: {err}");
    }
    ExitStatus::code(exit_code)
}

fn waybar_config_info(config: &Config) -> WaybarConfigInfo {
    let model_label = match config.asr_backend {
        shuvoice_core::AsrBackendKind::Sherpa => Some(config.sherpa_model_name.clone()),
        shuvoice_core::AsrBackendKind::Nemo => Some(config.model_name.clone()),
        shuvoice_core::AsrBackendKind::Moonshine => Some(config.moonshine_model_name.clone()),
        shuvoice_core::AsrBackendKind::OpenaiRealtime => Some(config.openai_realtime_model.clone()),
    };
    let device_label = match config.asr_backend {
        shuvoice_core::AsrBackendKind::Sherpa => Some(config.sherpa_provider.as_str().into()),
        shuvoice_core::AsrBackendKind::Nemo => Some(config.device.clone()),
        shuvoice_core::AsrBackendKind::Moonshine => Some(config.moonshine_provider.as_str().into()),
        shuvoice_core::AsrBackendKind::OpenaiRealtime => Some("cloud".into()),
    };

    WaybarConfigInfo {
        asr_backend: config.asr_backend.as_str().into(),
        instant_mode: config.instant_mode,
        model_label,
        device_label,
        tts_enabled: config.tts_enabled,
        tts_backend_label: Some(config.tts_backend.as_str().into()),
        tts_voice_label: Some(config.tts_default_voice_id.clone()),
        tts_speed_label: Some(format_tts_playback_speed(config.tts_playback_speed)),
        overlay_debug_mode: config.overlay_debug_mode,
    }
}
