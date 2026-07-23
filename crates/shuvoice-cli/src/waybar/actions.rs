//! Waybar action layer and stable public action vocabulary.

use std::sync::Arc;
use std::time::Duration;

use shuvoice_core::Config;
use shuvoice_io::waybar::{service_action, service_active_state};

use super::deps::{WaybarDeps, resolve_shuvoice_bin};
use crate::control::ControlCmd;

/// Internal action tokens shared by the CLI and menu.
pub const ACTION_STATUS: &str = "status";
pub const ACTION_MENU: &str = "menu";
pub const ACTION_TOGGLE_RECORD: &str = "toggle-record";
pub const ACTION_START_RECORD: &str = "start-record";
pub const ACTION_STOP_RECORD: &str = "stop-record";
pub const ACTION_LAUNCH_WIZARD: &str = "launch-wizard";
pub const ACTION_SERVICE_START: &str = "service-start";
pub const ACTION_SERVICE_STOP: &str = "service-stop";
pub const ACTION_SERVICE_RESTART: &str = "service-restart";
pub const ACTION_SERVICE_TOGGLE: &str = "service-toggle";
pub const ACTION_TOGGLE_DEBUG_OVERLAY: &str = "toggle-debug-overlay";

/// Wire-stable error strings consumed by shell integrations.
pub const ERR_SOCKET_AFTER_START: &str = "control socket not ready after starting service";
pub const ERR_SOCKET_AFTER_RESTART: &str = "control socket not ready after restarting service";
pub const ERR_NO_MENU: &str = "No menu launcher found (install/use omarchy-launch-walker, walker, wofi, rofi, bemenu, or dmenu)";

const STATUS_TIMEOUT: Duration = Duration::from_millis(350);
const TOGGLE_STATUS_TIMEOUT: Duration = Duration::from_millis(500);
const CONTROL_CMD_TIMEOUT: Duration = Duration::from_secs(1);
const PING_TIMEOUT: Duration = Duration::from_millis(200);

/// Wait until the control socket answers `ping`, or timeout elapses.
pub fn wait_for_control_socket(deps: &WaybarDeps, config: &Config, timeout: Duration) -> bool {
    let deadline = deps.clock.now() + timeout;
    while deps.clock.now() < deadline {
        if deps
            .control
            .send("ping", config.control_socket.as_deref(), PING_TIMEOUT)
            .is_ok()
        {
            return true;
        }
        deps.sleeper.sleep(deps.control_ready_poll);
    }
    false
}

/// Query control status body (strip `OK ` prefix).
pub fn query_control_state(
    deps: &WaybarDeps,
    config: &Config,
    timeout: Duration,
) -> Result<String, String> {
    let response = deps.control.send(
        ControlCmd::Status.as_str(),
        config.control_socket.as_deref(),
        timeout,
    )?;
    Ok(response
        .strip_prefix("OK ")
        .unwrap_or(response.as_str())
        .trim()
        .to_string())
}

/// Map control/service state into Waybar runtime state.
pub fn query_runtime_state(
    deps: &WaybarDeps,
    config: &Config,
    service: &str,
) -> (String, Option<String>, Option<String>) {
    match query_control_state(deps, config, STATUS_TIMEOUT) {
        Ok(state) => (state, None, None),
        Err(err) => {
            let service_state = service_active_state(service, Some(Arc::clone(&deps.runner)));
            let mapped = match service_state.as_str() {
                "active" | "activating" | "deactivating" | "reloading" => "starting",
                "failed" => "error:service_failed",
                "inactive" | "dead" => "stopped",
                _ if err.to_ascii_lowercase().contains("socket not found") => "stopped",
                _ => "error:control_unreachable",
            };
            (mapped.into(), Some(service_state), Some(err))
        }
    }
}

fn ensure_service_running(deps: &WaybarDeps, service: &str) -> Result<(), String> {
    let state = service_active_state(service, Some(Arc::clone(&deps.runner)));
    if state == "active" {
        return Ok(());
    }
    service_action(service, "start", Some(Arc::clone(&deps.runner)))
}

fn action_toggle_record(deps: &WaybarDeps, config: &Config, service: &str) -> Result<(), String> {
    match query_control_state(deps, config, TOGGLE_STATUS_TIMEOUT) {
        Ok(state) => {
            if state == "recording" {
                deps.control.send(
                    ControlCmd::Stop.as_str(),
                    config.control_socket.as_deref(),
                    CONTROL_CMD_TIMEOUT,
                )?;
            } else {
                deps.control.send(
                    ControlCmd::Start.as_str(),
                    config.control_socket.as_deref(),
                    CONTROL_CMD_TIMEOUT,
                )?;
            }
            Ok(())
        }
        Err(_exc) => {
            ensure_service_running(deps, service)?;
            if !wait_for_control_socket(deps, config, deps.control_ready_timeout) {
                return Err(ERR_SOCKET_AFTER_START.into());
            }
            deps.control.send(
                ControlCmd::Start.as_str(),
                config.control_socket.as_deref(),
                CONTROL_CMD_TIMEOUT,
            )?;
            Ok(())
        }
    }
}

fn action_start_record(deps: &WaybarDeps, config: &Config, service: &str) -> Result<(), String> {
    ensure_service_running(deps, service)?;
    if !wait_for_control_socket(deps, config, deps.control_ready_timeout) {
        return Err(ERR_SOCKET_AFTER_START.into());
    }
    deps.control.send(
        ControlCmd::Start.as_str(),
        config.control_socket.as_deref(),
        CONTROL_CMD_TIMEOUT,
    )?;
    Ok(())
}

fn action_stop_record(deps: &WaybarDeps, config: &Config, service: &str) -> Result<(), String> {
    match deps.control.send(
        ControlCmd::Stop.as_str(),
        config.control_socket.as_deref(),
        CONTROL_CMD_TIMEOUT,
    ) {
        Ok(_) => Ok(()),
        Err(err) => {
            let state = service_active_state(service, Some(Arc::clone(&deps.runner)));
            if state == "active" { Err(err) } else { Ok(()) }
        }
    }
}

fn action_launch_wizard(deps: &WaybarDeps) -> Result<(), String> {
    let bin = resolve_shuvoice_bin();
    deps.launcher.spawn_detached(&bin, &["wizard"])
}

fn action_service_start(deps: &WaybarDeps, config: &Config, service: &str) -> Result<(), String> {
    service_action(service, "start", Some(Arc::clone(&deps.runner)))?;
    if !wait_for_control_socket(deps, config, deps.control_ready_timeout) {
        return Err(ERR_SOCKET_AFTER_START.into());
    }
    Ok(())
}

fn action_service_stop(deps: &WaybarDeps, service: &str) -> Result<(), String> {
    service_action(service, "stop", Some(Arc::clone(&deps.runner)))
}

fn action_service_restart(deps: &WaybarDeps, config: &Config, service: &str) -> Result<(), String> {
    service_action(service, "restart", Some(Arc::clone(&deps.runner)))?;
    if !wait_for_control_socket(deps, config, deps.control_ready_timeout) {
        return Err(ERR_SOCKET_AFTER_RESTART.into());
    }
    Ok(())
}

fn action_service_toggle(deps: &WaybarDeps, service: &str) -> Result<(), String> {
    let state = service_active_state(service, Some(Arc::clone(&deps.runner)));
    if matches!(state.as_str(), "active" | "activating" | "reloading") {
        service_action(service, "stop", Some(Arc::clone(&deps.runner)))
    } else {
        service_action(service, "start", Some(Arc::clone(&deps.runner)))
    }
}

fn action_toggle_debug_overlay(
    deps: &WaybarDeps,
    config: &mut Config,
    service: &str,
) -> Result<(), String> {
    let new_enabled = !config.overlay_debug_mode;
    deps.config_writer
        .set_overlay_debug_mode(new_enabled)
        .map_err(|_| {
            format!(
                "failed to set overlay_debug_mode={}",
                if new_enabled { "true" } else { "false" }
            )
        })?;
    config.overlay_debug_mode = new_enabled;

    service_action(service, "restart", Some(Arc::clone(&deps.runner)))?;
    if !wait_for_control_socket(deps, config, deps.control_ready_timeout) {
        return Err(ERR_SOCKET_AFTER_RESTART.into());
    }
    Ok(())
}

/// Build the stable menu option labels and commands.
pub fn menu_options(
    runtime_state: &str,
    service_state: &str,
    overlay_debug_mode: bool,
) -> Vec<(String, String)> {
    let recording_label = if runtime_state == "recording" {
        "Stop recording"
    } else {
        "Start recording"
    };
    let recording_command = if runtime_state == "recording" {
        ACTION_STOP_RECORD
    } else {
        ACTION_START_RECORD
    };
    let service_label = if matches!(service_state, "active" | "activating" | "reloading") {
        "Stop service"
    } else {
        "Start service"
    };
    let debug_label = if overlay_debug_mode {
        "Disable debug overlay"
    } else {
        "Enable debug overlay"
    };

    vec![
        (recording_label.into(), recording_command.into()),
        ("Toggle recording".into(), ACTION_TOGGLE_RECORD.into()),
        (debug_label.into(), ACTION_TOGGLE_DEBUG_OVERLAY.into()),
        (service_label.into(), ACTION_SERVICE_TOGGLE.into()),
        ("Relaunch setup wizard".into(), ACTION_LAUNCH_WIZARD.into()),
        (
            "Restart service (advanced)".into(),
            ACTION_SERVICE_RESTART.into(),
        ),
    ]
}

fn action_menu(deps: &WaybarDeps, config: &mut Config, service: &str) -> Result<(), String> {
    let (runtime_state, _, _) = query_runtime_state(deps, config, service);
    let service_state = service_active_state(service, Some(Arc::clone(&deps.runner)));
    let options = menu_options(&runtime_state, &service_state, config.overlay_debug_mode);
    let labels: Vec<String> = options.iter().map(|(l, _)| l.clone()).collect();

    let choice = match deps.menu.prompt("ShuVoice", &labels) {
        Ok(c) => c,
        Err(err) => {
            // Normalize the no-menu case to the stable public error string.
            if err.contains("No menu launcher found") {
                return Err(ERR_NO_MENU.into());
            }
            return Err(err);
        }
    };
    let Some(choice) = choice else {
        return Ok(());
    };

    let action_map: std::collections::BTreeMap<&str, &str> = options
        .iter()
        .map(|(l, c)| (l.as_str(), c.as_str()))
        .collect();
    let Some(command) = action_map.get(choice.as_str()).copied() else {
        return Ok(());
    };
    perform_action(deps, command, config, service)
}

/// Dispatch a Waybar action by its stable command token.
pub fn perform_action(
    deps: &WaybarDeps,
    command: &str,
    config: &mut Config,
    service: &str,
) -> Result<(), String> {
    match command {
        ACTION_STATUS => Ok(()),
        ACTION_MENU => action_menu(deps, config, service),
        ACTION_TOGGLE_RECORD => action_toggle_record(deps, config, service),
        ACTION_START_RECORD => action_start_record(deps, config, service),
        ACTION_STOP_RECORD => action_stop_record(deps, config, service),
        ACTION_TOGGLE_DEBUG_OVERLAY => action_toggle_debug_overlay(deps, config, service),
        ACTION_LAUNCH_WIZARD => action_launch_wizard(deps),
        ACTION_SERVICE_START => action_service_start(deps, config, service),
        ACTION_SERVICE_STOP => action_service_stop(deps, service),
        ACTION_SERVICE_RESTART => action_service_restart(deps, config, service),
        ACTION_SERVICE_TOGGLE => action_service_toggle(deps, service),
        other => Err(format!("Unknown command: {other}")),
    }
}
