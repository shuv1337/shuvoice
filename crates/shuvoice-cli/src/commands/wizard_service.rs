//! Graphical-session handoff and end-to-end readiness after wizard completion.

use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

use shuvoice_control::{ControlCommand, send_control_command};
use shuvoice_io::process::StdCommandRunner;
use shuvoice_io::waybar::{run_systemctl_user, service_active_state};

fn validate_display(display: &str, runtime: &Path) -> Result<(), String> {
    if display.is_empty() {
        return Err("WAYLAND_DISPLAY is empty".into());
    }
    let socket = runtime.join(display);
    UnixStream::connect(socket)
        .map(drop)
        .map_err(|err| format!("current Wayland display is unavailable: {err}"))
}

pub(super) fn refresh_display_environment() -> Result<(), String> {
    let display = std::env::var("WAYLAND_DISPLAY")
        .map_err(|_| "wizard service startup requires WAYLAND_DISPLAY from the current desktop")?;
    let runtime = std::env::var("XDG_RUNTIME_DIR")
        .map_err(|_| "wizard service startup requires XDG_RUNTIME_DIR")?;
    validate_display(&display, Path::new(&runtime))?;

    // Import only graphical-session variables, never the whole environment
    // (which can contain credentials). Clear absent optional values as well.
    let names = ["WAYLAND_DISPLAY", "DISPLAY", "XAUTHORITY", "GDK_BACKEND"];
    let (present, absent): (Vec<_>, Vec<_>) = names
        .into_iter()
        .partition(|name| std::env::var_os(name).is_some());
    for (command, names) in [
        ("import-environment", present),
        ("unset-environment", absent),
    ] {
        if names.is_empty() {
            continue;
        }
        let mut args = vec![command];
        args.extend(names);
        let out = run_systemctl_user(&StdCommandRunner, &args, Duration::from_secs(3))
            .map_err(|err| format!("cannot refresh service display environment: {err}"))?;
        if !out.success {
            return Err("systemctl could not refresh the service display environment".into());
        }
    }
    Ok(())
}

fn response_is_ready(response: &str) -> bool {
    response
        .strip_prefix("OK ")
        .and_then(|json| serde_json::from_str::<serde_json::Value>(json).ok())
        .and_then(|json| json.get("ui_ready").and_then(serde_json::Value::as_bool))
        == Some(true)
}

pub(super) fn wait_until_ready(service: &str) -> Result<(), String> {
    let config = crate::config::load_config()?;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let state = service_active_state(service, None);
        if matches!(state.as_str(), "failed" | "inactive" | "dead" | "unknown") {
            return Err(format!(
                "{service} is {state} before the overlay became ready; inspect journalctl --user -u {service}"
            ));
        }
        if state == "active"
            && send_control_command(
                ControlCommand::DebugStatus,
                config.control_socket.as_deref(),
                Some(Duration::from_millis(500)),
            )
            .is_ok_and(|response| response_is_ready(&response))
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "{service} did not report overlay readiness within 30 seconds; inspect journalctl --user -u {service}"
            ));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_requires_explicit_ui_ready() {
        for response in [
            "OK pong",
            "OK {}",
            "OK {\"ui_ready\":false}",
            "ERROR unavailable",
        ] {
            assert!(!response_is_ready(response));
        }
        assert!(response_is_ready("OK {\"ui_ready\":true}"));
    }

    #[test]
    fn validates_live_wayland_socket_not_just_its_name() {
        let dir = tempfile::tempdir().unwrap();
        let listener =
            std::os::unix::net::UnixListener::bind(dir.path().join("wayland-1")).unwrap();
        assert!(validate_display("wayland-1", dir.path()).is_ok());
        assert!(validate_display("wayland-2", dir.path()).is_err());
        assert!(validate_display("", dir.path()).is_err());
        drop(listener);
        assert!(validate_display("wayland-1", dir.path()).is_err());
    }
}
