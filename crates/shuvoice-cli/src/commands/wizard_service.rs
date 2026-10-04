//! Graphical-session handoff and end-to-end readiness after wizard completion.

use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

use shuvoice_control::{ControlCommand, send_control_command};
use shuvoice_io::process::CommandRunner;
use shuvoice_io::waybar::run_systemctl_user;

/// Desktop-session variables the user service needs from the current session.
///
/// Only names present in the wizard's environment are imported. Absent names
/// are left untouched in the user manager, which other units share.
pub(crate) const SESSION_ENV: [&str; 7] = [
    "WAYLAND_DISPLAY",
    "DISPLAY",
    "XAUTHORITY",
    "GDK_BACKEND",
    "HYPRLAND_INSTANCE_SIGNATURE",
    "XDG_CURRENT_DESKTOP",
    "XDG_SESSION_TYPE",
];

/// How long the wizard waits for `debug_status.ui_ready` after start/restart.
pub(super) const READY_BUDGET: Duration = Duration::from_secs(30);
const POLL_INTERVAL: Duration = Duration::from_millis(100);
const SYSTEMCTL_TIMEOUT: Duration = Duration::from_secs(3);

fn validate_display(display: &str, runtime: &Path) -> Result<(), String> {
    if display.is_empty() {
        return Err("WAYLAND_DISPLAY is empty".into());
    }
    let socket = runtime.join(display);
    UnixStream::connect(socket)
        .map(drop)
        .map_err(|err| format!("current Wayland display {display} is unavailable: {err}"))
}

/// Validate the wizard's own Wayland socket, then import the present
/// [`SESSION_ENV`] names into the user manager.
pub(super) fn refresh_display_environment(
    runner: &dyn CommandRunner,
    env: impl Fn(&str) -> Option<String>,
) -> Result<(), String> {
    let display = env("WAYLAND_DISPLAY")
        .ok_or("wizard service startup requires WAYLAND_DISPLAY from the current desktop")?;
    let runtime =
        env("XDG_RUNTIME_DIR").ok_or("wizard service startup requires XDG_RUNTIME_DIR")?;
    validate_display(&display, Path::new(&runtime))?;

    let mut args = vec!["import-environment"];
    args.extend(SESSION_ENV.into_iter().filter(|name| env(name).is_some()));
    let out = run_systemctl_user(runner, &args, SYSTEMCTL_TIMEOUT)
        .map_err(|err| format!("cannot refresh service display environment: {err}"))?;
    if !out.success {
        return Err("systemctl could not refresh the service display environment".into());
    }
    Ok(())
}

/// Outcome of waiting for the restarted service's overlay host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Readiness {
    Ready,
    ReadyLegacyRevision,
    /// Unit is active and healthy so far but not ready within the budget
    /// (e.g. a slow worker model load). Not a failure.
    StillStarting(String),
    Failed(String),
}

/// One `debug_status` probe result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Probe {
    Ready,
    ReadyLegacyRevision,
    RevisionMismatch,
    NotReady,
    /// The service answered with diagnostics that have no `ui_ready` field.
    Legacy,
    Unreachable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct UnitSnapshot {
    pub state: String,
    pub restarts: Option<u64>,
}

pub(super) fn probe_from_response(response: &str) -> Probe {
    let Some(json) = response
        .strip_prefix("OK ")
        .and_then(|body| serde_json::from_str::<serde_json::Value>(body).ok())
        .filter(serde_json::Value::is_object)
    else {
        return Probe::NotReady;
    };
    match json.get("ui_ready") {
        Some(serde_json::Value::Bool(true)) => Probe::Ready,
        Some(_) => Probe::NotReady,
        None => Probe::Legacy,
    }
}

pub(super) fn unit_snapshot(runner: &dyn CommandRunner, service: &str) -> UnitSnapshot {
    let unknown = UnitSnapshot {
        state: "unknown".into(),
        restarts: None,
    };
    let Ok(out) = run_systemctl_user(
        runner,
        &["show", "-p", "ActiveState", "-p", "NRestarts", service],
        Duration::from_secs(2),
    ) else {
        return unknown;
    };
    if !out.success {
        return unknown;
    }
    let mut snapshot = unknown;
    for line in out.stdout_lossy().lines() {
        if let Some(state) = line.strip_prefix("ActiveState=") {
            if !state.trim().is_empty() {
                snapshot.state = state.trim().to_ascii_lowercase();
            }
        } else if let Some(n) = line.strip_prefix("NRestarts=") {
            snapshot.restarts = n.trim().parse().ok();
        }
    }
    snapshot
}

/// Real readiness wait: systemd state + control-socket `debug_status`.
pub(super) fn wait_until_ready(
    service: &str,
    runner: &dyn CommandRunner,
    control_socket: Option<&str>,
    previous_invocation: &str,
    expected_revision: Option<&str>,
) -> Readiness {
    let deadline = Instant::now() + READY_BUDGET;
    wait_until_ready_with(
        service,
        || unit_snapshot(runner, service),
        || match send_control_command(
            ControlCommand::DebugStatus,
            control_socket,
            Some(Duration::from_millis(500)),
        ) {
            Ok(response) => {
                let current = service_invocation_id(runner, service);
                readiness_probe(&response, previous_invocation, &current, expected_revision)
            }
            Err(_) => Probe::Unreachable,
        },
        || Instant::now() >= deadline,
        std::thread::sleep,
    )
}

pub(super) fn service_invocation_id(runner: &dyn CommandRunner, service: &str) -> String {
    run_systemctl_user(
        runner,
        &["show", "--value", "-p", "InvocationID", service],
        SYSTEMCTL_TIMEOUT,
    )
    .ok()
    .filter(|out| out.success)
    .map(|out| out.stdout_lossy().trim().to_string())
    .unwrap_or_default()
}

fn new_invocation_ready(response: &str, previous: &str, current: &str) -> bool {
    !previous.is_empty()
        && !current.is_empty()
        && current != previous
        && probe_from_response(response) == Probe::Ready
        && response
            .strip_prefix("OK ")
            .and_then(|body| serde_json::from_str::<serde_json::Value>(body).ok())
            .and_then(|v| v["invocation_id"].as_str().map(str::to_string))
            .as_deref()
            == Some(current)
}

fn readiness_probe(response: &str, previous: &str, current: &str, expected: Option<&str>) -> Probe {
    if previous.is_empty() || current.is_empty() || previous == current {
        return Probe::NotReady;
    }
    let probe = probe_from_response(response);
    if probe != Probe::Ready {
        return probe;
    }
    let fields: serde_json::Value =
        serde_json::from_str(response.strip_prefix("OK ").unwrap()).unwrap();
    // Older binaries lack socket invocation provenance. Only accept its absence
    // when revision provenance is absent too; a reported stale ID is never OK.
    if !new_invocation_ready(response, previous, current)
        && (fields.get("invocation_id").is_some() || fields.get("config_revision").is_some())
    {
        return Probe::NotReady;
    }
    match (expected, fields.get("config_revision")) {
        (Some(expected), Some(value)) if value.as_str() != Some(expected) => {
            Probe::RevisionMismatch
        }
        (Some(_), None) => Probe::ReadyLegacyRevision,
        _ => Probe::Ready,
    }
}

/// Readiness policy with injectable systemd/control/clock seams.
pub(super) fn wait_until_ready_with(
    service: &str,
    mut snapshot: impl FnMut() -> UnitSnapshot,
    mut probe: impl FnMut() -> Probe,
    mut expired: impl FnMut() -> bool,
    mut sleep: impl FnMut(Duration),
) -> Readiness {
    let journal = format!("journalctl --user -u {service} -b");
    let mut baseline = None;
    let mut saw_legacy = false;
    let last_state = loop {
        let snap = snapshot();
        if baseline.is_none() {
            baseline = snap.restarts;
        }
        if let (Some(before), Some(now)) = (baseline, snap.restarts)
            && now > before
        {
            return Readiness::Failed(format!(
                "{service} crashed and was restarted by systemd while starting; inspect {journal}"
            ));
        }
        match snap.state.as_str() {
            "failed" | "inactive" | "dead" => {
                return Readiness::Failed(format!(
                    "{service} is {} before the overlay became ready; inspect {journal}",
                    snap.state
                ));
            }
            "active" => match probe() {
                Probe::Ready => return Readiness::Ready,
                Probe::ReadyLegacyRevision => return Readiness::ReadyLegacyRevision,
                Probe::RevisionMismatch => return Readiness::Failed(
                    "The new service loaded a different config revision than the saved settings; reload settings and apply again.".into()
                ),
                Probe::Legacy => saw_legacy = true,
                Probe::NotReady | Probe::Unreachable => {}
            },
            // activating / reloading / deactivating / transient `unknown`.
            _ => {}
        }
        if expired() {
            break snap.state;
        }
        sleep(POLL_INTERVAL);
    };

    let secs = READY_BUDGET.as_secs();
    if last_state == "active" {
        let why = if saw_legacy {
            "the running binary does not report ui_ready; check the unit's ExecStart points at this build"
        } else {
            "it may still be loading the speech model"
        };
        Readiness::StillStarting(format!(
            "{service} is running but the overlay was not ready after {secs}s ({why}). \
             Check `shuvoice control debug_status` or {journal}"
        ))
    } else {
        Readiness::Failed(format!(
            "{service} did not become ready within {secs}s (state: {last_state}); inspect {journal}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shuvoice_io::process::{RunOutput, ScriptedRunner};
    use std::cell::Cell;
    use std::collections::HashMap;

    fn ok(stdout: &str) -> Result<RunOutput, shuvoice_io::ProcessError> {
        Ok(RunOutput {
            status_code: Some(0),
            stdout: stdout.as_bytes().to_vec(),
            stderr: Vec::new(),
            success: true,
        })
    }

    fn snap(state: &str, restarts: u64) -> UnitSnapshot {
        UnitSnapshot {
            state: state.into(),
            restarts: Some(restarts),
        }
    }

    /// Run the policy over scripted snapshots/probes; expires after the last snapshot.
    fn run(snaps: &[UnitSnapshot], probes: &[Probe]) -> (Readiness, usize) {
        let i = Cell::new(0usize);
        let p = Cell::new(0usize);
        let result = wait_until_ready_with(
            "shuvoice.service",
            || {
                let n = i.get();
                i.set(n + 1);
                snaps[n.min(snaps.len() - 1)].clone()
            },
            || {
                let n = p.get();
                p.set(n + 1);
                probes[n.min(probes.len() - 1)]
            },
            || i.get() >= snaps.len(),
            |_| {},
        );
        (result, p.get())
    }

    #[test]
    fn probe_requires_explicit_ui_ready() {
        for response in [
            "OK pong",
            "OK {\"ui_ready\":false}",
            "ERROR unavailable",
            "OK [1]",
        ] {
            assert_eq!(probe_from_response(response), Probe::NotReady, "{response}");
        }
        assert_eq!(probe_from_response("OK {}"), Probe::Legacy);
        assert_eq!(probe_from_response("OK {\"app\":{}}"), Probe::Legacy);
        assert_eq!(
            probe_from_response("OK {\"app\":{},\"ui_ready\":true}"),
            Probe::Ready
        );
    }

    #[test]
    fn readiness_rejects_old_invocation_and_old_socket() {
        let old = "OK {\"ui_ready\":true,\"invocation_id\":\"old\"}";
        let new = "OK {\"ui_ready\":true,\"invocation_id\":\"new\"}";
        assert!(!new_invocation_ready(old, "old", "old"));
        assert!(!new_invocation_ready(old, "old", "new"));
        assert!(!new_invocation_ready(new, "", "new"));
        assert!(new_invocation_ready(new, "<stopped>", "new"));
        assert!(!new_invocation_ready(new, "old", ""));
        assert!(!new_invocation_ready(
            "OK {\"ui_ready\":true}",
            "old",
            "new"
        ));
        assert!(new_invocation_ready(new, "old", "new"));
    }

    #[test]
    fn readiness_requires_saved_revision_and_documents_legacy_fallback() {
        let ready =
            "OK {\"ui_ready\":true,\"invocation_id\":\"new\",\"config_revision\":\"saved\"}";
        assert_eq!(
            readiness_probe(ready, "old", "new", Some("saved")),
            Probe::Ready
        );
        assert_eq!(
            readiness_probe(ready, "old", "new", Some("other")),
            Probe::RevisionMismatch
        );
        assert_eq!(
            readiness_probe(ready, "old", "old", Some("other")),
            Probe::NotReady
        );
        let legacy = "OK {\"ui_ready\":true}";
        assert_eq!(
            readiness_probe(legacy, "old", "new", Some("saved")),
            Probe::ReadyLegacyRevision
        );
        assert_eq!(
            readiness_probe(legacy, "old", "old", Some("saved")),
            Probe::NotReady
        );
        let stale = "OK {\"ui_ready\":true,\"invocation_id\":\"old\"}";
        assert_eq!(
            readiness_probe(stale, "old", "new", Some("saved")),
            Probe::NotReady
        );
        let missing_ui = "OK {\"invocation_id\":\"new\",\"config_revision\":\"saved\"}";
        assert_eq!(
            readiness_probe(missing_ui, "old", "new", Some("saved")),
            Probe::Legacy
        );
        let (failed, _) = run(&[snap("active", 0)], &[Probe::RevisionMismatch]);
        assert!(
            matches!(failed, Readiness::Failed(message) if message.contains("different config revision"))
        );
        assert_eq!(
            run(&[snap("active", 0)], &[Probe::ReadyLegacyRevision]).0,
            Readiness::ReadyLegacyRevision
        );
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

    #[test]
    fn refresh_imports_present_session_vars_and_never_unsets() {
        let dir = tempfile::tempdir().unwrap();
        let _listener =
            std::os::unix::net::UnixListener::bind(dir.path().join("wayland-1")).unwrap();
        let vars: HashMap<&str, String> = [
            ("WAYLAND_DISPLAY", "wayland-1".to_string()),
            ("XDG_RUNTIME_DIR", dir.path().display().to_string()),
            ("DISPLAY", ":0".to_string()),
            ("HYPRLAND_INSTANCE_SIGNATURE", "sig".to_string()),
        ]
        .into_iter()
        .collect();
        let runner = ScriptedRunner::new();
        runner.set_dynamic(|_| ok(""));
        refresh_display_environment(&runner, |name| vars.get(name).cloned()).unwrap();

        let calls = runner.calls();
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert_eq!(
            calls[0],
            [
                "systemctl",
                "--user",
                "import-environment",
                "WAYLAND_DISPLAY",
                "DISPLAY",
                "HYPRLAND_INSTANCE_SIGNATURE",
            ]
        );
    }

    #[test]
    fn stale_wizard_display_never_touches_the_manager() {
        let dir = tempfile::tempdir().unwrap();
        let runner = ScriptedRunner::new();
        runner.set_dynamic(|_| ok(""));
        let runtime = dir.path().display().to_string();
        let err = refresh_display_environment(&runner, |name| match name {
            "WAYLAND_DISPLAY" => Some("wayland-stale".into()),
            "XDG_RUNTIME_DIR" => Some(runtime.clone()),
            _ => None,
        })
        .unwrap_err();
        assert!(err.contains("wayland-stale"), "{err}");
        assert!(runner.calls().is_empty());
    }

    #[test]
    fn unit_snapshot_parses_state_and_restart_count() {
        let runner = ScriptedRunner::new();
        runner.set_dynamic(|_| ok("ActiveState=active\nNRestarts=3\n"));
        assert_eq!(
            unit_snapshot(&runner, "shuvoice.service"),
            snap("active", 3)
        );
    }

    #[test]
    fn ready_after_activating_and_transient_unknown() {
        let unknown = UnitSnapshot {
            state: "unknown".into(),
            restarts: None,
        };
        let (result, _) = run(
            &[
                snap("activating", 0),
                unknown,
                snap("active", 0),
                snap("active", 0),
            ],
            &[Probe::Unreachable, Probe::Ready],
        );
        assert_eq!(result, Readiness::Ready);
    }

    #[test]
    fn terminal_unit_state_fails_without_probing() {
        for state in ["failed", "inactive", "dead"] {
            let (result, probes) = run(&[snap(state, 0)], &[Probe::Ready]);
            assert!(matches!(result, Readiness::Failed(ref m) if m.contains(state)));
            assert_eq!(probes, 0);
        }
    }

    #[test]
    fn crash_restart_during_wait_is_a_failure_even_if_active() {
        let (result, _) = run(
            &[snap("active", 0), snap("activating", 1), snap("active", 1)],
            &[Probe::Unreachable],
        );
        assert!(matches!(result, Readiness::Failed(ref m) if m.contains("crashed")));
    }

    #[test]
    fn slow_but_healthy_service_is_still_starting_not_failed() {
        let (result, _) = run(&vec![snap("active", 0); 5], &[Probe::NotReady]);
        assert!(
            matches!(result, Readiness::StillStarting(ref m) if m.contains("loading")),
            "{result:?}"
        );
    }

    #[test]
    fn legacy_binary_without_ui_ready_is_named_in_the_hint() {
        let (result, _) = run(&vec![snap("active", 0); 3], &[Probe::Legacy]);
        assert!(
            matches!(result, Readiness::StillStarting(ref m) if m.contains("ExecStart")),
            "{result:?}"
        );
    }

    #[test]
    fn not_active_at_deadline_is_a_failure() {
        let (result, _) = run(&vec![snap("activating", 0); 4], &[Probe::Unreachable]);
        assert!(matches!(result, Readiness::Failed(ref m) if m.contains("activating")));
    }
}
