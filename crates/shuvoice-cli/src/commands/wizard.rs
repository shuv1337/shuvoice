//! Setup wizard command.

use std::sync::Arc;
use std::time::Duration;

use super::wizard_service::{self, Readiness};
use crate::error::{EXIT_DEPENDENCY, EXIT_FAILURE, EXIT_SUCCESS, ExitStatus};
use shuvoice_io::process::{CommandRunner, StdCommandRunner};
use shuvoice_io::waybar::{run_systemctl_user, service_action, service_active_state};

const SERVICE: &str = "shuvoice.service";

#[cfg(not(feature = "ui"))]
const NO_UI_MESSAGE: &str = "\
ERROR: setup wizard UI is not available in this build (missing `ui` feature / GTK4).\n\
Rebuild with: cargo build -p shuvoice-cli --features ui\n\
Or install a package built with UI support.";

#[cfg(feature = "ui")]
const NO_LAYER_SHELL_MESSAGE: &str = "\
ERROR: libgtk4-layer-shell.so not found.\n\
Install it with: pacman -S gtk4-layer-shell";

/// Launch the setup wizard (force reconfigure).
pub fn run_wizard_command() -> ExitStatus {
    if let Some(bin) = super::settings::installed_settings_bin() {
        return super::settings::launch_onboarding(&bin, false);
    }
    run_gtk_wizard_command()
}

pub fn run_gtk_wizard_command() -> ExitStatus {
    dispatch_wizard_launch(run_welcome_wizard(true), |service| {
        maybe_restart_running_service(service)
    })
}

/// Shared completion policy for `wizard` / first-run paths.
///
/// Restart/start is attempted **only** after [`WizardLaunch::Completed`].
/// Unavailable launches map to dependency exit code 78.
pub fn dispatch_wizard_launch(
    launch: WizardLaunch,
    on_completed: impl FnOnce(&str) -> &'static str,
) -> ExitStatus {
    match launch {
        WizardLaunch::Completed => {
            let result = on_completed(SERVICE);
            ExitStatus::code(if result == "failed" {
                EXIT_FAILURE
            } else {
                EXIT_SUCCESS
            })
        }
        WizardLaunch::Cancelled => ExitStatus::code(EXIT_SUCCESS),
        WizardLaunch::Unavailable { message, code } => {
            eprintln!("{message}");
            ExitStatus::code(code)
        }
    }
}

/// Outcome of attempting to launch the wizard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WizardLaunch {
    Completed,
    Cancelled,
    Unavailable { message: String, code: i32 },
}

/// Launch the setup wizard.
///
/// Returns whether the user completed setup (Launch). When the UI feature or
/// layer-shell runtime is missing, returns [`WizardLaunch::Unavailable`] with
/// dependency exit code 78.
pub fn run_welcome_wizard(force_reconfigure: bool) -> WizardLaunch {
    run_welcome_wizard_impl(force_reconfigure)
}

#[cfg(feature = "ui")]
fn run_welcome_wizard_impl(force_reconfigure: bool) -> WizardLaunch {
    if !layer_shell_present() {
        return WizardLaunch::Unavailable {
            message: NO_LAYER_SHELL_MESSAGE.into(),
            code: EXIT_DEPENDENCY,
        };
    }

    match shuvoice_ui::run_welcome_wizard_gtk_deferred(force_reconfigure) {
        Ok(true) => WizardLaunch::Completed,
        Ok(false) => WizardLaunch::Cancelled,
        Err(err) => WizardLaunch::Unavailable {
            message: format!(
                "ERROR: {err}\n\
                 The wizard needs a working Wayland/X11 display session with GTK4."
            ),
            code: EXIT_DEPENDENCY,
        },
    }
}

#[cfg(not(feature = "ui"))]
fn run_welcome_wizard_impl(_force_reconfigure: bool) -> WizardLaunch {
    WizardLaunch::Unavailable {
        message: NO_UI_MESSAGE.into(),
        code: EXIT_DEPENDENCY,
    }
}

/// Bool-shaped helper used by `run` (true = completed).
pub fn run_welcome_wizard_completed(force_reconfigure: bool) -> bool {
    matches!(
        run_welcome_wizard(force_reconfigure),
        WizardLaunch::Completed
    )
}

/// Start or restart `service` after the wizard completes (real systemctl).
///
/// Returns `"failed"` only when the handoff, the systemctl action, or overlay
/// readiness failed; a healthy service that is still loading returns
/// `"starting"`.
pub fn maybe_restart_running_service(service: &str) -> &'static str {
    report(service, &restart_live(service))
}

/// Start/restart `service` from the current desktop session: validate and
/// import its display environment, then wait for overlay readiness. Prints
/// nothing (the settings bridge uses stdout for its protocol).
pub fn restart_live(service: &str) -> RestartOutcome {
    restart_live_with_revision(service, None)
}

pub fn restart_live_with_revision(
    service: &str,
    expected_revision: Option<&str>,
) -> RestartOutcome {
    let mut previous_invocation = wizard_service::service_invocation_id(&StdCommandRunner, service);
    if previous_invocation.is_empty()
        && matches!(
            wizard_service::unit_snapshot(&StdCommandRunner, service)
                .state
                .as_str(),
            "inactive" | "failed" | "dead"
        )
    {
        // Proven stopped unit: a nonempty current InvocationID is a new start.
        previous_invocation = "<stopped>".into();
    }
    let control_socket = crate::config::load_config()
        .ok()
        .and_then(|config| config.control_socket);
    restart_service(
        service,
        None,
        || {
            wizard_service::refresh_display_environment(&StdCommandRunner, |name| {
                std::env::var(name).ok()
            })
        },
        || {
            wizard_service::wait_until_ready(
                service,
                &StdCommandRunner,
                control_socket.as_deref(),
                &previous_invocation,
                expected_revision,
            )
        },
    )
}

/// Injectable variant for tests (scripted `systemctl --user` runner).
pub fn maybe_restart_running_service_with(
    service: &str,
    runner: Option<Arc<dyn CommandRunner>>,
) -> &'static str {
    restart_with_checks(service, runner, || Ok(()), || Readiness::Ready)
}

/// Result of [`restart_service`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum RestartOutcome {
    /// Overlay reported ready.
    Ready { action: &'static str },
    #[serde(rename = "ready")]
    ReadyLegacyRevision {
        action: &'static str,
        message: String,
    },
    /// Unit healthy but not ready within the budget (e.g. slow model load).
    Starting {
        action: &'static str,
        message: String,
    },
    /// The current desktop's display could not be handed to the service.
    HandoffFailed {
        action: &'static str,
        message: String,
    },
    ActionFailed {
        action: &'static str,
        message: String,
    },
    ReadinessFailed {
        action: &'static str,
        message: String,
    },
    /// `systemctl --user` is not available.
    Unavailable,
    /// The unit is in a state we do not start/restart from.
    NotActive { state: String },
}

impl RestartOutcome {
    /// Stable status word used by the wizard's exit-code policy.
    pub fn status(&self) -> &'static str {
        match self {
            Self::Ready { action: "start" } => "started",
            Self::Ready { .. } => "restarted",
            Self::ReadyLegacyRevision {
                action: "start", ..
            } => "started",
            Self::ReadyLegacyRevision { .. } => "restarted",
            Self::Starting { .. } => "starting",
            Self::HandoffFailed { .. }
            | Self::ActionFailed { .. }
            | Self::ReadinessFailed { .. } => "failed",
            Self::Unavailable => "unavailable",
            Self::NotActive { .. } => "not_active",
        }
    }
}

fn past_tense(action: &str) -> &'static str {
    if action == "start" {
        "Started"
    } else {
        "Restarted"
    }
}

/// Print the wizard's human-readable result and return its status word.
fn report(service: &str, outcome: &RestartOutcome) -> &'static str {
    match outcome {
        RestartOutcome::Ready { action } | RestartOutcome::ReadyLegacyRevision { action, .. } => {
            println!(
                "✓ {} {service} so wizard changes take effect.",
                past_tense(action)
            );
        }
        RestartOutcome::Starting { action, message } => {
            println!(
                "✓ {} {service}; overlay not ready yet: {message}",
                past_tense(action)
            );
        }
        RestartOutcome::HandoffFailed { action, message } => eprintln!(
            "WARNING: did not {action} {service}: {message}\n         Run `shuvoice wizard` from a terminal in the current desktop session, or see \"Service fails after the wizard\" in docs/TROUBLESHOOTING.md."
        ),
        RestartOutcome::ActionFailed { action, message } => eprintln!(
            "WARNING: failed to {action} {service} automatically: {message}\n         Run `systemctl --user {action} {service}` manually for wizard changes to take effect."
        ),
        RestartOutcome::ReadinessFailed { action, message } => eprintln!(
            "WARNING: {} {service}, but the overlay did not become ready: {message}",
            past_tense(action)
        ),
        RestartOutcome::Unavailable | RestartOutcome::NotActive { .. } => {}
    }
    outcome.status()
}

fn restart_with_checks(
    service: &str,
    runner: Option<Arc<dyn CommandRunner>>,
    prepare: impl FnOnce() -> Result<(), String>,
    ready: impl FnOnce() -> Readiness,
) -> &'static str {
    report(service, &restart_service(service, runner, prepare, ready))
}

/// Start (stopped/failed unit) or restart (running unit) `service`.
fn restart_service(
    service: &str,
    runner: Option<Arc<dyn CommandRunner>>,
    prepare: impl FnOnce() -> Result<(), String>,
    ready: impl FnOnce() -> Readiness,
) -> RestartOutcome {
    let runner: Arc<dyn CommandRunner> = runner.unwrap_or_else(|| Arc::new(StdCommandRunner));
    let state = service_active_state(service, Some(Arc::clone(&runner)));
    if state == "unknown" {
        return RestartOutcome::Unavailable;
    }
    let action = if matches!(state.as_str(), "active" | "activating" | "reloading") {
        "restart"
    } else if matches!(
        state.as_str(),
        "inactive" | "failed" | "dead" | "deactivating"
    ) {
        "start"
    } else {
        return RestartOutcome::NotActive { state };
    };

    if let Err(message) = prepare() {
        return RestartOutcome::HandoffFailed { action, message };
    }
    if action == "start" && state == "failed" {
        // A unit that hit its start limit refuses an explicit start until reset.
        let _ = run_systemctl_user(
            runner.as_ref(),
            &["reset-failed", service],
            Duration::from_secs(3),
        );
    }
    if let Err(message) = service_action(service, action, Some(runner)) {
        return RestartOutcome::ActionFailed { action, message };
    }
    match ready() {
        Readiness::Ready => RestartOutcome::Ready { action },
        Readiness::ReadyLegacyRevision => RestartOutcome::ReadyLegacyRevision {
            action,
            message: "The running binary does not report config_revision; readiness used the new invocation and ui_ready only.".into(),
        },
        Readiness::StillStarting(message) => RestartOutcome::Starting { action, message },
        Readiness::Failed(message) => RestartOutcome::ReadinessFailed { action, message },
    }
}

#[cfg(feature = "ui")]
fn layer_shell_present() -> bool {
    for dir in ["/usr/lib", "/usr/lib64", "/usr/local/lib"] {
        if std::path::Path::new(dir)
            .join("libgtk4-layer-shell.so")
            .exists()
            || std::path::Path::new(dir)
                .join("libgtk4-layer-shell.so.0")
                .exists()
        {
            return true;
        }
    }
    // Also try opening the soname via libloading-less dlopen probe.
    #[cfg(unix)]
    if libc_dlopen_probe("libgtk4-layer-shell.so.0") {
        return true;
    }
    false
}

#[cfg(all(unix, feature = "ui"))]
fn libc_dlopen_probe(name: &str) -> bool {
    use std::ffi::CString;
    let Ok(c) = CString::new(name) else {
        return false;
    };
    // SAFETY: probe-only dlopen/dlclose of a well-formed soname; handle is closed immediately.
    unsafe {
        let h = libc::dlopen(c.as_ptr(), libc::RTLD_LAZY);
        if h.is_null() {
            false
        } else {
            libc::dlclose(h);
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shuvoice_io::process::{RunOutput, ScriptedRunner};
    use std::sync::Mutex;

    #[test]
    #[cfg(not(feature = "ui"))]
    fn wizard_unavailable_without_ui_feature() {
        let result = run_welcome_wizard(false);
        match result {
            WizardLaunch::Unavailable { message, code } => {
                assert_eq!(code, EXIT_DEPENDENCY);
                assert!(
                    message.contains("ui") || message.contains("UI") || message.contains("GTK")
                );
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_restarts_only_after_completed() {
        let calls = Mutex::new(Vec::<String>::new());
        let status = dispatch_wizard_launch(WizardLaunch::Completed, |svc| {
            calls.lock().unwrap().push(svc.to_string());
            "restarted"
        });
        assert_eq!(status.code, EXIT_SUCCESS);
        assert_eq!(calls.lock().unwrap().as_slice(), [SERVICE]);

        calls.lock().unwrap().clear();
        let status = dispatch_wizard_launch(WizardLaunch::Cancelled, |svc| {
            calls.lock().unwrap().push(svc.to_string());
            "restarted"
        });
        assert_eq!(status.code, EXIT_SUCCESS);
        assert!(calls.lock().unwrap().is_empty());
    }

    #[test]
    fn dispatch_unavailable_exits_78() {
        let calls = Mutex::new(0usize);
        let status = dispatch_wizard_launch(
            WizardLaunch::Unavailable {
                message: "no display".into(),
                code: EXIT_DEPENDENCY,
            },
            |_| {
                *calls.lock().unwrap() += 1;
                "restarted"
            },
        );
        assert_eq!(status.code, EXIT_DEPENDENCY);
        assert_eq!(*calls.lock().unwrap(), 0);
    }

    fn scripted_state_then_action(state: &'static str, action_ok: bool) -> Arc<ScriptedRunner> {
        let r = ScriptedRunner::new();
        r.set_dynamic(move |argv| {
            if argv.iter().any(|a| a == "show") {
                return Ok(RunOutput {
                    status_code: Some(0),
                    stdout: format!("{state}\n").into_bytes(),
                    stderr: Vec::new(),
                    success: true,
                });
            }
            // start/restart/stop
            Ok(RunOutput {
                status_code: Some(if action_ok { 0 } else { 1 }),
                stdout: Vec::new(),
                stderr: if action_ok {
                    Vec::new()
                } else {
                    b"boom\n".to_vec()
                },
                success: action_ok,
            })
        });
        Arc::new(r)
    }

    #[test]
    fn maybe_restart_active_restarts() {
        let r = scripted_state_then_action("active", true);
        let status = maybe_restart_running_service_with("shuvoice.service", Some(r.clone()));
        assert_eq!(status, "restarted");
        let calls = r.calls();
        assert!(calls.iter().any(|c| c.iter().any(|a| a == "restart")));
    }

    #[test]
    fn failed_display_handoff_does_not_restart_service() {
        let r = scripted_state_then_action("active", true);
        let status = restart_with_checks(
            SERVICE,
            Some(r.clone()),
            || Err("stale display".into()),
            || panic!("readiness must not run"),
        );
        assert_eq!(status, "failed");
        assert!(!r.calls().iter().any(|c| c.iter().any(|a| a == "restart")));
    }

    #[test]
    fn active_systemd_state_does_not_override_failed_overlay_readiness() {
        let r = scripted_state_then_action("active", true);
        let status = restart_with_checks(
            SERVICE,
            Some(r.clone()),
            || Ok(()),
            || Readiness::Failed("GTK did not become ready".into()),
        );
        assert_eq!(status, "failed");
        assert!(r.calls().iter().any(|c| c.iter().any(|a| a == "restart")));
        let result = dispatch_wizard_launch(WizardLaunch::Completed, |_| status);
        assert_eq!(result.code, EXIT_FAILURE);
    }

    #[test]
    fn slow_but_healthy_service_is_not_a_wizard_failure() {
        let r = scripted_state_then_action("active", true);
        let status = restart_with_checks(
            SERVICE,
            Some(r),
            || Ok(()),
            || Readiness::StillStarting("loading model".into()),
        );
        assert_eq!(status, "starting");
        let result = dispatch_wizard_launch(WizardLaunch::Completed, |_| status);
        assert_eq!(result.code, EXIT_SUCCESS);
    }

    #[test]
    fn revision_readiness_outcomes_are_visible_on_the_bridge_wire() {
        let legacy = restart_service(
            SERVICE,
            Some(scripted_state_then_action("active", true)),
            || Ok(()),
            || Readiness::ReadyLegacyRevision,
        );
        let value = serde_json::to_value(legacy).unwrap();
        assert_eq!(value["outcome"], "ready");
        assert!(
            value["message"]
                .as_str()
                .unwrap()
                .contains("does not report config_revision")
        );
        let failed = restart_service(
            SERVICE,
            Some(scripted_state_then_action("active", true)),
            || Ok(()),
            || Readiness::Failed("different config revision".into()),
        );
        let value = serde_json::to_value(failed).unwrap();
        assert_eq!(value["outcome"], "readiness_failed");
        assert_eq!(value["message"], "different config revision");
    }

    #[test]
    fn failed_unit_is_reset_before_explicit_start() {
        let r = scripted_state_then_action("failed", true);
        let status = restart_with_checks(SERVICE, Some(r.clone()), || Ok(()), || Readiness::Ready);
        assert_eq!(status, "started");
        let calls = r.calls();
        let pos = |verb: &str| calls.iter().position(|c| c.iter().any(|a| a == verb));
        assert!(
            pos("reset-failed").unwrap() < pos("start").unwrap(),
            "{calls:?}"
        );
    }

    #[test]
    fn maybe_restart_inactive_starts() {
        let r = scripted_state_then_action("inactive", true);
        let status = maybe_restart_running_service_with("shuvoice.service", Some(r.clone()));
        assert_eq!(status, "started");
        let calls = r.calls();
        assert!(calls.iter().any(|c| c.iter().any(|a| a == "start")));
        assert!(!calls.iter().any(|c| c.iter().any(|a| a == "restart")));
    }

    #[test]
    fn maybe_restart_failed_unit_starts() {
        let r = scripted_state_then_action("failed", true);
        assert_eq!(
            maybe_restart_running_service_with("shuvoice.service", Some(r)),
            "started"
        );
    }

    #[test]
    fn maybe_restart_unknown_is_unavailable() {
        let r = scripted_state_then_action("unknown", true);
        // service_active_state returns unknown on empty too; explicit unknown string works
        // when show succeeds with "unknown"
        assert_eq!(
            maybe_restart_running_service_with("shuvoice.service", Some(r)),
            "unavailable"
        );
    }

    #[test]
    fn maybe_restart_timeout_query_is_unavailable() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|_| {
            Err(shuvoice_io::ProcessError::Timeout {
                program: "systemctl".into(),
                timeout: std::time::Duration::from_secs(2),
            })
        });
        assert_eq!(
            maybe_restart_running_service_with("shuvoice.service", Some(Arc::new(r))),
            "unavailable"
        );
    }

    #[test]
    fn maybe_restart_action_failure_returns_failed() {
        let r = scripted_state_then_action("active", false);
        assert_eq!(
            maybe_restart_running_service_with("shuvoice.service", Some(r)),
            "failed"
        );
    }

    #[test]
    fn maybe_restart_other_state_is_not_active() {
        let r = scripted_state_then_action("maintenance", true);
        assert_eq!(
            maybe_restart_running_service_with("shuvoice.service", Some(r)),
            "not_active"
        );
    }

    #[test]
    fn run_wizard_command_unavailable_path_is_dependency_exit() {
        // Without mocking GTK, exercise dispatch policy used by run_wizard_command.
        let status = dispatch_wizard_launch(
            WizardLaunch::Unavailable {
                message: "missing ui".into(),
                code: EXIT_DEPENDENCY,
            },
            |_| panic!("restart must not run"),
        );
        assert_eq!(status.code, EXIT_DEPENDENCY);
    }
}
