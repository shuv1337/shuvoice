//! Setup wizard command.

use std::sync::Arc;

use crate::error::{EXIT_DEPENDENCY, EXIT_SUCCESS, ExitStatus};
use shuvoice_io::process::CommandRunner;
use shuvoice_io::waybar::{service_action, service_active_state};

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
            let _ = on_completed(SERVICE);
            ExitStatus::code(EXIT_SUCCESS)
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
pub fn maybe_restart_running_service(service: &str) -> &'static str {
    maybe_restart_running_service_with(service, None)
}

/// Injectable variant for tests (scripted `systemctl --user` runner).
pub fn maybe_restart_running_service_with(
    service: &str,
    runner: Option<Arc<dyn CommandRunner>>,
) -> &'static str {
    let state = service_active_state(service, runner.clone());
    if state == "unknown" {
        return "unavailable";
    }

    let (action, success_status, success_verb) =
        if matches!(state.as_str(), "active" | "activating" | "reloading") {
            ("restart", "restarted", "Restarted")
        } else if matches!(
            state.as_str(),
            "inactive" | "failed" | "dead" | "deactivating"
        ) {
            ("start", "started", "Started")
        } else {
            return "not_active";
        };

    match service_action(service, action, runner) {
        Ok(()) => {
            println!("✓ {success_verb} {service} so wizard changes take effect.");
            success_status
        }
        Err(err) => {
            eprintln!(
                "WARNING: failed to {action} {service} automatically: {err}\n         Run `systemctl --user {action} {service}` manually for wizard changes to take effect."
            );
            "failed"
        }
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
