//! systemd --user helpers for Waybar integration.

use std::sync::Arc;
use std::time::Duration;

use crate::error::ProcessError;
use crate::process::{CommandRunner, RunOptions, StdCommandRunner};

/// Run `systemctl --user …`.
pub fn run_systemctl_user(
    runner: &dyn CommandRunner,
    args: &[&str],
    timeout: Duration,
) -> Result<crate::process::RunOutput, ProcessError> {
    let mut full = vec!["systemctl".into(), "--user".into()];
    full.extend(args.iter().map(|s| (*s).to_string()));
    let opts = RunOptions {
        timeout,
        check: false,
        ..RunOptions::default()
    };
    runner.run(&full, &opts)
}

/// Return lowercased ActiveState or `"unknown"`.
#[must_use]
pub fn service_active_state(service: &str, runner: Option<Arc<dyn CommandRunner>>) -> String {
    let owned = runner.unwrap_or_else(|| Arc::new(StdCommandRunner));
    match run_systemctl_user(
        owned.as_ref(),
        &["show", "--property=ActiveState", "--value", service],
        Duration::from_secs(2),
    ) {
        Ok(out) if out.success => {
            let s = out.stdout_lossy();
            let trimmed = s.trim();
            if trimmed.is_empty() {
                "unknown".into()
            } else {
                trimmed.to_ascii_lowercase()
            }
        }
        Ok(_) | Err(ProcessError::Timeout { .. }) | Err(ProcessError::NotFound { .. }) => {
            "unknown".into()
        }
        Err(_) => "unknown".into(),
    }
}

/// Start/stop/restart a user service. Restart polls for immediate failure loops.
pub fn service_action(
    service: &str,
    action: &str,
    runner: Option<Arc<dyn CommandRunner>>,
) -> Result<(), String> {
    let owned = runner.unwrap_or_else(|| Arc::new(StdCommandRunner));
    let out = run_systemctl_user(owned.as_ref(), &[action, service], Duration::from_secs(3))
        .map_err(|e| e.to_string())?;
    if !out.success {
        let detail = {
            let err = out.stdout_lossy();
            let stderr = String::from_utf8_lossy(&out.stderr);
            let d = if !stderr.trim().is_empty() {
                stderr
            } else {
                err.into()
            };
            let t = d.trim();
            if t.is_empty() {
                "unknown error".into()
            } else {
                t.to_string()
            }
        };
        return Err(format!("systemctl {action} {service} failed: {detail}"));
    }

    if action != "restart" {
        return Ok(());
    }

    for _ in 0..8 {
        let state = service_active_state(service, Some(Arc::clone(&owned)));
        if state == "failed" {
            let _ = run_systemctl_user(owned.as_ref(), &["stop", service], Duration::from_secs(3));
            return Err(format!(
                "systemctl restart {service} entered failed state; stopped service to avoid restart loop"
            ));
        }
        if matches!(state.as_str(), "active" | "inactive" | "dead" | "unknown") {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::{RunOutput, ScriptedRunner, argv};

    #[test]
    fn active_state_unknown_on_timeout() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|_| {
            Err(ProcessError::Timeout {
                program: "systemctl".into(),
                timeout: Duration::from_secs(2),
            })
        });
        assert_eq!(
            service_active_state("shuvoice.service", Some(Arc::new(r))),
            "unknown"
        );
    }

    #[test]
    fn restart_stops_failed_service() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|argv| {
            // systemctl --user restart X → ok
            // systemctl --user show … → failed
            // systemctl --user stop X → ok
            if argv.iter().any(|a| a == "restart") {
                return Ok(RunOutput {
                    status_code: Some(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    success: true,
                });
            }
            if argv.iter().any(|a| a == "show") {
                return Ok(RunOutput {
                    status_code: Some(0),
                    stdout: b"failed\n".to_vec(),
                    stderr: Vec::new(),
                    success: true,
                });
            }
            if argv.iter().any(|a| a == "stop") {
                return Ok(RunOutput {
                    status_code: Some(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    success: true,
                });
            }
            Ok(RunOutput {
                status_code: Some(0),
                stdout: Vec::new(),
                stderr: Vec::new(),
                success: true,
            })
        });
        let err =
            service_action("shuvoice.service", "restart", Some(Arc::new(r.clone()))).unwrap_err();
        assert!(err.contains("restart loop") || err.contains("failed state"));
        let calls = r.calls();
        assert!(calls.iter().any(|c| c.iter().any(|a| a == "restart")));
        assert!(calls.iter().any(|c| c.iter().any(|a| a == "stop")));
    }

    #[test]
    fn _argv_helper_used() {
        let _ = argv(["systemctl"]);
    }
}
