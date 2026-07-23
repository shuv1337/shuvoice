use crate::application::Application;
use crate::commands::wizard::{WizardLaunch, run_welcome_wizard};
use crate::config::{RuntimeOverrides, apply_runtime_overrides, load_config, needs_wizard};
use crate::error::{EXIT_DEPENDENCY, EXIT_FAILURE, EXIT_SUCCESS, ExitStatus};

pub async fn run_app(overrides: &RuntimeOverrides) -> ExitStatus {
    let mut config = match load_config() {
        Ok(c) => c,
        Err(err) => {
            eprintln!("ERROR: {err}");
            return ExitStatus::code(EXIT_DEPENDENCY);
        }
    };
    if let Err(err) = apply_runtime_overrides(&mut config, overrides) {
        eprintln!("ERROR: {err}");
        return ExitStatus::code(EXIT_DEPENDENCY);
    }

    if needs_wizard() {
        // First-run is in-process only; only `shuvoice wizard` touches systemd.
        match run_welcome_wizard(false) {
            WizardLaunch::Completed => {
                // Reload config below and continue in-process. No systemd action.
            }
            WizardLaunch::Cancelled => {
                // User closed wizard without finishing — do not start the app.
                return ExitStatus::code(EXIT_SUCCESS);
            }
            WizardLaunch::Unavailable { message, code } => {
                eprintln!("{message}");
                return ExitStatus::code(code);
            }
        }
        config = match load_config() {
            Ok(c) => c,
            Err(err) => {
                eprintln!("ERROR: {err}");
                return ExitStatus::code(EXIT_DEPENDENCY);
            }
        };
        if let Err(err) = apply_runtime_overrides(&mut config, overrides) {
            eprintln!("ERROR: {err}");
            return ExitStatus::code(EXIT_DEPENDENCY);
        }
    }

    let app = match Application::new(config) {
        Ok(app) => app,
        Err(err) => {
            eprintln!("ERROR: {err}");
            return ExitStatus::code(EXIT_DEPENDENCY);
        }
    };

    let status = app.run().await;
    if status.code == 0 || status.code == EXIT_DEPENDENCY {
        return status;
    }
    if let Some(msg) = status.message {
        eprintln!("ERROR: {msg}");
    }
    ExitStatus::code(EXIT_FAILURE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::wizard::{WizardLaunch, dispatch_wizard_launch};
    use crate::error::EXIT_DEPENDENCY;
    use std::sync::Mutex;

    fn dispatch_first_run_wizard_launch(launch: WizardLaunch) -> ExitStatus {
        match launch {
            WizardLaunch::Completed | WizardLaunch::Cancelled => ExitStatus::code(EXIT_SUCCESS),
            WizardLaunch::Unavailable { message: _, code } => ExitStatus::code(code),
        }
    }

    #[test]
    fn first_run_completed_does_not_invoke_service_action() {
        let calls = Mutex::new(0usize);
        let status = dispatch_first_run_wizard_launch(WizardLaunch::Completed);
        assert_eq!(status.code, EXIT_SUCCESS);
        assert_eq!(*calls.lock().unwrap(), 0);

        let status = dispatch_wizard_launch(WizardLaunch::Completed, |_| {
            *calls.lock().unwrap() += 1;
            "restarted"
        });
        assert_eq!(status.code, EXIT_SUCCESS);
        assert_eq!(*calls.lock().unwrap(), 1);
    }

    #[test]
    fn first_run_unavailable_is_exit_78() {
        let status = dispatch_first_run_wizard_launch(WizardLaunch::Unavailable {
            message: "no ui".into(),
            code: EXIT_DEPENDENCY,
        });
        assert_eq!(status.code, EXIT_DEPENDENCY);
    }
}
