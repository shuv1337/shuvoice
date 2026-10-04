//! `shuvoice settings`: open the settings app, or the setup wizard when the
//! app is not installed.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{EXIT_FAILURE, ExitStatus};

const SETTINGS_BIN: &str = "shuvoice-settings";

fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.metadata()
        .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

/// Locate the settings app from the install layout (never `PATH`):
/// `SHUVOICE_SETTINGS_BIN`, next to this executable, then `/usr/bin`.
pub fn resolve_settings_bin(
    override_path: Option<PathBuf>,
    current_exe: Option<PathBuf>,
) -> Option<PathBuf> {
    if let Some(path) = override_path {
        return is_executable(&path).then_some(path);
    }
    let sibling = current_exe.and_then(|exe| exe.parent().map(|dir| dir.join(SETTINGS_BIN)));
    [sibling, Some(PathBuf::from("/usr/bin").join(SETTINGS_BIN))]
        .into_iter()
        .flatten()
        .find(|path| is_executable(path))
}

pub fn installed_settings_bin() -> Option<PathBuf> {
    resolve_settings_bin(
        std::env::var_os("SHUVOICE_SETTINGS_BIN").map(PathBuf::from),
        std::env::current_exe().ok(),
    )
}

/// Launch independently of the speech unit's cgroup. No shell interpretation.
pub fn onboarding_command(bin: &Path, detached: bool) -> Command {
    let mut command = if detached {
        let mut command = Command::new("systemd-run");
        command.args(["--user", "--collect", "--quiet"]);
        for name in super::wizard_service::SESSION_ENV.into_iter().chain([
            "XDG_RUNTIME_DIR",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "SHUVOICE_BIN",
        ]) {
            if let Some(value) = std::env::var_os(name) {
                let mut argument = std::ffi::OsString::from(format!("--setenv={name}="));
                argument.push(value);
                command.arg(argument);
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            command.arg(format!("--setenv=SHUVOICE_BIN={}", exe.display()));
        }
        command.arg("--").arg(bin);
        command
    } else {
        Command::new(bin)
    };
    command.arg("--onboarding");
    if let Ok(exe) = std::env::current_exe() {
        command.env("SHUVOICE_BIN", exe);
    }
    command
}

pub fn launch_onboarding(bin: &Path, detached: bool) -> ExitStatus {
    match onboarding_command(bin, detached).status() {
        Ok(status) if status.success() && detached => {
            ExitStatus::code(crate::error::EXIT_DEPENDENCY)
        }
        Ok(status) => ExitStatus::code(status.code().unwrap_or(EXIT_FAILURE)),
        Err(err) => {
            eprintln!("ERROR: could not launch onboarding: {err}");
            ExitStatus::code(if detached {
                crate::error::EXIT_DEPENDENCY
            } else {
                EXIT_FAILURE
            })
        }
    }
}

pub fn run_settings_command(onboarding: bool) -> ExitStatus {
    let current_exe = std::env::current_exe().ok();
    let override_path = std::env::var_os("SHUVOICE_SETTINGS_BIN").map(PathBuf::from);
    let Some(bin) = resolve_settings_bin(override_path, current_exe.clone()) else {
        eprintln!("{SETTINGS_BIN} is not installed; opening the setup wizard instead.");
        return super::wizard::run_gtk_wizard_command();
    };
    if onboarding {
        return launch_onboarding(&bin, false);
    }
    let mut command = Command::new(&bin);
    // The app runs its bridge with this same `shuvoice` binary.
    if let Some(exe) = current_exe {
        command.env("SHUVOICE_BIN", exe);
    }
    match command.status() {
        Ok(status) => ExitStatus::code(status.code().unwrap_or(EXIT_FAILURE)),
        Err(err) => {
            eprintln!("ERROR: could not start {}: {err}", bin.display());
            ExitStatus::code(EXIT_FAILURE)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn touch(path: &Path, mode: u32) {
        std::fs::write(path, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    }

    #[test]
    fn prefers_override_then_sibling_and_requires_executable() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("shuvoice");
        let sibling = dir.path().join(SETTINGS_BIN);

        touch(&sibling, 0o644);
        assert_ne!(
            resolve_settings_bin(None, Some(exe.clone())),
            Some(sibling.clone()),
            "non-executable sibling is ignored"
        );

        touch(&sibling, 0o755);
        assert_eq!(
            resolve_settings_bin(None, Some(exe.clone())),
            Some(sibling.clone())
        );

        let custom = dir.path().join("custom-settings");
        touch(&custom, 0o755);
        assert_eq!(
            resolve_settings_bin(Some(custom.clone()), Some(exe)),
            Some(custom)
        );
        assert_eq!(
            resolve_settings_bin(Some(dir.path().join("missing")), None),
            None,
            "a broken override does not silently fall through"
        );
    }
}
