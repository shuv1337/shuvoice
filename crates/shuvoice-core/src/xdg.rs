//! XDG path helpers matching the Python runtime layout.

use std::env;
use std::path::PathBuf;

/// Resolve `$XDG_CONFIG_HOME` or `~/.config`.
pub fn xdg_config_home() -> PathBuf {
    if let Ok(value) = env::var("XDG_CONFIG_HOME") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    home_dir().join(".config")
}

/// Resolve `$XDG_DATA_HOME` or `~/.local/share`.
pub fn xdg_data_home() -> PathBuf {
    if let Ok(value) = env::var("XDG_DATA_HOME") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    home_dir().join(".local").join("share")
}

/// Resolve `$XDG_RUNTIME_DIR` when set.
pub fn xdg_runtime_dir() -> Option<PathBuf> {
    env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from)
}

/// ShuVoice config directory: `$XDG_CONFIG_HOME/shuvoice`.
pub fn config_dir() -> PathBuf {
    xdg_config_home().join("shuvoice")
}

/// Default config file path.
pub fn config_path() -> PathBuf {
    config_dir().join("config.toml")
}

/// ShuVoice data directory: `$XDG_DATA_HOME/shuvoice`.
pub fn data_dir() -> PathBuf {
    xdg_data_home().join("shuvoice")
}

/// Local secrets file path (`local.dev`).
pub fn local_dev_env_path() -> PathBuf {
    config_dir().join("local.dev")
}

/// Wizard completion marker path.
pub fn wizard_done_path() -> PathBuf {
    data_dir().join(".wizard-done")
}

fn home_dir() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("USERPROFILE").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn config_paths_respect_xdg_overrides() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: ENV_LOCK serializes process-global env mutation in this test binary.
        unsafe {
            env::set_var("XDG_CONFIG_HOME", "/tmp/shuvoice-core-cfg");
            env::set_var("XDG_DATA_HOME", "/tmp/shuvoice-core-data");
        }
        assert_eq!(
            config_path(),
            PathBuf::from("/tmp/shuvoice-core-cfg/shuvoice/config.toml")
        );
        assert_eq!(
            data_dir(),
            PathBuf::from("/tmp/shuvoice-core-data/shuvoice")
        );
        // SAFETY: paired cleanup under the same ENV_LOCK.
        unsafe {
            env::remove_var("XDG_CONFIG_HOME");
            env::remove_var("XDG_DATA_HOME");
        }
    }
}
