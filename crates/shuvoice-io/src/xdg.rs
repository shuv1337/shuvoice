//! XDG path helpers (used when shuvoice-core does not yet expose them).

use std::env;
use std::path::PathBuf;

/// `$XDG_CONFIG_HOME` or `~/.config`.
#[must_use]
pub fn config_home() -> PathBuf {
    if let Some(v) = env::var_os("XDG_CONFIG_HOME")
        && !v.is_empty()
    {
        return PathBuf::from(v);
    }
    home_dir().join(".config")
}

/// `$XDG_RUNTIME_DIR` if set.
#[must_use]
pub fn runtime_dir() -> Option<PathBuf> {
    env::var_os("XDG_RUNTIME_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// `$XDG_DATA_HOME` or `~/.local/share`.
#[must_use]
pub fn data_home() -> PathBuf {
    if let Some(v) = env::var_os("XDG_DATA_HOME")
        && !v.is_empty()
    {
        return PathBuf::from(v);
    }
    home_dir().join(".local").join("share")
}

/// ShuVoice config directory: `$XDG_CONFIG_HOME/shuvoice`.
#[must_use]
pub fn shuvoice_config_dir() -> PathBuf {
    config_home().join("shuvoice")
}

/// ShuVoice data directory: `$XDG_DATA_HOME/shuvoice`.
#[must_use]
pub fn shuvoice_data_dir() -> PathBuf {
    data_home().join("shuvoice")
}

fn home_dir() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn config_home_respects_xdg() {
        let _g = LOCK.lock().unwrap();
        // SAFETY: serialized test env mutation.
        unsafe {
            env::set_var("XDG_CONFIG_HOME", "/tmp/xdg-config-home");
        }
        assert_eq!(config_home(), PathBuf::from("/tmp/xdg-config-home"));
        // SAFETY: serialized test env mutation.
        unsafe {
            env::remove_var("XDG_CONFIG_HOME");
        }
    }
}
