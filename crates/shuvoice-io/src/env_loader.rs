//! `~/.config/shuvoice/local.dev` environment loader.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use regex::Regex;
use tracing::warn;

use crate::xdg::shuvoice_config_dir;

/// Default path: `$XDG_CONFIG_HOME/shuvoice/local.dev`.
#[must_use]
pub fn local_dev_env_path() -> PathBuf {
    shuvoice_config_dir().join("local.dev")
}

/// Load `KEY=value` / `export KEY=value` lines from `local.dev`.
///
/// Returns the number of variables applied. Existing env values are preserved
/// unless `override_existing` is true.
pub fn load_local_dev_env(path: Option<&Path>, override_existing: bool) -> std::io::Result<usize> {
    let env_path = path
        .map(Path::to_path_buf)
        .unwrap_or_else(local_dev_env_path);
    if !env_path.is_file() {
        return Ok(0);
    }

    let text = match fs::read_to_string(&env_path) {
        Ok(t) => t,
        Err(err) => {
            warn!("Could not read {}: {err}", env_path.display());
            return Ok(0);
        }
    };

    let key_re = Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*$").expect("static regex");
    let mut loaded = 0usize;

    for (line_no, raw_line) in text.lines().enumerate() {
        let mut line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("export ") {
            line = rest.trim();
        }
        let Some((key_raw, value_raw)) = line.split_once('=') else {
            warn!(
                "Ignoring invalid line {}:{} (missing '=')",
                env_path.display(),
                line_no + 1
            );
            continue;
        };
        let key = key_raw.trim();
        let mut value = value_raw.trim().to_string();
        if !key_re.is_match(key) {
            warn!(
                "Ignoring invalid environment key {key:?} in {}:{}",
                env_path.display(),
                line_no + 1
            );
            continue;
        }
        if value.len() >= 2 {
            let bytes = value.as_bytes();
            let first = bytes[0];
            let last = bytes[value.len() - 1];
            if first == last && (first == b'\'' || first == b'"') {
                value = value[1..value.len() - 1].to_string();
            }
        }
        if !override_existing && env::var_os(key).is_some() {
            continue;
        }
        // SAFETY: intentional env mutation for app bootstrap (mirrors Python os.environ).
        unsafe {
            env::set_var(key, &value);
        }
        loaded += 1;
    }

    Ok(loaded)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]

    use super::*;
    use std::sync::Mutex;

    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn loads_keys_and_export_lines() {
        let _g = LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("local.dev");
        fs::write(
            &path,
            "# comment\nELEVENLABS_API_KEY=abc123\nexport OPENAI_API_KEY=\"xyz\"\n",
        )
        .unwrap();
        unsafe {
            env::remove_var("ELEVENLABS_API_KEY");
            env::remove_var("OPENAI_API_KEY");
        }
        let n = load_local_dev_env(Some(&path), false).unwrap();
        assert_eq!(n, 2);
        assert_eq!(env::var("ELEVENLABS_API_KEY").unwrap(), "abc123");
        assert_eq!(env::var("OPENAI_API_KEY").unwrap(), "xyz");
        unsafe {
            env::remove_var("ELEVENLABS_API_KEY");
            env::remove_var("OPENAI_API_KEY");
        }
    }

    #[test]
    fn does_not_override_by_default() {
        let _g = LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("local.dev");
        fs::write(&path, "ELEVENLABS_API_KEY=from_file\n").unwrap();
        unsafe {
            env::set_var("ELEVENLABS_API_KEY", "already_set");
        }
        let n = load_local_dev_env(Some(&path), false).unwrap();
        assert_eq!(n, 0);
        assert_eq!(env::var("ELEVENLABS_API_KEY").unwrap(), "already_set");
        unsafe {
            env::remove_var("ELEVENLABS_API_KEY");
        }
    }

    #[test]
    fn overrides_when_requested() {
        let _g = LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("local.dev");
        fs::write(&path, "ELEVENLABS_API_KEY=from_file\n").unwrap();
        unsafe {
            env::set_var("ELEVENLABS_API_KEY", "already_set");
        }
        let n = load_local_dev_env(Some(&path), true).unwrap();
        assert_eq!(n, 1);
        assert_eq!(env::var("ELEVENLABS_API_KEY").unwrap(), "from_file");
        unsafe {
            env::remove_var("ELEVENLABS_API_KEY");
        }
    }
}
