//! Low-level TOML config I/O helpers (atomic writes + backups).

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::Utc;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

use super::defaults::CURRENT_CONFIG_VERSION;
use crate::error::{CoreError, CoreResult};

static BARE_TOML_KEY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[A-Za-z0-9_-]+$").expect("bare key regex"));

/// Expand a leading `~/` (or bare `~`) using `$HOME` / `$USERPROFILE`.
///
/// Expands the common home-relative form used in ShuVoice configs. Non-`~`
/// paths are returned unchanged (absolute or relative).
pub fn expand_user_path(path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    let Some(s) = path.to_str() else {
        return path.to_path_buf();
    };
    if s == "~" {
        return home_dir();
    }
    if let Some(rest) = s.strip_prefix("~/") {
        return home_dir().join(rest);
    }
    if let Some(rest) = s.strip_prefix("~\\") {
        return home_dir().join(rest);
    }
    path.to_path_buf()
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

/// Load raw TOML config data from `path` into a JSON-compatible map.
///
/// Missing files return an empty vCurrent skeleton.
/// Existing unversioned files are tagged as `config_version = 0` for migration.
/// Paths are `~`-expanded.
pub fn load_raw(path: impl AsRef<Path>) -> CoreResult<Map<String, Value>> {
    let path = expand_user_path(path);
    if !path.exists() {
        let mut map = Map::new();
        map.insert("config_version".into(), Value::from(CURRENT_CONFIG_VERSION));
        return Ok(map);
    }

    let mut file = File::open(&path).map_err(|source| CoreError::Io {
        path: path.clone(),
        source,
    })?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|source| CoreError::Io {
            path: path.clone(),
            source,
        })?;
    let text = String::from_utf8_lossy(&bytes);
    let toml_value: toml::Value = toml::from_str(&text).map_err(|source| CoreError::TomlParse {
        path: path.clone(),
        source,
    })?;
    let value = toml_value_to_json(toml_value);
    let Some(mut map) = value.as_object().cloned() else {
        return Err(CoreError::validation(format!(
            "Config file must contain a top-level TOML table: {}",
            path.display()
        )));
    };
    if !map.contains_key("config_version") {
        map.insert("config_version".into(), Value::from(0));
    }
    Ok(map)
}

/// Create timestamped backup beside `path`.
pub fn backup_config(path: impl AsRef<Path>) -> CoreResult<Option<PathBuf>> {
    let path = expand_user_path(path);
    if !path.exists() {
        return Ok(None);
    }
    // Python: %Y%m%dT%H%M%SZ
    let stamp = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let backup_path = path.with_file_name(format!(
        "{}.bak-{stamp}",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("config.toml")
    ));
    fs::copy(&path, &backup_path).map_err(|source| CoreError::Io {
        path: backup_path.clone(),
        source,
    })?;
    #[cfg(unix)]
    {
        #[cfg(unix)]
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&backup_path, fs::Permissions::from_mode(0o600));
    }
    Ok(Some(backup_path))
}

/// Canonical float formatting for stable TOML dumps.
pub fn format_toml_float(value: f64) -> String {
    if !value.is_finite() {
        return if value.is_nan() {
            "nan".into()
        } else if value.is_sign_negative() {
            "-inf".into()
        } else {
            "inf".into()
        };
    }
    if value.fract() == 0.0 && value.abs() < 1e15 {
        return format!("{value:.1}");
    }
    let mut s = format!("{value}");
    if s.contains('e') || s.contains('E') {
        return s;
    }
    if s.contains('.') {
        while s.contains('.') && s.ends_with('0') {
            let trimmed = s.trim_end_matches('0');
            if trimmed.ends_with('.') {
                s = format!("{trimmed}0");
                break;
            }
            s = trimmed.to_string();
        }
    } else {
        s.push_str(".0");
    }
    s
}

fn format_toml_key(key: &str) -> String {
    if BARE_TOML_KEY_RE.is_match(key) {
        key.to_string()
    } else {
        serde_json::to_string(key).unwrap_or_else(|_| format!("\"{key}\""))
    }
}

fn format_toml_value(value: &Value) -> CoreResult<String> {
    match value {
        Value::Bool(v) => Ok(if *v { "true".into() } else { "false".into() }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_string())
            } else if let Some(u) = n.as_u64() {
                Ok(u.to_string())
            } else if let Some(f) = n.as_f64() {
                Ok(format_toml_float(f))
            } else {
                Err(CoreError::TomlSerialize(format!("unsupported number: {n}")))
            }
        }
        Value::String(s) => Ok(serde_json::to_string(s).unwrap_or_else(|_| format!("\"{s}\""))),
        Value::Array(items) => {
            let mut parts = Vec::with_capacity(items.len());
            for item in items {
                parts.push(format_toml_value(item)?);
            }
            Ok(format!("[{}]", parts.join(", ")))
        }
        Value::Null => Err(CoreError::TomlSerialize(
            "null is not supported in TOML dump".into(),
        )),
        Value::Object(_) => Err(CoreError::TomlSerialize(
            "nested object must be serialized as a table".into(),
        )),
    }
}

fn serialize_table(data: &Map<String, Value>, prefix: &[String]) -> CoreResult<Vec<String>> {
    let mut lines = Vec::new();
    if !prefix.is_empty() {
        let dotted = prefix
            .iter()
            .map(|p| format_toml_key(p))
            .collect::<Vec<_>>()
            .join(".");
        lines.push(format!("[{dotted}]"));
    }

    // Preserve caller insertion order for scalars and nested tables (section
    // field order from CONFIG_SECTION_FIELDS). Nested maps built from BTreeMap
    // arrive pre-sorted for stable dumps of free-form tables like replacements.
    let mut nested: Vec<(String, &Map<String, Value>)> = Vec::new();
    for (key, value) in data {
        if value.is_null() {
            continue;
        }
        if let Some(child) = value.as_object() {
            nested.push((key.clone(), child));
            continue;
        }
        lines.push(format!(
            "{} = {}",
            format_toml_key(key),
            format_toml_value(value)?
        ));
    }

    for (idx, (key, child)) in nested.iter().enumerate() {
        if !lines.is_empty() {
            lines.push(String::new());
        }
        let mut child_prefix = prefix.to_vec();
        child_prefix.push(key.clone());
        lines.extend(serialize_table(child, &child_prefix)?);
        if idx + 1 < nested.len() {
            lines.push(String::new());
        }
    }
    Ok(lines)
}

/// Serialize nested config data to TOML text.
pub fn toml_dumps(data: &Map<String, Value>) -> CoreResult<String> {
    let lines = serialize_table(data, &[])?;
    let mut text = lines.join("\n");
    text = text.trim().to_string();
    text.push('\n');
    Ok(text)
}

// Avoid unsafe FD tricks — open the directory path and sync_all.
#[cfg(unix)]
fn fsync_dir_path(dir: &Path) {
    if let Ok(file) = OpenOptions::new().read(true).open(dir) {
        let _ = file.sync_all();
    }
}

#[cfg(not(unix))]
fn fsync_dir_path(_dir: &Path) {}

/// Atomically write TOML data to `path`.
///
/// - Paths are `~`-expanded.
/// - On Unix, temp and final files are forced to mode `0o600` (independent of umask).
/// - File contents are `fsync`ed before rename; parent directory is `fsync`ed after rename.
///
/// Returns backup path when an existing file was backed up.
pub fn write_atomic(
    path: impl AsRef<Path>,
    data: &Map<String, Value>,
) -> CoreResult<Option<PathBuf>> {
    let path = expand_user_path(path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| CoreError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }

    let backup = backup_config(&path)?;
    let payload = toml_dumps(data)?;

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp_name = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("config"),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    let write_result = (|| -> CoreResult<()> {
        let mut opts = OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            // Force 0600 regardless of process umask (Python mkstemp semantics).
            opts.mode(0o600);
        }
        let mut file = opts.open(&tmp_name).map_err(|source| CoreError::Io {
            path: tmp_name.clone(),
            source,
        })?;
        file.write_all(payload.as_bytes())
            .map_err(|source| CoreError::Io {
                path: tmp_name.clone(),
                source,
            })?;
        file.sync_all().map_err(|source| CoreError::Io {
            path: tmp_name.clone(),
            source,
        })?;
        drop(file);

        #[cfg(unix)]
        {
            #[cfg(unix)]
            use std::os::unix::fs::PermissionsExt;
            // Belt-and-suspenders if filesystem ignored open mode.
            fs::set_permissions(&tmp_name, fs::Permissions::from_mode(0o600)).map_err(
                |source| CoreError::Io {
                    path: tmp_name.clone(),
                    source,
                },
            )?;
        }

        fs::rename(&tmp_name, &path).map_err(|source| CoreError::Io {
            path: path.clone(),
            source,
        })?;

        #[cfg(unix)]
        {
            #[cfg(unix)]
            use std::os::unix::fs::PermissionsExt;
            // Ensure final path is 0600 even if rename replaced a looser inode on
            // exotic FS behavior; normal rename keeps temp mode.
            let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o600));
        }

        fsync_dir_path(parent);
        Ok(())
    })();

    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_name);
        if let Some(backup_path) = &backup
            && !path.exists()
        {
            let _ = fs::copy(backup_path, &path);
        }
        return Err(err);
    }

    Ok(backup)
}

/// Convert a TOML value tree into serde_json map/value form.
pub fn toml_value_to_json(value: toml::Value) -> Value {
    match value {
        toml::Value::String(s) => Value::String(s),
        toml::Value::Integer(i) => Value::Number(i.into()),
        toml::Value::Float(f) => serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        toml::Value::Boolean(b) => Value::Bool(b),
        toml::Value::Datetime(dt) => Value::String(dt.to_string()),
        toml::Value::Array(items) => {
            Value::Array(items.into_iter().map(toml_value_to_json).collect())
        }
        toml::Value::Table(table) => {
            let mut map = Map::new();
            for (k, v) in table {
                map.insert(k, toml_value_to_json(v));
            }
            Value::Object(map)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use tempfile::tempdir;

    fn map(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    #[test]
    fn load_raw_missing_file_returns_current_schema() {
        let dir = tempdir().unwrap();
        let raw = load_raw(dir.path().join("config.toml")).unwrap();
        assert_eq!(
            raw.get("config_version").and_then(|v| v.as_u64()),
            Some(CURRENT_CONFIG_VERSION as u64)
        );
    }

    #[test]
    fn write_atomic_creates_backup_for_existing_file() {
        let dir = tempdir().unwrap();
        let config_file = dir.path().join("config.toml");
        fs::write(&config_file, "config_version = 1\n").unwrap();
        let backup = write_atomic(
            &config_file,
            &map(json!({
                "config_version": CURRENT_CONFIG_VERSION,
                "asr": {"asr_backend": "sherpa"}
            })),
        )
        .unwrap();
        assert!(backup.unwrap().exists());
        let content = fs::read_to_string(&config_file).unwrap();
        assert!(content.contains("[asr]"));
        assert!(content.contains("asr_backend = \"sherpa\""));
    }

    #[test]
    fn round_trip_load_write_reload() {
        let dir = tempdir().unwrap();
        let config_file = dir.path().join("config.toml");
        write_atomic(
            &config_file,
            &map(json!({
                "config_version": CURRENT_CONFIG_VERSION,
                "audio": {"sample_rate": 16000, "chunk_ms": 100},
                "asr": {
                    "asr_backend": "moonshine",
                    "moonshine_model_name": "moonshine/tiny"
                }
            })),
        )
        .unwrap();
        let raw = load_raw(&config_file).unwrap();
        assert_eq!(
            raw.get("config_version").and_then(|v| v.as_u64()),
            Some(CURRENT_CONFIG_VERSION as u64)
        );
        assert_eq!(raw["audio"]["sample_rate"].as_i64(), Some(16000));
        assert_eq!(raw["asr"]["asr_backend"].as_str(), Some("moonshine"));
    }

    #[test]
    fn write_atomic_quotes_non_bare_keys() {
        let dir = tempdir().unwrap();
        let config_file = dir.path().join("config.toml");
        write_atomic(
            &config_file,
            &map(json!({
                "config_version": CURRENT_CONFIG_VERSION,
                "typing": {
                    "text_replacements": {
                        "shove voice": "ShuVoice",
                        "high-per-land": "Hyprland"
                    }
                }
            })),
        )
        .unwrap();
        let content = fs::read_to_string(&config_file).unwrap();
        assert!(content.contains("\"shove voice\" = \"ShuVoice\""));
        assert!(
            content.contains("high-per-land = \"Hyprland\"")
                || content.contains("\"high-per-land\" = \"Hyprland\"")
        );
        let raw = load_raw(&config_file).unwrap();
        assert_eq!(
            raw["typing"]["text_replacements"]["shove voice"].as_str(),
            Some("ShuVoice")
        );
    }

    #[test]
    fn format_toml_float_is_stable() {
        assert_eq!(format_toml_float(1.0), "1.0");
        assert_eq!(format_toml_float(1.25), "1.25");
        assert_eq!(format_toml_float(0.15), "0.15");
    }

    #[test]
    fn toml_value_to_json_round_trips_tables() {
        let v: toml::Value = toml::from_str("a = 1\n[b]\nc = \"x\"\n").unwrap();
        let json = toml_value_to_json(v);
        assert_eq!(json["a"].as_i64(), Some(1));
        assert_eq!(json["b"]["c"].as_str(), Some("x"));
    }

    #[cfg(unix)]
    #[test]
    fn write_atomic_forces_0600_under_permissive_umask() {
        let dir = tempdir().unwrap();
        let config_file = dir.path().join("config.toml");

        // SAFETY: umask is process-global; this unit test restores the previous
        // value immediately after the write and does not run concurrent umask
        // mutators in this module.
        let previous = unsafe { libc::umask(0o022) };
        let result = write_atomic(
            &config_file,
            &map(json!({
                "config_version": CURRENT_CONFIG_VERSION,
                "asr": {"asr_backend": "sherpa"}
            })),
        );
        // SAFETY: restore process umask to the pre-test value.
        unsafe {
            let _ = libc::umask(previous);
        }
        result.unwrap();

        let mode = fs::metadata(&config_file).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "expected 0600, got {mode:o}");
    }

    #[test]
    fn expand_user_path_expands_tilde() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        assert_eq!(
            expand_user_path("~/shuvoice/config.toml"),
            PathBuf::from(home).join("shuvoice/config.toml")
        );
        assert_eq!(expand_user_path("/abs/path"), PathBuf::from("/abs/path"));
    }

    #[test]
    fn load_raw_and_write_atomic_expand_tilde() {
        let dir = tempdir().unwrap();
        let home = dir.path();
        let old_home = std::env::var_os("HOME");
        // SAFETY: HOME is process-global; restored before the test returns.
        unsafe {
            std::env::set_var("HOME", home);
        }
        let rel = PathBuf::from("~/cfg/config.toml");
        let result = write_atomic(
            &rel,
            &map(json!({
                "config_version": CURRENT_CONFIG_VERSION,
                "asr": {"asr_backend": "nemo"}
            })),
        );
        let raw = load_raw(&rel);
        // SAFETY: restore HOME to the pre-test value.
        unsafe {
            match &old_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
        result.unwrap();
        let raw = raw.unwrap();
        assert_eq!(raw["asr"]["asr_backend"].as_str(), Some("nemo"));
        assert!(home.join("cfg/config.toml").is_file());
    }
}
