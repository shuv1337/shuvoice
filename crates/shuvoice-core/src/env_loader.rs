//! Parse `local.dev` environment files (no process mutation required).

use std::collections::BTreeMap;
use std::path::Path;

use once_cell::sync::Lazy;
use regex::Regex;

use crate::error::{CoreError, CoreResult};
use crate::xdg::local_dev_env_path;

static ENV_KEY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*$").expect("env key regex"));

/// Parsed local.dev assignments in file order (last key wins in map).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LocalDevEnv {
    pub values: BTreeMap<String, String>,
    pub loaded_count: usize,
}

/// Load and parse a local.dev-style file.
pub fn parse_local_dev_env_text(text: &str) -> LocalDevEnv {
    let mut values = BTreeMap::new();
    let mut loaded = 0usize;
    for raw_line in text.lines() {
        let mut line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("export ") {
            line = rest.trim();
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let mut value = value.trim().to_string();
        if !ENV_KEY_RE.is_match(key) {
            continue;
        }
        if value.len() >= 2 {
            let bytes = value.as_bytes();
            let q = bytes[0];
            if (q == b'\'' || q == b'"') && bytes[value.len() - 1] == q {
                value = value[1..value.len() - 1].to_string();
            }
        }
        values.insert(key.to_string(), value);
        loaded += 1;
    }
    LocalDevEnv {
        values,
        loaded_count: loaded,
    }
}

/// Read local.dev from disk (default XDG path when `path` is None).
pub fn load_local_dev_env_file(path: Option<&Path>) -> CoreResult<LocalDevEnv> {
    let path = path
        .map(Path::to_path_buf)
        .unwrap_or_else(local_dev_env_path);
    if !path.is_file() {
        return Ok(LocalDevEnv::default());
    }
    let text = std::fs::read_to_string(&path).map_err(|source| CoreError::Io {
        path: path.clone(),
        source,
    })?;
    Ok(parse_local_dev_env_text(&text))
}

/// Apply parsed values into a mutable environment map.
/// Existing keys are preserved unless `override_existing` is true.
pub fn merge_into_env_map(
    parsed: &LocalDevEnv,
    env: &mut BTreeMap<String, String>,
    override_existing: bool,
) -> usize {
    let mut applied = 0usize;
    for (k, v) in &parsed.values {
        if !override_existing && env.contains_key(k) {
            continue;
        }
        env.insert(k.clone(), v.clone());
        applied += 1;
    }
    applied
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_export_and_quotes() {
        let text = r#"
# comment
export FOO=bar
BAZ="qux"
EMPTY=
BAD
"#;
        let parsed = parse_local_dev_env_text(text);
        assert_eq!(parsed.values.get("FOO").map(String::as_str), Some("bar"));
        assert_eq!(parsed.values.get("BAZ").map(String::as_str), Some("qux"));
        assert!(!parsed.values.contains_key("BAD"));
    }
}
