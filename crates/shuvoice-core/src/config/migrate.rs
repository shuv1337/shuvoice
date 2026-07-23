//! Config schema migrations.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use super::defaults::CURRENT_CONFIG_VERSION;
use crate::error::{CoreError, CoreResult};

/// Report describing a migration pass.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MigrationReport {
    pub from_version: u32,
    pub to_version: u32,
    pub changed_keys: Vec<String>,
}

fn detect_version(raw: &Map<String, Value>) -> u32 {
    match raw.get("config_version") {
        Some(Value::Number(n)) => n.as_u64().unwrap_or(0) as u32,
        Some(Value::String(s)) => s.parse().unwrap_or(0),
        _ => 0,
    }
}

fn migrate_v0_to_v1(raw: &Map<String, Value>) -> (Map<String, Value>, BTreeSet<String>) {
    let mut migrated = raw.clone();
    let mut changed = BTreeSet::new();
    let needs = match migrated.get("config_version") {
        Some(Value::Number(n)) => n.as_u64() != Some(1),
        _ => true,
    };
    if needs {
        migrated.insert("config_version".into(), Value::from(1));
        changed.insert("config_version".into());
    }
    (migrated, changed)
}

/// Migrate an arbitrary raw config map to `CURRENT_CONFIG_VERSION`.
pub fn migrate_to_latest(
    raw: &Map<String, Value>,
) -> CoreResult<(Map<String, Value>, MigrationReport)> {
    let mut current = raw.clone();
    let from_version = detect_version(&current);
    if from_version > CURRENT_CONFIG_VERSION {
        return Err(CoreError::migration(format!(
            "Config schema version is newer than this ShuVoice build supports (got {from_version}, max {CURRENT_CONFIG_VERSION})"
        )));
    }

    let mut changed_keys = BTreeSet::new();
    let mut version = from_version;
    while version < CURRENT_CONFIG_VERSION {
        let (next, changed) = match version {
            0 => migrate_v0_to_v1(&current),
            other => {
                return Err(CoreError::migration(format!(
                    "Missing migration step for config schema {other} -> {}",
                    other + 1
                )));
            }
        };
        current = next;
        changed_keys.extend(changed);
        version += 1;
    }

    let actual = detect_version(&current);
    if actual != CURRENT_CONFIG_VERSION {
        current.insert("config_version".into(), Value::from(CURRENT_CONFIG_VERSION));
        changed_keys.insert("config_version".into());
    }

    Ok((
        current,
        MigrationReport {
            from_version,
            to_version: CURRENT_CONFIG_VERSION,
            changed_keys: changed_keys.into_iter().collect(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn map(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    #[test]
    fn migrate_unversioned_config_to_latest() {
        let raw = map(json!({"asr": {"asr_backend": "sherpa"}}));
        let (migrated, report) = migrate_to_latest(&raw).unwrap();
        assert_eq!(
            migrated.get("config_version").and_then(|v| v.as_u64()),
            Some(CURRENT_CONFIG_VERSION as u64)
        );
        assert_eq!(report.from_version, 0);
        assert_eq!(report.to_version, CURRENT_CONFIG_VERSION);
        assert!(report.changed_keys.iter().any(|k| k == "config_version"));
    }

    #[test]
    fn migrate_current_version_is_noop() {
        let raw = map(json!({
            "config_version": CURRENT_CONFIG_VERSION,
            "audio": {"sample_rate": 16000}
        }));
        let (migrated, report) = migrate_to_latest(&raw).unwrap();
        assert_eq!(migrated, raw);
        assert_eq!(report.from_version, CURRENT_CONFIG_VERSION);
        assert_eq!(report.to_version, CURRENT_CONFIG_VERSION);
        assert!(report.changed_keys.is_empty());
    }

    #[test]
    fn migrate_rejects_future_schema_version() {
        let raw = map(json!({"config_version": CURRENT_CONFIG_VERSION + 1}));
        let err = migrate_to_latest(&raw).unwrap_err().to_string();
        assert!(err.contains("newer"));
    }
}
