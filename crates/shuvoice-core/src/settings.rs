//! Settings-panel contract: field metadata, snapshots, draft validation and
//! patch-based persistence for the settings app.
//!
//! Field ids are `section.key` of real config keys, except the virtual
//! [`SHERPA_PROFILE`] which maps to several `[asr]` keys. Persistence patches
//! only the changed keys of the migrated raw map, so unknown keys survive.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::config::{
    CURRENT_CONFIG_VERSION, Config, DEFAULT_SHERPA_MODEL_NAME, PARAKEET_TDT_V3_INT8_MODEL_NAME,
    expand_user_path, load_raw, migrate_to_latest, write_atomic,
};
use crate::tts_speed::{TTS_PLAYBACK_SPEED_MAX, TTS_PLAYBACK_SPEED_MIN};

/// Virtual field: Sherpa model + decode mode preset.
pub const SHERPA_PROFILE: &str = "asr.sherpa_profile";
/// Revision of a config file that does not exist yet.
pub const ABSENT_REVISION: &str = "absent";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Section {
    Speech,
    Typing,
    TextToSpeech,
    Audio,
    Appearance,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Choice {
    pub value: &'static str,
    pub label: &'static str,
}

const fn choice(value: &'static str, label: &'static str) -> Choice {
    Choice { value, label }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FieldKind {
    Bool,
    Int {
        min: i64,
        max: i64,
    },
    Float {
        min: f64,
        max: f64,
        step: f64,
    },
    Text {
        max_len: usize,
    },
    Choice {
        choices: &'static [Choice],
    },
    /// `null` (system default), a device name, or a device index.
    AudioDevice,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct FieldMeta {
    pub id: &'static str,
    pub section: Section,
    pub label: &'static str,
    /// Empty unless the label alone would plausibly cause a mistake.
    pub help: &'static str,
    pub unit: &'static str,
    pub kind: FieldKind,
}

const fn field(
    id: &'static str,
    section: Section,
    label: &'static str,
    kind: FieldKind,
) -> FieldMeta {
    FieldMeta {
        id,
        section,
        label,
        help: "",
        unit: "",
        kind,
    }
}

/// Fields shown by the settings app (v1). Every non-virtual id is a real
/// `section.key` from [`crate::config::config_section_fields`].
pub static FIELDS: &[FieldMeta] = &[
    field(
        "asr.asr_backend",
        Section::Speech,
        "Engine",
        FieldKind::Choice {
            choices: &[
                choice("sherpa", "Sherpa-ONNX (local)"),
                choice("nemo", "NeMo (needs worker)"),
                choice("moonshine", "Moonshine (needs worker)"),
                choice("openai_realtime", "OpenAI Realtime (cloud)"),
            ],
        },
    ),
    FieldMeta {
        help: "Instant transcribes on key release; Streaming shows live text.",
        ..field(
            SHERPA_PROFILE,
            Section::Speech,
            "Sherpa profile",
            FieldKind::Choice {
                choices: &[
                    choice("instant", "Instant (Parakeet TDT v3)"),
                    choice("streaming", "Streaming (Zipformer)"),
                ],
            },
        )
    },
    field(
        "typing.typing_final_injection_mode",
        Section::Typing,
        "Text insertion",
        FieldKind::Choice {
            choices: &[
                choice("auto", "Automatic"),
                choice("clipboard", "Paste via clipboard"),
                choice("direct", "Type directly"),
            ],
        },
    ),
    field(
        "typing.typing_text_case",
        Section::Typing,
        "Text case",
        FieldKind::Choice {
            choices: &[
                choice("default", "As transcribed"),
                choice("lowercase", "Lowercase"),
            ],
        },
    ),
    field(
        "tts.tts_enabled",
        Section::TextToSpeech,
        "Read aloud",
        FieldKind::Bool,
    ),
    field(
        "tts.tts_backend",
        Section::TextToSpeech,
        "Provider",
        FieldKind::Choice {
            choices: &[
                choice("kokoro", "Kokoro"),
                choice("elevenlabs", "ElevenLabs"),
                choice("openai", "OpenAI"),
                choice("local", "Piper (local)"),
                choice("melotts", "MeloTTS (needs worker)"),
            ],
        },
    ),
    field(
        "tts.tts_kokoro_base_url",
        Section::TextToSpeech,
        "Kokoro URL",
        FieldKind::Text { max_len: 512 },
    ),
    field(
        "tts.tts_default_voice_id",
        Section::TextToSpeech,
        "Voice",
        FieldKind::Text { max_len: 256 },
    ),
    FieldMeta {
        unit: "×",
        ..field(
            "tts.tts_playback_speed",
            Section::TextToSpeech,
            "Speed",
            FieldKind::Float {
                min: TTS_PLAYBACK_SPEED_MIN,
                max: TTS_PLAYBACK_SPEED_MAX,
                step: 0.05,
            },
        )
    },
    field(
        "audio.audio_device",
        Section::Audio,
        "Microphone",
        FieldKind::AudioDevice,
    ),
    FieldMeta {
        unit: "×",
        ..field(
            "audio.input_gain",
            Section::Audio,
            "Input gain",
            FieldKind::Float {
                min: 0.1,
                max: 10.0,
                step: 0.1,
            },
        )
    },
    field(
        "feedback.audio_feedback",
        Section::Audio,
        "Start/stop tones",
        FieldKind::Bool,
    ),
    FieldMeta {
        unit: "px",
        ..field(
            "overlay.font_size",
            Section::Appearance,
            "Caption size",
            FieldKind::Int { min: 8, max: 96 },
        )
    },
    FieldMeta {
        unit: "px",
        ..field(
            "overlay.bottom_margin",
            Section::Appearance,
            "Caption offset from bottom",
            FieldKind::Int { min: 0, max: 2000 },
        )
    },
];

pub fn field_meta(id: &str) -> Option<&'static FieldMeta> {
    FIELDS.iter().find(|f| f.id == id)
}

/// An error tied to a field, or to the whole draft when `field` is `None`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FieldError {
    pub field: Option<String>,
    pub message: String,
}

impl FieldError {
    fn on(field: &str, message: impl Into<String>) -> Self {
        Self {
            field: Some(field.to_string()),
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct OwnedChoice {
    pub value: String,
    pub label: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Snapshot {
    pub path: String,
    pub revision: String,
    /// Effective value for every field in [`FIELDS`].
    pub values: BTreeMap<String, Value>,
    /// Field ids explicitly set in the config file.
    pub explicit: Vec<String>,
    /// The existing file failed to load; values are best-effort and Apply can repair it.
    pub config_error: Option<String>,
    /// Choices that exist only for the current value (e.g. a custom Sherpa model).
    pub extra_choices: BTreeMap<String, Vec<OwnedChoice>>,
    /// Secret presence only — values are never included.
    pub secrets: Vec<SecretPresence>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SecretPresence {
    pub env: String,
    pub present: bool,
    pub used_by: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct Applied {
    pub revision: String,
    pub backup: Option<String>,
    pub changed: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ApplyError {
    /// The file changed since the snapshot the draft was based on.
    Conflict {
        current_revision: String,
    },
    Invalid {
        errors: Vec<FieldError>,
    },
    Io {
        message: String,
    },
}

/// Content revision of the config file (`absent` when missing).
pub fn revision(path: impl AsRef<Path>) -> Result<String, String> {
    let path = expand_user_path(path);
    match std::fs::read(&path) {
        Ok(bytes) => {
            let digest = Sha256::digest(&bytes);
            Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(ABSENT_REVISION.into()),
        Err(err) => Err(format!("cannot read {}: {err}", path.display())),
    }
}

fn split_id(id: &str) -> (&str, &str) {
    id.split_once('.').unwrap_or(("", id))
}

fn raw_get<'a>(raw: &'a Map<String, Value>, id: &str) -> Option<&'a Value> {
    let (section, key) = split_id(id);
    raw.get(section)?.as_object()?.get(key)
}

fn sherpa_profile_of(model: &str) -> &'static str {
    if model == PARAKEET_TDT_V3_INT8_MODEL_NAME {
        "instant"
    } else if model == DEFAULT_SHERPA_MODEL_NAME {
        "streaming"
    } else {
        "custom"
    }
}

fn read_field(config: &Config, id: &str) -> Value {
    if id == SHERPA_PROFILE {
        return Value::from(sherpa_profile_of(&config.sherpa_model_name));
    }
    config.field_to_value(split_id(id).1).unwrap_or(Value::Null)
}

/// Read the current config for the settings app.
///
/// `env_present` reports whether a secret env var is set (values never leave it).
pub fn snapshot(
    path: impl AsRef<Path>,
    env_present: impl Fn(&str) -> bool,
) -> Result<Snapshot, String> {
    let path = expand_user_path(path);
    let revision = revision(&path)?;
    let (raw, load_error) = match load_raw(&path) {
        Ok(raw) => (raw, None),
        Err(err) => (Map::new(), Some(err.to_string())),
    };
    let migrated = migrate_to_latest(&raw)
        .map(|(migrated, _)| migrated)
        .unwrap_or_else(|_| raw.clone());

    let (config, config_error) = match (load_error, Config::resolve_raw(&raw)) {
        (Some(err), _) => (Config::default(), Some(err)),
        (None, Ok(resolved)) => (resolved.config, None),
        (None, Err(err)) => (Config::default(), Some(err.to_string())),
    };

    let mut values = BTreeMap::new();
    let mut explicit = Vec::new();
    for meta in FIELDS {
        let present_key = if meta.id == SHERPA_PROFILE {
            "asr.sherpa_model_name"
        } else {
            meta.id
        };
        let raw_value = raw_get(&migrated, present_key);
        if raw_value.is_some() {
            explicit.push(meta.id.to_string());
        }
        // With an invalid file, show what the file says where possible.
        let value = match (&config_error, raw_value) {
            (Some(_), Some(raw)) if meta.id == SHERPA_PROFILE => {
                Value::from(sherpa_profile_of(raw.as_str().unwrap_or_default()))
            }
            (Some(_), Some(raw)) => raw.clone(),
            _ => read_field(&config, meta.id),
        };
        values.insert(meta.id.to_string(), value);
    }

    let mut extra_choices = BTreeMap::new();
    if values.get(SHERPA_PROFILE).and_then(Value::as_str) == Some("custom") {
        extra_choices.insert(
            SHERPA_PROFILE.to_string(),
            vec![OwnedChoice {
                value: "custom".into(),
                label: format!("Custom ({})", config.sherpa_model_name),
            }],
        );
    }

    let secrets = vec![
        SecretPresence {
            present: env_present(&config.openai_realtime_api_key_env),
            env: config.openai_realtime_api_key_env.clone(),
            used_by: "asr.asr_backend:openai_realtime",
        },
        SecretPresence {
            present: env_present(&config.tts_api_key_env),
            env: config.tts_api_key_env.clone(),
            used_by: "tts.tts_backend",
        },
    ];

    Ok(Snapshot {
        path: path.display().to_string(),
        revision,
        values,
        explicit,
        config_error,
        extra_choices,
        secrets,
    })
}

fn check_kind(meta: &FieldMeta, value: &Value) -> Result<(), String> {
    match meta.kind {
        FieldKind::Bool => value
            .as_bool()
            .map(drop)
            .ok_or("must be true or false".into()),
        FieldKind::Int { min, max } => match value.as_i64() {
            Some(n) if (min..=max).contains(&n) => Ok(()),
            Some(_) => Err(format!("must be between {min} and {max}")),
            None => Err("must be a whole number".into()),
        },
        FieldKind::Float { min, max, .. } => match value.as_f64() {
            Some(n) if n.is_finite() && n >= min && n <= max => Ok(()),
            Some(_) => Err(format!("must be between {min} and {max}")),
            None => Err("must be a number".into()),
        },
        FieldKind::Text { max_len } => match value.as_str() {
            Some(s) if s.trim().is_empty() => Err("must not be empty".into()),
            Some(s) if s.chars().count() > max_len => {
                Err(format!("must be at most {max_len} characters"))
            }
            Some(s) if s.chars().any(char::is_control) => {
                Err("must not contain control characters".into())
            }
            Some(_) => Ok(()),
            None => Err("must be text".into()),
        },
        FieldKind::Choice { choices } => match value.as_str() {
            Some(v) if choices.iter().any(|c| c.value == v) => Ok(()),
            _ => Err(format!(
                "must be one of: {}",
                choices
                    .iter()
                    .map(|c| c.value)
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        },
        FieldKind::AudioDevice => match value {
            Value::Null => Ok(()),
            Value::Number(n) if n.as_i64().is_some_and(|i| i >= 0) => Ok(()),
            Value::String(s) if !s.trim().is_empty() && !s.chars().any(char::is_control) => Ok(()),
            _ => Err("must be the default, a device name, or a device index".into()),
        },
    }
}

fn table<'a>(raw: &'a mut Map<String, Value>, section: &str) -> &'a mut Map<String, Value> {
    let entry = raw
        .entry(section.to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if !entry.is_object() {
        *entry = Value::Object(Map::new());
    }
    entry.as_object_mut().expect("table is an object")
}

fn write_change(raw: &mut Map<String, Value>, id: &str, value: &Value) {
    if id == SHERPA_PROFILE {
        let asr = table(raw, "asr");
        let (model, instant, mode) = match value.as_str() {
            Some("instant") => (PARAKEET_TDT_V3_INT8_MODEL_NAME, true, "offline_instant"),
            _ => (DEFAULT_SHERPA_MODEL_NAME, false, "auto"),
        };
        asr.insert("sherpa_model_name".into(), Value::from(model));
        asr.insert("instant_mode".into(), Value::from(instant));
        asr.insert("sherpa_decode_mode".into(), Value::from(mode));
        asr.insert(
            "sherpa_enable_parakeet_streaming".into(),
            Value::from(false),
        );
        return;
    }
    let (section, key) = split_id(id);
    let tbl = table(raw, section);
    if value.is_null() {
        tbl.remove(key);
        return;
    }
    tbl.insert(key.to_string(), value.clone());
    if id == "typing.typing_final_injection_mode" {
        // Keep the legacy flag consistent, like `shuvoice config set`.
        tbl.insert(
            "use_clipboard_for_final".into(),
            Value::from(value.as_str() != Some("direct")),
        );
    }
}

/// Attribute a whole-config validation message to the field whose key it names.
fn attribute(message: String) -> FieldError {
    let field = FIELDS
        .iter()
        .filter(|f| f.id != SHERPA_PROFILE)
        .find(|f| message.contains(split_id(f.id).1))
        .map(|f| f.id.to_string());
    FieldError { field, message }
}

/// Validate `changes` on top of the migrated raw config. Returns the patched raw map.
pub fn validate_changes(
    migrated: &Map<String, Value>,
    current_values: &BTreeMap<String, Value>,
    changes: &BTreeMap<String, Value>,
) -> Result<Map<String, Value>, Vec<FieldError>> {
    let mut errors = Vec::new();
    for (id, value) in changes {
        let Some(meta) = field_meta(id) else {
            errors.push(FieldError::on(id, "unknown setting"));
            continue;
        };
        // A custom Sherpa model can be kept but not newly chosen.
        if id == SHERPA_PROFILE
            && value.as_str() == Some("custom")
            && current_values.get(id) == Some(value)
        {
            continue;
        }
        if let Err(message) = check_kind(meta, value) {
            errors.push(FieldError::on(id, message));
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    let mut patched = migrated.clone();
    for (id, value) in changes {
        if id == SHERPA_PROFILE && value.as_str() == Some("custom") {
            continue;
        }
        write_change(&mut patched, id, value);
    }
    patched.insert("config_version".into(), Value::from(CURRENT_CONFIG_VERSION));
    Config::resolve_raw(&patched).map_err(|err| vec![attribute(err.to_string())])?;
    Ok(patched)
}

/// Validate and persist `changes` if the file still has `expected_revision`.
pub fn apply(
    path: impl AsRef<Path>,
    expected_revision: &str,
    changes: &BTreeMap<String, Value>,
) -> Result<Applied, ApplyError> {
    let path = expand_user_path(path);
    let io = |message: String| ApplyError::Io { message };

    let current = revision(&path).map_err(io)?;
    if current != expected_revision {
        return Err(ApplyError::Conflict {
            current_revision: current,
        });
    }
    let raw = load_raw(&path).map_err(|e| io(e.to_string()))?;
    let (migrated, _) = migrate_to_latest(&raw).map_err(|e| io(e.to_string()))?;
    let current_values = snapshot(&path, |_| false)
        .map(|s| s.values)
        .unwrap_or_default();
    let patched = validate_changes(&migrated, &current_values, changes)
        .map_err(|errors| ApplyError::Invalid { errors })?;

    // Narrow the race with other writers: recheck right before the atomic write.
    let again = revision(&path).map_err(io)?;
    if again != expected_revision {
        return Err(ApplyError::Conflict {
            current_revision: again,
        });
    }
    let backup = write_atomic(&path, &patched).map_err(|e| io(e.to_string()))?;
    Ok(Applied {
        revision: revision(&path).map_err(io)?,
        backup: backup.map(|p| p.display().to_string()),
        changed: changes.keys().cloned().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::config_section_fields;
    use serde_json::json;

    fn changes(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    fn write(dir: &tempfile::TempDir, text: &str) -> std::path::PathBuf {
        let path = dir.path().join("config.toml");
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn every_field_maps_to_a_real_config_key() {
        let known: Vec<(String, String)> = config_section_fields()
            .iter()
            .flat_map(|(s, keys)| keys.iter().map(move |k| (s.to_string(), k.to_string())))
            .collect();
        for meta in FIELDS.iter().filter(|f| f.id != SHERPA_PROFILE) {
            let (s, k) = split_id(meta.id);
            assert!(
                known.contains(&(s.to_string(), k.to_string())),
                "{} is not a config key",
                meta.id
            );
        }
    }

    #[test]
    fn snapshot_reads_effective_values_and_explicit_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(
            &dir,
            "config_version = 1\n[tts]\ntts_backend = \"openai\"\ntts_playback_speed = 1.25\n",
        );
        let snap = snapshot(&path, |name| name == "OPENAI_API_KEY").unwrap();
        assert_eq!(snap.values["tts.tts_backend"], json!("openai"));
        assert_eq!(snap.values["tts.tts_playback_speed"], json!(1.25));
        assert_eq!(
            snap.values["overlay.font_size"],
            json!(Config::default().font_size)
        );
        assert!(snap.explicit.contains(&"tts.tts_backend".to_string()));
        assert!(!snap.explicit.contains(&"overlay.font_size".to_string()));
        assert_eq!(snap.values.len(), FIELDS.len());
        assert!(snap.config_error.is_none());
        let rendered = serde_json::to_string(&snap).unwrap();
        assert!(
            !rendered.contains("sk-"),
            "snapshots never carry secret values"
        );
    }

    #[test]
    fn apply_patches_only_changed_keys_and_preserves_unknown_ones() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(
            &dir,
            "config_version = 1\n[tts]\ntts_backend = \"kokoro\"\nmy_note = \"keep\"\n[custom]\nfoo = 1\n",
        );
        let rev = revision(&path).unwrap();
        let applied = apply(
            &path,
            &rev,
            &changes(&[
                ("tts.tts_playback_speed", json!(1.5)),
                ("overlay.font_size", json!(30)),
            ]),
        )
        .unwrap();
        assert_ne!(applied.revision, rev);
        assert!(applied.backup.is_some());

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("my_note = \"keep\""), "{text}");
        assert!(
            text.contains("[custom]") && text.contains("foo = 1"),
            "{text}"
        );
        assert!(text.contains("tts_playback_speed = 1.5"), "{text}");
        assert!(text.contains("font_size = 30"), "{text}");
        // Untouched defaults are not materialized into the file.
        assert!(!text.contains("bottom_margin"), "{text}");
        let cfg = Config::load_from_path(&path).unwrap();
        assert_eq!(cfg.font_size, 30);
    }

    #[test]
    fn stale_revision_is_a_conflict_and_leaves_the_file_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(&dir, "config_version = 1\n");
        let rev = revision(&path).unwrap();
        std::fs::write(&path, "config_version = 1\n[overlay]\nfont_size = 40\n").unwrap();
        let before = std::fs::read_to_string(&path).unwrap();
        let err = apply(&path, &rev, &changes(&[("overlay.font_size", json!(20))])).unwrap_err();
        assert!(matches!(err, ApplyError::Conflict { .. }));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
    }

    #[test]
    fn invalid_values_are_field_addressed_and_not_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(&dir, "config_version = 1\n");
        let rev = revision(&path).unwrap();
        let err = apply(
            &path,
            &rev,
            &changes(&[
                ("audio.input_gain", json!(0)),
                ("tts.tts_backend", json!("espeak")),
                ("tts.tts_default_voice_id", json!("  ")),
                ("nope.key", json!(1)),
            ]),
        )
        .unwrap_err();
        let ApplyError::Invalid { errors } = err else {
            panic!("expected Invalid, got {err:?}");
        };
        let fields: Vec<_> = errors.iter().filter_map(|e| e.field.clone()).collect();
        for id in [
            "audio.input_gain",
            "tts.tts_backend",
            "tts.tts_default_voice_id",
            "nope.key",
        ] {
            assert!(
                fields.contains(&id.to_string()),
                "{id} missing in {errors:?}"
            );
        }
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "config_version = 1\n"
        );
    }

    #[test]
    fn sherpa_profile_writes_the_matching_asr_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(&dir, "config_version = 1\n");
        let rev = revision(&path).unwrap();
        apply(&path, &rev, &changes(&[(SHERPA_PROFILE, json!("instant"))])).unwrap();
        let cfg = Config::load_from_path(&path).unwrap();
        assert_eq!(cfg.sherpa_model_name, PARAKEET_TDT_V3_INT8_MODEL_NAME);
        assert!(cfg.instant_mode);
        assert_eq!(
            snapshot(&path, |_| false).unwrap().values[SHERPA_PROFILE],
            json!("instant")
        );

        let rev = revision(&path).unwrap();
        apply(
            &path,
            &rev,
            &changes(&[(SHERPA_PROFILE, json!("streaming"))]),
        )
        .unwrap();
        let cfg = Config::load_from_path(&path).unwrap();
        assert_eq!(cfg.sherpa_model_name, DEFAULT_SHERPA_MODEL_NAME);
        assert!(!cfg.instant_mode);
    }

    #[test]
    fn custom_sherpa_model_is_kept_unless_changed() {
        let dir = tempfile::tempdir().unwrap();
        let model = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";
        let path = write(
            &dir,
            &format!("config_version = 1\n[asr]\nsherpa_model_name = \"{model}\"\n"),
        );
        let snap = snapshot(&path, |_| false).unwrap();
        assert_eq!(snap.values[SHERPA_PROFILE], json!("custom"));
        assert!(snap.extra_choices[SHERPA_PROFILE][0].label.contains(model));
        // Sending the unchanged custom value with another change keeps the model.
        apply(
            &path,
            &snap.revision,
            &changes(&[
                (SHERPA_PROFILE, json!("custom")),
                ("overlay.font_size", json!(24)),
            ]),
        )
        .unwrap();
        assert_eq!(
            Config::load_from_path(&path).unwrap().sherpa_model_name,
            model
        );
    }

    #[test]
    fn default_microphone_unsets_the_key_and_injection_mode_keeps_legacy_flag() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(
            &dir,
            "config_version = 1\n[audio]\naudio_device = \"USB Mic\"\n",
        );
        let rev = revision(&path).unwrap();
        apply(
            &path,
            &rev,
            &changes(&[
                ("audio.audio_device", Value::Null),
                ("typing.typing_final_injection_mode", json!("direct")),
            ]),
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("audio_device"), "{text}");
        assert!(text.contains("use_clipboard_for_final = false"), "{text}");
        assert!(
            Config::load_from_path(&path)
                .unwrap()
                .audio_device
                .is_none()
        );
    }

    #[test]
    fn invalid_existing_config_is_inspectable_and_repairable() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(&dir, "config_version = 1\n[overlay]\nfont_size = 0\n");
        let snap = snapshot(&path, |_| false).unwrap();
        assert!(snap.config_error.is_some());
        assert_eq!(snap.values["overlay.font_size"], json!(0));
        apply(
            &path,
            &snap.revision,
            &changes(&[("overlay.font_size", json!(22))]),
        )
        .unwrap();
        assert_eq!(Config::load_from_path(&path).unwrap().font_size, 22);
    }

    #[test]
    fn missing_file_has_absent_revision_and_apply_creates_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shuvoice/config.toml");
        let snap = snapshot(&path, |_| false).unwrap();
        assert_eq!(snap.revision, ABSENT_REVISION);
        apply(
            &path,
            ABSENT_REVISION,
            &changes(&[("overlay.font_size", json!(26))]),
        )
        .unwrap();
        assert_eq!(Config::load_from_path(&path).unwrap().font_size, 26);
    }
}
