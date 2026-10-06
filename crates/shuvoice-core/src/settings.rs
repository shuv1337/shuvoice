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
    expand_user_path, load_raw, migrate_to_latest, toml_dumps, write_atomic,
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
    Vocabulary,
    Typing,
    TextToSpeech,
    Audio,
    Appearance,
    Advanced,
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
    OptionalText {
        max_len: usize,
    },
    StringList {
        item_max_len: usize,
        max_items: usize,
    },
    StringMap {
        key_max_len: usize,
        value_max_len: usize,
        max_entries: usize,
    },
    Choice {
        choices: &'static [Choice],
    },
    /// `null` (system default), a device name, or a device index.
    AudioDevice {
        direction: &'static str,
    },
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct FieldMeta {
    pub id: &'static str,
    pub section: Section,
    pub label: &'static str,
    /// Empty unless the label alone would plausibly cause a mistake.
    pub help: &'static str,
    pub unit: &'static str,
    pub advanced: bool,
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
        advanced: false,
        kind,
    }
}

/// Fields shown by the settings app (v1). Every non-virtual id is a real
/// `section.key` from [`crate::config::config_section_fields`].
const BASIC_FIELDS: &[FieldMeta] = &[
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
    FieldMeta {
        help: "Shift + push-to-talk dictates in the other case.",
        ..field(
            "typing.typing_text_case",
            Section::Typing,
            "Text case",
            FieldKind::Choice {
                choices: &[
                    choice("default", "As transcribed"),
                    choice("lowercase", "Lowercase"),
                ],
            },
        )
    },
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
        FieldKind::AudioDevice { direction: "input" },
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

/// Display label and unit for fields generated from `config_section_fields()`.
const GENERATED_LABELS: &[(&str, &str, &str)] = &[
    ("recognition_hints", "Preferred terms", ""),
    ("sample_rate", "Sample rate", "Hz"),
    ("chunk_ms", "Chunk length", "ms"),
    ("fallback_sample_rate", "Fallback sample rate", "Hz"),
    ("audio_queue_max_size", "Audio queue size", "chunks"),
    ("recording_preroll_ms", "Pre-roll", "ms"),
    ("silence_rms_threshold", "Silence threshold (RMS)", ""),
    ("silence_rms_multiplier", "Silence multiplier", "×"),
    ("min_speech_ms", "Minimum speech", "ms"),
    ("auto_gain_target_peak", "Auto-gain target peak", ""),
    ("auto_gain_max", "Auto-gain maximum", "×"),
    ("auto_gain_settle_chunks", "Auto-gain settle", "chunks"),
    ("instant_mode", "Instant mode", ""),
    ("model_name", "NeMo model", ""),
    ("right_context", "NeMo right context", "frames"),
    ("device", "NeMo device", ""),
    ("use_cuda_graph_decoder", "NeMo CUDA graph decoder", ""),
    ("sherpa_model_name", "Sherpa model", ""),
    ("sherpa_model_dir", "Sherpa model folder", ""),
    ("sherpa_decode_mode", "Sherpa decode mode", ""),
    ("sherpa_enable_parakeet_streaming", "Parakeet streaming", ""),
    ("sherpa_provider", "Sherpa compute", ""),
    ("sherpa_num_threads", "Sherpa threads", ""),
    ("sherpa_chunk_ms", "Sherpa chunk length", "ms"),
    (
        "sherpa_offline_max_utterance_sec",
        "Sherpa max utterance",
        "s",
    ),
    ("moonshine_model_name", "Moonshine model", ""),
    ("moonshine_model_dir", "Moonshine model folder", ""),
    ("moonshine_model_precision", "Moonshine precision", ""),
    ("moonshine_chunk_ms", "Moonshine chunk length", "ms"),
    ("moonshine_max_window_sec", "Moonshine max window", "s"),
    ("moonshine_max_tokens", "Moonshine max tokens", ""),
    ("moonshine_provider", "Moonshine compute", ""),
    ("moonshine_onnx_threads", "Moonshine threads", ""),
    ("openai_realtime_model", "OpenAI model", ""),
    ("openai_realtime_api_key_env", "OpenAI API key variable", ""),
    ("openai_realtime_language", "OpenAI language", ""),
    (
        "openai_realtime_latency_target_sec",
        "OpenAI latency target",
        "s",
    ),
    (
        "openai_realtime_turn_detection",
        "OpenAI turn detection",
        "",
    ),
    ("openai_realtime_vad_eagerness", "OpenAI VAD eagerness", ""),
    (
        "openai_realtime_request_timeout_sec",
        "OpenAI request timeout",
        "s",
    ),
    (
        "openai_realtime_commit_timeout_sec",
        "OpenAI commit timeout",
        "s",
    ),
    ("font_family", "Caption font", ""),
    ("bg_opacity", "Background opacity", ""),
    ("border_radius", "Corner radius", "px"),
    ("overlay_debug_mode", "Debug overlay", ""),
    ("overlay_debug_max_lines", "Debug overlay lines", ""),
    ("control_socket", "Control socket path", ""),
    ("tts_model_id", "TTS model", ""),
    ("tts_api_key_env", "TTS API key variable", ""),
    ("tts_output_format", "Audio format", ""),
    ("tts_max_chars", "Max characters", ""),
    ("tts_request_timeout_sec", "Request timeout", "s"),
    ("tts_playback_device", "Speaker", ""),
    ("tts_overlay_auto_hide_sec", "Hide overlay after", "s"),
    ("tts_local_model_path", "Piper model path", ""),
    ("tts_local_voice", "Piper voice", ""),
    ("tts_local_device", "Piper output device", ""),
    ("tts_melotts_device", "MeloTTS compute", ""),
    ("tts_melotts_venv_path", "MeloTTS environment", ""),
    ("output_mode", "Output mode", ""),
    ("preserve_clipboard", "Restore clipboard after paste", ""),
    (
        "typing_clipboard_settle_delay_ms",
        "Clipboard settle delay",
        "ms",
    ),
    ("typing_retry_attempts", "Typing retries", ""),
    ("typing_retry_delay_ms", "Retry delay", "ms"),
    ("typing_subprocess_timeout", "Typing command timeout", "s"),
    ("auto_capitalize", "Capitalize first letter", ""),
    ("text_replacements", "Corrections", ""),
    ("streaming_stall_guard", "Stall guard", ""),
    ("streaming_stall_chunks", "Stall detection window", "chunks"),
    ("streaming_stall_rms_ratio", "Stall RMS ratio", ""),
    ("streaming_stall_flush_chunks", "Stall flush", "chunks"),
    ("feedback_start_freq", "Start tone", "Hz"),
    ("feedback_stop_freq", "Stop tone", "Hz"),
    ("feedback_duration_ms", "Tone length", "ms"),
    ("feedback_volume", "Tone volume", ""),
];

fn generated_label(key: &'static str) -> (&'static str, &'static str) {
    GENERATED_LABELS
        .iter()
        .find(|(k, _, _)| *k == key)
        .map_or((key, ""), |(_, label, unit)| (*label, *unit))
}

/// Legacy compatibility switch is derived from the injection-mode field.
pub const EXCLUDED: &[(&str, &str)] = &[(
    "typing.use_clipboard_for_final",
    "Legacy alias; use typing.typing_final_injection_mode",
)];

/// All real config fields are discoverable. More specialized metadata above
/// takes precedence over the type inferred from the core default.
pub static FIELDS: once_cell::sync::Lazy<Vec<FieldMeta>> = once_cell::sync::Lazy::new(|| {
    const DECODE: &[Choice] = &[
        choice("auto", "Automatic"),
        choice("streaming", "Streaming"),
        choice("offline_instant", "Offline instant"),
    ];
    const PROVIDER: &[Choice] = &[choice("cpu", "CPU"), choice("cuda", "CUDA")];
    const MELO: &[Choice] = &[
        choice("auto", "Automatic"),
        choice("cpu", "CPU"),
        choice("cuda", "CUDA"),
    ];
    const OUTPUT: &[Choice] = &[
        choice("final_only", "Final only"),
        choice("streaming_partial", "Streaming partial"),
    ];
    const MODELS: &[Choice] = &[
        choice("gpt-4o-transcribe", "GPT-4o Transcribe"),
        choice("gpt-4o-mini-transcribe", "GPT-4o Mini Transcribe"),
        choice("gpt-4o-transcribe-latest", "GPT-4o Transcribe latest"),
        choice("whisper-1", "Whisper"),
    ];
    const TURN: &[Choice] = &[
        choice("manual", "Manual"),
        choice("server_vad", "Server VAD"),
        choice("semantic_vad", "Semantic VAD"),
    ];
    const EAGER: &[Choice] = &[
        choice("auto", "Automatic"),
        choice("low", "Low"),
        choice("medium", "Medium"),
        choice("high", "High"),
    ];
    let defaults = Config::default();
    let mut fields = BASIC_FIELDS.to_vec();
    for (section, keys) in crate::config::config_section_fields() {
        for key in *keys {
            let id = format!("{section}.{key}");
            if fields.iter().any(|f| f.id == id)
                || EXCLUDED.iter().any(|(excluded, _)| *excluded == id)
            {
                continue;
            }
            // One allocation per field for the process-lifetime schema.
            let id: &'static str = Box::leak(id.into_boxed_str());
            let kind = match *key {
                "sherpa_decode_mode" => FieldKind::Choice { choices: DECODE },
                "sherpa_provider" | "moonshine_provider" => FieldKind::Choice { choices: PROVIDER },
                "tts_melotts_device" => FieldKind::Choice { choices: MELO },
                "output_mode" => FieldKind::Choice { choices: OUTPUT },
                "openai_realtime_model" => FieldKind::Choice { choices: MODELS },
                "openai_realtime_turn_detection" => FieldKind::Choice { choices: TURN },
                "openai_realtime_vad_eagerness" => FieldKind::Choice { choices: EAGER },
                "text_replacements" => FieldKind::StringMap {
                    key_max_len: 256,
                    value_max_len: 1024,
                    max_entries: 1024,
                },
                "recognition_hints" => FieldKind::StringList {
                    item_max_len: 100,
                    max_items: 100,
                },
                "tts_playback_device" | "tts_local_device" => FieldKind::AudioDevice {
                    direction: "output",
                },
                _ => match defaults.field_to_value(key).unwrap_or(Value::Null) {
                    Value::Bool(_) => FieldKind::Bool,
                    Value::Number(n) if n.is_u64() || n.is_i64() => FieldKind::Int {
                        min: 0,
                        max: u32::MAX.into(),
                    },
                    Value::Number(_) => FieldKind::Float {
                        min: 0.0,
                        max: 1e9,
                        step: 0.01,
                    },
                    Value::Null => FieldKind::OptionalText { max_len: 4096 },
                    _ => FieldKind::Text { max_len: 4096 },
                },
            };
            let group = match *section {
                "asr" => Section::Speech,
                "vocabulary" => Section::Vocabulary,
                "typing" if *key == "text_replacements" => Section::Vocabulary,
                "typing" => Section::Typing,
                "tts" => Section::TextToSpeech,
                "audio" | "feedback" => Section::Audio,
                "overlay" => Section::Appearance,
                _ => Section::Advanced,
            };
            let (label, unit) = generated_label(key);
            fields.push(FieldMeta {
                advanced: !matches!(
                    *key,
                    "recognition_hints" | "text_replacements" | "tts_playback_device"
                ),
                unit,
                ..field(id, group, label, kind)
            });
        }
    }
    fields
});

pub fn schema_fields() -> Vec<Value> {
    let defaults = Config::default();
    FIELDS
        .iter()
        .map(|meta| {
            let mut value = serde_json::to_value(meta).expect("field metadata serializes");
            value["default"] = read_field(&defaults, meta.id);
            value
        })
        .collect()
}

/// Conservative allowlist: aliases and unimplemented local/worker adapters
/// remain unsupported even if their native libraries expose hotword knobs.
pub fn openai_hint_model_supported(model: &str) -> bool {
    matches!(
        model,
        "gpt-4o-transcribe" | "gpt-4o-mini-transcribe" | "whisper-1"
    )
}

pub fn vocabulary_capability(config: &Config) -> Value {
    let supported = config.asr_backend == crate::types::AsrBackendKind::OpenaiRealtime
        && openai_hint_model_supported(&config.openai_realtime_model);
    serde_json::json!({ "supported": supported, "detail": if supported { "Recognition hints are sent as OpenAI transcription prompt context; they are not guaranteed output." } else { "This backend/model/decode mode has no tested recognition-hint adapter. Correction rules still apply after transcription." } })
}

pub fn draft(
    path: impl AsRef<Path>,
    changes: &BTreeMap<String, Value>,
) -> Result<Config, ApplyError> {
    let path = path.as_ref();
    let raw = load_raw(path).map_err(|e| ApplyError::Io {
        message: e.to_string(),
    })?;
    let (migrated, _) = migrate_to_latest(&raw).map_err(|e| ApplyError::Io {
        message: e.to_string(),
    })?;
    let current = profile_current_value(&migrated);
    let patched = patch_changes(&migrated, &current, changes)
        .map_err(|errors| ApplyError::Invalid { errors })?;
    Config::resolve_raw(&patched)
        .map(|r| r.config)
        .map_err(|e| ApplyError::Invalid {
            errors: vec![attribute(e.to_string())],
        })
}

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
    pub source: Option<&'static str>,
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
        Ok(bytes) => Ok(revision_bytes(&bytes)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(ABSENT_REVISION.into()),
        Err(err) => Err(format!("cannot read {}: {err}", path.display())),
    }
}

/// Hash the exact bytes consumed by a config loader, without reopening its path.
pub(crate) fn revision_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
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
    for meta in FIELDS.iter() {
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

    let mut secrets = vec![
        SecretPresence {
            present: env_present(&config.openai_realtime_api_key_env),
            env: config.openai_realtime_api_key_env.clone(),
            used_by: "asr.asr_backend:openai_realtime",
            source: env_present(&config.openai_realtime_api_key_env).then_some("env"),
        },
        SecretPresence {
            present: env_present(&config.tts_api_key_env),
            env: config.tts_api_key_env.clone(),
            used_by: "tts.tts_backend",
            source: env_present(&config.tts_api_key_env).then_some("env"),
        },
    ];
    for secret in &mut secrets {
        if secret.present {
            continue;
        }
        for file in ["local.dev", "local.env"] {
            let local = path.parent().unwrap_or(Path::new(".")).join(file);
            if secret_in_file(&local, &secret.env) {
                secret.present = true;
                secret.source = Some(file);
                break;
            }
        }
    }

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

fn secret_in_file(path: &Path, name: &str) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    text.lines().any(|line| {
        let line = line
            .trim()
            .strip_prefix("export ")
            .unwrap_or(line.trim())
            .trim();
        let Some((key, value)) = line.split_once('=') else {
            return false;
        };
        let value = value.trim();
        let value = if value.len() >= 2
            && ((value.starts_with('"') && value.ends_with('"'))
                || (value.starts_with('\'') && value.ends_with('\'')))
        {
            &value[1..value.len() - 1]
        } else {
            value
        };
        key.trim() == name && !value.is_empty() && !key.trim().starts_with('#')
    })
}

pub fn onboarding_defaults(snapshot: &Snapshot) -> BTreeMap<String, Value> {
    use crate::config::wizard as w;
    let mut values = snapshot.values.clone();
    for (id, value) in [
        ("asr.asr_backend", Value::from(w::ASR_BACKEND)),
        ("asr.sherpa_model_name", Value::from(w::SHERPA_MODEL_NAME)),
        ("asr.sherpa_provider", Value::from(w::SHERPA_PROVIDER)),
        ("asr.instant_mode", Value::from(w::INSTANT_MODE)),
        ("asr.sherpa_decode_mode", Value::from(w::SHERPA_DECODE_MODE)),
        ("typing.output_mode", Value::from(w::OUTPUT_MODE)),
        (
            "typing.typing_final_injection_mode",
            Value::from(w::TYPING_FINAL_INJECTION_MODE),
        ),
        ("typing.typing_text_case", Value::from(w::TYPING_TEXT_CASE)),
        ("tts.tts_backend", Value::from(w::TTS_BACKEND)),
        (
            "tts.tts_default_voice_id",
            Value::from(w::TTS_DEFAULT_VOICE_ID),
        ),
        (
            "tts.tts_kokoro_base_url",
            Value::from(w::TTS_KOKORO_BASE_URL),
        ),
        ("tts.tts_playback_speed", Value::from(w::TTS_PLAYBACK_SPEED)),
    ] {
        if !snapshot.explicit.iter().any(|explicit| explicit == id) {
            values.insert(id.into(), value);
        }
    }
    values.insert(
        SHERPA_PROFILE.into(),
        Value::from(sherpa_profile_of(
            values["asr.sherpa_model_name"].as_str().unwrap_or_default(),
        )),
    );
    values
}

fn check_kind(meta: &FieldMeta, value: &Value) -> Result<(), String> {
    if value.is_null() {
        return Ok(());
    }
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
        FieldKind::Text { max_len } | FieldKind::OptionalText { max_len } => match value.as_str() {
            Some(s) if s.trim().is_empty() && meta.id != "asr.openai_realtime_language" => {
                Err("must not be empty".into())
            }
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
        FieldKind::StringList {
            item_max_len,
            max_items,
        } => match value.as_array() {
            Some(items)
                if items.len() <= max_items
                    && items.iter().all(|v| {
                        v.as_str().is_some_and(|s| {
                            !s.trim().is_empty()
                                && s.chars().count() <= item_max_len
                                && !s.chars().any(char::is_control)
                        })
                    }) =>
            {
                Ok(())
            }
            _ => Err(format!(
                "must be a list of at most {max_items} nonempty strings (at most {item_max_len} characters each)"
            )),
        },
        FieldKind::StringMap {
            key_max_len,
            value_max_len,
            max_entries,
        } => match value.as_object() {
            Some(items)
                if items.len() <= max_entries
                    && items.iter().all(|(k, v)| {
                        !k.trim().is_empty()
                            && k.chars().count() <= key_max_len
                            && !k.chars().any(char::is_control)
                            && v.as_str().is_some_and(|s| {
                                s.chars().count() <= value_max_len
                                    && !s.chars().any(char::is_control)
                            })
                    }) =>
            {
                Ok(())
            }
            _ => Err("must be a bounded map of nonempty keys to string values".into()),
        },
        FieldKind::AudioDevice { .. } => match value {
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
        if value.is_null() {
            for key in [
                "sherpa_model_name",
                "instant_mode",
                "sherpa_decode_mode",
                "sherpa_enable_parakeet_streaming",
            ] {
                asr.remove(key);
            }
            return;
        }
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
    let patched = patch_changes(migrated, current_values, changes)?;
    Config::resolve_raw(&patched).map_err(|err| vec![attribute(err.to_string())])?;
    Ok(patched)
}

fn profile_current_value(raw: &Map<String, Value>) -> BTreeMap<String, Value> {
    let model = raw_get(raw, "asr.sherpa_model_name")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_SHERPA_MODEL_NAME);
    BTreeMap::from([(SHERPA_PROFILE.into(), Value::from(sherpa_profile_of(model)))])
}

fn patch_changes(
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
    // Preset first, explicit raw keys second: raw draft overrides win,
    // independently of lexical ordering of the request object.
    if let Some(value) = changes.get(SHERPA_PROFILE)
        && value.as_str() != Some("custom")
    {
        write_change(&mut patched, SHERPA_PROFILE, value);
    }
    for (id, value) in changes {
        if id == SHERPA_PROFILE {
            continue;
        }
        write_change(&mut patched, id, value);
    }
    patched.insert("config_version".into(), Value::from(CURRENT_CONFIG_VERSION));
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
    // Cooperating bridge writers serialize the revision check and rename.
    // Editors ignoring this lock can still race the final filesystem rename.
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| io(e.to_string()))?;
    }
    use std::os::unix::fs::OpenOptionsExt;
    let lock_path = path.with_extension("toml.settings-lock");
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .mode(0o600)
        .open(lock_path)
        .map_err(|e| io(e.to_string()))?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        match lock.try_lock() {
            Ok(()) => break,
            Err(std::fs::TryLockError::WouldBlock) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(10))
            }
            Err(error) => return Err(io(format!("cannot acquire settings writer lock: {error}"))),
        }
    }

    let current = revision(&path).map_err(io)?;
    if current != expected_revision {
        return Err(ApplyError::Conflict {
            current_revision: current,
        });
    }
    let raw = load_raw(&path).map_err(|e| io(e.to_string()))?;
    let (migrated, _) = migrate_to_latest(&raw).map_err(|e| io(e.to_string()))?;
    let current_values = profile_current_value(&migrated);
    let patched = validate_changes(&migrated, &current_values, changes)
        .map_err(|errors| ApplyError::Invalid { errors })?;

    // Narrow the race with other writers: recheck right before the atomic write.
    let again = revision(&path).map_err(io)?;
    if again != expected_revision {
        return Err(ApplyError::Conflict {
            current_revision: again,
        });
    }
    // Identify the payload we commit, not a later read that could observe an
    // uncooperative editor's replacement instead of our saved settings.
    let saved_revision = revision_bytes(
        toml_dumps(&patched)
            .map_err(|e| io(e.to_string()))?
            .as_bytes(),
    );
    let backup = write_atomic(&path, &patched).map_err(|e| io(e.to_string()))?;
    Ok(Applied {
        revision: saved_revision,
        backup: backup.map(|p| p.display().to_string()),
        changed: changes.keys().cloned().collect(),
    })
}

#[cfg(test)]
mod tests {

    #[test]
    fn every_field_has_a_human_label() {
        for meta in FIELDS.iter() {
            let key = split_id(meta.id).1;
            assert_ne!(meta.label, key, "{} has no display label", meta.id);
        }
    }
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
        for (section, key) in known {
            let id = format!("{section}.{key}");
            assert_eq!(
                FIELDS.iter().filter(|f| f.id == id).count()
                    + EXCLUDED
                        .iter()
                        .filter(|(excluded, reason)| *excluded == id && !reason.is_empty())
                        .count(),
                1,
                "{id} must be covered exactly once"
            );
        }
        let fields = schema_fields();
        assert!(
            fields
                .iter()
                .all(|f| f.get("default").is_some() && f["advanced"].is_boolean())
        );
    }

    #[test]
    fn raw_sherpa_changes_override_the_virtual_preset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        apply(
            &path,
            ABSENT_REVISION,
            &changes(&[
                (SHERPA_PROFILE, json!("instant")),
                ("asr.sherpa_decode_mode", json!("streaming")),
                ("asr.instant_mode", json!(false)),
            ]),
        )
        .unwrap();
        let config = Config::load_from_path(path).unwrap();
        assert!(!config.instant_mode);
        assert_eq!(config.sherpa_decode_mode.as_str(), "streaming");
    }

    #[test]
    fn recognition_hints_validate_roundtrip_and_unsupported_modes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        apply(
            &path,
            ABSENT_REVISION,
            &changes(&[(
                "vocabulary.recognition_hints",
                json!([" ShuVoice ", "Hyprland"]),
            )]),
        )
        .unwrap();
        let config = Config::load_from_path(&path).unwrap();
        assert_eq!(config.recognition_hints, ["ShuVoice", "Hyprland"]);
        assert_eq!(vocabulary_capability(&config)["supported"], false);
        for list in [
            json!(["a", "A"]),
            json!([""]),
            json!(["x\ny"]),
            json!(["<x>"]),
            json!(["x".repeat(101)]),
        ] {
            assert!(draft(&path, &changes(&[("vocabulary.recognition_hints", list)])).is_err());
        }
        let cloud = draft(
            &path,
            &changes(&[("asr.asr_backend", json!("openai_realtime"))]),
        )
        .unwrap();
        assert_eq!(vocabulary_capability(&cloud)["supported"], true);
    }

    #[test]
    fn secrets_report_sources_and_never_values() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(&dir, "config_version = 1\n");
        std::fs::write(
            dir.path().join("local.env"),
            "export OPENAI_API_KEY='secret-value'\n",
        )
        .unwrap();
        let snap = snapshot(&path, |_| false).unwrap();
        assert_eq!(snap.secrets[0].source, Some("local.env"));
        assert!(snap.secrets[0].present);
        assert!(
            !serde_json::to_string(&snap)
                .unwrap()
                .contains("secret-value")
        );
        std::fs::write(dir.path().join("local.dev"), "OPENAI_API_KEY=dev-value\n").unwrap();
        assert_eq!(
            snapshot(&path, |_| false).unwrap().secrets[0].source,
            Some("local.dev")
        );
        assert_eq!(
            snapshot(&path, |_| true).unwrap().secrets[0].source,
            Some("env")
        );
    }

    #[test]
    fn onboarding_does_not_overwrite_explicit_settings() {
        let dir = tempfile::tempdir().unwrap();
        let fresh = snapshot(dir.path().join("config.toml"), |_| false).unwrap();
        assert_eq!(onboarding_defaults(&fresh)["asr.sherpa_profile"], "instant");
        assert_eq!(onboarding_defaults(&fresh)["tts.tts_playback_speed"], 1.25);
        let path = write(
            &dir,
            "config_version=1\n[asr]\nsherpa_model_name='custom'\n[tts]\ntts_playback_speed=1.5\n",
        );
        let current = snapshot(&path, |_| false).unwrap();
        let defaults = onboarding_defaults(&current);
        assert_eq!(defaults["tts.tts_playback_speed"], 1.5);
        assert_eq!(defaults["asr.sherpa_model_name"], "custom");
    }

    #[test]
    fn parallel_writers_get_one_commit_and_one_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let handles: Vec<_> = [24, 30]
            .into_iter()
            .map(|size| {
                let (path, barrier) = (path.clone(), barrier.clone());
                std::thread::spawn(move || {
                    barrier.wait();
                    apply(
                        path,
                        ABSENT_REVISION,
                        &changes(&[("overlay.font_size", json!(size))]),
                    )
                })
            })
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|r| matches!(r, Err(ApplyError::Conflict { .. })))
                .count(),
            1
        );
    }

    #[test]
    fn pre_save_validation_latency_measurement() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(&dir, "config_version=1\n[overlay]\nfont_size=22\n");
        let changes = changes(&[("overlay.font_size", json!(24))]);
        let before = std::time::Instant::now();
        for _ in 0..200 {
            // Previous bridge validation: load + migrate + full snapshot
            // (including revision hashing and another load/migration).
            let raw = load_raw(&path).unwrap();
            let (migrated, _) = migrate_to_latest(&raw).unwrap();
            let values = snapshot(&path, |_| false).unwrap().values;
            validate_changes(&migrated, &values, &changes).unwrap();
        }
        let before = before.elapsed();
        let after = std::time::Instant::now();
        for _ in 0..200 {
            draft(&path, &changes).unwrap();
        }
        eprintln!(
            "pre-save validation: before={:.1}us/op after={:.1}us/op (200 iterations, fixture config)",
            before.as_secs_f64() * 1e6 / 200.0,
            after.elapsed().as_secs_f64() * 1e6 / 200.0
        );
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
