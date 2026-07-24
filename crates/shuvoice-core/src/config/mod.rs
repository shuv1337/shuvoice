//! Configuration load, migration, validation, and persistence.

mod defaults;
mod io;
mod migrate;
mod model;

pub use defaults::{
    CURRENT_CONFIG_VERSION, DEFAULT_ELEVENLABS_TTS_API_KEY_ENV, DEFAULT_ELEVENLABS_TTS_MODEL_ID,
    DEFAULT_ELEVENLABS_TTS_VOICE_ID, DEFAULT_KOKORO_TTS_BASE_URL, DEFAULT_KOKORO_TTS_MODEL_ID,
    DEFAULT_KOKORO_TTS_VOICE_ID, DEFAULT_LOCAL_TTS_MODEL_ID, DEFAULT_LOCAL_TTS_VOICE_ID,
    DEFAULT_MELOTTS_MODEL_ID, DEFAULT_MELOTTS_VOICE_ID, DEFAULT_OPENAI_TTS_API_KEY_ENV,
    DEFAULT_OPENAI_TTS_MODEL_ID, DEFAULT_OPENAI_TTS_VOICE_ID, DEFAULT_SHERPA_MODEL_NAME,
    DEFAULT_TEXT_REPLACEMENTS, PARAKEET_TDT_V3_INT8_MODEL_NAME, config_section_fields, wizard,
};
pub use io::{
    backup_config, expand_user_path, format_toml_float, load_raw, toml_dumps, toml_value_to_json,
    write_atomic,
};
pub use migrate::{MigrationReport, migrate_to_latest};
pub use model::{Config, ConfigLoadReport};
