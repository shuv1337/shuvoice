//! TTS traits, provider adapters, worker clients, and playback for ShuVoice.
//!
//! # Overview
//!
//! - [`TtsBackend`] / [`SynthesisStream`] — async provider contract with
//!   authoritative PCM rate/encoding metadata
//! - [`TtsPlayer`] — generation-token state machine; playback on a dedicated OS thread
//! - Optional [`CpalAudioOutputFactory`] behind feature `cpal-output`
//!
//! Selection capture lives in `shuvoice-io`, not this crate.

#![allow(clippy::module_name_repetitions)]

pub mod backend;
pub mod error;
pub mod metrics;
pub mod mp3;
pub mod player;
pub mod registry;
pub mod speed;
pub mod types;

pub use backend::{
    CHILD_ENV_ALLOWLIST, ElevenLabsConfig, ElevenLabsTtsBackend, KokoroConfig, KokoroTtsBackend,
    MeloTtsBackend, MeloTtsConfig, MeloWireMode, MeloWorkerSpawn, OpenAiConfig, OpenAiTtsBackend,
    PcmStream, PiperConfig, PiperTtsBackend, SharedBackend, SynthesisStream, TtsBackend,
    build_isolated_child_env, find_piper_binary, piper_sample_rate_from_sidecar, redact_for_ui,
};
pub use error::TtsError;
pub use metrics::{CountingMetrics, Counts, NoopMetrics, TtsMetrics};
pub use mp3::{DecodedPcm, decode_mp3_to_pcm, pcm_samples_to_le_bytes, reject_mp3_player_input};
pub use player::{
    AudioOutput, AudioOutputFactory, FakeAudioOutput, FakeAudioOutputFactory, NullAudioOutput,
    NullAudioOutputFactory, PLAYER_QUEUE_CAPACITY, TtsPlayer, TtsPlayerBuilder,
    WORKER_JOIN_DEADLINE, chunk_to_samples, parse_sample_rate, resample_linear_i16,
};
#[cfg(feature = "cpal-output")]
pub use player::{CpalAudioOutput, CpalAudioOutputFactory, CpalOutputConfig, OutputDeviceInfo};
pub use registry::{
    TtsBackendSettings, create_elevenlabs_for_test, create_kokoro_for_test, create_openai_for_test,
    create_tts_backend, parse_backend_name,
};
pub use speed::{
    TTS_PLAYBACK_SPEED_DEFAULT, TTS_PLAYBACK_SPEED_MAX, TTS_PLAYBACK_SPEED_MIN,
    TTS_PLAYBACK_SPEED_STEP, format_tts_playback_speed, normalize_tts_playback_speed,
    step_tts_playback_speed, validate_tts_playback_speed,
};
pub use types::{
    AudioEncoding, BackendId, Capabilities, DEFAULT_ELEVENLABS_TTS_BASE_URL,
    DEFAULT_ELEVENLABS_TTS_MODEL_ID, DEFAULT_ELEVENLABS_TTS_VOICE_ID, DEFAULT_KOKORO_TTS_BASE_URL,
    DEFAULT_KOKORO_TTS_MODEL_ID, DEFAULT_KOKORO_TTS_VOICE_ID, DEFAULT_LOCAL_TTS_MODEL_ID,
    DEFAULT_LOCAL_TTS_VOICE_ID, DEFAULT_MELOTTS_MODEL_ID, DEFAULT_MELOTTS_VOICE_ID,
    DEFAULT_OPENAI_TTS_BASE_URL, DEFAULT_OPENAI_TTS_MODEL_ID, DEFAULT_OPENAI_TTS_VOICE_ID,
    EventInfo, PlayerEvent, PlayerState, StatusPayload, SynthesisRequest, VOICE_CACHE_TTL_SECS,
    VoiceInfo,
};
