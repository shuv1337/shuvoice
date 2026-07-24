//! Headless domain policy for ShuVoice.
//!
//! This crate intentionally contains **no** GTK, audio device I/O, or network
//! code. Sibling crates (`shuvoice-io`, `shuvoice-asr`, `shuvoice-ui`, …) consume
//! these pure types and algorithms.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod audio_math;
pub mod circuit_breaker;
pub mod config;
pub mod env_loader;
pub mod error;
pub mod feedback;
pub mod flush;
pub mod metrics;
pub mod postprocess;
pub mod runtime;
pub mod streaming_health;
pub mod transcript;
pub mod tts_speed;
pub mod types;
pub mod utterance;
pub mod xdg;

pub use audio_math::{
    apply_utterance_gain, audio_rms, compute_utterance_rms_threshold, ms_to_samples,
    observe_recording_chunk, select_preroll_chunks, update_noise_floor, utterance_gain_is_noop,
};
pub use circuit_breaker::{
    ASR_CIRCUIT_COOLDOWN, ASR_MAX_FAILURES, BreakerAction, CircuitBreaker, ERROR_TOAST_SECONDS,
    MIN_SPLASH_VISIBLE, PTT_REARM_GRACE, remaining_splash_ms, should_ignore_start_during_rearm,
};
pub use config::{
    CURRENT_CONFIG_VERSION, Config, ConfigLoadReport, DEFAULT_TEXT_REPLACEMENTS, MigrationReport,
    PARAKEET_TDT_V3_INT8_MODEL_NAME, config_section_fields, expand_user_path, format_toml_float,
    load_raw, migrate_to_latest, toml_dumps, toml_value_to_json, wizard, write_atomic,
};
pub use env_loader::{
    LocalDevEnv, load_local_dev_env_file, merge_into_env_map, parse_local_dev_env_text,
};
pub use error::{CoreError, CoreResult};
pub use feedback::generate_tone;
pub use flush::{
    FLUSH_NOISE_ESCALATION, FLUSH_NOISE_MAX_RMS, FLUSH_NOISE_MIN_RMS, TailFlushDecision,
    evaluate_tail_flush_step, flush_noise_escalation, make_flush_noise,
};
pub use metrics::{MetricsCollector, MetricsSnapshot, metrics_to_human, metrics_to_json};
pub use postprocess::{
    CompiledReplacement, CompiledTextReplacements, RenderOptions, apply_text_replacements,
    capitalize_first, compile_text_replacements, find_bounded_phrase_matches, is_word_char,
    lowercase_text, render_transcript_text, sanitize_final_injection_text,
};
pub use runtime::{
    ASR_LOOP_POLL_TIMEOUT, BeginUtteranceParams, METRICS_LOG_PERIOD, STOP_TAIL_GRACE, StartGate,
    begin_utterance, capture_preroll, evaluate_start_gate, is_silent_utterance, recording_status,
};
pub use streaming_health::should_trigger_stall_flush;
pub use transcript::{MIN_OVERLAP_CHARS, MIN_OVERLAP_WORDS, prefer_transcript};
pub use tts_speed::{
    TTS_PLAYBACK_SPEED_DEFAULT, TTS_PLAYBACK_SPEED_MAX, TTS_PLAYBACK_SPEED_MIN,
    TTS_PLAYBACK_SPEED_STEP, format_tts_playback_speed, normalize_tts_playback_speed,
    normalize_tts_playback_speed_lossy, step_tts_playback_speed, validate_tts_playback_speed,
    validate_tts_playback_speed_str,
};
pub use types::{
    AsrBackendKind, AsrCapabilities, CONTROL_COMMANDS, ComputeProvider, DEPENDENCY_EXIT_CODE,
    DeviceRef, ExpectedChunking, FinalizationMode, InjectionMode, MeloTtsDevice,
    OpenaiTurnDetection, OpenaiVadEagerness, OutputMode, OverlayState, RecordingStatus,
    ResolvedSherpaDecodeMode, SherpaDecodeMode, TtsBackendKind, TtsCapabilities,
    TtsSynthesisRequest, TypingTextCase, VoiceInfo, is_parakeet_model, is_valid_control_command,
    looks_like_cuda_oom_error, overlay_state_class,
};
pub use utterance::UtteranceState;
pub use xdg::{
    config_dir, config_path, data_dir, local_dev_env_path, wizard_done_path, xdg_config_home,
    xdg_data_home, xdg_runtime_dir,
};

/// Exit status used for dependency/configuration failures that systemd must not restart.
pub const DEPENDENCY_EXIT_CODE_U8: u8 = DEPENDENCY_EXIT_CODE;
