//! Default constants and brand text replacements.

use std::collections::BTreeMap;

use once_cell::sync::Lazy;

pub const CURRENT_CONFIG_VERSION: u32 = 1;

pub const DEFAULT_ELEVENLABS_TTS_VOICE_ID: &str = "zNsotODqUhvbJ5wMG7Ei";
pub const DEFAULT_ELEVENLABS_TTS_MODEL_ID: &str = "eleven_flash_v2_5";
pub const DEFAULT_ELEVENLABS_TTS_API_KEY_ENV: &str = "ELEVENLABS_API_KEY";
pub const DEFAULT_OPENAI_TTS_VOICE_ID: &str = "onyx";
pub const DEFAULT_OPENAI_TTS_MODEL_ID: &str = "gpt-4o-mini-tts";
pub const DEFAULT_OPENAI_TTS_API_KEY_ENV: &str = "OPENAI_API_KEY";
pub const DEFAULT_MELOTTS_VOICE_ID: &str = "EN-US";
pub const DEFAULT_MELOTTS_MODEL_ID: &str = "melotts";
pub const DEFAULT_KOKORO_TTS_VOICE_ID: &str = "af_heart";
pub const DEFAULT_KOKORO_TTS_MODEL_ID: &str = "kokoro";
pub const DEFAULT_KOKORO_TTS_BASE_URL: &str = "http://localhost:8880/v1";
pub const DEFAULT_LOCAL_TTS_VOICE_ID: &str = "default";
pub const DEFAULT_LOCAL_TTS_MODEL_ID: &str = "piper";

pub const DEFAULT_SHERPA_MODEL_NAME: &str = "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06";
pub const PARAKEET_TDT_V3_INT8_MODEL_NAME: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8";

/// Wizard screenshot / stable-instant ASR profile defaults.
pub mod wizard {
    use super::*;

    pub const ASR_BACKEND: &str = "sherpa";
    pub const SHERPA_MODEL_NAME: &str = PARAKEET_TDT_V3_INT8_MODEL_NAME;
    pub const SHERPA_PROVIDER: &str = "cpu";
    pub const INSTANT_MODE: bool = true;
    pub const SHERPA_DECODE_MODE: &str = "offline_instant";
    pub const OUTPUT_MODE: &str = "final_only";
    pub const TYPING_FINAL_INJECTION_MODE: &str = "auto";
    pub const TYPING_TEXT_CASE: &str = "default";
    pub const TTS_BACKEND: &str = "kokoro";
    pub const TTS_DEFAULT_VOICE_ID: &str = DEFAULT_KOKORO_TTS_VOICE_ID;
    pub const TTS_KOKORO_BASE_URL: &str = DEFAULT_KOKORO_TTS_BASE_URL;
    pub const TTS_PLAYBACK_SPEED: f64 = 1.25;
}

pub static DEFAULT_TEXT_REPLACEMENTS: Lazy<BTreeMap<String, String>> = Lazy::new(|| {
    let pairs = [
        ("shove voice", "ShuVoice"),
        ("shove-voice", "ShuVoice"),
        ("shovevoice", "ShuVoice"),
        ("shu voice", "ShuVoice"),
        ("shu-voice", "ShuVoice"),
        ("shuvoice", "ShuVoice"),
        ("shoo voice", "ShuVoice"),
        ("shoo-voice", "ShuVoice"),
        ("shoovoice", "ShuVoice"),
        ("shoe voice", "ShuVoice"),
        ("shoe-voice", "ShuVoice"),
        ("shoevoice", "ShuVoice"),
        ("show voice", "ShuVoice"),
        ("show-voice", "ShuVoice"),
        ("showvoice", "ShuVoice"),
        ("hyper land", "Hyprland"),
        ("hyper-land", "Hyprland"),
        ("hyperland", "Hyprland"),
        ("hypr land", "Hyprland"),
        ("hypr-land", "Hyprland"),
        ("hyprland", "Hyprland"),
        ("hype land", "Hyprland"),
        ("hype-land", "Hyprland"),
        ("high per land", "Hyprland"),
        ("high-per-land", "Hyprland"),
        ("highper land", "Hyprland"),
        ("highper-land", "Hyprland"),
        ("highperland", "Hyprland"),
        ("hyper lend", "Hyprland"),
        ("hyper-lend", "Hyprland"),
        ("hyperlend", "Hyprland"),
    ];
    pairs
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
});

/// Nested TOML section → field names (for serialization).
pub fn config_section_fields() -> &'static [(&'static str, &'static [&'static str])] {
    &[
        (
            "audio",
            &[
                "sample_rate",
                "chunk_ms",
                "fallback_sample_rate",
                "audio_device",
                "input_gain",
                "audio_queue_max_size",
                "recording_preroll_ms",
                "silence_rms_threshold",
                "silence_rms_multiplier",
                "min_speech_ms",
                "auto_gain_target_peak",
                "auto_gain_max",
                "auto_gain_settle_chunks",
            ],
        ),
        (
            "asr",
            &[
                "asr_backend",
                "instant_mode",
                "model_name",
                "right_context",
                "device",
                "use_cuda_graph_decoder",
                "sherpa_model_name",
                "sherpa_model_dir",
                "sherpa_decode_mode",
                "sherpa_enable_parakeet_streaming",
                "sherpa_provider",
                "sherpa_num_threads",
                "sherpa_chunk_ms",
                "sherpa_offline_max_utterance_sec",
                "moonshine_model_name",
                "moonshine_model_dir",
                "moonshine_model_precision",
                "moonshine_chunk_ms",
                "moonshine_max_window_sec",
                "moonshine_max_tokens",
                "moonshine_provider",
                "moonshine_onnx_threads",
                "openai_realtime_model",
                "openai_realtime_api_key_env",
                "openai_realtime_language",
                "openai_realtime_latency_target_sec",
                "openai_realtime_turn_detection",
                "openai_realtime_vad_eagerness",
                "openai_realtime_request_timeout_sec",
                "openai_realtime_commit_timeout_sec",
            ],
        ),
        (
            "overlay",
            &[
                "font_size",
                "font_family",
                "bg_opacity",
                "border_radius",
                "bottom_margin",
                "overlay_debug_mode",
                "overlay_debug_max_lines",
            ],
        ),
        ("control", &["control_socket"]),
        (
            "tts",
            &[
                "tts_enabled",
                "tts_backend",
                "tts_default_voice_id",
                "tts_model_id",
                "tts_api_key_env",
                "tts_output_format",
                "tts_max_chars",
                "tts_request_timeout_sec",
                "tts_playback_speed",
                "tts_playback_device",
                "tts_overlay_auto_hide_sec",
                "tts_local_model_path",
                "tts_local_voice",
                "tts_local_device",
                "tts_melotts_device",
                "tts_melotts_venv_path",
                "tts_kokoro_base_url",
            ],
        ),
        (
            "typing",
            &[
                "output_mode",
                "typing_final_injection_mode",
                "typing_text_case",
                "use_clipboard_for_final",
                "preserve_clipboard",
                "typing_clipboard_settle_delay_ms",
                "typing_retry_attempts",
                "typing_retry_delay_ms",
                "typing_subprocess_timeout",
                "auto_capitalize",
                "text_replacements",
            ],
        ),
        (
            "streaming",
            &[
                "streaming_stall_guard",
                "streaming_stall_chunks",
                "streaming_stall_rms_ratio",
                "streaming_stall_flush_chunks",
            ],
        ),
        (
            "feedback",
            &[
                "audio_feedback",
                "feedback_start_freq",
                "feedback_stop_freq",
                "feedback_duration_ms",
                "feedback_volume",
            ],
        ),
    ]
}
