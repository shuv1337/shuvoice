//! High-risk config/default/validation/io/migration parity matrix.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::{Map, Value, json};
use shuvoice_core::config::{
    CURRENT_CONFIG_VERSION, Config, DEFAULT_ELEVENLABS_TTS_VOICE_ID, DEFAULT_OPENAI_TTS_VOICE_ID,
    DEFAULT_TEXT_REPLACEMENTS, PARAKEET_TDT_V3_INT8_MODEL_NAME, config_section_fields,
    format_toml_float, load_raw, migrate_to_latest, toml_dumps, write_atomic,
};
use shuvoice_core::{
    AsrBackendKind, ComputeProvider, DeviceRef, InjectionMode, MeloTtsDevice, OutputMode,
    ResolvedSherpaDecodeMode, SherpaDecodeMode, TtsBackendKind, TypingTextCase,
    apply_text_replacements, is_parakeet_model,
};
use tempfile::tempdir;

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples")
}

fn map(v: Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap()
}

fn write_cfg(home: &Path, body: &str) -> PathBuf {
    let dir = home.join("shuvoice");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.toml");
    fs::write(&path, body).unwrap();
    path
}

#[test]
fn loads_every_tracked_example_config_toml() {
    let dir = examples_dir();
    assert!(dir.is_dir(), "examples dir missing at {}", dir.display());
    let mut paths: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("toml")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n.starts_with("config"))
        })
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "expected examples/config*.toml fixtures");

    for path in paths {
        let cfg = Config::load_from_path(&path)
            .unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));
        assert_eq!(cfg.config_version, CURRENT_CONFIG_VERSION);
        let dumped = cfg.to_toml_string().unwrap();
        let tmp = tempdir().unwrap();
        let out = tmp.path().join("round.toml");
        fs::write(&out, dumped).unwrap();
        Config::load_from_path(&out)
            .unwrap_or_else(|e| panic!("round-trip reload failed for {}: {e}", path.display()));
    }
}

#[test]
fn load_defaults_when_config_missing() {
    let tmp = tempdir().unwrap();
    let _g = ENV_LOCK.lock().unwrap();
    // SAFETY: ENV_LOCK serializes process-global env mutation in this test binary.
    unsafe {
        std::env::set_var("XDG_CONFIG_HOME", tmp.path());
    }
    let cfg = Config::load().unwrap();
    // SAFETY: paired cleanup under ENV_LOCK.
    unsafe {
        std::env::remove_var("XDG_CONFIG_HOME");
    }
    assert_eq!(cfg.sample_rate, 16000);
    assert_eq!(cfg.output_mode, OutputMode::FinalOnly);
    assert_eq!(cfg.typing_final_injection_mode, InjectionMode::Auto);
    assert_eq!(cfg.typing_text_case, TypingTextCase::Default);
    assert_eq!(cfg.audio_queue_max_size, 200);
    assert!(cfg.audio_feedback);
    assert_eq!(cfg.text_replacements, *DEFAULT_TEXT_REPLACEMENTS);
    assert_eq!(cfg.asr_backend, AsrBackendKind::Sherpa);
    assert_eq!(cfg.tts_backend, TtsBackendKind::Elevenlabs);
    assert_eq!(cfg.tts_default_voice_id, DEFAULT_ELEVENLABS_TTS_VOICE_ID);
}

#[test]
fn load_flattens_sections_devices_and_replacements() {
    let tmp = tempdir().unwrap();
    let path = write_cfg(
        tmp.path(),
        r#"
[audio]
chunk_ms = 80
audio_queue_max_size = 55
recording_preroll_ms = 180
silence_rms_threshold = 0.007
silence_rms_multiplier = 2.2
min_speech_ms = 90
auto_gain_target_peak = 0.11
auto_gain_max = 7.5
auto_gain_settle_chunks = 3
unknown_audio_key = 999

[asr]
asr_backend = "sherpa"
sherpa_model_name = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
sherpa_provider = "cuda"
sherpa_num_threads = 4
sherpa_chunk_ms = 120

[overlay]
font_size = 28
font_family = "JetBrains Mono"
overlay_debug_mode = true
overlay_debug_max_lines = 20

[tts]
tts_backend = "local"
tts_default_voice_id = "voice-test"
tts_model_id = "model-test"
tts_api_key_env = "TEST_TTS_KEY"
tts_max_chars = 1234
tts_request_timeout_sec = 12.5
tts_playback_speed = 1.3
tts_playback_device = "2"
tts_local_model_path = "/tmp/piper-models"
tts_local_voice = "amy"
tts_local_device = "3"

[typing]
output_mode = "streaming_partial"
typing_text_case = "lowercase"
use_clipboard_for_final = true
preserve_clipboard = true
typing_retry_attempts = 3
typing_retry_delay_ms = 20
auto_capitalize = false

[typing.text_replacements]
"shove voice" = "ShuVoice"
"um" = ""

[streaming]
streaming_stall_guard = false
streaming_stall_chunks = 6
streaming_stall_rms_ratio = 0.9
streaming_stall_flush_chunks = 2

[feedback]
audio_feedback = false
feedback_start_freq = 500

[extra]
foo = "bar"
"#,
    );
    let (cfg, report) = Config::load_from_path_with_report(&path).unwrap();
    assert!(report.ignored_keys.iter().any(|k| k == "unknown_audio_key"));
    assert_eq!(cfg.chunk_ms, 80);
    assert_eq!(cfg.audio_queue_max_size, 55);
    assert_eq!(cfg.sherpa_provider, ComputeProvider::Cuda);
    assert_eq!(cfg.font_family.as_deref(), Some("JetBrains Mono"));
    assert!(cfg.overlay_debug_mode);
    assert_eq!(cfg.tts_backend, TtsBackendKind::Local);
    assert_eq!(cfg.tts_playback_device, Some(DeviceRef::Index(2)));
    assert_eq!(cfg.tts_local_device, Some(DeviceRef::Index(3)));
    assert!((cfg.tts_playback_speed - 1.3).abs() < 1e-12);
    assert_eq!(cfg.output_mode, OutputMode::StreamingPartial);
    assert_eq!(cfg.typing_text_case, TypingTextCase::Lowercase);
    assert_eq!(cfg.typing_final_injection_mode, InjectionMode::Auto);
    assert_eq!(
        cfg.text_replacements.get("shove voice").map(String::as_str),
        Some("ShuVoice")
    );
    assert_eq!(
        cfg.text_replacements.get("um").map(String::as_str),
        Some("")
    );
    assert_eq!(
        cfg.text_replacements.get("shu voice").map(String::as_str),
        Some("ShuVoice")
    );
    assert!(!cfg.streaming_stall_guard);
    assert!(!cfg.audio_feedback);
}

#[test]
fn legacy_clipboard_mapping_matrix() {
    let tmp = tempdir().unwrap();
    let path = write_cfg(tmp.path(), "[typing]\nuse_clipboard_for_final = false\n");
    let cfg = Config::load_from_path(&path).unwrap();
    assert_eq!(cfg.typing_final_injection_mode, InjectionMode::Direct);

    let path = write_cfg(tmp.path(), "[typing]\nuse_clipboard_for_final = true\n");
    let (cfg, report) = Config::load_from_path_with_report(&path).unwrap();
    assert_eq!(cfg.typing_final_injection_mode, InjectionMode::Auto);
    assert!(report.derived_mode_from_legacy);
    assert!(report.persist_attempted);
    assert!(report.persist_error.is_none());
    let persisted = fs::read_to_string(&path).unwrap();
    assert!(persisted.contains("typing_final_injection_mode = \"auto\""));

    let path = write_cfg(
        tmp.path(),
        "[typing]\ntyping_final_injection_mode = \"auto\"\nuse_clipboard_for_final = false\n",
    );
    let (cfg, report) = Config::load_from_path_with_report(&path).unwrap();
    assert_eq!(cfg.typing_final_injection_mode, InjectionMode::Auto);
    assert!(!cfg.use_clipboard_for_final);
    assert!(!report.derived_mode_from_legacy);
}

#[test]
fn migrate_v0_to_v1_persists_and_future_rejected() {
    let tmp = tempdir().unwrap();
    let path = write_cfg(tmp.path(), "[asr]\nasr_backend = \"sherpa\"\n");
    let (cfg, report) = Config::load_from_path_with_report(&path).unwrap();
    assert_eq!(cfg.config_version, CURRENT_CONFIG_VERSION);
    assert_eq!(report.migration.from_version, 0);
    assert!(report.persist_attempted);
    let raw = load_raw(&path).unwrap();
    assert_eq!(
        raw.get("config_version").and_then(|v| v.as_u64()),
        Some(CURRENT_CONFIG_VERSION as u64)
    );

    let current = map(json!({
        "config_version": CURRENT_CONFIG_VERSION,
        "audio": {"sample_rate": 16000}
    }));
    let (migrated, report) = migrate_to_latest(&current).unwrap();
    assert_eq!(migrated, current);
    assert!(report.changed_keys.is_empty());

    let future = map(json!({"config_version": CURRENT_CONFIG_VERSION + 1}));
    assert!(
        migrate_to_latest(&future)
            .unwrap_err()
            .to_string()
            .contains("newer")
    );
}

#[test]
fn tts_backend_default_remapping() {
    let cfg = Config::try_with(|c| c.tts_backend = TtsBackendKind::Openai).unwrap();
    assert_eq!(cfg.tts_default_voice_id, DEFAULT_OPENAI_TTS_VOICE_ID);
    assert_eq!(cfg.tts_model_id, "gpt-4o-mini-tts");
    assert_eq!(cfg.tts_api_key_env, "OPENAI_API_KEY");

    let cfg = Config::try_with(|c| c.tts_backend = TtsBackendKind::Local).unwrap();
    assert_eq!(cfg.tts_default_voice_id, "default");
    assert_eq!(cfg.tts_model_id, "piper");

    let cfg = Config::try_with(|c| {
        c.tts_backend = TtsBackendKind::Local;
        c.tts_local_voice = Some("amy".into());
    })
    .unwrap();
    assert_eq!(cfg.tts_default_voice_id, "amy");

    let cfg = Config::try_with(|c| c.tts_backend = TtsBackendKind::Melotts).unwrap();
    assert_eq!(cfg.tts_default_voice_id, "EN-US");
    assert_eq!(cfg.tts_model_id, "melotts");

    let cfg = Config::try_with(|c| c.tts_backend = TtsBackendKind::Kokoro).unwrap();
    assert_eq!(cfg.tts_default_voice_id, "af_heart");
    assert_eq!(cfg.tts_model_id, "kokoro");
}

#[test]
fn instant_mode_and_sherpa_resolve_matrix() {
    let cfg = Config::try_with(|c| {
        c.asr_backend = AsrBackendKind::Nemo;
        c.right_context = 13;
        c.instant_mode = true;
    })
    .unwrap();
    assert_eq!(cfg.right_context, 0);

    let cfg = Config::try_with(|c| {
        c.asr_backend = AsrBackendKind::Sherpa;
        c.sherpa_decode_mode = SherpaDecodeMode::Streaming;
        c.sherpa_chunk_ms = 120;
        c.instant_mode = true;
    })
    .unwrap();
    assert_eq!(cfg.sherpa_chunk_ms, 80);

    let cfg = Config::try_with(|c| {
        c.asr_backend = AsrBackendKind::Moonshine;
        c.instant_mode = true;
        c.moonshine_model_name = "moonshine/base".into();
        c.moonshine_max_window_sec = 5.0;
        c.moonshine_max_tokens = 64;
    })
    .unwrap();
    assert_eq!(cfg.moonshine_model_name, "moonshine/tiny");
    assert!((cfg.moonshine_max_window_sec - 3.0).abs() < 1e-12);
    assert_eq!(cfg.moonshine_max_tokens, 48);

    assert!(
        Config::try_with(|c| c.asr_backend = AsrBackendKind::Nemo)
            .unwrap()
            .resolved_sherpa_decode_mode()
            .is_none()
    );

    let cfg = Config::try_with(|c| {
        c.asr_backend = AsrBackendKind::Sherpa;
        c.sherpa_decode_mode = SherpaDecodeMode::Streaming;
        c.sherpa_model_name = PARAKEET_TDT_V3_INT8_MODEL_NAME.into();
        c.instant_mode = true;
    })
    .unwrap();
    assert_eq!(
        cfg.resolved_sherpa_decode_mode(),
        Some(ResolvedSherpaDecodeMode::Streaming)
    );

    let cfg = Config::try_with(|c| {
        c.asr_backend = AsrBackendKind::Sherpa;
        c.sherpa_decode_mode = SherpaDecodeMode::Auto;
        c.sherpa_model_name = PARAKEET_TDT_V3_INT8_MODEL_NAME.into();
        c.instant_mode = true;
        c.sherpa_chunk_ms = 120;
    })
    .unwrap();
    assert_eq!(
        cfg.resolved_sherpa_decode_mode(),
        Some(ResolvedSherpaDecodeMode::OfflineInstant)
    );
    assert_eq!(cfg.sherpa_chunk_ms, 120);

    let cfg = Config::try_with(|c| {
        c.asr_backend = AsrBackendKind::Sherpa;
        c.sherpa_decode_mode = SherpaDecodeMode::Auto;
        c.sherpa_model_name = "SHERPA-ONNX-NEMO-PARAKEET-TDT-0.6B-V3-INT8".into();
        c.instant_mode = true;
    })
    .unwrap();
    assert_eq!(
        cfg.resolved_sherpa_decode_mode(),
        Some(ResolvedSherpaDecodeMode::OfflineInstant)
    );
    assert!(is_parakeet_model(&cfg.sherpa_model_name));
}

#[test]
fn validation_enum_and_range_errors() {
    assert!(Config::try_with(|c| c.font_size = 0).is_err());
    assert!(Config::try_with(|c| c.bg_opacity = 1.1).is_err());
    assert!(Config::try_with(|c| c.input_gain = 0.0).is_err());
    assert!(Config::try_with(|c| c.sample_rate = 0).is_err());
    assert!(Config::try_with(|c| c.overlay_debug_max_lines = 0).is_err());
    assert!(Config::try_with(|c| c.tts_max_chars = 0).is_err());
    assert!(Config::try_with(|c| c.tts_request_timeout_sec = 0.0).is_err());
    assert!(Config::try_with(|c| c.tts_playback_speed = 0.25).is_err());
    assert!(Config::try_with(|c| c.tts_overlay_auto_hide_sec = -0.1).is_err());
    assert!(Config::try_with(|c| c.sherpa_chunk_ms = 0).is_err());
    assert!(Config::try_with(|c| c.sherpa_num_threads = 0).is_err());
    assert!(Config::try_with(|c| c.sherpa_model_name = "   ".into()).is_err());
    assert!(Config::try_with(|c| c.moonshine_chunk_ms = 0).is_err());
    assert!(Config::try_with(|c| c.moonshine_max_window_sec = 0.0).is_err());
    assert!(Config::try_with(|c| c.moonshine_max_tokens = 0).is_err());
    assert!(Config::try_with(|c| c.audio_queue_max_size = 0).is_err());
    assert!(Config::try_with(|c| c.auto_gain_target_peak = 0.0).is_err());
    assert!(Config::try_with(|c| c.auto_gain_max = 0.9).is_err());
    assert!(Config::try_with(|c| c.auto_gain_settle_chunks = 0).is_err());
    assert!(Config::try_with(|c| c.streaming_stall_chunks = 0).is_err());
    assert!(Config::try_with(|c| c.streaming_stall_rms_ratio = 0.0).is_err());
    assert!(Config::try_with(|c| c.openai_realtime_commit_timeout_sec = 0.0).is_err());
    assert!(Config::try_with(|c| c.tts_api_key_env = "   ".into()).is_err());
    assert!(
        Config::try_with(|c| {
            c.font_family = Some("Sans\"; color: red;".into());
        })
        .is_err()
    );
    assert!(
        Config::try_with(|c| c.font_family = Some("   ".into()))
            .unwrap()
            .font_family
            .is_none()
    );

    for d in [MeloTtsDevice::Auto, MeloTtsDevice::Cpu, MeloTtsDevice::Cuda] {
        assert_eq!(
            Config::try_with(|c| c.tts_melotts_device = d)
                .unwrap()
                .tts_melotts_device,
            d
        );
    }
    assert!(
        Config::try_with(|c| c.tts_melotts_venv_path = Some("   ".into()))
            .unwrap()
            .tts_melotts_venv_path
            .is_none()
    );
}

#[test]
fn text_replacements_and_section_fields() {
    let cfg = Config::try_with(|c| {
        c.text_replacements = BTreeMap::from([(" shove voice ".into(), " ShuVoice ".into())]);
    })
    .unwrap();
    assert_eq!(
        cfg.text_replacements.get("shove voice").map(String::as_str),
        Some("ShuVoice")
    );
    assert_eq!(
        cfg.text_replacements.get("hyper land").map(String::as_str),
        Some("Hyprland")
    );

    let asr = config_section_fields()
        .iter()
        .find(|(n, _)| *n == "asr")
        .unwrap()
        .1;
    assert!(asr.contains(&"openai_realtime_model"));
    let overlay = config_section_fields()
        .iter()
        .find(|(n, _)| *n == "overlay")
        .unwrap()
        .1;
    assert!(overlay.contains(&"overlay_debug_mode"));
    let tts = config_section_fields()
        .iter()
        .find(|(n, _)| *n == "tts")
        .unwrap()
        .1;
    assert!(tts.contains(&"tts_melotts_device"));
}

#[test]
fn atomic_backup_stable_toml_and_full_nested_round_trip() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("config.toml");
    fs::write(&path, "config_version = 1\n").unwrap();
    let backup = write_atomic(
        &path,
        &map(json!({
            "config_version": CURRENT_CONFIG_VERSION,
            "audio": {"sample_rate": 16000, "input_gain": 1.0, "silence_rms_threshold": 0.15},
            "asr": {"asr_backend": "sherpa"}
        })),
    )
    .unwrap();
    assert!(backup.unwrap().exists());
    let content = fs::read_to_string(&path).unwrap();
    assert!(content.contains("input_gain = 1.0"));
    assert!(content.contains("silence_rms_threshold = 0.15"));
    assert_eq!(format_toml_float(1.25), "1.25");
    assert_eq!(format_toml_float(1.0), "1.0");

    let cfg = Config::try_with(|c| {
        c.asr_backend = AsrBackendKind::Sherpa;
        c.sherpa_decode_mode = SherpaDecodeMode::OfflineInstant;
        c.sherpa_enable_parakeet_streaming = true;
        c.tts_backend = TtsBackendKind::Local;
        c.tts_playback_speed = 1.4;
        c.tts_local_model_path = Some("/tmp/models".into());
        c.overlay_debug_mode = true;
        c.overlay_debug_max_lines = 15;
        c.typing_final_injection_mode = InjectionMode::Clipboard;
        c.text_replacements
            .insert("custom phrase".into(), "Custom".into());
    })
    .unwrap();
    let nested = cfg.to_nested_map(false);
    assert_eq!(
        nested["asr"]["sherpa_decode_mode"].as_str(),
        Some("offline_instant")
    );
    let dumped = toml_dumps(&nested).unwrap();
    assert!(dumped.contains("\"custom phrase\" = \"Custom\""));
    let out = tmp.path().join("round.toml");
    fs::write(&out, dumped).unwrap();
    let reloaded = Config::load_from_path(&out).unwrap();
    assert_eq!(
        reloaded.sherpa_decode_mode,
        SherpaDecodeMode::OfflineInstant
    );
    assert!(reloaded.sherpa_enable_parakeet_streaming);
    assert_eq!(reloaded.tts_backend, TtsBackendKind::Local);
    assert!((reloaded.tts_playback_speed - 1.4).abs() < 1e-12);
    assert_eq!(reloaded.overlay_debug_max_lines, 15);
    assert_eq!(
        reloaded.typing_final_injection_mode,
        InjectionMode::Clipboard
    );
    assert_eq!(
        reloaded
            .text_replacements
            .get("custom phrase")
            .map(String::as_str),
        Some("Custom")
    );
}

#[test]
fn default_brand_replacements_apply() {
    let cfg = Config::default();
    let text = apply_text_replacements(
        "Shove Voice on Hyperland and high per land",
        Some(&cfg.text_replacements),
        Some(&cfg.compiled_text_replacements),
    );
    assert!(text.contains("ShuVoice"));
    assert!(text.contains("Hyprland"));
    assert!(!text.to_ascii_lowercase().contains("hyperland"));
}

#[test]
fn config_helpers_create_dirs() {
    let tmp = tempdir().unwrap();
    let _g = ENV_LOCK.lock().unwrap();
    // SAFETY: ENV_LOCK serializes process-global env mutation in this test binary.
    unsafe {
        std::env::set_var("XDG_CONFIG_HOME", tmp.path().join("cfg"));
        std::env::set_var("XDG_DATA_HOME", tmp.path().join("data"));
    }
    assert!(Config::config_dir().is_dir());
    assert!(Config::data_dir().is_dir());
    // SAFETY: paired cleanup under ENV_LOCK.
    unsafe {
        std::env::remove_var("XDG_CONFIG_HOME");
        std::env::remove_var("XDG_DATA_HOME");
    }
}

#[test]
fn openai_realtime_rejects_bad_model() {
    assert!(
        Config::try_with(|c| {
            c.asr_backend = AsrBackendKind::OpenaiRealtime;
            c.openai_realtime_model = "gpt-realtime-whisper".into();
        })
        .is_err()
    );
    let cfg = Config::try_with(|c| c.asr_backend = AsrBackendKind::OpenaiRealtime).unwrap();
    assert_eq!(cfg.openai_realtime_model, "gpt-4o-transcribe");
}

#[test]
fn audio_device_digit_string_stays_name_tts_digits_become_index() {
    let tmp = tempdir().unwrap();
    let path = write_cfg(
        tmp.path(),
        r#"
config_version = 1
[audio]
audio_device = "2"
[tts]
tts_playback_device = "2"
tts_local_device = "3"
"#,
    );
    let cfg = Config::load_from_path(&path).unwrap();
    assert_eq!(cfg.audio_device, Some(DeviceRef::Name("2".into())));
    assert_eq!(cfg.tts_playback_device, Some(DeviceRef::Index(2)));
    assert_eq!(cfg.tts_local_device, Some(DeviceRef::Index(3)));
}

#[test]
fn integer_fields_accept_integral_floats_reject_fractional() {
    let tmp = tempdir().unwrap();
    let path = write_cfg(
        tmp.path(),
        r#"
config_version = 1
[audio]
chunk_ms = 80.0
sample_rate = 16000.0
audio_queue_max_size = 55.0
"#,
    );
    let cfg = Config::load_from_path(&path).unwrap();
    assert_eq!(cfg.chunk_ms, 80);
    assert_eq!(cfg.sample_rate, 16000);
    assert_eq!(cfg.audio_queue_max_size, 55);

    let path = write_cfg(
        tmp.path(),
        r#"
config_version = 1
[audio]
chunk_ms = 80.5
"#,
    );
    assert!(Config::load_from_path(&path).is_err());
}

#[test]
fn empty_whitespace_optional_fields_normalize_to_none() {
    // Intentional runtime normalization: trim blank optional values to `None`.
    let cfg = Config::try_with(|c| {
        c.tts_local_model_path = Some("   ".into());
        c.tts_local_voice = Some("".into());
        c.tts_melotts_venv_path = Some("\t  ".into());
        c.sherpa_model_dir = Some("  ".into());
        c.moonshine_model_dir = Some("\n".into());
        c.control_socket = Some("   ".into());
        c.font_family = Some("   ".into());
    })
    .unwrap();
    assert!(cfg.tts_local_model_path.is_none());
    assert!(cfg.tts_local_voice.is_none());
    assert!(cfg.tts_melotts_venv_path.is_none());
    assert!(cfg.sherpa_model_dir.is_none());
    assert!(cfg.moonshine_model_dir.is_none());
    assert!(cfg.control_socket.is_none());
    assert!(cfg.font_family.is_none());
}

#[test]
fn fail_closed_output_mode_bool_and_u32_validation() {
    // Intentional fail-closed improvement: reject garbage instead of coercing.
    let tmp = tempdir().unwrap();
    let path = write_cfg(
        tmp.path(),
        "config_version = 1\n[typing]\noutput_mode = \"nope\"\n",
    );
    assert!(Config::load_from_path(&path).is_err());

    let path = write_cfg(
        tmp.path(),
        "config_version = 1\n[typing]\nuse_clipboard_for_final = \"yes\"\n",
    );
    assert!(Config::load_from_path(&path).is_err());

    let path = write_cfg(
        tmp.path(),
        "config_version = 1\n[audio]\nsample_rate = -1\n",
    );
    assert!(Config::load_from_path(&path).is_err());

    let path = write_cfg(
        tmp.path(),
        "config_version = 1\n[audio]\nsample_rate = \"16000; drop\"\n",
    );
    assert!(Config::load_from_path(&path).is_err());
}
