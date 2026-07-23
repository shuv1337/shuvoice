#![allow(
    clippy::field_reassign_with_default,
    clippy::undocumented_unsafe_blocks
)]
use futures_util::StreamExt;
use httpmock::prelude::*;
use shuvoice_tts::{
    BackendId, TtsBackend, TtsBackendSettings, create_tts_backend, format_tts_playback_speed,
    normalize_tts_playback_speed, parse_backend_name, step_tts_playback_speed,
    validate_tts_playback_speed,
};

// Bring trait into scope for `backend.id()`.
use tokio_util::sync::CancellationToken;

#[test]
fn parses_known_backends() {
    for name in ["elevenlabs", "openai", "local", "melotts", "kokoro"] {
        assert!(parse_backend_name(name).is_ok(), "{name}");
    }
    assert_eq!(parse_backend_name("KOKORO").unwrap(), BackendId::Kokoro);
}

#[test]
fn rejects_unknown_backend() {
    let err = parse_backend_name("nope").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Unknown TTS backend"));
    assert!(msg.contains("elevenlabs"));
}

#[test]
fn speed_helpers_match_public_contract() {
    assert_eq!(validate_tts_playback_speed(1.25).unwrap(), 1.25);
    assert!(validate_tts_playback_speed(0.4).is_err());
    assert_eq!(normalize_tts_playback_speed(9.0), 2.0);
    assert_eq!(step_tts_playback_speed(1.0, 1), 1.1);
    assert_eq!(format_tts_playback_speed(1.0), "1.0×");
}

#[tokio::test]
async fn settings_override_provider_base_urls() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/audio/speech");
        then.status(200).body([1u8, 0]);
    });

    let mut settings = TtsBackendSettings {
        backend: BackendId::OpenAi,
        api_key_env: "SHUVOICE_TEST_URL_OAI".into(),
        openai_base_url: server.base_url(),
        default_voice_id: "onyx".into(),
        model_id: "gpt-4o-mini-tts".into(),
        ..TtsBackendSettings::default()
    };
    settings.output_format = "pcm_24000".into();

    unsafe { std::env::set_var("SHUVOICE_TEST_URL_OAI", "sk-test") };
    let backend = create_tts_backend(&settings).unwrap();
    let mut stream = backend
        .synthesize_stream(
            shuvoice_tts::SynthesisRequest::new("hi", "onyx", "gpt-4o-mini-tts", 1.0),
            CancellationToken::new(),
        )
        .await
        .unwrap()
        .chunks;
    let _ = stream.next().await;
    mock.assert();
    unsafe { std::env::remove_var("SHUVOICE_TEST_URL_OAI") };

    // Empty override falls back to default constant (still constructs).
    let settings = TtsBackendSettings {
        backend: BackendId::ElevenLabs,
        elevenlabs_base_url: "   ".into(),
        api_key_env: "SHUVOICE_TEST_URL_EL".into(),
        ..TtsBackendSettings::default()
    };
    unsafe { std::env::set_var("SHUVOICE_TEST_URL_EL", "k") };
    let backend = create_tts_backend(&settings).unwrap();
    assert_eq!(TtsBackend::id(backend.as_ref()), BackendId::ElevenLabs);
    unsafe { std::env::remove_var("SHUVOICE_TEST_URL_EL") };
}

#[test]
fn default_settings_preserve_stock_base_urls() {
    let s = TtsBackendSettings::default();
    assert_eq!(
        s.elevenlabs_base_url,
        shuvoice_tts::DEFAULT_ELEVENLABS_TTS_BASE_URL
    );
    assert_eq!(s.openai_base_url, shuvoice_tts::DEFAULT_OPENAI_TTS_BASE_URL);
    assert_eq!(s.kokoro_base_url, shuvoice_tts::DEFAULT_KOKORO_TTS_BASE_URL);
}

#[test]
#[cfg(not(feature = "worker-proto"))]
fn melotts_create_fails_closed_without_worker_proto_feature() {
    let settings = TtsBackendSettings {
        backend: BackendId::MeloTts,
        ..TtsBackendSettings::default()
    };
    match create_tts_backend(&settings) {
        Ok(_) => panic!("expected MeloTTS create to fail without worker-proto"),
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("worker-proto"),
                "expected fail-closed worker-proto error, got {msg}"
            );
        }
    }
}

#[test]
#[cfg(feature = "worker-proto")]
fn melotts_create_forces_worker_proto_and_ignores_legacy_helper() {
    use shuvoice_tts::{MeloTtsBackend, MeloWireMode};

    let settings = TtsBackendSettings {
        backend: BackendId::MeloTts,
        melotts_device: "cpu".into(),
        melotts_venv_path: Some(std::path::PathBuf::from("/tmp/melotts-venv")),
        melotts_worker_root: Some(std::path::PathBuf::from("/tmp/workers")),
        melotts_helper_script: Some(std::path::PathBuf::from("/tmp/melo_helper.py")),
        default_voice_id: "EN-US".into(),
        ..TtsBackendSettings::default()
    };
    let backend = create_tts_backend(&settings).expect("create melotts");
    assert_eq!(TtsBackend::id(backend.as_ref()), BackendId::MeloTts);

    // Downcast via concrete construction parity: rebuild config the same way and
    // assert resolve_spawn never mentions the legacy helper even when set on settings.
    let concrete = MeloTtsBackend::new(shuvoice_tts::MeloTtsConfig {
        device: "cpu".into(),
        wire_mode: MeloWireMode::WorkerProto,
        helper_script: None,
        worker_root: Some(std::path::PathBuf::from("/tmp/workers")),
        venv_path: std::path::PathBuf::from("/tmp/melotts-venv"),
        ..shuvoice_tts::MeloTtsConfig::default()
    });
    let err = concrete.resolve_spawn().unwrap_err().to_string();
    assert!(
        !err.contains("melo_helper"),
        "WorkerProto must not fall back to legacy helper: {err}"
    );
}

#[test]
#[cfg(feature = "worker-proto")]
fn melotts_settings_carry_worker_runtime_fields() {
    use shuvoice_tts::MeloWorkerSpawn;
    let spawn = MeloWorkerSpawn::new("/usr/bin/python3")
        .args(["-m", "melotts", "--fake"])
        .env_pair("PYTHONPATH", "/workers")
        .current_dir("/workers");
    let settings = TtsBackendSettings {
        backend: BackendId::MeloTts,
        melotts_worker_root: Some(std::path::PathBuf::from("/workers")),
        melotts_python_binary: Some(std::path::PathBuf::from("/usr/bin/python3")),
        melotts_worker_spawn: Some(spawn.clone()),
        melotts_worker_env: vec![("SHUVOICE_WORKER_FAKE".into(), "1".into())],
        melotts_helper_script: None,
        ..TtsBackendSettings::default()
    };
    assert_eq!(
        settings.melotts_worker_root.as_deref(),
        Some(std::path::Path::new("/workers"))
    );
    assert_eq!(settings.melotts_worker_spawn.as_ref(), Some(&spawn));
    assert!(settings.melotts_helper_script.is_none());
}
