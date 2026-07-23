#![allow(
    clippy::field_reassign_with_default,
    clippy::undocumented_unsafe_blocks
)]
use std::time::Duration;

use futures_util::StreamExt;
use httpmock::prelude::*;
use serde_json::json;
use shuvoice_tts::{
    ElevenLabsConfig, ElevenLabsTtsBackend, KokoroConfig, KokoroTtsBackend, OpenAiConfig,
    OpenAiTtsBackend, SynthesisRequest, TtsBackend,
};
use tokio_util::sync::CancellationToken;

fn pcm_request(speed: f64) -> SynthesisRequest {
    SynthesisRequest::new("hello world", "voice-1", "model-1", speed)
}

#[tokio::test]
async fn elevenlabs_shapes_request_and_streams() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/text-to-speech/voice-1/stream")
            .header("xi-api-key", "test-key");
        then.status(200)
            .header("content-type", "application/octet-stream")
            .body([1u8, 0, 2, 0]);
    });

    let env = "SHUVOICE_TEST_EL_KEY";
    unsafe { std::env::set_var(env, "test-key") };

    let mut cfg = ElevenLabsConfig::default();
    cfg.base_url = server.base_url();
    cfg.api_key_env = env.into();
    cfg.request_timeout = Duration::from_secs(5);
    let backend = ElevenLabsTtsBackend::new(cfg).unwrap();
    assert_eq!(backend.sample_rate_hz(), 24_000);

    let mut stream = backend
        .synthesize_stream(pcm_request(1.25), CancellationToken::new())
        .await
        .unwrap()
        .chunks;
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        bytes.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(bytes, vec![1, 0, 2, 0]);
    mock.assert();
    unsafe { std::env::remove_var(env) };
}

#[tokio::test]
async fn elevenlabs_lists_and_caches_voices() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(GET).path("/voices");
        then.status(200).json_body(json!({
            "voices": [
                {"voice_id": "v1", "name": "One", "description": "d"},
                {"voice_id": "", "name": "skip"},
            ]
        }));
    });

    let env = "SHUVOICE_TEST_EL_KEY2";
    unsafe { std::env::set_var(env, "k") };
    let mut cfg = ElevenLabsConfig::default();
    cfg.base_url = server.base_url();
    cfg.api_key_env = env.into();
    let backend = ElevenLabsTtsBackend::new(cfg).unwrap();
    let voices = backend.list_voices().await.unwrap();
    assert_eq!(voices.len(), 1);
    assert_eq!(voices[0].id, "v1");
    let _ = backend.list_voices().await.unwrap();
    mock.assert_calls(1);
    unsafe { std::env::remove_var(env) };
}

#[tokio::test]
async fn elevenlabs_http_error_classification() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/text-to-speech/voice-1/stream");
        then.status(401);
    });
    let env = "SHUVOICE_TEST_EL_KEY3";
    unsafe { std::env::set_var(env, "k") };
    let mut cfg = ElevenLabsConfig::default();
    cfg.base_url = server.base_url();
    cfg.api_key_env = env.into();
    let backend = ElevenLabsTtsBackend::new(cfg).unwrap();
    let err = backend
        .synthesize_stream(pcm_request(1.0), CancellationToken::new())
        .await
        .err()
        .unwrap();
    assert!(err.to_string().contains("authentication failed (401)"));
    unsafe { std::env::remove_var(env) };
}

#[tokio::test]
async fn openai_shapes_request_and_clamps_speed() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/audio/speech");
        then.status(200).body([3u8, 0]);
    });
    let env = "SHUVOICE_TEST_OAI_KEY";
    unsafe { std::env::set_var(env, "sk-test") };
    let mut cfg = OpenAiConfig::default();
    cfg.base_url = server.base_url();
    cfg.api_key_env = env.into();
    let backend = OpenAiTtsBackend::new(cfg).unwrap();

    let mut stream = backend
        .synthesize_stream(pcm_request(1.0), CancellationToken::new())
        .await
        .unwrap()
        .chunks;
    let chunk = stream.next().await.unwrap().unwrap();
    assert_eq!(&chunk[..], &[3, 0]);
    mock.assert();

    let voices = backend.list_voices().await.unwrap();
    assert!(voices.iter().any(|v| v.id == "onyx"));
    unsafe { std::env::remove_var(env) };
}

#[tokio::test]
async fn openai_rejects_non_pcm() {
    let mut cfg = OpenAiConfig::default();
    cfg.output_format = "mp3".into();
    cfg.api_key_env = "SHUVOICE_TEST_OAI_KEY_X".into();
    unsafe { std::env::set_var("SHUVOICE_TEST_OAI_KEY_X", "k") };
    let backend = OpenAiTtsBackend::new(cfg).unwrap();
    let err = backend
        .synthesize_stream(pcm_request(1.0), CancellationToken::new())
        .await
        .err()
        .unwrap();
    assert!(err.to_string().contains("raw PCM"));
    unsafe { std::env::remove_var("SHUVOICE_TEST_OAI_KEY_X") };
}

#[tokio::test]
async fn kokoro_voices_string_list_and_cache() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(GET).path("/audio/voices");
        then.status(200)
            .json_body(json!({"voices": ["af_heart", "bm_george", ""]}));
    });
    let mut cfg = KokoroConfig::default();
    cfg.base_url = server.base_url();
    let backend = KokoroTtsBackend::new(cfg).unwrap();
    let voices = backend.list_voices().await.unwrap();
    assert_eq!(voices.len(), 2);
    assert_eq!(voices[0].id, "af_heart");
    let _ = backend.list_voices().await.unwrap();
    mock.assert_calls(1);
}

#[tokio::test]
async fn kokoro_pcm_stream_and_speed() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/audio/speech");
        then.status(200).body([9u8, 0, 8, 0]);
    });
    let mut cfg = KokoroConfig::default();
    cfg.base_url = server.base_url();
    let backend = KokoroTtsBackend::new(cfg).unwrap();
    assert!(backend.capabilities().supports_speed_control);
    assert!(!backend.capabilities().requires_api_key);

    let mut stream = backend
        .synthesize_stream(pcm_request(1.5), CancellationToken::new())
        .await
        .unwrap()
        .chunks;
    let mut out = Vec::new();
    while let Some(c) = stream.next().await {
        out.extend_from_slice(&c.unwrap());
    }
    assert_eq!(out, vec![9, 0, 8, 0]);
    mock.assert();
}

#[tokio::test]
async fn kokoro_rejects_unsupported_format() {
    let mut cfg = KokoroConfig::default();
    cfg.output_format = "ogg".into();
    let backend = KokoroTtsBackend::new(cfg).unwrap();
    let err = backend
        .synthesize_stream(pcm_request(1.0), CancellationToken::new())
        .await
        .err()
        .unwrap();
    assert!(err.to_string().contains("supported output format"));
}

#[tokio::test]
async fn empty_text_rejected() {
    let mut cfg = KokoroConfig::default();
    cfg.base_url = "http://127.0.0.1:1".into();
    let backend = KokoroTtsBackend::new(cfg).unwrap();
    let err = backend
        .synthesize_stream(
            SynthesisRequest::new("   ", "v", "m", 1.0),
            CancellationToken::new(),
        )
        .await
        .err()
        .unwrap();
    assert!(matches!(err, shuvoice_tts::TtsError::EmptyText));
}
