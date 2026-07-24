//! Comprehensive protocol unit, round-trip, malformation, and property tests.

use std::io::Cursor;

use proptest::prelude::*;
use shuvoice_worker_proto::{
    Ack, AsrCapabilities, AudioStreamEvent, BackendKind, CloseRequest, ControlMessage,
    ErrorResponse, FinishRequest, Frame, FrameKind, FramedConnection, FramedReader, Hello, HelloOk,
    IdRequest, LoadRequest, MAX_FRAME_LEN, PROTOCOL_VERSION, PcmEncoding, ProcessAudioRequest,
    ProtocolError, SynthesizeRequest, TranscriptEvent, TtsCapabilities, VoiceInfo, VoicesResponse,
    WorkerClient, WorkerManifest, accept_handshake,
};
use tokio::io::{AsyncWriteExt, duplex};
use uuid::Uuid;

fn asr_manifest() -> WorkerManifest {
    WorkerManifest::asr(
        "sherpa",
        AsrCapabilities {
            supports_model_download: true,
            wants_raw_audio: false,
            supports_streaming: true,
            supports_offline_utterance: true,
            supports_cancel: true,
            native_sample_rate_hz: Some(16_000),
            native_chunk_samples: Some(1600),
            extra: Default::default(),
        },
    )
}

fn tts_manifest() -> WorkerManifest {
    WorkerManifest::tts(
        "kokoro",
        TtsCapabilities {
            requires_api_key: false,
            supports_native_speed: true,
            supports_streaming_audio: true,
            supports_list_voices: true,
            supports_cancel: true,
            default_sample_rate_hz: Some(24_000),
            max_chars: Some(5000),
            extra: Default::default(),
        },
    )
}

// ── Frame codec ────────────────────────────────────────────────────────

#[test]
fn json_frame_roundtrip_bytes() {
    let msg = ControlMessage::Hello(Hello::new("test-client"));
    let frame = msg.to_frame().unwrap();
    let encoded = frame.encode().unwrap();
    let (decoded, n) = Frame::decode_from(&encoded).unwrap();
    assert_eq!(n, encoded.len());
    assert_eq!(decoded.kind, FrameKind::Json);
    let parsed = ControlMessage::from_frame(&decoded).unwrap();
    assert_eq!(parsed, msg);
}

#[test]
fn pcm_f32_frame_roundtrip() {
    let id = Uuid::nil();
    let samples = vec![0.0f32, 0.5, -0.25, 1.0];
    let frame = Frame::pcm_f32le(id, &samples).unwrap();
    let encoded = frame.encode().unwrap();
    let (decoded, _) = Frame::decode_from(&encoded).unwrap();
    let (rid, body) = decoded.split_binary_payload().unwrap();
    assert_eq!(rid, id);
    let out = Frame::decode_f32le_samples(&body).unwrap();
    assert_eq!(out, samples);
}

#[test]
fn pcm_i16_frame_roundtrip() {
    let id = Uuid::from_u128(0xdead_beef);
    let samples = vec![0i16, 1, -1, 32767, -32768];
    let frame = Frame::pcm_i16le(id, &samples).unwrap();
    let (decoded, _) = Frame::decode_from(&frame.encode().unwrap()).unwrap();
    let (rid, body) = decoded.split_binary_payload().unwrap();
    assert_eq!(rid, id);
    assert_eq!(Frame::decode_i16le_samples(&body).unwrap(), samples);
}

#[test]
fn read_write_stdio_style_cursor() {
    let frame = ControlMessage::Reset(IdRequest::new(Uuid::nil()))
        .to_frame()
        .unwrap();
    let mut buf = Vec::new();
    frame.write_to(&mut buf).unwrap();
    let decoded = Frame::read_from(Cursor::new(&buf)).unwrap();
    assert_eq!(decoded, frame);
}

#[test]
fn rejects_zero_length_frame() {
    let buf = 0u32.to_be_bytes();
    let err = Frame::decode_from(&buf).unwrap_err();
    assert!(matches!(err, ProtocolError::FrameTooSmall { .. }));
}

#[test]
fn rejects_oversize_frame_length() {
    let mut buf = (MAX_FRAME_LEN + 1).to_be_bytes().to_vec();
    buf.push(1);
    let err = Frame::decode_from(&buf).unwrap_err();
    assert!(matches!(
        err,
        ProtocolError::FrameTooLarge {
            max: MAX_FRAME_LEN,
            ..
        }
    ));
}

#[test]
fn rejects_oversize_on_read_from_before_body_alloc() {
    let data = (MAX_FRAME_LEN + 1).to_be_bytes().to_vec();
    let err = Frame::read_from(Cursor::new(data)).unwrap_err();
    assert!(matches!(err, ProtocolError::FrameTooLarge { .. }));
}

#[test]
fn rejects_unknown_frame_kind() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&1u32.to_be_bytes());
    buf.push(0xFF);
    let err = Frame::decode_from(&buf).unwrap_err();
    assert!(matches!(err, ProtocolError::UnsupportedFrameKind(0xFF)));
}

#[test]
fn rejects_truncated_after_length() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&10u32.to_be_bytes());
    buf.push(1); // kind
    let err = Frame::decode_from(&buf).unwrap_err();
    assert!(matches!(err, ProtocolError::TruncatedFrame { .. }));
}

#[test]
fn rejects_truncated_length_prefix() {
    let err = Frame::decode_from(&[0x00, 0x01]).unwrap_err();
    assert!(matches!(err, ProtocolError::TruncatedFrame { .. }));
}

#[test]
fn rejects_binary_without_request_id() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&2u32.to_be_bytes());
    buf.push(FrameKind::PcmF32Le as u8);
    buf.push(0);
    let err = Frame::decode_from(&buf).unwrap_err();
    assert!(matches!(err, ProtocolError::InvalidBinaryPayload(_)));
}

#[test]
fn eof_on_empty_stream() {
    let err = Frame::read_from(Cursor::new(Vec::<u8>::new())).unwrap_err();
    assert!(matches!(err, ProtocolError::UnexpectedEof { .. }));
    assert!(err.is_clean_eof());
}

#[test]
fn truncated_body_on_read_from() {
    let mut data = Vec::new();
    data.extend_from_slice(&5u32.to_be_bytes());
    data.push(1);
    data.push(b'{');
    let err = Frame::read_from(Cursor::new(data)).unwrap_err();
    assert!(matches!(err, ProtocolError::TruncatedFrame { .. }));
}

// ── JSON messages / forward compatibility ──────────────────────────────

#[test]
fn control_message_json_roundtrips() {
    let id = Uuid::nil();
    let cases = vec![
        ControlMessage::Hello(Hello::new("cli")),
        ControlMessage::HelloOk(HelloOk {
            protocol_version: PROTOCOL_VERSION,
            manifest: asr_manifest(),
            extra: Default::default(),
        }),
        ControlMessage::Load(LoadRequest {
            request_id: id,
            config: serde_json::json!({"model": "x"}),
            extra: Default::default(),
        }),
        ControlMessage::Reset(IdRequest::new(id)),
        ControlMessage::ProcessChunk(ProcessAudioRequest {
            request_id: id,
            sample_rate_hz: 16000,
            channels: 1,
            encoding: PcmEncoding::F32Le,
            end: true,
            extra: Default::default(),
        }),
        ControlMessage::ProcessUtterance(ProcessAudioRequest {
            request_id: id,
            sample_rate_hz: 16000,
            channels: 1,
            encoding: PcmEncoding::F32Le,
            end: true,
            extra: Default::default(),
        }),
        ControlMessage::Finish(FinishRequest {
            request_id: id,
            timeout_ms: Some(1000),
            extra: Default::default(),
        }),
        ControlMessage::Cancel(IdRequest::new(id)),
        ControlMessage::Close(CloseRequest::default()),
        ControlMessage::Synthesize(SynthesizeRequest {
            request_id: id,
            text: "hi".into(),
            voice_id: Some("af_heart".into()),
            speed: Some(1.25),
            output_encoding: Some(PcmEncoding::F32Le),
            extra: Default::default(),
        }),
        ControlMessage::ListVoices(IdRequest::new(id)),
        ControlMessage::Ack(Ack {
            request_id: id,
            result: None,
            extra: Default::default(),
        }),
        ControlMessage::Error(ErrorResponse {
            request_id: Some(id),
            code: "boom".into(),
            message: "nope".into(),
            extra: Default::default(),
        }),
        ControlMessage::PartialTranscript(TranscriptEvent {
            request_id: id,
            text: "partial".into(),
            extra: Default::default(),
        }),
        ControlMessage::FinalTranscript(TranscriptEvent {
            request_id: id,
            text: "final".into(),
            extra: Default::default(),
        }),
        ControlMessage::Voices(VoicesResponse {
            request_id: id,
            voices: vec![VoiceInfo {
                id: "v1".into(),
                name: Some("Voice".into()),
                locale: Some("en".into()),
                extra: Default::default(),
            }],
            extra: Default::default(),
        }),
        ControlMessage::AudioStart(AudioStreamEvent {
            request_id: id,
            sample_rate_hz: Some(24000),
            channels: Some(1),
            encoding: Some(PcmEncoding::F32Le),
            extra: Default::default(),
        }),
        ControlMessage::AudioEnd(AudioStreamEvent {
            request_id: id,
            sample_rate_hz: None,
            channels: None,
            encoding: None,
            extra: Default::default(),
        }),
    ];

    for msg in cases {
        let bytes = serde_json::to_vec(&msg).unwrap();
        let back: ControlMessage = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, msg, "roundtrip failed for {msg:?}");
    }
}

#[test]
fn unknown_message_type_is_forward_compatible() {
    let raw = br#"{"type":"future_extension","foo":1}"#;
    let msg = ControlMessage::from_slice(raw).unwrap();
    assert_eq!(msg, ControlMessage::Unknown);
}

#[test]
fn unknown_manifest_fields_are_captured_in_extra() {
    let raw = r#"{
        "backend_id": "sherpa",
        "kind": "asr",
        "asr": {"supports_streaming": true, "new_cap": true},
        "brand_new_field": 42
    }"#;
    let manifest: WorkerManifest = serde_json::from_str(raw).unwrap();
    assert_eq!(manifest.backend_id, "sherpa");
    assert_eq!(manifest.kind, BackendKind::Asr);
    assert!(manifest.extra.contains_key("brand_new_field"));
    assert_eq!(
        manifest.asr.as_ref().unwrap().extra.get("new_cap"),
        Some(&serde_json::json!(true))
    );
}

#[test]
fn capability_flags_default_false_when_omitted() {
    let raw = r#"{
        "backend_id": "x",
        "kind": "asr",
        "asr": {}
    }"#;
    let manifest: WorkerManifest = serde_json::from_str(raw).unwrap();
    let asr = manifest.asr.unwrap();
    assert!(!asr.supports_streaming);
    assert!(!asr.supports_cancel);
    assert!(!asr.wants_raw_audio);
    assert!(!asr.supports_offline_utterance);

    let raw_tts = r#"{
        "backend_id": "y",
        "kind": "tts",
        "tts": {}
    }"#;
    let m2: WorkerManifest = serde_json::from_str(raw_tts).unwrap();
    let tts = m2.tts.unwrap();
    assert!(!tts.supports_streaming_audio);
    assert!(!tts.supports_list_voices);
    assert!(!tts.supports_native_speed);
}

#[test]
fn json_payload_over_max_rejected() {
    use shuvoice_worker_proto::MAX_JSON_PAYLOAD_LEN;
    let huge = vec![b'x'; (MAX_JSON_PAYLOAD_LEN as usize) + 8];
    let err = Frame::json_bytes(huge).unwrap_err();
    assert!(matches!(err, ProtocolError::JsonTooLarge { .. }), "{err:?}");
}

// ── Async codec + handshake + client ───────────────────────────────────

#[tokio::test]
async fn handshake_and_load_roundtrip() {
    let (client_side, server_side) = duplex(64 * 1024);
    let (cr, cw) = tokio::io::split(client_side);
    let (sr, sw) = tokio::io::split(server_side);

    let server = tokio::spawn(async move {
        let mut conn = FramedConnection::new(sr, sw);
        accept_handshake(&mut conn, asr_manifest()).await.unwrap();

        match conn.read_message().await.unwrap() {
            ControlMessage::Load(req) => {
                conn.write_message(&ControlMessage::Ack(Ack {
                    request_id: req.request_id,
                    result: None,
                    extra: Default::default(),
                }))
                .await
                .unwrap();
            }
            other => panic!("unexpected {other:?}"),
        }

        match conn.read_message().await.unwrap() {
            ControlMessage::ProcessUtterance(meta) => {
                let frame = conn.read_frame().await.unwrap();
                assert_eq!(frame.kind, FrameKind::PcmF32Le);
                let (rid, body) = frame.split_binary_payload().unwrap();
                assert_eq!(rid, meta.request_id);
                assert!(!body.is_empty());
                conn.write_message(&ControlMessage::FinalTranscript(TranscriptEvent {
                    request_id: meta.request_id,
                    text: "hello world".into(),
                    extra: Default::default(),
                }))
                .await
                .unwrap();
            }
            other => panic!("unexpected {other:?}"),
        }

        match conn.read_message().await.unwrap() {
            ControlMessage::Close(c) => {
                if let Some(id) = c.request_id {
                    conn.write_message(&ControlMessage::Ack(Ack {
                        request_id: id,
                        result: None,
                        extra: Default::default(),
                    }))
                    .await
                    .unwrap();
                }
            }
            other => panic!("unexpected {other:?}"),
        }
    });

    let mut client = WorkerClient::new(cr, cw);
    let session = client.handshake("shuvoice-test").await.unwrap().clone();
    assert_eq!(session.protocol_version, PROTOCOL_VERSION);
    assert_eq!(session.manifest.backend_id, "sherpa");

    client
        .load(serde_json::json!({"model": "parakeet"}))
        .await
        .unwrap();
    let samples = vec![0.1f32; 320];
    let tr = client.process_utterance(&samples, 16_000).await.unwrap();
    assert_eq!(tr.text, "hello world");
    client.close().await.unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn handshake_version_mismatch_is_rejected() {
    let (client_side, server_side) = duplex(4096);
    let (cr, cw) = tokio::io::split(client_side);
    let (sr, sw) = tokio::io::split(server_side);

    let server = tokio::spawn(async move {
        let mut conn = FramedConnection::new(sr, sw);
        let hello = conn.read_message().await.unwrap();
        assert!(matches!(hello, ControlMessage::Hello(_)));
        conn.write_message(&ControlMessage::HelloOk(HelloOk {
            protocol_version: 99,
            manifest: asr_manifest(),
            extra: Default::default(),
        }))
        .await
        .unwrap();
    });

    let mut client = WorkerClient::new(cr, cw);
    let err = client.handshake("cli").await.unwrap_err();
    assert!(matches!(
        err,
        ProtocolError::UnsupportedVersion {
            remote: 99,
            local: PROTOCOL_VERSION
        }
    ));
    server.await.unwrap();
}

#[tokio::test]
async fn accept_handshake_rejects_bad_client_version() {
    let (a, b) = duplex(4096);
    let (ar, aw) = tokio::io::split(a);
    let (br, bw) = tokio::io::split(b);

    let server = tokio::spawn(async move {
        let mut conn = FramedConnection::new(br, bw);
        let err = accept_handshake(&mut conn, asr_manifest())
            .await
            .unwrap_err();
        assert!(matches!(err, ProtocolError::UnsupportedVersion { .. }));
    });

    let mut client_conn = FramedConnection::new(ar, aw);
    client_conn
        .write_message(&ControlMessage::Hello(Hello {
            protocol_version: 0,
            client_name: Some("old".into()),
            client_version: None,
            extra: Default::default(),
        }))
        .await
        .unwrap();
    let reply = client_conn.read_message().await.unwrap();
    assert!(matches!(reply, ControlMessage::HelloErr(_)));
    server.await.unwrap();
}

#[tokio::test]
async fn tts_synthesize_and_list_voices() {
    let (client_side, server_side) = duplex(64 * 1024);
    let (cr, cw) = tokio::io::split(client_side);
    let (sr, sw) = tokio::io::split(server_side);

    let server = tokio::spawn(async move {
        let mut conn = FramedConnection::new(sr, sw);
        accept_handshake(&mut conn, tts_manifest()).await.unwrap();

        match conn.read_message().await.unwrap() {
            ControlMessage::ListVoices(req) => {
                conn.write_message(&ControlMessage::Voices(VoicesResponse {
                    request_id: req.request_id,
                    voices: vec![VoiceInfo {
                        id: "af_heart".into(),
                        name: Some("Heart".into()),
                        locale: Some("en-US".into()),
                        extra: Default::default(),
                    }],
                    extra: Default::default(),
                }))
                .await
                .unwrap();
            }
            other => panic!("{other:?}"),
        }

        match conn.read_message().await.unwrap() {
            ControlMessage::Synthesize(req) => {
                conn.write_message(&ControlMessage::AudioStart(AudioStreamEvent {
                    request_id: req.request_id,
                    sample_rate_hz: Some(24_000),
                    channels: Some(1),
                    encoding: Some(PcmEncoding::F32Le),
                    extra: Default::default(),
                }))
                .await
                .unwrap();
                let pcm = Frame::pcm_f32le(req.request_id, &[0.0, 0.25, 0.5]).unwrap();
                conn.write_frame(&pcm).await.unwrap();
                conn.write_message(&ControlMessage::AudioEnd(AudioStreamEvent {
                    request_id: req.request_id,
                    sample_rate_hz: Some(24_000),
                    channels: Some(1),
                    encoding: Some(PcmEncoding::F32Le),
                    extra: Default::default(),
                }))
                .await
                .unwrap();
            }
            other => panic!("{other:?}"),
        }
    });

    let mut client = WorkerClient::new(cr, cw);
    client.handshake("tts-client").await.unwrap();
    let voices = client.list_voices().await.unwrap();
    assert_eq!(voices.len(), 1);
    assert_eq!(voices[0].id, "af_heart");

    let result = client
        .synthesize("hello", Some("af_heart".into()), Some(1.0))
        .await
        .unwrap();
    assert_eq!(result.sample_rate_hz, Some(24_000));
    assert_eq!(result.pcm.len(), 12);
    server.await.unwrap();
}

#[tokio::test]
async fn process_chunk_and_finish_flow() {
    let (client_side, server_side) = duplex(64 * 1024);
    let (cr, cw) = tokio::io::split(client_side);
    let (sr, sw) = tokio::io::split(server_side);

    let server = tokio::spawn(async move {
        let mut conn = FramedConnection::new(sr, sw);
        accept_handshake(&mut conn, asr_manifest()).await.unwrap();

        match conn.read_message().await.unwrap() {
            ControlMessage::ProcessChunk(meta) => {
                let _pcm = conn.read_frame().await.unwrap();
                conn.write_message(&ControlMessage::PartialTranscript(TranscriptEvent {
                    request_id: meta.request_id,
                    text: "hel".into(),
                    extra: Default::default(),
                }))
                .await
                .unwrap();
            }
            other => panic!("{other:?}"),
        }

        match conn.read_message().await.unwrap() {
            ControlMessage::Finish(req) => {
                conn.write_message(&ControlMessage::FinalTranscript(TranscriptEvent {
                    request_id: req.request_id,
                    text: "hello".into(),
                    extra: Default::default(),
                }))
                .await
                .unwrap();
            }
            other => panic!("{other:?}"),
        }

        match conn.read_message().await.unwrap() {
            ControlMessage::Cancel(req) => {
                conn.write_message(&ControlMessage::Ack(Ack {
                    request_id: req.request_id,
                    result: None,
                    extra: Default::default(),
                }))
                .await
                .unwrap();
            }
            other => panic!("{other:?}"),
        }

        match conn.read_message().await.unwrap() {
            ControlMessage::Reset(req) => {
                conn.write_message(&ControlMessage::Ack(Ack {
                    request_id: req.request_id,
                    result: None,
                    extra: Default::default(),
                }))
                .await
                .unwrap();
            }
            other => panic!("{other:?}"),
        }
    });

    let mut client = WorkerClient::new(cr, cw);
    client.handshake("c").await.unwrap();
    let partial = client.process_chunk(&[0.01; 80], 16_000).await.unwrap();
    assert_eq!(partial.text, "hel");
    let final_tr = client.finish(Some(500)).await.unwrap();
    assert_eq!(final_tr.text, "hello");
    let cancel_id = Uuid::nil();
    client.cancel(cancel_id).await.unwrap();
    client.reset().await.unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn worker_error_surfaces() {
    let (client_side, server_side) = duplex(4096);
    let (cr, cw) = tokio::io::split(client_side);
    let (sr, sw) = tokio::io::split(server_side);

    let server = tokio::spawn(async move {
        let mut conn = FramedConnection::new(sr, sw);
        accept_handshake(&mut conn, asr_manifest()).await.unwrap();
        match conn.read_message().await.unwrap() {
            ControlMessage::Load(req) => {
                conn.write_message(&ControlMessage::Error(ErrorResponse {
                    request_id: Some(req.request_id),
                    code: "model_missing".into(),
                    message: "not found".into(),
                    extra: Default::default(),
                }))
                .await
                .unwrap();
            }
            other => panic!("{other:?}"),
        }
    });

    let mut client = WorkerClient::new(cr, cw);
    client.handshake("c").await.unwrap();
    let err = client.load(serde_json::json!({})).await.unwrap_err();
    match err {
        ProtocolError::Worker { code, message, .. } => {
            assert_eq!(code, "model_missing");
            assert_eq!(message, "not found");
        }
        other => panic!("{other:?}"),
    }
    server.await.unwrap();
}

#[tokio::test]
async fn async_reader_eof_and_truncation() {
    // clean EOF
    let (w, r) = duplex(64);
    drop(w);
    let mut reader = FramedReader::new(r);
    let err = reader.read_frame().await.unwrap_err();
    assert!(err.is_clean_eof());

    // truncated body
    let (mut w, r) = duplex(64);
    let mut reader = FramedReader::new(r);
    w.write_all(&10u32.to_be_bytes()).await.unwrap();
    w.write_all(&[1, b'{']).await.unwrap();
    drop(w);
    let err = reader.read_frame().await.unwrap_err();
    assert!(matches!(err, ProtocolError::TruncatedFrame { .. }));

    // oversize rejected
    let (mut w, r) = duplex(64);
    let mut reader = FramedReader::new(r);
    w.write_all(&(MAX_FRAME_LEN + 1).to_be_bytes())
        .await
        .unwrap();
    let err = reader.read_frame().await.unwrap_err();
    assert!(matches!(err, ProtocolError::FrameTooLarge { .. }));
}

// ── Property tests ─────────────────────────────────────────────────────

proptest! {
    #[test]
    fn prop_json_hello_roundtrip(name in ".{0,64}") {
        let msg = ControlMessage::Hello(Hello {
            protocol_version: PROTOCOL_VERSION,
            client_name: Some(name),
            client_version: None,
            extra: Default::default(),
        });
        let frame = msg.to_frame().unwrap();
        let encoded = frame.encode().unwrap();
        let (decoded, n) = Frame::decode_from(&encoded).unwrap();
        prop_assert_eq!(n, encoded.len());
        let back = ControlMessage::from_frame(&decoded).unwrap();
        prop_assert_eq!(back, msg);
    }

    #[test]
    fn prop_pcm_f32_roundtrip(samples in proptest::collection::vec(-1.0f32..1.0f32, 0..256)) {
        let id = Uuid::from_u128(0x1234);
        let samples: Vec<f32> = samples.into_iter().filter(|s| s.is_finite()).collect();
        let frame = Frame::pcm_f32le(id, &samples).unwrap();
        let encoded = frame.encode().unwrap();
        let (decoded, _) = Frame::decode_from(&encoded).unwrap();
        let (rid, body) = decoded.split_binary_payload().unwrap();
        prop_assert_eq!(rid, id);
        let out = Frame::decode_f32le_samples(&body).unwrap();
        prop_assert_eq!(out.len(), samples.len());
        for (a, b) in out.iter().zip(samples.iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn prop_length_prefix_never_accepts_oversize(len in (MAX_FRAME_LEN + 1)..=u32::MAX) {
        let mut buf = len.to_be_bytes().to_vec();
        buf.push(1);
        let err = Frame::decode_from(&buf).unwrap_err();
        let ok = matches!(err, ProtocolError::FrameTooLarge { .. });
        prop_assert!(ok, "expected FrameTooLarge, got {:?}", err);
    }

    #[test]
    fn prop_encode_decode_identity_for_byte_payloads(
        body in proptest::collection::vec(any::<u8>(), 0..512)
    ) {
        let id = Uuid::from_u128(0xabcdef);
        let frame = Frame::binary(FrameKind::Bytes, id, &body).unwrap();
        let encoded = frame.encode().unwrap();
        prop_assert_eq!(encoded.len(), frame.encoded_len());
        let (decoded, n) = Frame::decode_from(&encoded).unwrap();
        prop_assert_eq!(n, encoded.len());
        prop_assert_eq!(decoded.kind, FrameKind::Bytes);
        let (rid, got) = decoded.split_binary_payload().unwrap();
        prop_assert_eq!(rid, id);
        prop_assert_eq!(&got[..], &body[..]);
    }

    #[test]
    fn prop_request_id_in_messages(raw in any::<u128>()) {
        let id = Uuid::from_u128(raw);
        let msg = ControlMessage::Cancel(IdRequest::new(id));
        prop_assert_eq!(msg.request_id(), Some(id));
        let bytes = serde_json::to_vec(&msg).unwrap();
        let back = ControlMessage::from_slice(&bytes).unwrap();
        prop_assert_eq!(back.request_id(), Some(id));
    }
}

#[test]
fn binary_kind_rejects_json_constructor_path() {
    let err = Frame::binary(FrameKind::Json, Uuid::nil(), b"x").unwrap_err();
    assert!(matches!(err, ProtocolError::InvalidBinaryPayload(_)));
}

#[test]
fn f32_body_rejects_bad_alignment() {
    let err = Frame::decode_f32le_samples(&[0, 1, 2]).unwrap_err();
    assert!(matches!(err, ProtocolError::InvalidBinaryPayload(_)));
}

#[test]
fn i16_body_rejects_bad_alignment() {
    let err = Frame::decode_i16le_samples(&[0]).unwrap_err();
    assert!(matches!(err, ProtocolError::InvalidBinaryPayload(_)));
}

#[test]
fn multi_frame_buffer_decode_consumes_prefix_only() {
    let f1 = ControlMessage::Hello(Hello::new("a")).to_frame().unwrap();
    let f2 = ControlMessage::Hello(Hello::new("b")).to_frame().unwrap();
    let mut buf = f1.encode().unwrap().to_vec();
    buf.extend_from_slice(&f2.encode().unwrap());
    let (d1, n1) = Frame::decode_from(&buf).unwrap();
    let (d2, n2) = Frame::decode_from(&buf[n1..]).unwrap();
    assert_eq!(n1 + n2, buf.len());
    assert_eq!(
        ControlMessage::from_frame(&d1).unwrap(),
        ControlMessage::Hello(Hello::new("a"))
    );
    assert_eq!(
        ControlMessage::from_frame(&d2).unwrap(),
        ControlMessage::Hello(Hello::new("b"))
    );
}

#[tokio::test]
async fn synthesize_requires_audio_end_and_nonempty_pcm() {
    let (client_side, server_side) = duplex(64 * 1024);
    let (cr, cw) = tokio::io::split(client_side);
    let (sr, sw) = tokio::io::split(server_side);

    let server = tokio::spawn(async move {
        let mut conn = FramedConnection::new(sr, sw);
        accept_handshake(&mut conn, tts_manifest()).await.unwrap();
        match conn.read_message().await.unwrap() {
            ControlMessage::Synthesize(req) => {
                // AudioStart + Ack only — must NOT complete successfully.
                conn.write_message(&ControlMessage::AudioStart(AudioStreamEvent {
                    request_id: req.request_id,
                    sample_rate_hz: Some(24_000),
                    channels: Some(1),
                    encoding: Some(PcmEncoding::F32Le),
                    extra: Default::default(),
                }))
                .await
                .unwrap();
                conn.write_message(&ControlMessage::Ack(Ack {
                    request_id: req.request_id,
                    result: None,
                    extra: Default::default(),
                }))
                .await
                .unwrap();
                // Eventually end with empty stream
                conn.write_message(&ControlMessage::AudioEnd(AudioStreamEvent {
                    request_id: req.request_id,
                    sample_rate_hz: Some(24_000),
                    channels: Some(1),
                    encoding: Some(PcmEncoding::F32Le),
                    extra: Default::default(),
                }))
                .await
                .unwrap();
            }
            other => panic!("{other:?}"),
        }
    });

    let mut client = WorkerClient::with_options(
        cr,
        cw,
        shuvoice_worker_proto::ClientOptions {
            rpc_timeout: std::time::Duration::from_secs(5),
            load_timeout: std::time::Duration::from_secs(5),
            max_ignored_messages: 8,
            ..Default::default()
        },
    );
    client.handshake("tts-empty").await.unwrap();
    let err = client
        .synthesize("hi", None, None)
        .await
        .expect_err("empty pcm must fail");
    assert!(matches!(err, ProtocolError::EmptyAudio), "got {err:?}");
    server.await.unwrap();
}

#[tokio::test]
async fn rpc_timeout_on_hanging_worker() {
    let (client_side, server_side) = duplex(4096);
    let (cr, cw) = tokio::io::split(client_side);
    let (_sr, _sw) = tokio::io::split(server_side);
    // Server never replies after handshake would be needed — just hang on load.
    let mut client = WorkerClient::with_options(
        cr,
        cw,
        shuvoice_worker_proto::ClientOptions {
            rpc_timeout: std::time::Duration::from_millis(80),
            load_timeout: std::time::Duration::from_millis(80),
            max_ignored_messages: 4,
            ..Default::default()
        },
    );
    // load writes and waits for ack with rpc/load timeout.
    let err = client.load(serde_json::json!({})).await.unwrap_err();
    assert!(
        matches!(err, ProtocolError::RpcTimeout { .. }),
        "got {err:?}"
    );
}

#[tokio::test]
async fn hello_err_unsupported_version_is_typed() {
    let (client_side, server_side) = duplex(4096);
    let (cr, cw) = tokio::io::split(client_side);
    let (sr, sw) = tokio::io::split(server_side);

    let server = tokio::spawn(async move {
        let mut conn = FramedConnection::new(sr, sw);
        let _ = conn.read_message().await.unwrap();
        conn.write_message(&ControlMessage::HelloErr(shuvoice_worker_proto::HelloErr {
            message: "nope".into(),
            code: Some("unsupported_version".into()),
            protocol_version: Some(99),
            extra: Default::default(),
        }))
        .await
        .unwrap();
    });

    let mut client = WorkerClient::new(cr, cw);
    let err = client.handshake("cli").await.unwrap_err();
    assert!(
        matches!(
            err,
            ProtocolError::UnsupportedVersion {
                remote: 99,
                local: PROTOCOL_VERSION
            }
        ),
        "got {err:?}"
    );
    server.await.unwrap();
}

#[tokio::test]
async fn synthesize_encoding_mismatch_errors() {
    let (client_side, server_side) = duplex(64 * 1024);
    let (cr, cw) = tokio::io::split(client_side);
    let (sr, sw) = tokio::io::split(server_side);

    let server = tokio::spawn(async move {
        let mut conn = FramedConnection::new(sr, sw);
        accept_handshake(&mut conn, tts_manifest()).await.unwrap();
        match conn.read_message().await.unwrap() {
            ControlMessage::Synthesize(req) => {
                conn.write_message(&ControlMessage::AudioStart(AudioStreamEvent {
                    request_id: req.request_id,
                    sample_rate_hz: Some(24_000),
                    channels: Some(1),
                    encoding: Some(PcmEncoding::F32Le),
                    extra: Default::default(),
                }))
                .await
                .unwrap();
                // Send i16 body while declaring f32
                let pcm = Frame::pcm_i16le(req.request_id, &[1, 2, 3]).unwrap();
                conn.write_frame(&pcm).await.unwrap();
                conn.write_message(&ControlMessage::AudioEnd(AudioStreamEvent {
                    request_id: req.request_id,
                    sample_rate_hz: Some(24_000),
                    channels: Some(1),
                    encoding: Some(PcmEncoding::F32Le),
                    extra: Default::default(),
                }))
                .await
                .unwrap();
            }
            other => panic!("{other:?}"),
        }
    });

    let mut client = WorkerClient::new(cr, cw);
    client.handshake("enc").await.unwrap();
    let err = client.synthesize("x", None, None).await.unwrap_err();
    assert!(
        matches!(err, ProtocolError::EncodingMismatch(_)),
        "got {err:?}"
    );
    server.await.unwrap();
}

#[tokio::test]
async fn synthesize_rejects_unbounded_audio_stream() {
    // A worker streaming matching-request frames without audio_end must hit
    // the aggregate cap instead of growing supervisor memory until OOM.
    let (client_side, server_side) = duplex(64 * 1024);
    let (cr, cw) = tokio::io::split(client_side);
    let (sr, sw) = tokio::io::split(server_side);

    let server = tokio::spawn(async move {
        let mut conn = FramedConnection::new(sr, sw);
        accept_handshake(&mut conn, tts_manifest()).await.unwrap();
        match conn.read_message().await.unwrap() {
            ControlMessage::Synthesize(req) => {
                conn.write_message(&ControlMessage::AudioStart(AudioStreamEvent {
                    request_id: req.request_id,
                    sample_rate_hz: Some(24_000),
                    channels: Some(1),
                    encoding: Some(PcmEncoding::F32Le),
                    extra: Default::default(),
                }))
                .await
                .unwrap();
                // Never send audio_end; keep streaming until the client bails.
                let samples = vec![0.0f32; 256];
                loop {
                    let pcm = Frame::pcm_f32le(req.request_id, &samples).unwrap();
                    if conn.write_frame(&pcm).await.is_err() {
                        break;
                    }
                }
            }
            other => panic!("{other:?}"),
        }
    });

    let mut client = WorkerClient::with_options(
        cr,
        cw,
        shuvoice_worker_proto::ClientOptions {
            rpc_timeout: std::time::Duration::from_secs(5),
            max_synthesis_audio_bytes: 8 * 1024,
            ..Default::default()
        },
    );
    client.handshake("tts-flood").await.unwrap();
    let err = client.synthesize("x", None, None).await.unwrap_err();
    assert!(
        matches!(err, ProtocolError::AudioTooLarge { limit: 8192 }),
        "got {err:?}"
    );
    server.abort();
}
