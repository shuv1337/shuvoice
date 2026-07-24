//! In-process mock ASR worker speaking `shuvoice-worker-proto`.

use tokio::io::duplex;

use shuvoice_worker_proto::{
    AsrCapabilities as ProtoAsrCaps, ControlMessage, Frame, FramedConnection, WorkerManifest,
    accept_handshake,
};

use crate::error::AsrResult;

/// Spawn a duplex mock worker. Returns (reader, writer) for the client side.
pub async fn spawn_mock_worker(
    runtime: &str,
) -> AsrResult<(
    tokio::io::ReadHalf<tokio::io::DuplexStream>,
    tokio::io::WriteHalf<tokio::io::DuplexStream>,
)> {
    let (client, server) = duplex(256 * 1024);
    let runtime = runtime.to_owned();
    tokio::spawn(async move {
        if let Err(e) = run_mock(server, runtime).await {
            tracing::debug!("mock worker ended: {e}");
        }
    });
    let (r, w) = tokio::io::split(client);
    Ok((r, w))
}

async fn run_mock(stream: tokio::io::DuplexStream, runtime: String) -> AsrResult<()> {
    let (r, w) = tokio::io::split(stream);
    let mut conn = FramedConnection::new(r, w);

    let mut asr_caps = ProtoAsrCaps {
        wants_raw_audio: true,
        supports_streaming: true,
        supports_offline_utterance: true,
        supports_cancel: true,
        supports_model_download: false,
        native_sample_rate_hz: Some(16_000),
        native_chunk_samples: Some(if runtime == "moonshine" { 1600 } else { 1280 }),
        ..Default::default()
    };
    if runtime == "moonshine" {
        asr_caps.supports_streaming = true;
    }

    let mut manifest = WorkerManifest::asr(runtime.clone(), asr_caps);
    manifest.runtime_version = Some("mock-0".into());

    accept_handshake(&mut conn, manifest)
        .await
        .map_err(|e| crate::error::AsrError::protocol(e.to_string()))?;

    let mut loaded = false;
    let mut last_text = String::new();

    loop {
        let frame = match conn.read_frame().await {
            Ok(f) => f,
            Err(e) if e.is_clean_eof() => break,
            Err(e) => {
                tracing::debug!("mock read error: {e}");
                break;
            }
        };

        match frame.kind {
            shuvoice_worker_proto::FrameKind::Json => {
                let msg = ControlMessage::from_frame(&frame)
                    .map_err(|e| crate::error::AsrError::protocol(e.to_string()))?;
                match msg {
                    ControlMessage::Load(req) => {
                        loaded = true;
                        last_text.clear();
                        let chunk = if runtime == "moonshine" { 1600 } else { 1280 };
                        let result = serde_json::json!({
                            "wants_raw_audio": true,
                            "native_chunk_samples": chunk,
                            "native_sample_rate_hz": 16_000,
                        });
                        conn.write_message(&ControlMessage::Ack(shuvoice_worker_proto::Ack {
                            request_id: req.request_id,
                            result: Some(result),
                            extra: Default::default(),
                        }))
                        .await
                        .map_err(|e| crate::error::AsrError::protocol(e.to_string()))?;
                    }
                    ControlMessage::Reset(req) => {
                        last_text.clear();
                        conn.write_message(&ControlMessage::Ack(shuvoice_worker_proto::Ack {
                            request_id: req.request_id,
                            result: None,
                            extra: Default::default(),
                        }))
                        .await
                        .map_err(|e| crate::error::AsrError::protocol(e.to_string()))?;
                    }
                    ControlMessage::ProcessChunk(meta) => {
                        // Expect following PCM frame.
                        let pcm_frame = conn
                            .read_frame()
                            .await
                            .map_err(|e| crate::error::AsrError::protocol(e.to_string()))?;
                        let (_rid, body) = pcm_frame
                            .split_binary_payload()
                            .map_err(|e| crate::error::AsrError::protocol(e.to_string()))?;
                        if !loaded {
                            conn.write_message(&ControlMessage::Error(
                                shuvoice_worker_proto::ErrorResponse {
                                    request_id: Some(meta.request_id),
                                    code: "internal".into(),
                                    message: "not loaded".into(),
                                    extra: Default::default(),
                                },
                            ))
                            .await
                            .ok();
                            continue;
                        }
                        let has_energy = body
                            .chunks_exact(4)
                            .any(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) != 0.0);
                        if has_energy {
                            if last_text.is_empty() {
                                last_text = "hello".into();
                            } else if !last_text.contains("world") {
                                last_text = format!("{last_text} world");
                            }
                        }
                        conn.write_message(&ControlMessage::PartialTranscript(
                            shuvoice_worker_proto::TranscriptEvent {
                                request_id: meta.request_id,
                                text: last_text.clone(),
                                extra: Default::default(),
                            },
                        ))
                        .await
                        .map_err(|e| crate::error::AsrError::protocol(e.to_string()))?;
                    }
                    ControlMessage::ProcessUtterance(meta) => {
                        let pcm_frame = conn
                            .read_frame()
                            .await
                            .map_err(|e| crate::error::AsrError::protocol(e.to_string()))?;
                        let (_rid, body) = pcm_frame
                            .split_binary_payload()
                            .map_err(|e| crate::error::AsrError::protocol(e.to_string()))?;
                        let text = if body.is_empty() {
                            String::new()
                        } else {
                            "utterance".into()
                        };
                        conn.write_message(&ControlMessage::FinalTranscript(
                            shuvoice_worker_proto::TranscriptEvent {
                                request_id: meta.request_id,
                                text,
                                extra: Default::default(),
                            },
                        ))
                        .await
                        .map_err(|e| crate::error::AsrError::protocol(e.to_string()))?;
                    }
                    ControlMessage::Finish(req) => {
                        conn.write_message(&ControlMessage::FinalTranscript(
                            shuvoice_worker_proto::TranscriptEvent {
                                request_id: req.request_id,
                                text: last_text.clone(),
                                extra: Default::default(),
                            },
                        ))
                        .await
                        .map_err(|e| crate::error::AsrError::protocol(e.to_string()))?;
                    }
                    ControlMessage::Close(req) => {
                        if let Some(id) = req.request_id {
                            conn.write_message(&ControlMessage::Ack(shuvoice_worker_proto::Ack {
                                request_id: id,
                                result: None,
                                extra: Default::default(),
                            }))
                            .await
                            .ok();
                        }
                        break;
                    }
                    ControlMessage::Cancel(req) => {
                        conn.write_message(&ControlMessage::Ack(shuvoice_worker_proto::Ack {
                            request_id: req.request_id,
                            result: None,
                            extra: Default::default(),
                        }))
                        .await
                        .ok();
                    }
                    _ => {}
                }
            }
            _ => {
                // Unexpected bare binary without meta — ignore.
                let _ = Frame::json_bytes(Vec::new());
            }
        }
    }
    Ok(())
}
