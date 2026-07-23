//! High-level async client helpers for native hosts talking to workers.

use std::time::Duration;

use bytes::Bytes;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::time::timeout;
use uuid::Uuid;

use crate::codec::FramedConnection;
use crate::error::ProtocolError;
use crate::frame::{Frame, FrameKind};
use crate::limits::{
    DEFAULT_LOAD_TIMEOUT, DEFAULT_MAX_IGNORED_MESSAGES, DEFAULT_RPC_TIMEOUT, PROTOCOL_VERSION,
};
use crate::manifest::WorkerManifest;
use crate::messages::{
    Ack, CloseRequest, ControlMessage, FinishRequest, Hello, HelloOk, IdRequest, LoadRequest,
    PcmEncoding, ProcessAudioRequest, RequestId, SynthesizeRequest, TranscriptEvent, VoiceInfo,
    VoicesResponse,
};

/// Tunables for [`WorkerClient`] RPC waits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientOptions {
    /// Deadline for ordinary RPCs (chunk, finish, synthesize, cancel, …).
    pub rpc_timeout: Duration,
    /// Deadline for `load` (model download / heavy init).
    pub load_timeout: Duration,
    /// Max unrelated frames/messages skipped while awaiting a correlated reply.
    pub max_ignored_messages: u32,
}

impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            rpc_timeout: DEFAULT_RPC_TIMEOUT,
            load_timeout: DEFAULT_LOAD_TIMEOUT,
            max_ignored_messages: DEFAULT_MAX_IGNORED_MESSAGES,
        }
    }
}

/// Result of a successful protocol handshake.
#[derive(Debug, Clone, PartialEq)]
pub struct NegotiatedSession {
    pub protocol_version: u16,
    pub manifest: WorkerManifest,
}

/// Async client bound to a framed reader/writer pair (stdio, socket, pipe, …).
#[derive(Debug)]
pub struct WorkerClient<R, W> {
    conn: FramedConnection<R, W>,
    session: Option<NegotiatedSession>,
    options: ClientOptions,
}

impl<R, W> WorkerClient<R, W> {
    #[must_use]
    pub fn new(reader: R, writer: W) -> Self {
        Self::with_options(reader, writer, ClientOptions::default())
    }

    #[must_use]
    pub fn with_options(reader: R, writer: W, options: ClientOptions) -> Self {
        Self {
            conn: FramedConnection::new(reader, writer),
            session: None,
            options,
        }
    }

    #[must_use]
    pub fn options(&self) -> &ClientOptions {
        &self.options
    }

    #[must_use]
    pub fn options_mut(&mut self) -> &mut ClientOptions {
        &mut self.options
    }

    #[must_use]
    pub fn session(&self) -> Option<&NegotiatedSession> {
        self.session.as_ref()
    }

    #[must_use]
    pub fn connection(&self) -> &FramedConnection<R, W> {
        &self.conn
    }

    #[must_use]
    pub fn connection_mut(&mut self) -> &mut FramedConnection<R, W> {
        &mut self.conn
    }

    #[must_use]
    pub fn into_inner(self) -> (R, W, Option<NegotiatedSession>) {
        let (r, w) = self.conn.into_inner();
        (r, w, self.session)
    }
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> WorkerClient<R, W> {
    /// Perform the versioned handshake and store the worker manifest.
    pub async fn handshake(
        &mut self,
        client_name: impl Into<String>,
    ) -> Result<&NegotiatedSession, ProtocolError> {
        let name = client_name.into();
        let rpc_timeout = self.options.rpc_timeout;
        let fut = async {
            let hello = ControlMessage::Hello(Hello::new(name));
            self.conn.write_message(&hello).await?;
            let reply = self.conn.read_message().await?;
            match reply {
                ControlMessage::HelloOk(HelloOk {
                    protocol_version,
                    manifest,
                    ..
                }) => {
                    if protocol_version != PROTOCOL_VERSION {
                        return Err(ProtocolError::UnsupportedVersion {
                            remote: protocol_version,
                            local: PROTOCOL_VERSION,
                        });
                    }
                    self.session = Some(NegotiatedSession {
                        protocol_version,
                        manifest,
                    });
                    Ok(())
                }
                ControlMessage::HelloErr(err) => {
                    let code = err.code.as_deref().unwrap_or("");
                    if code == "unsupported_version" {
                        let remote = err.protocol_version.unwrap_or(0);
                        return Err(ProtocolError::UnsupportedVersion {
                            remote,
                            local: PROTOCOL_VERSION,
                        });
                    }
                    Err(ProtocolError::Handshake(
                        err.code
                            .map(|c| format!("{c}: {}", err.message))
                            .unwrap_or(err.message),
                    ))
                }
                other => Err(ProtocolError::UnexpectedMessage(message_type_name(&other))),
            }
        };
        with_timeout(rpc_timeout, fut).await?;
        Ok(self.session.as_ref().expect("set on success"))
    }

    /// ASR/TTS: load backend resources.
    pub async fn load(&mut self, config: serde_json::Value) -> Result<RequestId, ProtocolError> {
        let request_id = Uuid::now_v7();
        let timeout = self.options.load_timeout;
        let fut = async {
            self.conn
                .write_message(&ControlMessage::Load(LoadRequest {
                    request_id,
                    config,
                    extra: Default::default(),
                }))
                .await?;
            self.wait_ack(request_id).await?;
            Ok(request_id)
        };
        with_timeout(timeout, fut).await
    }

    /// ASR: reset streaming state.
    pub async fn reset(&mut self) -> Result<RequestId, ProtocolError> {
        let request_id = Uuid::now_v7();
        let timeout = self.options.rpc_timeout;
        let fut = async {
            self.conn
                .write_message(&ControlMessage::Reset(IdRequest::new(request_id)))
                .await?;
            self.wait_ack(request_id).await?;
            Ok(request_id)
        };
        with_timeout(timeout, fut).await
    }

    /// ASR: process one streaming chunk of f32 LE mono PCM; returns partial/final text if any.
    pub async fn process_chunk(
        &mut self,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<TranscriptEvent, ProtocolError> {
        let request_id = Uuid::now_v7();
        let timeout = self.options.rpc_timeout;
        let fut = async {
            self.conn
                .write_message(&ControlMessage::ProcessChunk(ProcessAudioRequest {
                    request_id,
                    sample_rate_hz,
                    channels: 1,
                    encoding: PcmEncoding::F32Le,
                    end: true,
                    extra: Default::default(),
                }))
                .await?;
            self.conn
                .write_frame(&Frame::pcm_f32le(request_id, samples)?)
                .await?;
            self.wait_transcript(request_id).await
        };
        with_timeout(timeout, fut).await
    }

    /// ASR: one-shot offline utterance decode.
    pub async fn process_utterance(
        &mut self,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<TranscriptEvent, ProtocolError> {
        let request_id = Uuid::now_v7();
        let timeout = self.options.rpc_timeout;
        let fut = async {
            self.conn
                .write_message(&ControlMessage::ProcessUtterance(ProcessAudioRequest {
                    request_id,
                    sample_rate_hz,
                    channels: 1,
                    encoding: PcmEncoding::F32Le,
                    end: true,
                    extra: Default::default(),
                }))
                .await?;
            self.conn
                .write_frame(&Frame::pcm_f32le(request_id, samples)?)
                .await?;
            self.wait_final_transcript(request_id).await
        };
        with_timeout(timeout, fut).await
    }

    /// ASR: finish the current streaming utterance.
    pub async fn finish(
        &mut self,
        timeout_ms: Option<u64>,
    ) -> Result<TranscriptEvent, ProtocolError> {
        let request_id = Uuid::now_v7();
        let rpc_timeout = timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(self.options.rpc_timeout)
            .max(Duration::from_millis(1));
        let fut = async {
            self.conn
                .write_message(&ControlMessage::Finish(FinishRequest {
                    request_id,
                    timeout_ms,
                    extra: Default::default(),
                }))
                .await?;
            self.wait_final_transcript(request_id).await
        };
        with_timeout(rpc_timeout, fut).await
    }

    /// Cancel an in-flight request (ASR or TTS).
    pub async fn cancel(&mut self, request_id: RequestId) -> Result<(), ProtocolError> {
        let timeout = self.options.rpc_timeout;
        let fut = async {
            self.conn
                .write_message(&ControlMessage::Cancel(IdRequest::new(request_id)))
                .await?;
            self.wait_ack(request_id).await?;
            Ok(())
        };
        with_timeout(timeout, fut).await
    }

    /// Ask the worker to shut down cleanly.
    pub async fn close(&mut self) -> Result<(), ProtocolError> {
        let request_id = Uuid::now_v7();
        let timeout = self.options.rpc_timeout;
        let fut = async {
            self.conn
                .write_message(&ControlMessage::Close(CloseRequest {
                    request_id: Some(request_id),
                    extra: Default::default(),
                }))
                .await?;
            // Close may return ack or EOF.
            match self.conn.read_message().await {
                Ok(ControlMessage::Ack(Ack {
                    request_id: rid, ..
                })) if rid == request_id => Ok(()),
                Ok(ControlMessage::Error(err)) => {
                    if err.request_id.is_none() || err.request_id == Some(request_id) {
                        Err(err.into_protocol_error())
                    } else {
                        Err(ProtocolError::UnexpectedMessage("error"))
                    }
                }
                Ok(other) => Err(ProtocolError::UnexpectedMessage(message_type_name(&other))),
                Err(e) if e.is_clean_eof() => Ok(()),
                Err(e) => Err(e),
            }
        };
        with_timeout(timeout, fut).await
    }

    /// TTS: list available voices.
    pub async fn list_voices(&mut self) -> Result<Vec<VoiceInfo>, ProtocolError> {
        let request_id = Uuid::now_v7();
        let max_ignored = self.options.max_ignored_messages;
        let timeout = self.options.rpc_timeout;
        let fut = async {
            self.conn
                .write_message(&ControlMessage::ListVoices(IdRequest::new(request_id)))
                .await?;
            let mut ignored = 0u32;
            loop {
                match self.conn.read_message().await? {
                    ControlMessage::Voices(VoicesResponse {
                        request_id: rid,
                        voices,
                        ..
                    }) if rid == request_id => return Ok(voices),
                    ControlMessage::Progress(p) if p.request_id == request_id => continue,
                    ControlMessage::Error(err)
                        if err.request_id.is_none() || err.request_id == Some(request_id) =>
                    {
                        return Err(err.into_protocol_error());
                    }
                    ControlMessage::Event(_) => {
                        ignored = bump_ignored(ignored, max_ignored)?;
                    }
                    other => {
                        if other.request_id() == Some(request_id) {
                            return Err(ProtocolError::UnexpectedMessage(message_type_name(
                                &other,
                            )));
                        }
                        ignored = bump_ignored(ignored, max_ignored)?;
                    }
                }
            }
        };
        with_timeout(timeout, fut).await
    }

    /// TTS: synthesize text and collect streamed PCM bytes (raw body, no request-id prefix).
    ///
    /// Requires a matching `audio_end` and a non-empty PCM body. Encoding advertised
    /// in `audio_start`/`audio_end` must agree with binary frame kinds when both present.
    pub async fn synthesize(
        &mut self,
        text: impl Into<String>,
        voice_id: Option<String>,
        speed: Option<f32>,
    ) -> Result<SynthesizeResult, ProtocolError> {
        let request_id = Uuid::now_v7();
        let timeout = self.options.rpc_timeout;
        let max_ignored = self.options.max_ignored_messages;
        let fut = async {
            self.conn
                .write_message(&ControlMessage::Synthesize(SynthesizeRequest {
                    request_id,
                    text: text.into(),
                    voice_id,
                    speed,
                    output_encoding: Some(PcmEncoding::F32Le),
                    extra: Default::default(),
                }))
                .await?;

            let mut sample_rate_hz = None;
            let mut channels = None;
            let mut encoding: Option<PcmEncoding> = None;
            let mut audio = Vec::new();
            let mut ignored = 0u32;

            loop {
                let frame = self.conn.read_frame().await?;
                match frame.kind {
                    FrameKind::Json => {
                        let msg = ControlMessage::from_frame(&frame)?;
                        match msg {
                            ControlMessage::AudioStart(ev) if ev.request_id == request_id => {
                                sample_rate_hz = ev.sample_rate_hz;
                                channels = ev.channels;
                                if let Some(enc) = ev.encoding {
                                    encoding = Some(enc);
                                }
                            }
                            ControlMessage::AudioEnd(ev) if ev.request_id == request_id => {
                                if let Some(sr) = ev.sample_rate_hz {
                                    sample_rate_hz = Some(sr);
                                }
                                if let Some(ch) = ev.channels {
                                    channels = Some(ch);
                                }
                                if let Some(enc) = ev.encoding {
                                    if let Some(prior) = encoding {
                                        if prior != enc {
                                            return Err(ProtocolError::EncodingMismatch(
                                                "audio_end encoding disagrees with stream",
                                            ));
                                        }
                                    } else {
                                        encoding = Some(enc);
                                    }
                                }
                                break;
                            }
                            ControlMessage::Error(err)
                                if err.request_id.is_none()
                                    || err.request_id == Some(request_id) =>
                            {
                                return Err(err.into_protocol_error());
                            }
                            ControlMessage::Progress(p) if p.request_id == request_id => continue,
                            ControlMessage::Event(_) => {
                                ignored = bump_ignored(ignored, max_ignored)?;
                            }
                            ControlMessage::Ack(Ack {
                                request_id: rid, ..
                            }) if rid == request_id => {
                                // Ack is not a substitute for audio_end.
                                ignored = bump_ignored(ignored, max_ignored)?;
                            }
                            other if other.request_id() == Some(request_id) => {
                                return Err(ProtocolError::UnexpectedMessage(message_type_name(
                                    &other,
                                )));
                            }
                            _ => {
                                ignored = bump_ignored(ignored, max_ignored)?;
                            }
                        }
                    }
                    FrameKind::PcmF32Le | FrameKind::PcmI16Le | FrameKind::Bytes => {
                        let (rid, body) = frame.split_binary_payload()?;
                        if rid != request_id {
                            ignored = bump_ignored(ignored, max_ignored)?;
                            continue;
                        }
                        let frame_enc = match frame.kind {
                            FrameKind::PcmF32Le => Some(PcmEncoding::F32Le),
                            FrameKind::PcmI16Le => Some(PcmEncoding::I16Le),
                            FrameKind::Bytes => None,
                            FrameKind::Json => None,
                        };
                        if let (Some(declared), Some(got)) = (encoding, frame_enc)
                            && declared != got
                        {
                            return Err(ProtocolError::EncodingMismatch(
                                "binary frame kind disagrees with audio_start encoding",
                            ));
                        }
                        if encoding.is_none() {
                            encoding = frame_enc;
                        }
                        audio.extend_from_slice(&body);
                    }
                }
            }

            if audio.is_empty() {
                return Err(ProtocolError::EmptyAudio);
            }

            Ok(SynthesizeResult {
                request_id,
                sample_rate_hz,
                channels,
                encoding: encoding.unwrap_or(PcmEncoding::F32Le),
                pcm: Bytes::from(audio),
            })
        };
        with_timeout(timeout, fut).await
    }

    /// Low-level: wait until an `ack` for `request_id` arrives.
    pub async fn wait_ack(&mut self, request_id: RequestId) -> Result<Ack, ProtocolError> {
        let max_ignored = self.options.max_ignored_messages;
        let mut ignored = 0u32;
        loop {
            match self.conn.read_message().await? {
                ControlMessage::Ack(ack) if ack.request_id == request_id => return Ok(ack),
                ControlMessage::Error(err) => {
                    if err.request_id.is_none() || err.request_id == Some(request_id) {
                        return Err(err.into_protocol_error());
                    }
                    ignored = bump_ignored(ignored, max_ignored)?;
                }
                ControlMessage::Progress(p) if p.request_id == request_id => continue,
                ControlMessage::Event(_) => {
                    ignored = bump_ignored(ignored, max_ignored)?;
                }
                other if other.request_id() == Some(request_id) => {
                    return Err(ProtocolError::UnexpectedMessage(message_type_name(&other)));
                }
                _ => {
                    ignored = bump_ignored(ignored, max_ignored)?;
                }
            }
        }
    }

    async fn wait_transcript(
        &mut self,
        request_id: RequestId,
    ) -> Result<TranscriptEvent, ProtocolError> {
        let max_ignored = self.options.max_ignored_messages;
        let mut ignored = 0u32;
        loop {
            match self.conn.read_message().await? {
                ControlMessage::PartialTranscript(ev) if ev.request_id == request_id => {
                    return Ok(ev);
                }
                ControlMessage::FinalTranscript(ev) if ev.request_id == request_id => {
                    return Ok(ev);
                }
                ControlMessage::Ack(Ack {
                    request_id: rid,
                    result,
                    ..
                }) if rid == request_id => {
                    let text = result
                        .as_ref()
                        .and_then(|v| v.get("text"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    return Ok(TranscriptEvent {
                        request_id,
                        text,
                        extra: Default::default(),
                    });
                }
                ControlMessage::Error(err) => {
                    if err.request_id.is_none() || err.request_id == Some(request_id) {
                        return Err(err.into_protocol_error());
                    }
                    ignored = bump_ignored(ignored, max_ignored)?;
                }
                ControlMessage::Progress(p) if p.request_id == request_id => continue,
                ControlMessage::Event(_) => {
                    ignored = bump_ignored(ignored, max_ignored)?;
                }
                other if other.request_id() == Some(request_id) => {
                    return Err(ProtocolError::UnexpectedMessage(message_type_name(&other)));
                }
                _ => {
                    ignored = bump_ignored(ignored, max_ignored)?;
                }
            }
        }
    }

    async fn wait_final_transcript(
        &mut self,
        request_id: RequestId,
    ) -> Result<TranscriptEvent, ProtocolError> {
        let max_ignored = self.options.max_ignored_messages;
        let mut ignored = 0u32;
        loop {
            match self.conn.read_message().await? {
                ControlMessage::FinalTranscript(ev) if ev.request_id == request_id => {
                    return Ok(ev);
                }
                ControlMessage::PartialTranscript(ev) if ev.request_id == request_id => {
                    let _ = ev;
                    continue;
                }
                ControlMessage::Ack(Ack {
                    request_id: rid,
                    result,
                    ..
                }) if rid == request_id => {
                    let text = result
                        .as_ref()
                        .and_then(|v| v.get("text"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    return Ok(TranscriptEvent {
                        request_id,
                        text,
                        extra: Default::default(),
                    });
                }
                ControlMessage::Error(err) => {
                    if err.request_id.is_none() || err.request_id == Some(request_id) {
                        return Err(err.into_protocol_error());
                    }
                    ignored = bump_ignored(ignored, max_ignored)?;
                }
                ControlMessage::Progress(p) if p.request_id == request_id => continue,
                ControlMessage::Event(_) => {
                    ignored = bump_ignored(ignored, max_ignored)?;
                }
                other if other.request_id() == Some(request_id) => {
                    return Err(ProtocolError::UnexpectedMessage(message_type_name(&other)));
                }
                _ => {
                    ignored = bump_ignored(ignored, max_ignored)?;
                }
            }
        }
    }
}

/// Collected TTS audio from [`WorkerClient::synthesize`].
#[derive(Debug, Clone, PartialEq)]
pub struct SynthesizeResult {
    pub request_id: RequestId,
    pub sample_rate_hz: Option<u32>,
    pub channels: Option<u16>,
    pub encoding: PcmEncoding,
    /// Raw PCM bytes (without request-id prefix).
    pub pcm: Bytes,
}

fn bump_ignored(current: u32, limit: u32) -> Result<u32, ProtocolError> {
    let next = current.saturating_add(1);
    if next > limit {
        return Err(ProtocolError::TooManyIgnoredMessages { limit });
    }
    Ok(next)
}

async fn with_timeout<T, F>(dur: Duration, fut: F) -> Result<T, ProtocolError>
where
    F: std::future::Future<Output = Result<T, ProtocolError>>,
{
    match timeout(dur, fut).await {
        Ok(inner) => inner,
        Err(_) => Err(ProtocolError::RpcTimeout { timeout: dur }),
    }
}

fn message_type_name(msg: &ControlMessage) -> &'static str {
    match msg {
        ControlMessage::Hello(_) => "hello",
        ControlMessage::HelloOk(_) => "hello_ok",
        ControlMessage::HelloErr(_) => "hello_err",
        ControlMessage::Load(_) => "load",
        ControlMessage::Reset(_) => "reset",
        ControlMessage::ProcessChunk(_) => "process_chunk",
        ControlMessage::ProcessUtterance(_) => "process_utterance",
        ControlMessage::Finish(_) => "finish",
        ControlMessage::Cancel(_) => "cancel",
        ControlMessage::Close(_) => "close",
        ControlMessage::Synthesize(_) => "synthesize",
        ControlMessage::ListVoices(_) => "list_voices",
        ControlMessage::Ack(_) => "ack",
        ControlMessage::Error(_) => "error",
        ControlMessage::PartialTranscript(_) => "partial_transcript",
        ControlMessage::FinalTranscript(_) => "final_transcript",
        ControlMessage::Progress(_) => "progress",
        ControlMessage::Voices(_) => "voices",
        ControlMessage::AudioStart(_) => "audio_start",
        ControlMessage::AudioEnd(_) => "audio_end",
        ControlMessage::Event(_) => "event",
        ControlMessage::Unknown => "unknown",
    }
}

/// Server-side helper: accept a client `hello` and reply with `hello_ok`.
pub async fn accept_handshake<R, W>(
    conn: &mut FramedConnection<R, W>,
    manifest: WorkerManifest,
) -> Result<u16, ProtocolError>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let msg = conn.read_message().await?;
    let ControlMessage::Hello(hello) = msg else {
        return Err(ProtocolError::UnexpectedMessage("expected hello"));
    };
    if hello.protocol_version != PROTOCOL_VERSION {
        let err = ControlMessage::HelloErr(crate::messages::HelloErr {
            message: format!(
                "unsupported protocol version {} (server supports {})",
                hello.protocol_version, PROTOCOL_VERSION
            ),
            code: Some("unsupported_version".into()),
            protocol_version: Some(hello.protocol_version),
            extra: Default::default(),
        });
        conn.write_message(&err).await?;
        return Err(ProtocolError::UnsupportedVersion {
            remote: hello.protocol_version,
            local: PROTOCOL_VERSION,
        });
    }
    conn.write_message(&ControlMessage::HelloOk(HelloOk {
        protocol_version: PROTOCOL_VERSION,
        manifest,
        extra: Default::default(),
    }))
    .await?;
    Ok(PROTOCOL_VERSION)
}
