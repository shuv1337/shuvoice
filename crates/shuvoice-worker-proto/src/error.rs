//! Protocol and process supervision errors.

use std::io;
use std::time::Duration;

use thiserror::Error;

/// Errors produced while encoding, decoding, or negotiating the worker protocol.
#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("unexpected end of stream while reading {context}")]
    UnexpectedEof { context: &'static str },

    #[error("frame truncated: declared payload {declared} bytes, got {got} bytes")]
    TruncatedFrame { declared: usize, got: usize },

    #[error("frame length {length} exceeds maximum {max}")]
    FrameTooLarge { length: u32, max: u32 },

    #[error("frame length {length} is below minimum {min}")]
    FrameTooSmall { length: u32, min: u32 },

    #[error("JSON payload length {length} exceeds maximum {max}")]
    JsonTooLarge { length: u32, max: u32 },

    #[error("unsupported frame kind: {0}")]
    UnsupportedFrameKind(u8),

    #[error("unsupported protocol version: remote={remote}, local={local}")]
    UnsupportedVersion { remote: u16, local: u16 },

    #[error("protocol version mismatch after negotiation: expected {expected}, got {got}")]
    VersionMismatch { expected: u16, got: u16 },

    #[error("invalid binary payload: {0}")]
    InvalidBinaryPayload(&'static str),

    #[error("invalid JSON control payload: {0}")]
    InvalidJson(#[from] serde_json::Error),

    #[error("unexpected control message type: {0}")]
    UnexpectedMessage(&'static str),

    #[error("request id mismatch: expected {expected}, got {got}")]
    RequestIdMismatch { expected: String, got: String },

    #[error("worker error ({code}): {message}")]
    Worker {
        code: String,
        message: String,
        request_id: Option<String>,
    },

    #[error("handshake failed: {0}")]
    Handshake(String),

    #[error("RPC timed out after {timeout:?}")]
    RpcTimeout { timeout: Duration },

    #[error("too many ignored messages while waiting for response (limit {limit})")]
    TooManyIgnoredMessages { limit: u32 },

    #[error("TTS audio stream ended without audio_end")]
    MissingAudioEnd,

    #[error("TTS audio stream produced no PCM samples")]
    EmptyAudio,

    #[error("audio encoding mismatch: {0}")]
    EncodingMismatch(&'static str),

    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

impl ProtocolError {
    /// True when the peer closed the stream cleanly at a frame boundary.
    #[must_use]
    pub fn is_clean_eof(&self) -> bool {
        matches!(self, Self::UnexpectedEof { context } if *context == "frame length")
    }

    #[must_use]
    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::RpcTimeout { .. })
    }
}

/// Errors from spawning, handshaking, supervising, or shutting down a worker process.
///
/// Display / debug forms must never include protocol payloads or transcripts.
/// `stderr_tail` fields are already redacted by the capture path.
#[derive(Debug, Error)]
pub enum WorkerProcessError {
    #[error("failed to spawn worker process: {0}")]
    Spawn(#[source] io::Error),

    #[error("worker handshake timed out")]
    HandshakeTimeout { stderr_tail: String },

    #[error("worker handshake failed: {message}")]
    Handshake {
        message: String,
        stderr_tail: String,
    },

    #[error("unsupported worker protocol version: remote={remote}, local={local}")]
    UnsupportedVersion {
        remote: u16,
        local: u16,
        stderr_tail: String,
    },

    #[error("worker dependency missing: {message}")]
    DependencyMissing {
        message: String,
        stderr_tail: String,
    },

    #[error("worker process crashed (exit={exit_code:?})")]
    Crashed {
        exit_code: Option<i32>,
        stderr_tail: String,
    },

    #[error("worker shutdown timed out")]
    ShutdownTimeout { stderr_tail: String },

    #[error("restart deferred for {delay:?}")]
    RestartDeferred { delay: Duration },

    #[error("restart attempts exhausted after {consecutive_failures} consecutive failures")]
    RestartExhausted { consecutive_failures: u32 },

    #[error("worker I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("worker I/O error: {source}")]
    IoWithStderr {
        #[source]
        source: io::Error,
        stderr_tail: String,
    },

    #[error("worker protocol error: {0}")]
    Protocol(#[from] ProtocolError),
}

impl WorkerProcessError {
    /// Redacted stderr tail when available.
    #[must_use]
    pub fn stderr_tail(&self) -> Option<&str> {
        match self {
            Self::HandshakeTimeout { stderr_tail }
            | Self::Handshake { stderr_tail, .. }
            | Self::UnsupportedVersion { stderr_tail, .. }
            | Self::DependencyMissing { stderr_tail, .. }
            | Self::Crashed { stderr_tail, .. }
            | Self::ShutdownTimeout { stderr_tail }
            | Self::IoWithStderr { stderr_tail, .. } => Some(stderr_tail.as_str()),
            _ => None,
        }
    }

    #[must_use]
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Crashed { .. }
                | Self::HandshakeTimeout { .. }
                | Self::ShutdownTimeout { .. }
                | Self::Io(_)
                | Self::IoWithStderr { .. }
                | Self::RestartDeferred { .. }
        )
    }
}
