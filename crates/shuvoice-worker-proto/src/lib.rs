//! Versioned framed protocol for external ShuVoice model workers.
//!
//! # Wire format
//!
//! Every frame is length-prefixed:
//!
//! ```text
//! ┌─────────────────┬──────────┬──────────────────────────┐
//! │ length u32 BE   │ kind u8  │ payload (length-1 bytes) │
//! └─────────────────┴──────────┴──────────────────────────┘
//! ```
//!
//! `length` covers the kind byte plus payload and is rejected unless it falls
//! in `MIN_FRAME_LEN..=MAX_FRAME_LEN` **before** any payload allocation.
//!
//! | kind | meaning |
//! |------|---------|
//! | `1`  | JSON control message ([`ControlMessage`]) |
//! | `2`  | `f32` LE mono PCM, payload = `request_id[16] \|\| samples` |
//! | `3`  | `i16` LE mono PCM, payload = `request_id[16] \|\| samples` |
//! | `4`  | opaque bytes, payload = `request_id[16] \|\| data` |
//!
//! # Handshake
//!
//! 1. Client sends `hello` with [`PROTOCOL_VERSION`].
//! 2. Worker replies `hello_ok` + [`WorkerManifest`] or `hello_err`.
//!
//! # Roles
//!
//! - Native hosts use [`WorkerClient`] over stdio/sockets, or [`WorkerProcess`] /
//!   [`WorkerSupervisor`] to spawn an explicit executable (no shell; isolated child env).
//! - Bundled/reference workers use [`FramedConnection`] + [`accept_handshake`].

#![forbid(unsafe_code)]

mod client;
mod codec;
mod error;
mod frame;
mod limits;
mod manifest;
mod messages;
mod process;
mod restart;
mod stderr_tail;
mod supervisor;

pub use client::{
    ClientOptions, NegotiatedSession, SynthesizeResult, WorkerClient, accept_handshake,
};
pub use codec::{FramedConnection, FramedReader, FramedWriter};
pub use error::{ProtocolError, WorkerProcessError};
pub use frame::{Frame, FrameKind};
pub use limits::{
    BINARY_REQUEST_ID_LEN, DEFAULT_LOAD_TIMEOUT, DEFAULT_MAX_IGNORED_MESSAGES, DEFAULT_RPC_TIMEOUT,
    MAX_FRAME_LEN, MAX_JSON_PAYLOAD_LEN, MIN_BINARY_PAYLOAD_LEN, MIN_FRAME_LEN, PROTOCOL_VERSION,
};
pub use manifest::{AsrCapabilities, BackendKind, TtsCapabilities, WorkerManifest};
pub use messages::{
    Ack, AudioStreamEvent, CloseRequest, ControlMessage, ErrorResponse, FinishRequest, Hello,
    HelloErr, HelloOk, IdRequest, LoadRequest, PcmEncoding, ProcessAudioRequest, ProgressEvent,
    RequestId, SynthesizeRequest, TranscriptEvent, VoiceInfo, VoicesResponse, WorkerEvent,
};
pub use process::{
    CHILD_ENV_ALLOWLIST, DEFAULT_HANDSHAKE_TIMEOUT, DEFAULT_KILL_TIMEOUT, DEFAULT_SHUTDOWN_TIMEOUT,
    WorkerExitStatus, WorkerProcess, WorkerSpawnConfig, build_isolated_child_env,
    build_isolated_child_env_from, format_argv_for_log,
};
pub use restart::{RestartDecision, RestartPolicy, RestartState};
pub use stderr_tail::{DEFAULT_STDERR_TAIL_BYTES, redact_stderr_tail, redact_text};
pub use supervisor::{WorkerSupervisor, honor_delay};
