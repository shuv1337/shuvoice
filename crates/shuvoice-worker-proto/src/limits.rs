//! Hard limits that keep framed I/O allocation-bounded.

use std::time::Duration;

/// Current wire protocol version negotiated during handshake.
pub const PROTOCOL_VERSION: u16 = 1;

/// Maximum value of the 32-bit frame length field (kind byte + payload).
///
/// Chosen to allow multi-second float32 PCM utterance frames while remaining
/// far below pathological multi-gigabyte allocations.
pub const MAX_FRAME_LEN: u32 = 16 * 1024 * 1024;

/// Hard maximum for JSON control frame payloads (kind excluded).
///
/// Enforced on encode and on decode after the kind byte is known, before JSON
/// parsing. PCM/binary frames may still use up to [`MAX_FRAME_LEN`].
pub const MAX_JSON_PAYLOAD_LEN: u32 = 1024 * 1024;

/// Bytes occupied by a UUID request id prefix on binary frames.
pub const BINARY_REQUEST_ID_LEN: usize = 16;

/// Minimum binary payload size (request id only).
pub const MIN_BINARY_PAYLOAD_LEN: usize = BINARY_REQUEST_ID_LEN;

/// Minimum legal frame length field: one kind byte.
pub const MIN_FRAME_LEN: u32 = 1;

/// Default per-RPC deadline for non-load operations on [`crate::WorkerClient`].
pub const DEFAULT_RPC_TIMEOUT: Duration = Duration::from_secs(120);

/// Default deadline for `load` (model download / init can be slow).
pub const DEFAULT_LOAD_TIMEOUT: Duration = Duration::from_secs(600);

/// Default max unrelated control/binary messages skipped while awaiting a reply.
pub const DEFAULT_MAX_IGNORED_MESSAGES: u32 = 64;
