"""Wire constants — keep in lockstep with crates/shuvoice-worker-proto/src/limits.rs."""

from __future__ import annotations

PROTOCOL_VERSION: int = 1
MAX_FRAME_LEN: int = 16 * 1024 * 1024
MAX_JSON_PAYLOAD_LEN: int = 1024 * 1024
BINARY_REQUEST_ID_LEN: int = 16
MIN_BINARY_PAYLOAD_LEN: int = BINARY_REQUEST_ID_LEN
MIN_FRAME_LEN: int = 1

FRAME_KIND_JSON: int = 1
FRAME_KIND_PCM_F32_LE: int = 2
FRAME_KIND_PCM_I16_LE: int = 3
FRAME_KIND_BYTES: int = 4

BINARY_KINDS = frozenset(
    {
        FRAME_KIND_PCM_F32_LE,
        FRAME_KIND_PCM_I16_LE,
        FRAME_KIND_BYTES,
    }
)

# Default client-side waits (seconds) — advisory for Python test client.
DEFAULT_RPC_TIMEOUT_SEC: float = 120.0
DEFAULT_LOAD_TIMEOUT_SEC: float = 600.0
DEFAULT_MAX_IGNORED_MESSAGES: int = 64
