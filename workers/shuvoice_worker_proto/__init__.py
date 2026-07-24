"""ShuVoice external-worker protocol (v1) — Python reference implementation.

Matches ``crates/shuvoice-worker-proto`` wire format exactly so optional
Python model runtimes can speak to the Rust shell over stdio.
"""

from .constants import (
    BINARY_REQUEST_ID_LEN,
    FRAME_KIND_BYTES,
    FRAME_KIND_JSON,
    FRAME_KIND_PCM_F32_LE,
    FRAME_KIND_PCM_I16_LE,
    MAX_FRAME_LEN,
    MIN_FRAME_LEN,
    PROTOCOL_VERSION,
)
from .errors import ProtocolError
from .framing import Frame, decode_f32le_samples, decode_i16le_samples, read_frame, write_frame
from .io import FramedStdio
from .messages import parse_control_message
from .server import WorkerServer

__all__ = [
    "BINARY_REQUEST_ID_LEN",
    "FRAME_KIND_BYTES",
    "FRAME_KIND_JSON",
    "FRAME_KIND_PCM_F32_LE",
    "FRAME_KIND_PCM_I16_LE",
    "Frame",
    "FramedStdio",
    "MAX_FRAME_LEN",
    "MIN_FRAME_LEN",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "WorkerServer",
    "decode_f32le_samples",
    "decode_i16le_samples",
    "parse_control_message",
    "read_frame",
    "write_frame",
]

__version__ = "0.1.3"
