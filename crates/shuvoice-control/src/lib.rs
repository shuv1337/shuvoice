//! Secure Unix-socket control protocol for ShuVoice.
//!
//! Behavior is intentionally locked to the Python `shuvoice.control` module:
//! AF_UNIX stream socket, single-line lowercased commands, `OK`/`ERROR` responses,
//! path jail under `$XDG_RUNTIME_DIR` or `/tmp`, and user-only directory modes.
//!
//! Security and lifecycle guarantees:
//! - post-bind force+verify socket mode `0600`
//! - hard-fail directory mode `0700` / uid / no-follow symlink walk
//! - client path resolve does not create directories
//! - synchronous bind readiness from [`ControlServer::start`]
//! - panic-isolated handlers, capped/sanitized responses
//! - same-uid `SO_PEERCRED` on Linux
//! - bounded stop join
//!
//! ## Residual portability notes
//! - `SO_PEERCRED` is provided by rustix on Linux. Non-Linux targets are outside
//!   ShuVoice's supported surface; filesystem mode `0600`/`0700` remains the
//!   primary auth boundary.
//! - Bind still uses a path string (not `bindat` on a verified dirfd). Symlink
//!   components on the logical path are refused before bind, and the socket
//!   inode is force-chmod'd to `0600` immediately after bind.

#![forbid(unsafe_op_in_unsafe_fn)]

mod client;
mod commands;
mod error;
mod handlers;
mod path;
mod protocol;
mod server;

pub use client::{
    request_bytes, resolve_path, send_control_command, send_control_command_str,
    send_control_command_to,
};
pub use commands::{CONTROL_COMMANDS, ControlCommand};
pub use error::ControlError;
pub use handlers::{ControlHandlers, FnControlHandlers, dispatch};
pub use path::{
    allowed_control_roots, default_control_socket_path, default_control_socket_path_logical,
    ensure_secure_directory, force_socket_mode, prepare_control_socket_path,
    resolve_control_socket_path,
};
pub use protocol::{
    ControlResponse, DEFAULT_CLIENT_TIMEOUT, MAX_HANDLER_RESPONSE_BYTES, MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES, SERVER_ACCEPT_TIMEOUT, SERVER_CONN_TIMEOUT, SERVER_STOP_JOIN_TIMEOUT,
    fixed, parse_request, sanitize_response_line,
};
pub use server::ControlServer;
