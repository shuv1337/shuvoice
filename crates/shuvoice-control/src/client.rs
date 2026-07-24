//! Blocking control-socket client.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::commands::ControlCommand;
use crate::error::ControlError;
use crate::path::resolve_control_socket_path;
use crate::protocol::{
    self, ControlResponse, DEFAULT_CLIENT_TIMEOUT, MAX_RESPONSE_BYTES, encode_request,
};

/// Send a control command and return the full response line (`OK …`).
///
/// ERROR responses are returned as [`ControlError::Remote`].
///
/// Path resolution does **not** create directories (client-safe).
pub fn send_control_command(
    command: ControlCommand,
    socket_path: Option<&str>,
    timeout: Option<Duration>,
) -> Result<String, ControlError> {
    let path = resolve_control_socket_path(socket_path)?;
    send_control_command_to(&path, command, timeout.unwrap_or(DEFAULT_CLIENT_TIMEOUT))
}

/// Send to an already-resolved socket path.
pub fn send_control_command_to(
    path: &Path,
    command: ControlCommand,
    timeout: Duration,
) -> Result<String, ControlError> {
    let stream = UnixStream::connect(path).map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            ControlError::SocketNotFound(path.to_path_buf())
        } else {
            ControlError::Connect {
                path: path.to_path_buf(),
                source: err,
            }
        }
    })?;

    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;

    let mut stream = stream;
    stream.write_all(&encode_request(command))?;
    // Match Python: shutdown write half after sending.
    let _ = stream.shutdown(std::net::Shutdown::Write);

    let mut buf = vec![0u8; MAX_RESPONSE_BYTES];
    let n = stream.read(&mut buf)?;
    if n == 0 {
        return Err(ControlError::EmptyResponse);
    }
    let text = String::from_utf8_lossy(&buf[..n]);
    // Only the first line is significant.
    let line = text.lines().next().unwrap_or("").trim();
    ControlResponse::parse_line(line)?.into_result()
}

/// Convenience: parse a string command then send.
pub fn send_control_command_str(
    command: &str,
    socket_path: Option<&str>,
    timeout: Option<Duration>,
) -> Result<String, ControlError> {
    let cmd = command.parse::<ControlCommand>()?;
    send_control_command(cmd, socket_path, timeout)
}

/// Resolve path helper re-export for callers that only have the client API open.
///
/// Does **not** create directories.
pub fn resolve_path(socket_path: Option<&str>) -> Result<PathBuf, ControlError> {
    resolve_control_socket_path(socket_path)
}

/// Build the on-wire request bytes (test helper / low-level clients).
#[must_use]
pub fn request_bytes(command: ControlCommand) -> Vec<u8> {
    protocol::encode_request(command)
}
