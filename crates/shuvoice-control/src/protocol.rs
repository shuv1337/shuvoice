//! Wire protocol helpers (request/response lines).

use crate::commands::ControlCommand;
use crate::error::ControlError;

/// Maximum request payload accepted by the server (bytes).
pub const MAX_REQUEST_BYTES: usize = 1024;

/// Maximum response payload read by the client (bytes).
pub const MAX_RESPONSE_BYTES: usize = 4096;

/// Maximum handler response body emitted by the server (bytes, excluding `OK `/`ERROR ` prefix).
pub const MAX_HANDLER_RESPONSE_BYTES: usize = 3500;

/// Default client connect/read timeout.
pub const DEFAULT_CLIENT_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(1500);

/// Accept-loop timeout used by the server to check shutdown.
pub const SERVER_ACCEPT_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);

/// Per-connection read timeout on the server.
pub const SERVER_CONN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);

/// Join timeout when stopping the control server thread.
pub const SERVER_STOP_JOIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);

/// Normalize a raw request buffer into a [`ControlCommand`].
pub fn parse_request(payload: &[u8]) -> Result<ControlCommand, ControlError> {
    let text = String::from_utf8_lossy(payload);
    let normalized = text.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Err(ControlError::InvalidCommand(String::new()));
    }
    // Server accepts only a single token command (no args on the wire today).
    let token = normalized.split_whitespace().next().unwrap_or("");
    // Reject control characters / newlines in the token.
    if token.bytes().any(|b| b < 0x20 || b == 0x7f) {
        return Err(ControlError::InvalidCommand(String::new()));
    }
    ControlCommand::parse_token(token)
}

/// Encode a command for the wire (`"{cmd}\n"`).
#[must_use]
pub fn encode_request(command: ControlCommand) -> Vec<u8> {
    let mut out = command.as_str().as_bytes().to_vec();
    out.push(b'\n');
    out
}

/// Collapse handler output to a single safe wire line and enforce size caps.
#[must_use]
pub fn sanitize_response_line(raw: &str) -> String {
    let mut cleaned = String::with_capacity(raw.len().min(MAX_HANDLER_RESPONSE_BYTES + 16));
    for ch in raw.chars() {
        if ch == '\n' || ch == '\r' || ch == '\0' {
            cleaned.push(' ');
        } else if ch.is_control() {
            continue;
        } else {
            cleaned.push(ch);
        }
        if cleaned.len() >= MAX_HANDLER_RESPONSE_BYTES + 16 {
            break;
        }
    }
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        return fixed::INTERNAL.to_string();
    }
    // Ensure OK/ERROR prefix is preserved; truncate body only.
    if let Some(rest) = trimmed.strip_prefix("OK") {
        let body = rest.strip_prefix(' ').unwrap_or(rest).trim();
        if body.is_empty() {
            return "OK".to_string();
        }
        let body = truncate_bytes(body, MAX_HANDLER_RESPONSE_BYTES);
        return format!("OK {body}");
    }
    if let Some(rest) = trimmed.strip_prefix("ERROR") {
        let body = rest.strip_prefix(' ').unwrap_or(rest).trim();
        if body.is_empty() {
            return "ERROR".to_string();
        }
        let body = truncate_bytes(body, MAX_HANDLER_RESPONSE_BYTES);
        return format!("ERROR {body}");
    }
    // Bare body → OK-prefixed.
    let body = truncate_bytes(trimmed, MAX_HANDLER_RESPONSE_BYTES);
    format!("OK {body}")
}

fn truncate_bytes(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// A successful or error control response line (without trailing newline).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlResponse {
    /// `OK …` body (the text after the `OK` prefix, trimmed; may be empty for bare OK).
    Ok(String),
    /// Full `ERROR …` line.
    Err(String),
}

impl ControlResponse {
    /// Parse a response line from the peer.
    pub fn parse_line(line: &str) -> Result<Self, ControlError> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Err(ControlError::EmptyResponse);
        }
        if let Some(rest) = trimmed.strip_prefix("OK") {
            let body = rest.strip_prefix(' ').unwrap_or(rest).to_string();
            return Ok(Self::Ok(body));
        }
        if trimmed.starts_with("ERROR") {
            return Ok(Self::Err(trimmed.to_string()));
        }
        // Be liberal in what we accept: treat unknown non-empty as OK body.
        Ok(Self::Ok(trimmed.to_string()))
    }

    /// Wire line including no trailing newline.
    #[must_use]
    pub fn to_wire(&self) -> String {
        match self {
            Self::Ok(body) if body.is_empty() => "OK".to_string(),
            Self::Ok(body) => format!("OK {body}"),
            Self::Err(msg) => msg.clone(),
        }
    }

    /// Convert to client-facing result (ERROR lines become [`ControlError::Remote`]).
    pub fn into_result(self) -> Result<String, ControlError> {
        match self {
            Self::Ok(body) => {
                if body.is_empty() {
                    Ok("OK".to_string())
                } else {
                    Ok(format!("OK {body}"))
                }
            }
            Self::Err(msg) => Err(ControlError::Remote(msg)),
        }
    }
}

/// Fixed success responses matching the Python server.
pub mod fixed {
    pub const STARTED: &str = "OK started";
    pub const STOPPED: &str = "OK stopped";
    pub const TOGGLED: &str = "OK toggled";
    pub const PONG: &str = "OK pong";
    pub const TTS_NOT_AVAILABLE: &str = "ERROR tts not available";
    pub const TIMEOUT: &str = "ERROR timeout";
    pub const INTERNAL: &str = "ERROR internal error";
    pub const INVALID_REQUEST: &str = "ERROR invalid request";
    pub const PEER_REJECTED: &str = "ERROR peer rejected";

    #[must_use]
    pub fn unknown_command(cmd: &str) -> String {
        // Only echo a short sanitized token — never raw binary/control payload.
        let safe: String = cmd
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
            .take(64)
            .collect();
        if safe.is_empty() {
            "ERROR unknown command".to_string()
        } else {
            format!("ERROR unknown command: {safe}")
        }
    }

    #[must_use]
    pub fn ok_status(status: &str) -> String {
        format!("OK {status}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_request_trims_and_lowercases() {
        let cmd = parse_request(b"  TTS_SPEAK \n").unwrap();
        assert_eq!(cmd, ControlCommand::TtsSpeak);
    }

    #[test]
    fn parse_request_rejects_control_chars() {
        assert!(parse_request(b"sta\x00rt\n").is_err());
    }

    #[test]
    fn response_round_trip() {
        let line = "OK recording";
        let parsed = ControlResponse::parse_line(line).unwrap();
        assert_eq!(parsed, ControlResponse::Ok("recording".into()));
        assert_eq!(parsed.to_wire(), line);
        assert_eq!(parsed.into_result().unwrap(), line);
    }

    #[test]
    fn error_response_becomes_remote_error() {
        let parsed = ControlResponse::parse_line("ERROR tts disabled").unwrap();
        let err = parsed.into_result().unwrap_err();
        assert!(matches!(err, ControlError::Remote(msg) if msg == "ERROR tts disabled"));
    }

    #[test]
    fn sanitize_strips_newlines_and_caps() {
        let huge = format!("OK {}\nINJECT", "x".repeat(10_000));
        let out = sanitize_response_line(&huge);
        assert!(out.starts_with("OK "));
        assert!(!out.contains('\n'));
        assert!(out.len() <= MAX_HANDLER_RESPONSE_BYTES + 16);
    }

    #[test]
    fn unknown_command_sanitizes_token() {
        let msg = fixed::unknown_command("evil\ncmd;rm");
        assert!(!msg.contains('\n'));
        assert!(msg.contains("unknown command"));
    }
}
