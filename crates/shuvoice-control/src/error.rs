//! Error types for the control protocol.

use std::io;
use std::path::PathBuf;

use thiserror::Error;

/// Errors produced while resolving paths, serving, or calling the control socket.
#[derive(Debug, Error)]
pub enum ControlError {
    #[error("invalid control command: {0}")]
    InvalidCommand(String),

    #[error("control socket path must be absolute")]
    PathNotAbsolute,

    #[error("control socket path must be a .sock file, not a directory")]
    PathIsDirectory,

    #[error("control socket path must end with '.sock'")]
    PathBadSuffix,

    #[error("control socket parent must live under: {0}")]
    PathOutsideJail(String),

    #[error("control socket directory {0} is not owned by current user")]
    DirectoryNotOwned(PathBuf),

    #[error("control socket directory {path} has insecure mode {mode:#o} (want 0700)")]
    DirectoryInsecureMode { path: PathBuf, mode: u32 },

    #[error("control socket directory {0} is a symlink (refused)")]
    DirectoryIsSymlink(PathBuf),

    #[error("control socket directory {0} is not a directory")]
    NotADirectory(PathBuf),

    #[error("control socket {path} has insecure mode {mode:#o} (want 0600)")]
    SocketInsecureMode { path: PathBuf, mode: u32 },

    #[error("control socket not found at {0}. Is shuvoice running?")]
    SocketNotFound(PathBuf),

    #[error("failed to connect to control socket {path}: {source}")]
    Connect {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("empty response from control socket")]
    EmptyResponse,

    #[error("{0}")]
    Remote(String),

    #[error("control peer rejected (uid mismatch)")]
    PeerRejected,

    #[error("control server failed to become ready: {0}")]
    NotReady(String),

    #[error("control I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("{0}")]
    Other(String),
}

impl ControlError {
    /// True when the peer returned an `ERROR …` response line.
    #[must_use]
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::Remote(_))
    }
}
