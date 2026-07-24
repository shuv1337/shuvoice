//! Hardened, thread-based blocking AF_UNIX control server.

use std::io::{Read, Write};
use std::os::fd::AsFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use rustix::fs::Mode;
use rustix::net::sockopt;
use rustix::process::{getuid, umask};

use crate::error::ControlError;
use crate::handlers::{ControlHandlers, dispatch};
use crate::path::{force_socket_mode, prepare_control_socket_path};
use crate::protocol::{
    MAX_REQUEST_BYTES, SERVER_ACCEPT_TIMEOUT, SERVER_CONN_TIMEOUT, SERVER_STOP_JOIN_TIMEOUT, fixed,
    parse_request,
};

/// Secure Unix-domain control server.
pub struct ControlServer {
    socket_path: PathBuf,
    handlers: Arc<dyn ControlHandlers>,
    running: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl ControlServer {
    /// Create a server bound to `socket_path` (None → default path).
    ///
    /// Prepares the parent directory (`0700`) but does not bind until [`start`].
    pub fn new(
        socket_path: Option<&str>,
        handlers: Arc<dyn ControlHandlers>,
    ) -> Result<Self, ControlError> {
        let path = prepare_control_socket_path(socket_path)?;
        Ok(Self {
            socket_path: path,
            handlers,
            running: Arc::new(AtomicBool::new(false)),
            thread: None,
        })
    }

    /// Resolved socket path.
    #[must_use]
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Bind + start the accept loop on a background thread.
    ///
    /// Returns an error if bind/chmod fails (synchronous readiness).
    pub fn start(&mut self) -> Result<(), ControlError> {
        if self.thread.as_ref().is_some_and(|t| !t.is_finished()) {
            return Ok(());
        }

        crate::path::ensure_secure_directory(
            self.socket_path
                .parent()
                .ok_or_else(|| ControlError::Other("socket path has no parent".into()))?,
        )?;

        // Bind on the calling thread so failures surface synchronously.
        let listener = bind_listener(&self.socket_path)?;

        self.running.store(true, Ordering::SeqCst);
        let running = Arc::clone(&self.running);
        let handlers = Arc::clone(&self.handlers);
        let socket_path = self.socket_path.clone();

        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();

        self.thread = Some(
            thread::Builder::new()
                .name("control-socket".into())
                .spawn(move || {
                    // Signal that the accept loop owns the listener.
                    let _ = ready_tx.send(Ok(()));
                    if let Err(err) = run_server_loop(listener, &socket_path, handlers, running) {
                        tracing::error!("control server exited: {err}");
                    }
                })
                .map_err(ControlError::Io)?,
        );

        match ready_rx.recv_timeout(Duration::from_secs(2)) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(msg)) => {
                self.running.store(false, Ordering::SeqCst);
                Err(ControlError::NotReady(msg))
            }
            Err(_) => {
                self.running.store(false, Ordering::SeqCst);
                Err(ControlError::NotReady(
                    "accept thread failed to start".into(),
                ))
            }
        }
    }

    /// Stop the server and remove the socket file.
    ///
    /// Joins the accept thread with a bounded timeout so process teardown cannot hang.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);

        // Wake the accept loop (Python pings during stop).
        let _ = crate::client::send_control_command_to(
            &self.socket_path,
            crate::commands::ControlCommand::Ping,
            Duration::from_millis(200),
        );

        if let Some(handle) = self.thread.take() {
            let deadline = Instant::now() + SERVER_STOP_JOIN_TIMEOUT;
            loop {
                if handle.is_finished() {
                    let _ = handle.join();
                    break;
                }
                if Instant::now() >= deadline {
                    tracing::warn!(
                        "control server thread did not exit within {:?}; abandoning join",
                        SERVER_STOP_JOIN_TIMEOUT
                    );
                    // Detach: drop JoinHandle without join (thread may outlive briefly).
                    // Leak the handle intentionally to avoid blocking Drop.
                    std::mem::forget(handle);
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
        }
        cleanup_socket_file(&self.socket_path);
    }
}

impl Drop for ControlServer {
    fn drop(&mut self) {
        self.stop();
    }
}

fn cleanup_socket_file(path: &Path) {
    // remove_file on a path: if it's a symlink, removes the symlink only.
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => {
            tracing::debug!("Failed to remove control socket {}: {err}", path.display());
        }
    }
}

fn bind_listener(socket_path: &Path) -> Result<UnixListener, ControlError> {
    cleanup_socket_file(socket_path);

    let old = umask(Mode::from_raw_mode(0o077));
    let bind_result = UnixListener::bind(socket_path);
    let _ = umask(old);
    let listener = bind_result.map_err(ControlError::Io)?;

    // Force + verify 0600 regardless of concurrent umask races.
    force_socket_mode(socket_path)?;

    listener.set_nonblocking(true)?;
    tracing::info!("Control socket listening: {}", socket_path.display());
    Ok(listener)
}

fn run_server_loop(
    listener: UnixListener,
    socket_path: &Path,
    handlers: Arc<dyn ControlHandlers>,
    running: Arc<AtomicBool>,
) -> Result<(), ControlError> {
    while running.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((conn, _)) => {
                if let Err(err) = handle_connection(conn, &handlers) {
                    tracing::debug!("control connection error: {err}");
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(SERVER_ACCEPT_TIMEOUT);
            }
            Err(err) => {
                if running.load(Ordering::SeqCst) {
                    tracing::error!("Control socket accept failed: {err}");
                }
                break;
            }
        }
    }

    cleanup_socket_file(socket_path);
    tracing::info!("Control socket stopped");
    Ok(())
}

fn peer_uid_allowed(conn: &UnixStream) -> bool {
    match sockopt::socket_peercred(conn.as_fd()) {
        Ok(cred) => cred.uid == getuid(),
        Err(err) => {
            // Residual: non-Linux or kernels without SO_PEERCRED — refuse closed-form
            // rather than fail-open when the call errors unexpectedly.
            tracing::warn!("SO_PEERCRED unavailable/failed ({err}); rejecting connection");
            false
        }
    }
}

fn handle_connection(
    conn: UnixStream,
    handlers: &Arc<dyn ControlHandlers>,
) -> Result<(), ControlError> {
    if !peer_uid_allowed(&conn) {
        let mut conn = conn;
        let _ = conn.set_nonblocking(false);
        let mut out = fixed::PEER_REJECTED.as_bytes().to_vec();
        out.push(b'\n');
        let _ = conn.write_all(&out);
        return Err(ControlError::PeerRejected);
    }

    conn.set_read_timeout(Some(SERVER_CONN_TIMEOUT))?;
    conn.set_write_timeout(Some(SERVER_CONN_TIMEOUT))?;
    conn.set_nonblocking(false)?;

    let mut conn = conn;
    let mut buf = vec![0u8; MAX_REQUEST_BYTES];
    let response = match conn.read(&mut buf) {
        Ok(0) => fixed::INVALID_REQUEST.to_string(),
        Ok(n) => match parse_request(&buf[..n]) {
            Ok(cmd) => dispatch(handlers, cmd),
            Err(_) => {
                let text = String::from_utf8_lossy(&buf[..n]);
                let token = text.trim().to_ascii_lowercase();
                let token = token.split_whitespace().next().unwrap_or("");
                if token.is_empty() {
                    fixed::INVALID_REQUEST.to_string()
                } else {
                    fixed::unknown_command(token)
                }
            }
        },
        Err(err)
            if err.kind() == std::io::ErrorKind::TimedOut
                || err.kind() == std::io::ErrorKind::WouldBlock =>
        {
            tracing::warn!("Control command timed out");
            fixed::TIMEOUT.to_string()
        }
        Err(err) => {
            tracing::error!("Error handling control command: {err}");
            fixed::INTERNAL.to_string()
        }
    };

    let mut out = response.into_bytes();
    out.push(b'\n');
    conn.write_all(&out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::send_control_command_to;
    use crate::commands::ControlCommand;
    use crate::handlers::FnControlHandlers;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    fn test_handlers(state: Arc<Mutex<String>>) -> Arc<dyn ControlHandlers> {
        let state_start = Arc::clone(&state);
        let state_stop = Arc::clone(&state);
        let state_toggle = Arc::clone(&state);
        let state_status = Arc::clone(&state);
        Arc::new(FnControlHandlers {
            on_start: move || {
                *state_start.lock().unwrap() = "recording".into();
            },
            on_stop: move || {
                *state_stop.lock().unwrap() = "idle".into();
            },
            on_toggle: move || {
                let mut g = state_toggle.lock().unwrap();
                if g.as_str() == "recording" {
                    *g = "idle".into();
                } else {
                    *g = "recording".into();
                }
            },
            on_status: move || state_status.lock().unwrap().clone(),
            on_metrics: || r#"{"chunks":12}"#.into(),
            on_debug_status: || r#"{"ok":true}"#.into(),
            on_tts_command: Some(|cmd: ControlCommand| format!("OK handled:{}", cmd.as_str())),
        })
    }

    #[test]
    fn start_status_stop_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("control.sock");
        let state = Arc::new(Mutex::new("idle".to_string()));
        let mut server = ControlServer::new(
            Some(sock.to_str().unwrap()),
            test_handlers(Arc::clone(&state)),
        )
        .unwrap();
        server.start().unwrap();

        // Exact socket mode 0600.
        let meta = std::fs::metadata(server.socket_path()).unwrap();
        assert_eq!(meta.permissions().mode() & 0o777, 0o600);

        let path = server.socket_path().to_path_buf();
        let resp =
            send_control_command_to(&path, ControlCommand::Status, Duration::from_secs(1)).unwrap();
        assert_eq!(resp, "OK idle");

        let resp =
            send_control_command_to(&path, ControlCommand::Start, Duration::from_secs(1)).unwrap();
        assert_eq!(resp, "OK started");
        assert_eq!(state.lock().unwrap().as_str(), "recording");

        let resp =
            send_control_command_to(&path, ControlCommand::Stop, Duration::from_secs(1)).unwrap();
        assert_eq!(resp, "OK stopped");

        server.stop();
        assert!(!path.exists());
    }

    #[test]
    fn metrics_and_tts_routing() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("control.sock");
        let state = Arc::new(Mutex::new("idle".to_string()));
        let mut server =
            ControlServer::new(Some(sock.to_str().unwrap()), test_handlers(state)).unwrap();
        server.start().unwrap();
        let path = server.socket_path().to_path_buf();

        let resp = send_control_command_to(&path, ControlCommand::Metrics, Duration::from_secs(1))
            .unwrap();
        assert_eq!(resp, r#"OK {"chunks":12}"#);

        let resp =
            send_control_command_to(&path, ControlCommand::TtsStatus, Duration::from_secs(1))
                .unwrap();
        assert_eq!(resp, "OK handled:tts_status");

        let resp =
            send_control_command_to(&path, ControlCommand::Ping, Duration::from_secs(1)).unwrap();
        assert_eq!(resp, "OK pong");

        server.stop();
    }

    #[test]
    fn tts_without_callback_errors() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("control.sock");
        let calls = Arc::new(AtomicUsize::new(0));
        let c2 = Arc::clone(&calls);
        let handlers: Arc<dyn ControlHandlers> = Arc::new(FnControlHandlers {
            on_start: move || {
                c2.fetch_add(1, Ordering::SeqCst);
            },
            on_stop: || {},
            on_toggle: || {},
            on_status: || "idle".into(),
            on_metrics: || "metrics unavailable".into(),
            on_debug_status: || "debug unavailable".into(),
            on_tts_command: None::<fn(ControlCommand) -> String>,
        });
        let mut server = ControlServer::new(Some(sock.to_str().unwrap()), handlers).unwrap();
        server.start().unwrap();
        let path = server.socket_path().to_path_buf();
        let err = send_control_command_to(&path, ControlCommand::TtsSpeak, Duration::from_secs(1))
            .unwrap_err();
        assert!(matches!(err, ControlError::Remote(msg) if msg.contains("tts not available")));
        let _ = calls;
        server.stop();
    }

    #[test]
    fn handler_panic_does_not_kill_server() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("control.sock");
        let n = Arc::new(AtomicUsize::new(0));
        let n2 = Arc::clone(&n);
        let handlers: Arc<dyn ControlHandlers> = Arc::new(FnControlHandlers {
            on_start: move || {
                if n2.fetch_add(1, Ordering::SeqCst) == 0 {
                    panic!("first start panics");
                }
            },
            on_stop: || {},
            on_toggle: || {},
            on_status: || "idle".into(),
            on_metrics: || "m".into(),
            on_debug_status: || "d".into(),
            on_tts_command: None::<fn(ControlCommand) -> String>,
        });
        let mut server = ControlServer::new(Some(sock.to_str().unwrap()), handlers).unwrap();
        server.start().unwrap();
        let path = server.socket_path().to_path_buf();

        let err = send_control_command_to(&path, ControlCommand::Start, Duration::from_secs(1))
            .unwrap_err();
        assert!(matches!(err, ControlError::Remote(msg) if msg.contains("internal error")));

        // Server still alive.
        let resp =
            send_control_command_to(&path, ControlCommand::Ping, Duration::from_secs(1)).unwrap();
        assert_eq!(resp, "OK pong");

        let resp =
            send_control_command_to(&path, ControlCommand::Start, Duration::from_secs(1)).unwrap();
        assert_eq!(resp, "OK started");
        server.stop();
    }

    #[test]
    fn stop_is_bounded_with_slow_handler() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("control.sock");
        let handlers: Arc<dyn ControlHandlers> = Arc::new(FnControlHandlers {
            on_start: || thread::sleep(Duration::from_secs(30)),
            on_stop: || {},
            on_toggle: || {},
            on_status: || "idle".into(),
            on_metrics: || "m".into(),
            on_debug_status: || "d".into(),
            on_tts_command: None::<fn(ControlCommand) -> String>,
        });
        let mut server = ControlServer::new(Some(sock.to_str().unwrap()), handlers).unwrap();
        server.start().unwrap();
        let path = server.socket_path().to_path_buf();

        // Fire-and-forget a slow start in a client thread.
        let path2 = path.clone();
        thread::spawn(move || {
            let _ = send_control_command_to(&path2, ControlCommand::Start, Duration::from_secs(60));
        });
        thread::sleep(Duration::from_millis(100));

        let t0 = Instant::now();
        server.stop();
        assert!(
            t0.elapsed() < Duration::from_secs(3),
            "stop hung for {:?}",
            t0.elapsed()
        );
    }
}
