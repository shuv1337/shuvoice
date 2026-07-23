//! IPC end-to-end: real `shuvoice` child against a temporary ControlServer.
//!
//! Isolated under a private XDG runtime/config/data tree (serial + RAII env guard).

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;
use serial_test::serial;
use shuvoice_control::ControlHandlers;
use shuvoice_control::{ControlCommand, ControlServer};
use tempfile::TempDir;

/// RAII env restore for XDG isolation.
struct EnvGuard {
    saved: Vec<(String, Option<OsString>)>,
}

impl EnvGuard {
    fn set(pairs: &[(&str, PathBuf)]) -> Self {
        let mut saved = Vec::new();
        for (key, value) in pairs {
            saved.push(((*key).to_string(), std::env::var_os(key)));
            // SAFETY: tests run under #[serial]; single-threaded env mutation.
            unsafe {
                std::env::set_var(key, value);
            }
        }
        Self { saved }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, prev) in self.saved.drain(..) {
            // SAFETY: paired with set under #[serial].
            unsafe {
                match prev {
                    Some(v) => std::env::set_var(&key, v),
                    None => std::env::remove_var(&key),
                }
            }
        }
    }
}

struct TestState {
    status: Mutex<String>,
    starts: AtomicUsize,
    stops: AtomicUsize,
    toggles: AtomicUsize,
    tts: Mutex<Vec<String>>,
}

impl TestState {
    fn new(status: &str) -> Arc<Self> {
        Arc::new(Self {
            status: Mutex::new(status.into()),
            starts: AtomicUsize::new(0),
            stops: AtomicUsize::new(0),
            toggles: AtomicUsize::new(0),
            tts: Mutex::new(Vec::new()),
        })
    }
}

struct TestHandlers {
    state: Arc<TestState>,
}

impl ControlHandlers for TestHandlers {
    fn on_start(&self) {
        self.state.starts.fetch_add(1, Ordering::SeqCst);
        *self.state.status.lock().unwrap() = "recording".into();
    }

    fn on_stop(&self) {
        self.state.stops.fetch_add(1, Ordering::SeqCst);
        *self.state.status.lock().unwrap() = "idle".into();
    }

    fn on_toggle(&self) {
        self.state.toggles.fetch_add(1, Ordering::SeqCst);
        let mut s = self.state.status.lock().unwrap();
        if s.as_str() == "recording" {
            *s = "idle".into();
        } else {
            *s = "recording".into();
        }
    }

    fn on_status(&self) -> String {
        self.state.status.lock().unwrap().clone()
    }

    fn on_metrics(&self) -> String {
        "chunks=0".into()
    }

    fn on_debug_status(&self) -> String {
        "debug ok".into()
    }

    fn on_tts_command(&self, command: ControlCommand) -> String {
        self.state
            .tts
            .lock()
            .unwrap()
            .push(command.as_str().to_string());
        match command {
            ControlCommand::TtsSpeak => "OK tts_started source=selection".into(),
            ControlCommand::TtsSpeakClipboard => "OK tts_started source=clipboard".into(),
            ControlCommand::TtsStatus => "OK tts_idle".into(),
            ControlCommand::TtsStop => "OK tts_stopped".into(),
            ControlCommand::TtsPause => "OK tts_paused".into(),
            ControlCommand::TtsResume => "OK tts_resumed".into(),
            ControlCommand::TtsTogglePause => "OK tts_toggle_pause".into(),
            ControlCommand::TtsRestart => "OK tts_restarted".into(),
            _ => "ERROR tts not available".into(),
        }
    }
}

fn setup_xdg() -> (TempDir, EnvGuard, PathBuf) {
    let tmp = TempDir::new().unwrap();
    let config = tmp.path().join("config");
    let data = tmp.path().join("data");
    let runtime = tmp.path().join("runtime");
    std::fs::create_dir_all(&config).unwrap();
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&runtime).unwrap();
    // Minimal config so Config::load succeeds.
    let cfg_dir = config.join("shuvoice");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(cfg_dir.join("config.toml"), "config_version = 1\n").unwrap();

    let guard = EnvGuard::set(&[
        ("XDG_CONFIG_HOME", config),
        ("XDG_DATA_HOME", data),
        ("XDG_RUNTIME_DIR", runtime.clone()),
    ]);
    (tmp, guard, runtime)
}

fn start_server(runtime: &Path, state: Arc<TestState>) -> ControlServer {
    let sock = runtime.join("shuvoice").join("control.sock");
    std::fs::create_dir_all(sock.parent().unwrap()).unwrap();
    let handlers = Arc::new(TestHandlers { state });
    let mut server =
        ControlServer::new(Some(sock.to_str().unwrap()), handlers).expect("server new");
    server.start().expect("server start");
    server
}

fn shuvoice_control(args: &[&str]) -> assert_cmd::assert::Assert {
    let mut cmd = cargo_bin_cmd!("shuvoice");
    cmd.args(args);
    cmd.assert()
}

#[test]
#[serial]
fn control_start_status_stop_roundtrip() {
    let (_tmp, _guard, runtime) = setup_xdg();
    let state = TestState::new("idle");
    let mut server = start_server(&runtime, Arc::clone(&state));

    shuvoice_control(&["control", "start", "--control-wait-sec", "0"])
        .success()
        .stdout(predicate::str::contains("OK started"));
    assert_eq!(state.starts.load(Ordering::SeqCst), 1);

    shuvoice_control(&["control", "status", "--control-wait-sec", "0"])
        .success()
        .stdout(predicate::str::contains("OK recording"));

    shuvoice_control(&["control", "stop", "--control-wait-sec", "0"])
        .success()
        .stdout(predicate::str::contains("OK stopped"));
    assert_eq!(state.stops.load(Ordering::SeqCst), 1);

    shuvoice_control(&["control", "status"])
        .success()
        .stdout(predicate::str::contains("OK idle"));

    server.stop();
}

#[test]
#[serial]
fn control_toggle_and_ping() {
    let (_tmp, _guard, runtime) = setup_xdg();
    let state = TestState::new("idle");
    let mut server = start_server(&runtime, Arc::clone(&state));

    shuvoice_control(&["control", "ping"])
        .success()
        .stdout(predicate::str::contains("OK pong"));

    shuvoice_control(&["control", "toggle", "--control-wait-sec", "0"])
        .success()
        .stdout(predicate::str::contains("OK toggled"));
    assert_eq!(state.toggles.load(Ordering::SeqCst), 1);
    assert_eq!(state.status.lock().unwrap().as_str(), "recording");

    shuvoice_control(&["control", "toggle", "--control-wait-sec", "0"])
        .success()
        .stdout(predicate::str::contains("OK toggled"));
    assert_eq!(state.status.lock().unwrap().as_str(), "idle");

    server.stop();
}

#[test]
#[serial]
fn control_tts_selection_and_clipboard() {
    let (_tmp, _guard, runtime) = setup_xdg();
    let state = TestState::new("idle");
    let mut server = start_server(&runtime, Arc::clone(&state));

    shuvoice_control(&["control", "tts_speak", "--control-wait-sec", "0"])
        .success()
        .stdout(predicate::str::contains("OK tts_started source=selection"));

    shuvoice_control(&["control", "tts_speak_clipboard", "--control-wait-sec", "0"])
        .success()
        .stdout(predicate::str::contains("OK tts_started source=clipboard"));

    let tts = state.tts.lock().unwrap().clone();
    assert!(tts.iter().any(|c| c == "tts_speak"));
    assert!(tts.iter().any(|c| c == "tts_speak_clipboard"));

    shuvoice_control(&["control", "tts_status"])
        .success()
        .stdout(predicate::str::contains("OK tts_idle"));

    server.stop();
}

#[test]
#[serial]
fn control_missing_socket_errors_secret_safe() {
    let (_tmp, _guard, _runtime) = setup_xdg();
    // No server started — socket missing under isolated runtime.
    shuvoice_control(&["control", "ping", "--control-wait-sec", "0"])
        .failure()
        .code(predicate::eq(1))
        .stderr(
            predicate::str::contains("ERROR:")
                .and(predicate::str::contains("socket not found"))
                .and(predicate::str::contains("Is shuvoice running?"))
                // Secret-safe: no env dump markers.
                .and(predicate::str::contains("ELEVENLABS_API_KEY").not())
                .and(predicate::str::contains("OPENAI_API_KEY").not()),
        );
}

#[test]
#[serial]
fn control_metrics_and_debug_status() {
    let (_tmp, _guard, runtime) = setup_xdg();
    let state = TestState::new("idle");
    let mut server = start_server(&runtime, Arc::clone(&state));

    shuvoice_control(&["control", "metrics"])
        .success()
        .stdout(predicate::str::contains("OK chunks=0"));
    shuvoice_control(&["control", "debug_status"])
        .success()
        .stdout(predicate::str::contains("OK debug ok"));

    // diagnostics aggregates status/metrics/debug_status
    let mut cmd = cargo_bin_cmd!("shuvoice");
    cmd.args(["diagnostics", "--json"]);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("\"status\"").and(predicate::str::contains("OK")));

    server.stop();
}

#[test]
#[serial]
fn legacy_control_flag_still_works() {
    let (_tmp, _guard, runtime) = setup_xdg();
    let state = TestState::new("idle");
    let mut server = start_server(&runtime, Arc::clone(&state));

    let mut cmd = cargo_bin_cmd!("shuvoice");
    cmd.args(["--control", "ping"]);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("OK pong"));

    server.stop();
}
