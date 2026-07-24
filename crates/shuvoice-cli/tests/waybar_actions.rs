//! Waybar action-layer unit tests with injectable runners (no real systemctl/menu).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serial_test::serial;
use shuvoice_cli::waybar::{
    ACTION_LAUNCH_WIZARD, ACTION_MENU, ACTION_SERVICE_RESTART, ACTION_SERVICE_START,
    ACTION_SERVICE_STOP, ACTION_SERVICE_TOGGLE, ACTION_START_RECORD, ACTION_STOP_RECORD,
    ACTION_TOGGLE_DEBUG_OVERLAY, ACTION_TOGGLE_RECORD, BinaryLookup, Clock, ConfigWriter,
    ControlClient, ERR_NO_MENU, ERR_SOCKET_AFTER_RESTART, ERR_SOCKET_AFTER_START, MenuPrompt,
    ProcessLauncher, Sleeper, WaybarDeps, menu_options, perform_action, wait_for_control_socket,
};
use shuvoice_core::Config;
use shuvoice_io::process::{CommandRunner, RunOutput, ScriptedRunner};
use tempfile::TempDir;

// ─── fakes ──────────────────────────────────────────────────────────────────

struct FakeSleeper {
    calls: Mutex<Vec<Duration>>,
}

impl Sleeper for FakeSleeper {
    fn sleep(&self, duration: Duration) {
        self.calls.lock().unwrap().push(duration);
    }
}

/// Clock that advances by `step` on every `now()` after the first.
struct FakeClock {
    start: Instant,
    step: Duration,
    ticks: AtomicUsize,
}

impl Clock for FakeClock {
    fn now(&self) -> Instant {
        let n = self.ticks.fetch_add(1, Ordering::SeqCst);
        self.start + self.step * (n as u32)
    }
}

struct FakeControl {
    /// Queue of results per send (front popped).
    responses: Mutex<VecDeque<Result<String, String>>>,
    log: Mutex<Vec<String>>,
}

impl FakeControl {
    fn new(responses: Vec<Result<String, String>>) -> Arc<Self> {
        Arc::new(Self {
            responses: Mutex::new(responses.into()),
            log: Mutex::new(Vec::new()),
        })
    }

    fn log(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }
}

impl ControlClient for FakeControl {
    fn send(
        &self,
        command: &str,
        _socket_path: Option<&str>,
        _timeout: Duration,
    ) -> Result<String, String> {
        self.log.lock().unwrap().push(command.to_string());
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| {
                Err("control socket not found at /tmp/x. Is shuvoice running?".into())
            })
    }
}

struct FakeLauncher {
    calls: Mutex<Vec<(PathBuf, Vec<String>)>>,
    fail: AtomicBool,
}

impl ProcessLauncher for FakeLauncher {
    fn spawn_detached(&self, program: &Path, args: &[&str]) -> Result<(), String> {
        self.calls.lock().unwrap().push((
            program.to_path_buf(),
            args.iter().map(|s| (*s).to_string()).collect(),
        ));
        if self.fail.load(Ordering::SeqCst) {
            return Err("failed to launch wizard: simulated".into());
        }
        Ok(())
    }
}

struct FakeMenu {
    choice: Mutex<Option<String>>,
    fail_no_menu: AtomicBool,
}

impl MenuPrompt for FakeMenu {
    fn prompt(&self, _title: &str, _options: &[String]) -> Result<Option<String>, String> {
        if self.fail_no_menu.load(Ordering::SeqCst) {
            return Err(ERR_NO_MENU.into());
        }
        Ok(self.choice.lock().unwrap().clone())
    }
}

struct FakeConfigWriter {
    calls: Mutex<Vec<bool>>,
    fail: AtomicBool,
}

impl ConfigWriter for FakeConfigWriter {
    fn set_overlay_debug_mode(&self, enabled: bool) -> Result<(), String> {
        self.calls.lock().unwrap().push(enabled);
        if self.fail.load(Ordering::SeqCst) {
            return Err(format!(
                "failed to set overlay_debug_mode={}",
                if enabled { "true" } else { "false" }
            ));
        }
        Ok(())
    }
}

struct AlwaysMissingLookup;
impl BinaryLookup for AlwaysMissingLookup {
    fn exists(&self, _name: &str) -> bool {
        false
    }
}

fn base_deps(
    control: Arc<dyn ControlClient>,
    runner: Arc<dyn CommandRunner>,
) -> (
    WaybarDeps,
    Arc<FakeSleeper>,
    Arc<FakeLauncher>,
    Arc<FakeMenu>,
    Arc<FakeConfigWriter>,
) {
    let sleeper = Arc::new(FakeSleeper {
        calls: Mutex::new(Vec::new()),
    });
    let launcher = Arc::new(FakeLauncher {
        calls: Mutex::new(Vec::new()),
        fail: AtomicBool::new(false),
    });
    let menu = Arc::new(FakeMenu {
        choice: Mutex::new(None),
        fail_no_menu: AtomicBool::new(false),
    });
    let config_writer = Arc::new(FakeConfigWriter {
        calls: Mutex::new(Vec::new()),
        fail: AtomicBool::new(false),
    });
    let clock = Arc::new(FakeClock {
        start: Instant::now(),
        step: Duration::from_millis(100),
        ticks: AtomicUsize::new(0),
    });

    let deps = WaybarDeps {
        runner,
        control,
        sleeper: Arc::clone(&sleeper) as Arc<dyn Sleeper>,
        clock: clock as Arc<dyn Clock>,
        launcher: Arc::clone(&launcher) as Arc<dyn ProcessLauncher>,
        menu: Arc::clone(&menu) as Arc<dyn MenuPrompt>,
        config_writer: Arc::clone(&config_writer) as Arc<dyn ConfigWriter>,
        control_ready_timeout: Duration::from_millis(500),
        control_ready_poll: Duration::from_millis(50),
    };
    (deps, sleeper, launcher, menu, config_writer)
}

fn systemctl_scripted(active_state: &str) -> Arc<ScriptedRunner> {
    let state = active_state.to_string();
    let r = ScriptedRunner::new();
    r.set_dynamic(move |argv| {
        // systemctl --user show --property=ActiveState --value SERVICE
        if argv.iter().any(|a| a == "show") {
            return Ok(RunOutput {
                status_code: Some(0),
                stdout: format!("{state}\n").into_bytes(),
                stderr: Vec::new(),
                success: true,
            });
        }
        // start/stop/restart
        Ok(RunOutput {
            status_code: Some(0),
            stdout: Vec::new(),
            stderr: Vec::new(),
            success: true,
        })
    });
    Arc::new(r)
}

fn cfg() -> Config {
    let mut c = Config::default();
    c.control_socket = Some("/tmp/shuvoice-test/control.sock".into());
    c.overlay_debug_mode = false;
    c
}

// ─── menu option labels ─────────────────────────────────────────────────────

#[test]
fn menu_option_labels_match_public_contract() {
    let opts = menu_options("idle", "inactive", false);
    let labels: Vec<&str> = opts.iter().map(|(l, _)| l.as_str()).collect();
    assert_eq!(
        labels,
        [
            "Start recording",
            "Toggle recording",
            "Enable debug overlay",
            "Start service",
            "Relaunch setup wizard",
            "Restart service (advanced)",
        ]
    );
    let opts = menu_options("recording", "active", true);
    let labels: Vec<&str> = opts.iter().map(|(l, _)| l.as_str()).collect();
    assert_eq!(labels[0], "Stop recording");
    assert_eq!(labels[2], "Disable debug overlay");
    assert_eq!(labels[3], "Stop service");
    assert_eq!(opts[2].1, ACTION_TOGGLE_DEBUG_OVERLAY);
}

// ─── wait_for_control_socket ────────────────────────────────────────────────

#[test]
fn wait_for_control_socket_times_out() {
    let control = FakeControl::new(vec![
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
    ]);
    let runner = systemctl_scripted("inactive");
    let (deps, sleeper, _, _, _) = base_deps(control, runner);
    let config = cfg();
    assert!(!wait_for_control_socket(
        &deps,
        &config,
        Duration::from_millis(200)
    ));
    assert!(!sleeper.calls.lock().unwrap().is_empty());
}

#[test]
fn wait_for_control_socket_succeeds_on_ping() {
    let control = FakeControl::new(vec![Err("socket not found".into()), Ok("OK pong".into())]);
    let runner = systemctl_scripted("active");
    let (deps, _, _, _, _) = base_deps(control, runner);
    let config = cfg();
    assert!(wait_for_control_socket(
        &deps,
        &config,
        Duration::from_secs(2)
    ));
}

// ─── record actions ─────────────────────────────────────────────────────────

#[test]
fn start_record_waits_for_socket_then_starts() {
    let control = FakeControl::new(vec![
        // wait_for_control_socket pings
        Ok("OK pong".into()),
        // start
        Ok("OK started".into()),
    ]);
    let runner = systemctl_scripted("inactive");
    let (deps, _, _, _, _) = base_deps(control.clone(), runner);
    let mut config = cfg();
    perform_action(&deps, ACTION_START_RECORD, &mut config, "shuvoice.service").unwrap();
    let log = control.log();
    assert!(log.iter().any(|c| c == "ping"));
    assert!(log.iter().any(|c| c == "start"));
}

#[test]
fn start_record_errors_when_socket_never_returns() {
    let control = FakeControl::new(vec![
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
        Err("socket not found".into()),
    ]);
    let runner = systemctl_scripted("inactive");
    let (deps, _, _, _, _) = base_deps(control, runner);
    let mut config = cfg();
    let err =
        perform_action(&deps, ACTION_START_RECORD, &mut config, "shuvoice.service").unwrap_err();
    assert_eq!(err, ERR_SOCKET_AFTER_START);
}

#[test]
fn toggle_record_stops_when_recording() {
    let control = FakeControl::new(vec![
        Ok("OK recording".into()), // status
        Ok("OK stopped".into()),   // stop
    ]);
    let runner = systemctl_scripted("active");
    let (deps, _, _, _, _) = base_deps(control.clone(), runner);
    let mut config = cfg();
    perform_action(&deps, ACTION_TOGGLE_RECORD, &mut config, "shuvoice.service").unwrap();
    assert_eq!(
        control.log(),
        vec!["status".to_string(), "stop".to_string()]
    );
}

#[test]
fn toggle_record_starts_when_idle() {
    let control = FakeControl::new(vec![Ok("OK idle".into()), Ok("OK started".into())]);
    let runner = systemctl_scripted("active");
    let (deps, _, _, _, _) = base_deps(control.clone(), runner);
    let mut config = cfg();
    perform_action(&deps, ACTION_TOGGLE_RECORD, &mut config, "shuvoice.service").unwrap();
    assert_eq!(
        control.log(),
        vec!["status".to_string(), "start".to_string()]
    );
}

#[test]
fn toggle_record_starts_service_when_socket_missing() {
    let control = FakeControl::new(vec![
        Err("control socket not found at x. Is shuvoice running?".into()), // status
        Ok("OK pong".into()),                                              // wait ping
        Ok("OK started".into()),                                           // start
    ]);
    let runner = systemctl_scripted("inactive");
    let (deps, _, _, _, _) = base_deps(control.clone(), runner);
    let mut config = cfg();
    perform_action(&deps, ACTION_TOGGLE_RECORD, &mut config, "shuvoice.service").unwrap();
    let log = control.log();
    assert_eq!(log[0], "status");
    assert!(log.iter().any(|c| c == "ping"));
    assert!(log.iter().any(|c| c == "start"));
}

#[test]
fn stop_record_noop_when_service_inactive() {
    let control = FakeControl::new(vec![Err("socket not found".into())]);
    let runner = systemctl_scripted("inactive");
    let (deps, _, _, _, _) = base_deps(control, runner);
    let mut config = cfg();
    perform_action(&deps, ACTION_STOP_RECORD, &mut config, "shuvoice.service").unwrap();
}

// ─── service actions ────────────────────────────────────────────────────────

#[test]
fn service_start_waits_for_socket() {
    let control = FakeControl::new(vec![Ok("OK pong".into())]);
    let runner = systemctl_scripted("inactive");
    let (deps, _, _, _, _) = base_deps(control, runner);
    let mut config = cfg();
    perform_action(&deps, ACTION_SERVICE_START, &mut config, "shuvoice.service").unwrap();
}

#[test]
fn service_start_errors_when_socket_never_ready() {
    let control = FakeControl::new(vec![
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
    ]);
    let runner = systemctl_scripted("inactive");
    let (deps, _, _, _, _) = base_deps(control, runner);
    let mut config = cfg();
    let err =
        perform_action(&deps, ACTION_SERVICE_START, &mut config, "shuvoice.service").unwrap_err();
    assert_eq!(err, ERR_SOCKET_AFTER_START);
}

#[test]
fn service_restart_errors_with_exact_string_when_socket_missing() {
    let control = FakeControl::new(vec![
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
        Err("nope".into()),
    ]);
    let runner = systemctl_scripted("active");
    let (deps, _, _, _, _) = base_deps(control, runner);
    let mut config = cfg();
    let err = perform_action(
        &deps,
        ACTION_SERVICE_RESTART,
        &mut config,
        "shuvoice.service",
    )
    .unwrap_err();
    assert_eq!(err, ERR_SOCKET_AFTER_RESTART);
}

#[test]
fn service_toggle_stop_when_active() {
    let control = FakeControl::new(vec![]);
    let runner = systemctl_scripted("active");
    let (deps, _, _, _, _) = base_deps(control, runner);
    let mut config = cfg();
    perform_action(
        &deps,
        ACTION_SERVICE_TOGGLE,
        &mut config,
        "shuvoice.service",
    )
    .unwrap();
}

#[test]
fn service_stop_ok() {
    let control = FakeControl::new(vec![]);
    let runner = systemctl_scripted("active");
    let (deps, _, _, _, _) = base_deps(control, runner);
    let mut config = cfg();
    perform_action(&deps, ACTION_SERVICE_STOP, &mut config, "shuvoice.service").unwrap();
}

// ─── wizard launch ──────────────────────────────────────────────────────────

#[test]
fn launch_wizard_detached_uses_wizard_arg() {
    let control = FakeControl::new(vec![]);
    let runner = systemctl_scripted("inactive");
    let (deps, _, launcher, _, _) = base_deps(control, runner);
    let mut config = cfg();
    perform_action(&deps, ACTION_LAUNCH_WIZARD, &mut config, "shuvoice.service").unwrap();
    let calls = launcher.calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].1, vec!["wizard".to_string()]);
}

#[test]
fn launch_wizard_surfaces_failure() {
    let control = FakeControl::new(vec![]);
    let runner = systemctl_scripted("inactive");
    let (deps, _, launcher, _, _) = base_deps(control, runner);
    launcher.fail.store(true, Ordering::SeqCst);
    let mut config = cfg();
    let err =
        perform_action(&deps, ACTION_LAUNCH_WIZARD, &mut config, "shuvoice.service").unwrap_err();
    assert!(err.starts_with("failed to launch wizard:"), "{err}");
}

// ─── debug overlay toggle ───────────────────────────────────────────────────

#[test]
#[serial]
fn toggle_debug_overlay_writes_config_and_restarts() {
    // Isolate config writes.
    let tmp = TempDir::new().unwrap();
    let config_home = tmp.path().join("config");
    let data_home = tmp.path().join("data");
    let runtime = tmp.path().join("runtime");
    std::fs::create_dir_all(config_home.join("shuvoice")).unwrap();
    std::fs::create_dir_all(&data_home).unwrap();
    std::fs::create_dir_all(&runtime).unwrap();
    std::fs::write(
        config_home.join("shuvoice/config.toml"),
        "config_version = 1\n\n[overlay]\noverlay_debug_mode = false\n",
    )
    .unwrap();

    let prev_cfg = std::env::var_os("XDG_CONFIG_HOME");
    let prev_data = std::env::var_os("XDG_DATA_HOME");
    let prev_rt = std::env::var_os("XDG_RUNTIME_DIR");
    // SAFETY: this test is serialized, and all three process-global values
    // are restored before returning.
    unsafe {
        std::env::set_var("XDG_CONFIG_HOME", &config_home);
        std::env::set_var("XDG_DATA_HOME", &data_home);
        std::env::set_var("XDG_RUNTIME_DIR", &runtime);
    }

    let control = FakeControl::new(vec![Ok("OK pong".into())]); // wait after restart
    let runner = systemctl_scripted("active");
    let (mut deps, _, _, _, writer) = base_deps(control, runner);
    // Use real config writer for this test.
    deps.config_writer = Arc::new(shuvoice_cli::waybar::StdConfigWriter);

    let mut config = Config::load().unwrap();
    assert!(!config.overlay_debug_mode);
    perform_action(
        &deps,
        ACTION_TOGGLE_DEBUG_OVERLAY,
        &mut config,
        "shuvoice.service",
    )
    .unwrap();
    assert!(config.overlay_debug_mode);

    let reloaded = Config::load().unwrap();
    assert!(reloaded.overlay_debug_mode);

    // writer fake was replaced; just ensure real write happened.
    let _ = writer;

    // SAFETY: this test is serialized and restores the exact prior values.
    unsafe {
        match prev_cfg {
            Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
        match prev_data {
            Some(v) => std::env::set_var("XDG_DATA_HOME", v),
            None => std::env::remove_var("XDG_DATA_HOME"),
        }
        match prev_rt {
            Some(v) => std::env::set_var("XDG_RUNTIME_DIR", v),
            None => std::env::remove_var("XDG_RUNTIME_DIR"),
        }
    }
}

#[test]
fn toggle_debug_overlay_errors_on_write_failure() {
    let control = FakeControl::new(vec![]);
    let runner = systemctl_scripted("active");
    let (deps, _, _, _, writer) = base_deps(control, runner);
    writer.fail.store(true, Ordering::SeqCst);
    let mut config = cfg();
    let err = perform_action(
        &deps,
        ACTION_TOGGLE_DEBUG_OVERLAY,
        &mut config,
        "shuvoice.service",
    )
    .unwrap_err();
    assert_eq!(err, "failed to set overlay_debug_mode=true");
}

// ─── menu dispatch ──────────────────────────────────────────────────────────

#[test]
fn menu_dispatches_selected_action() {
    let control = FakeControl::new(vec![
        // query_runtime_state status for menu build
        Ok("OK idle".into()),
        // start-record path: wait ping + start
        Ok("OK pong".into()),
        Ok("OK started".into()),
    ]);
    let runner = systemctl_scripted("active");
    let (deps, _, _, menu, _) = base_deps(control.clone(), runner);
    *menu.choice.lock().unwrap() = Some("Start recording".into());
    let mut config = cfg();
    perform_action(&deps, ACTION_MENU, &mut config, "shuvoice.service").unwrap();
    let log = control.log();
    assert!(log.iter().any(|c| c == "start"), "{log:?}");
}

#[test]
fn menu_cancel_is_noop() {
    let control = FakeControl::new(vec![Ok("OK idle".into())]);
    let runner = systemctl_scripted("inactive");
    let (deps, _, _, menu, _) = base_deps(control, runner);
    *menu.choice.lock().unwrap() = None;
    let mut config = cfg();
    perform_action(&deps, ACTION_MENU, &mut config, "shuvoice.service").unwrap();
}

#[test]
fn menu_no_launcher_exact_error() {
    let control = FakeControl::new(vec![Ok("OK idle".into())]);
    let runner = systemctl_scripted("inactive");
    let (mut deps, _, _, _, _) = base_deps(control, runner);
    deps.menu = Arc::new(shuvoice_cli::waybar::StdMenuPrompt {
        lookup: Arc::new(AlwaysMissingLookup),
        runner: systemctl_scripted("inactive"),
    });
    let mut config = cfg();
    let err = perform_action(&deps, ACTION_MENU, &mut config, "shuvoice.service").unwrap_err();
    assert_eq!(err, ERR_NO_MENU);
}

#[test]
fn unknown_command_exact_error() {
    let control = FakeControl::new(vec![]);
    let runner = systemctl_scripted("inactive");
    let (deps, _, _, _, _) = base_deps(control, runner);
    let mut config = cfg();
    let err =
        perform_action(&deps, "not-a-real-action", &mut config, "shuvoice.service").unwrap_err();
    assert_eq!(err, "Unknown command: not-a-real-action");
}
