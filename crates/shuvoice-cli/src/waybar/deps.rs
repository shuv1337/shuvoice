//! Injectable seams for Waybar actions (tests avoid real systemctl/menu/service).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use shuvoice_io::process::{CommandRunner, StdCommandRunner};

use crate::control::{ControlCmd, send_cmd_str};

/// Wall-clock sleeper (injectable for tests).
pub trait Sleeper: Send + Sync {
    fn sleep(&self, duration: Duration);
}

/// Default sleeper using the OS clock.
#[derive(Debug, Default, Clone, Copy)]
pub struct StdSleeper;

impl Sleeper for StdSleeper {
    fn sleep(&self, duration: Duration) {
        std::thread::sleep(duration);
    }
}

/// Monotonic clock (injectable for tests).
pub trait Clock: Send + Sync {
    fn now(&self) -> std::time::Instant;
}

/// Default clock.
#[derive(Debug, Default, Clone, Copy)]
pub struct StdClock;

impl Clock for StdClock {
    fn now(&self) -> std::time::Instant {
        std::time::Instant::now()
    }
}

/// Detached process launcher (wizard).
pub trait ProcessLauncher: Send + Sync {
    fn spawn_detached(&self, program: &Path, args: &[&str]) -> Result<(), String>;
}

/// Default launcher using `std::process::Command`.
#[derive(Debug, Default, Clone, Copy)]
pub struct StdProcessLauncher;

impl ProcessLauncher for StdProcessLauncher {
    fn spawn_detached(&self, program: &Path, args: &[&str]) -> Result<(), String> {
        std::process::Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| format!("failed to launch wizard: {e}"))?;
        Ok(())
    }
}

/// Control-socket client seam.
pub trait ControlClient: Send + Sync {
    fn send(
        &self,
        command: &str,
        socket_path: Option<&str>,
        timeout: Duration,
    ) -> Result<String, String>;
}

/// Default client over `shuvoice-control`.
#[derive(Debug, Default, Clone, Copy)]
pub struct StdControlClient;

impl ControlClient for StdControlClient {
    fn send(
        &self,
        command: &str,
        socket_path: Option<&str>,
        timeout: Duration,
    ) -> Result<String, String> {
        send_cmd_str(command, socket_path, Some(timeout))
    }
}

/// Menu prompt seam (dmenu/wofi/etc.).
pub trait MenuPrompt: Send + Sync {
    /// Present `options` and return the selected label, or `None` on cancel.
    fn prompt(&self, title: &str, options: &[String]) -> Result<Option<String>, String>;
}

/// PATH lookup seam.
pub trait BinaryLookup: Send + Sync {
    fn exists(&self, name: &str) -> bool;
}

/// Default PATH lookup via `which`.
#[derive(Debug, Default, Clone, Copy)]
pub struct WhichLookup;

impl BinaryLookup for WhichLookup {
    fn exists(&self, name: &str) -> bool {
        which::which(name).is_ok()
    }
}

/// Default menu prompt with stable launcher ordering and labels.
#[derive(Clone)]
pub struct StdMenuPrompt {
    pub lookup: Arc<dyn BinaryLookup>,
    pub runner: Arc<dyn CommandRunner>,
}

impl Default for StdMenuPrompt {
    fn default() -> Self {
        Self {
            lookup: Arc::new(WhichLookup),
            runner: Arc::new(StdCommandRunner),
        }
    }
}

/// Ordered menu launchers (binary, argv template with `{prompt}`).
pub const MENU_LAUNCHERS: &[(&str, &[&str])] = &[
    (
        "omarchy-launch-walker",
        &["omarchy-launch-walker", "--dmenu", "-p", "{prompt}"],
    ),
    ("walker", &["walker", "--dmenu", "-p", "{prompt}"]),
    ("wofi", &["wofi", "--dmenu", "--prompt", "{prompt}"]),
    ("rofi", &["rofi", "-dmenu", "-p", "{prompt}"]),
    ("bemenu", &["bemenu", "-p", "{prompt}"]),
    ("dmenu", &["dmenu", "-p", "{prompt}"]),
];

impl MenuPrompt for StdMenuPrompt {
    fn prompt(&self, title: &str, options: &[String]) -> Result<Option<String>, String> {
        let menu_input = options.join("\n") + "\n";
        for (binary, template) in MENU_LAUNCHERS {
            if !self.lookup.exists(binary) {
                continue;
            }
            let argv: Vec<String> = template
                .iter()
                .map(|arg| arg.replace("{prompt}", title))
                .collect();
            let opts = shuvoice_io::process::RunOptions {
                timeout: Duration::from_secs(20),
                stdin_data: Some(menu_input.as_bytes().to_vec()),
                capture_stdout: true,
                capture_stderr: true,
                check: false,
                ..shuvoice_io::process::RunOptions::default()
            };
            let out = self
                .runner
                .run(&argv, &opts)
                .map_err(|e| format!("{binary} failed: {e}"))?;
            if !out.success {
                // User cancel is common (Esc / click outside).
                return Ok(None);
            }
            let selection = out.stdout_text().trim().to_string();
            return Ok(if selection.is_empty() {
                None
            } else {
                Some(selection)
            });
        }
        Err(
            "No menu launcher found (install/use omarchy-launch-walker, walker, wofi, rofi, bemenu, or dmenu)"
                .into(),
        )
    }
}

/// Config mutation seam (debug overlay toggle).
pub trait ConfigWriter: Send + Sync {
    fn set_overlay_debug_mode(&self, enabled: bool) -> Result<(), String>;
}

/// Default writer using core config I/O (atomic set of `overlay_debug_mode`).
#[derive(Debug, Default, Clone, Copy)]
pub struct StdConfigWriter;

impl ConfigWriter for StdConfigWriter {
    fn set_overlay_debug_mode(&self, enabled: bool) -> Result<(), String> {
        use crate::config::cmd_set;
        use crate::error::EXIT_SUCCESS;
        use crate::parser::{ConfigSetKey, ConfigSetValue};

        let value = if enabled {
            ConfigSetValue::True
        } else {
            ConfigSetValue::False
        };
        let status = cmd_set(ConfigSetKey::OverlayDebugMode, value);
        if status.code != EXIT_SUCCESS {
            return Err(format!(
                "failed to set overlay_debug_mode={}",
                if enabled { "true" } else { "false" }
            ));
        }
        Ok(())
    }
}

/// Resolve the `shuvoice` binary used for detached wizard launch.
pub fn resolve_shuvoice_bin() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let sibling = parent.join("shuvoice");
            if sibling.exists() {
                return sibling;
            }
        }
        return exe;
    }
    PathBuf::from("shuvoice")
}

/// Bundle of injectable dependencies for Waybar actions.
#[derive(Clone)]
pub struct WaybarDeps {
    pub runner: Arc<dyn CommandRunner>,
    pub control: Arc<dyn ControlClient>,
    pub sleeper: Arc<dyn Sleeper>,
    pub clock: Arc<dyn Clock>,
    pub launcher: Arc<dyn ProcessLauncher>,
    pub menu: Arc<dyn MenuPrompt>,
    pub config_writer: Arc<dyn ConfigWriter>,
    /// Bounded wait used after service start/restart before control calls.
    pub control_ready_timeout: Duration,
    /// Poll interval while waiting for the control socket.
    pub control_ready_poll: Duration,
}

impl Default for WaybarDeps {
    fn default() -> Self {
        let runner: Arc<dyn CommandRunner> = Arc::new(StdCommandRunner);
        Self {
            runner: Arc::clone(&runner),
            control: Arc::new(StdControlClient),
            sleeper: Arc::new(StdSleeper),
            clock: Arc::new(StdClock),
            launcher: Arc::new(StdProcessLauncher),
            menu: Arc::new(StdMenuPrompt {
                lookup: Arc::new(WhichLookup),
                runner,
            }),
            config_writer: Arc::new(StdConfigWriter),
            control_ready_timeout: Duration::from_secs(2),
            control_ready_poll: Duration::from_millis(80),
        }
    }
}

impl WaybarDeps {
    /// Convenience: send a typed control command.
    pub fn send_control(
        &self,
        cmd: ControlCmd,
        socket: Option<&str>,
        timeout: Duration,
    ) -> Result<String, String> {
        self.control.send(cmd.as_str(), socket, timeout)
    }
}
