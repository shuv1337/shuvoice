//! Streaming text injection via wtype / xdotool / ydotool / wl-clipboard.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use tracing::{debug, error, info, warn};

use crate::error::ProcessError;
use crate::inject::error::{CommitOutcome, InjectError};
use crate::inject::sanitize_final_injection_text;
use crate::process::{CommandRunner, RunOptions, StdCommandRunner, argv};

const BACKSPACE_BATCH_SIZE: usize = 50;
const YDOTOOL_KEY_DELAY_MS: u32 = 0;
const YDOTOOL_HOLD_DELAY_MS: u32 = 0;
const KEY_BACKSPACE: u32 = 14;
const KEY_LEFTCTRL: u32 = 29;
const KEY_V: u32 = 47;
/// ARG_MAX headroom: above this, fail closed for xdotool argv (no stdin path).
pub const MAX_ARGV_PAYLOAD_BYTES: usize = 100_000;

/// Final text injection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FinalInjectionMode {
    #[default]
    Auto,
    Clipboard,
    Direct,
}

impl FinalInjectionMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "clipboard" => Some(Self::Clipboard),
            "direct" => Some(Self::Direct),
            _ => None,
        }
    }
}

/// Typer configuration (Python `StreamingTyper` ctor parity).
#[derive(Debug, Clone)]
pub struct TyperConfig {
    pub final_injection_mode: FinalInjectionMode,
    pub preserve_clipboard: bool,
    pub clipboard_settle_delay: Duration,
    pub retry_attempts: u32,
    pub retry_delay: Duration,
    pub subprocess_timeout: Duration,
}

impl Default for TyperConfig {
    fn default() -> Self {
        Self {
            final_injection_mode: FinalInjectionMode::Auto,
            preserve_clipboard: false,
            clipboard_settle_delay: Duration::from_millis(40),
            retry_attempts: 2,
            retry_delay: Duration::from_millis(40),
            subprocess_timeout: Duration::from_secs(5),
        }
    }
}

/// Sleep abstraction (settle / retry delays) — injectable for tests.
pub trait Sleeper: Send + Sync {
    fn sleep(&self, duration: Duration);
}

/// Default sleeper using `std::thread::sleep`.
#[derive(Debug, Default, Clone, Copy)]
pub struct StdSleeper;

impl Sleeper for StdSleeper {
    fn sleep(&self, duration: Duration) {
        if !duration.is_zero() {
            std::thread::sleep(duration);
        }
    }
}

/// Recording sleeper for tests (never blocks).
#[derive(Debug, Default, Clone)]
pub struct RecordingSleeper {
    inner: Arc<Mutex<Vec<Duration>>>,
}

impl RecordingSleeper {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn sleeps(&self) -> Vec<Duration> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

impl Sleeper for RecordingSleeper {
    fn sleep(&self, duration: Duration) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(duration);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InjectBackend {
    Xdotool,
    Ydotool,
    Wtype,
}

/// Inject text into the focused window.
pub struct StreamingTyper {
    cfg: TyperConfig,
    runner: Arc<dyn CommandRunner>,
    sleeper: Arc<dyn Sleeper>,
    /// When set, error/warn lines are also pushed here (tests; never includes payload).
    log_capture: Option<Arc<Mutex<Vec<String>>>>,
    pub last_partial_len: usize,
    pub last_partial_text: String,
    watchers_detected: Option<bool>,
    watchers_checked_at: Option<Instant>,
    watchers_ttl: Duration,
    xdotool_available: Option<bool>,
    ydotool_available: Option<bool>,
    active_window: Option<Value>,
    active_window_checked_at: Option<Instant>,
    active_window_ttl: Duration,
    /// Optional overrides for tests.
    pub force_xwayland: Option<bool>,
    pub force_xdotool_window_id: Option<String>,
}

impl StreamingTyper {
    #[must_use]
    pub fn new(cfg: TyperConfig, runner: Arc<dyn CommandRunner>) -> Self {
        Self::new_with_sleeper(cfg, runner, Arc::new(StdSleeper))
    }

    #[must_use]
    pub fn new_with_sleeper(
        cfg: TyperConfig,
        runner: Arc<dyn CommandRunner>,
        sleeper: Arc<dyn Sleeper>,
    ) -> Self {
        Self {
            cfg,
            runner,
            sleeper,
            log_capture: None,
            last_partial_len: 0,
            last_partial_text: String::new(),
            watchers_detected: None,
            watchers_checked_at: None,
            watchers_ttl: Duration::from_secs(30),
            xdotool_available: None,
            ydotool_available: None,
            active_window: None,
            active_window_checked_at: None,
            active_window_ttl: Duration::from_secs(1),
            force_xwayland: None,
            force_xdotool_window_id: None,
        }
    }

    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(TyperConfig::default(), Arc::new(StdCommandRunner))
    }

    /// Install a log capture sink (tests only). Captured lines never include payload text.
    pub fn set_log_capture(&mut self, sink: Arc<Mutex<Vec<String>>>) {
        self.log_capture = Some(sink);
    }

    fn emit_error(&self, msg: String) {
        if let Some(cap) = &self.log_capture {
            cap.lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(msg.clone());
        }
        error!("{msg}");
    }

    fn emit_warn(&self, msg: String) {
        if let Some(cap) = &self.log_capture {
            cap.lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(msg.clone());
        }
        warn!("{msg}");
    }

    fn run_opts(&self) -> RunOptions {
        RunOptions {
            timeout: self.cfg.subprocess_timeout.max(Duration::from_secs(1)),
            check: true,
            ..RunOptions::default()
        }
    }

    fn run(&self, args: Vec<String>, op: &str, attempts: Option<u32>) -> bool {
        self.run_with_opts(args, None, op, attempts)
    }

    fn run_with_stdin(
        &self,
        args: Vec<String>,
        stdin: Vec<u8>,
        op: &str,
        attempts: Option<u32>,
    ) -> bool {
        self.run_with_opts(args, Some(stdin), op, attempts)
    }

    fn run_with_opts(
        &self,
        args: Vec<String>,
        stdin: Option<Vec<u8>>,
        op: &str,
        attempts: Option<u32>,
    ) -> bool {
        let attempts = attempts.unwrap_or(self.cfg.retry_attempts).max(1);
        let mut opts = self.run_opts();
        opts.stdin_data = stdin;
        for attempt in 1..=attempts {
            match self.runner.run(&args, &opts) {
                Ok(_) => return true,
                Err(err) => {
                    // Never format args / stdin into the message.
                    let err_msg = redact_process_error(&err, args.first().map(String::as_str));
                    if attempt == attempts {
                        self.emit_error(format!(
                            "{op} failed after {attempts} attempt(s): {err_msg}"
                        ));
                        return false;
                    }
                    self.emit_warn(format!(
                        "{op} attempt {attempt}/{attempts} failed: {err_msg}"
                    ));
                    if !self.cfg.retry_delay.is_zero() {
                        self.sleeper.sleep(self.cfg.retry_delay);
                    }
                }
            }
        }
        false
    }

    fn xdotool_installed(&mut self) -> bool {
        if self.xdotool_available.is_none() {
            self.xdotool_available = Some(which::which("xdotool").is_ok());
        }
        self.xdotool_available.unwrap_or(false)
    }

    fn ydotool_installed(&mut self) -> bool {
        if self.ydotool_available.is_none() {
            self.ydotool_available = Some(which::which("ydotool").is_ok());
        }
        self.ydotool_available.unwrap_or(false)
    }

    /// Test helper: force tool availability flags.
    pub fn set_tool_availability(&mut self, xdotool: bool, ydotool: bool) {
        self.xdotool_available = Some(xdotool);
        self.ydotool_available = Some(ydotool);
    }

    fn detect_active_window(&mut self) -> &Value {
        let now = Instant::now();
        let fresh = self
            .active_window_checked_at
            .is_some_and(|t| now.duration_since(t) < self.active_window_ttl);
        if !(self.active_window.is_some() && fresh) {
            let payload = {
                let args = argv(["hyprctl", "activewindow", "-j"]);
                match self.runner.run(&args, &self.run_opts()) {
                    Ok(out) => serde_json::from_slice::<Value>(&out.stdout)
                        .unwrap_or(Value::Object(Default::default())),
                    Err(err) => {
                        debug!("Could not inspect active window for injection backend: {err}");
                        Value::Object(Default::default())
                    }
                }
            };
            self.active_window = Some(payload);
            self.active_window_checked_at = Some(now);
        }
        self.active_window
            .as_ref()
            .expect("active_window populated")
    }

    fn active_window_is_xwayland(&mut self) -> bool {
        if let Some(force) = self.force_xwayland {
            return force;
        }
        self.detect_active_window()
            .get("xwayland")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    fn active_xdotool_window_id(&mut self) -> Option<String> {
        if let Some(id) = &self.force_xdotool_window_id {
            return Some(id.clone());
        }
        let pid = self
            .detect_active_window()
            .get("pid")
            .and_then(Value::as_i64)?;
        if pid <= 0 {
            return None;
        }
        let args = argv([
            "xdotool",
            "search",
            "--onlyvisible",
            "--pid",
            &pid.to_string(),
        ]);
        match self.runner.run(&args, &self.run_opts()) {
            Ok(out) => {
                let text = out.stdout_lossy();
                text.lines()
                    .map(str::trim)
                    .rfind(|l| !l.is_empty())
                    .map(ToOwned::to_owned)
            }
            Err(err) => {
                debug!("Could not resolve X11 window id for pid {pid}: {err}");
                None
            }
        }
    }

    fn prefer_xdotool(&mut self) -> bool {
        self.xdotool_installed() && self.active_window_is_xwayland()
    }

    fn prefer_ydotool(&mut self) -> bool {
        self.ydotool_installed() && self.active_window_is_xwayland()
    }

    fn select_backend(&mut self) -> InjectBackend {
        if self.prefer_xdotool() {
            InjectBackend::Xdotool
        } else if self.prefer_ydotool() {
            InjectBackend::Ydotool
        } else {
            InjectBackend::Wtype
        }
    }

    fn backspace_args(count: usize) -> Vec<String> {
        let mut args = argv(["wtype"]);
        for _ in 0..count {
            args.push("-k".into());
            args.push("BackSpace".into());
        }
        args
    }

    fn ydotool_backspace_args(count: usize) -> Vec<String> {
        let mut args = argv(["ydotool", "key", "-d", &YDOTOOL_KEY_DELAY_MS.to_string()]);
        for _ in 0..count {
            args.push(format!("{KEY_BACKSPACE}:1"));
            args.push(format!("{KEY_BACKSPACE}:0"));
        }
        args
    }

    fn xdotool_key_args(
        key_sequence: &str,
        window_id: Option<&str>,
        repeat: Option<usize>,
    ) -> Vec<String> {
        let mut args = argv(["xdotool", "key", "--clearmodifiers", "--delay", "0"]);
        if let Some(id) = window_id {
            args.push("--window".into());
            args.push(id.to_string());
        }
        if let Some(r) = repeat.filter(|r| *r > 1) {
            args.push("--repeat".into());
            args.push(r.to_string());
            args.push("--repeat-delay".into());
            args.push("0".into());
        }
        // Terminate options before the key sequence (hardening vs Python).
        args.push("--".into());
        args.push(key_sequence.into());
        args
    }

    /// Safe xdotool type argv: options terminated, payload via `--args 1`.
    fn xdotool_type_args(text: &str, window_id: Option<&str>) -> Vec<String> {
        let mut args = argv(["xdotool", "type", "--clearmodifiers", "--delay", "0"]);
        if let Some(id) = window_id {
            args.push("--window".into());
            args.push(id.to_string());
        }
        args.push("--args".into());
        args.push("1".into());
        args.push(text.into());
        args
    }

    fn send_backspaces_via_xdotool(&mut self, count: usize, op: &str) -> bool {
        if count == 0 {
            return true;
        }
        let window_id = self.active_xdotool_window_id();
        let mut remaining = count;
        while remaining > 0 {
            let batch = remaining.min(BACKSPACE_BATCH_SIZE);
            let args = Self::xdotool_key_args("BackSpace", window_id.as_deref(), Some(batch));
            if !self.run(args, op, Some(1)) {
                return false;
            }
            remaining -= batch;
        }
        true
    }

    fn send_backspaces_via_ydotool(&mut self, count: usize, op: &str) -> bool {
        if count == 0 {
            return true;
        }
        let mut remaining = count;
        while remaining > 0 {
            let batch = remaining.min(BACKSPACE_BATCH_SIZE);
            if !self.run(Self::ydotool_backspace_args(batch), op, None) {
                return false;
            }
            remaining -= batch;
        }
        true
    }

    fn send_backspaces_backend(&mut self, count: usize, backend: InjectBackend, op: &str) -> bool {
        if count == 0 {
            return true;
        }
        match backend {
            InjectBackend::Xdotool => self.send_backspaces_via_xdotool(count, op),
            InjectBackend::Ydotool => self.send_backspaces_via_ydotool(count, op),
            InjectBackend::Wtype => {
                let mut remaining = count;
                while remaining > 0 {
                    let batch = remaining.min(BACKSPACE_BATCH_SIZE);
                    if !self.run(Self::backspace_args(batch), op, None) {
                        return false;
                    }
                    remaining -= batch;
                }
                true
            }
        }
    }

    fn send_backspaces(&mut self, count: usize, op: &str) -> bool {
        if count == 0 {
            return true;
        }
        let primary = self.select_backend();
        if self.send_backspaces_backend(count, primary, op) {
            return true;
        }
        // Fallback order mirrors type_direct: wtype → xdotool(XWayland) → ydotool.
        for backend in [
            InjectBackend::Wtype,
            InjectBackend::Xdotool,
            InjectBackend::Ydotool,
        ] {
            if backend == primary {
                continue;
            }
            match backend {
                InjectBackend::Xdotool
                    if !self.xdotool_installed() || !self.active_window_is_xwayland() =>
                {
                    continue;
                }
                InjectBackend::Ydotool if !self.ydotool_installed() => continue,
                _ => {}
            }
            if self.send_backspaces_backend(count, backend, op) {
                return true;
            }
        }
        false
    }

    fn type_text_backend(&mut self, text: &str, backend: InjectBackend, op: &str) -> bool {
        if text.is_empty() {
            return true;
        }
        // All current type backends put text on argv (no stdin type path).
        // Fail closed above ARG_MAX headroom rather than risk truncation/injection.
        if text.len() > MAX_ARGV_PAYLOAD_BYTES {
            self.emit_warn(format!("{op}: payload too large for argv injection"));
            return false;
        }
        match backend {
            InjectBackend::Xdotool => {
                info!("Using xdotool direct typing for focused XWayland window.");
                let wid = self.active_xdotool_window_id();
                self.run(Self::xdotool_type_args(text, wid.as_deref()), op, None)
            }
            InjectBackend::Ydotool => {
                info!("Using ydotool direct typing for focused XWayland window.");
                let args = argv([
                    "ydotool",
                    "type",
                    "--key-delay",
                    &YDOTOOL_KEY_DELAY_MS.to_string(),
                    "--key-hold",
                    &YDOTOOL_HOLD_DELAY_MS.to_string(),
                    "--",
                    text,
                ]);
                self.run(args, op, None)
            }
            InjectBackend::Wtype => self.run(argv(["wtype", "--", text]), op, None),
        }
    }

    fn type_direct(&mut self, text: &str) -> bool {
        if text.is_empty() {
            return true;
        }
        if text.len() > MAX_ARGV_PAYLOAD_BYTES {
            self.emit_warn("direct type: payload too large for argv injection".into());
            return false;
        }
        let primary = self.select_backend();
        if self.type_text_backend(text, primary, "direct type") {
            return true;
        }
        // Python order when preferred path fails: wtype → xdotool(XWayland) → ydotool.
        for backend in [
            InjectBackend::Wtype,
            InjectBackend::Xdotool,
            InjectBackend::Ydotool,
        ] {
            if backend == primary {
                continue;
            }
            match backend {
                InjectBackend::Xdotool
                    if !self.xdotool_installed() || !self.active_window_is_xwayland() =>
                {
                    continue;
                }
                InjectBackend::Ydotool if !self.ydotool_installed() => continue,
                _ => {}
            }
            if self.type_text_backend(text, backend, "direct type") {
                return true;
            }
        }
        false
    }

    fn backspace_partial(&mut self) -> bool {
        let n = self.last_partial_len;
        self.send_backspaces(n, "partial backspace")
    }

    /// Unicode-scalar common-prefix length (Python `len` parity for partial tracking).
    #[must_use]
    pub fn common_prefix_char_len(left: &str, right: &str) -> usize {
        left.chars()
            .zip(right.chars())
            .take_while(|(a, b)| a == b)
            .count()
    }

    fn detect_clipboard_watchers(&mut self) -> bool {
        let now = Instant::now();
        if let (Some(v), Some(t)) = (self.watchers_detected, self.watchers_checked_at)
            && now.duration_since(t) < self.watchers_ttl
        {
            return v;
        }
        // Anchored patterns — avoid bare "elephant" false positives where possible.
        let args = argv([
            "pgrep",
            "-a",
            "-f",
            r"wl-paste[[:space:]]+--watch|wl-clip-persist|[[:space:]]elephant([[:space:]]|$)",
        ]);
        let opts = RunOptions {
            check: false,
            timeout: self.cfg.subprocess_timeout,
            ..RunOptions::default()
        };
        let detected = match self.runner.run(&args, &opts) {
            Ok(out) => out.success,
            Err(err) => {
                debug!("Failed to detect clipboard watchers: {err}");
                false
            }
        };
        if detected {
            info!("Detected clipboard watcher(s), enabling direct final typing.");
        }
        self.watchers_detected = Some(detected);
        self.watchers_checked_at = Some(now);
        detected
    }

    /// Test helper to force watcher detection cache.
    pub fn set_watchers_detected(&mut self, detected: bool) {
        self.watchers_detected = Some(detected);
        self.watchers_checked_at = Some(Instant::now());
    }

    fn wl_copy_set(&mut self, text: &str, op: &str) -> bool {
        // Prefer stdin to avoid ARG_MAX and /proc cmdline exposure of transcripts.
        self.run_with_stdin(argv(["wl-copy", "--"]), text.as_bytes().to_vec(), op, None)
    }

    fn paste_via_clipboard(&mut self, text: &str) -> bool {
        if text.is_empty() {
            return true;
        }
        if !self.wl_copy_set(text, "wl-copy set") {
            return false;
        }
        if !self.cfg.clipboard_settle_delay.is_zero() {
            self.sleeper.sleep(self.cfg.clipboard_settle_delay);
        }

        let primary = self.select_backend();
        if self.paste_key_backend(primary) {
            return true;
        }
        for backend in [
            InjectBackend::Wtype,
            InjectBackend::Xdotool,
            InjectBackend::Ydotool,
        ] {
            if backend == primary {
                continue;
            }
            match backend {
                InjectBackend::Xdotool
                    if !self.xdotool_installed() || !self.active_window_is_xwayland() =>
                {
                    continue;
                }
                InjectBackend::Ydotool if !self.ydotool_installed() => continue,
                _ => {}
            }
            if self.paste_key_backend(backend) {
                return true;
            }
        }
        false
    }

    fn paste_key_backend(&mut self, backend: InjectBackend) -> bool {
        match backend {
            InjectBackend::Xdotool => {
                info!("Using xdotool Ctrl+V paste for focused XWayland window.");
                let wid = self.active_xdotool_window_id();
                self.run(
                    Self::xdotool_key_args("ctrl+v", wid.as_deref(), None),
                    "xdotool ctrl+v",
                    None,
                )
            }
            InjectBackend::Ydotool => {
                info!("Using ydotool Ctrl+V paste for focused XWayland window.");
                let args = argv([
                    "ydotool",
                    "key",
                    "-d",
                    &YDOTOOL_KEY_DELAY_MS.to_string(),
                    &format!("{KEY_LEFTCTRL}:1"),
                    &format!("{KEY_V}:1"),
                    &format!("{KEY_V}:0"),
                    &format!("{KEY_LEFTCTRL}:0"),
                ]);
                self.run(args, "ydotool ctrl+v", None)
            }
            InjectBackend::Wtype => self.run(
                argv(["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"]),
                "wtype ctrl+v",
                None,
            ),
        }
    }

    /// Capture clipboard for preservation.
    ///
    /// Returns:
    /// - `None` if capture failed (missing tool / hard error) — do not clear on restore
    /// - `Some("")` if clipboard was empty
    /// - `Some(content)` if clipboard had content
    fn capture_clipboard(&self) -> Option<String> {
        let args = argv(["wl-paste", "--no-newline"]);
        let mut opts = self.run_opts();
        opts.check = false;
        match self.runner.run(&args, &opts) {
            Ok(out) if out.success => Some(out.stdout_lossy()),
            Ok(_) => {
                // Non-zero exit typically means empty selection on wl-paste.
                Some(String::new())
            }
            Err(err) => {
                debug!("Could not capture clipboard for preservation: {err}");
                None
            }
        }
    }

    /// Restore previously captured clipboard content.
    ///
    /// Returns `true` when restore is unnecessary or succeeds. Returns `false`
    /// only when a restore command was attempted and failed. Capture-miss
    /// (`prior == None`) intentionally leaves the clipboard alone and returns
    /// `true` — we cannot restore what we never captured.
    fn restore_clipboard(&mut self, prior: Option<&str>) -> bool {
        if !self.cfg.preserve_clipboard {
            return true;
        }
        match prior {
            None => true, // capture failed — leave clipboard alone
            Some("") => self.run(argv(["wl-copy", "--clear"]), "wl-copy clear", Some(1)),
            Some(content) => self.run_with_stdin(
                argv(["wl-copy", "--"]),
                content.as_bytes().to_vec(),
                "wl-copy restore",
                Some(1),
            ),
        }
    }

    /// Replace previous partial text using a diff-based suffix update.
    ///
    /// Backspace and insert use the **same** backend selection so XWayland
    /// windows do not receive split xdotool/wtype sequences.
    ///
    /// # Errors
    ///
    /// Returns [`InjectError`] when backspace or type fails. Local tracking is
    /// updated to match what is believed to be on screen so a subsequent call
    /// (or commit retry) can re-diff honestly instead of double-inserting.
    ///
    /// - Backspace failure: prior `last_partial_*` is left unchanged.
    /// - Type failure after a successful backspace: `last_partial_*` becomes the
    ///   retained common-prefix (what remains after the erase).
    pub fn update_partial(&mut self, new_text: &str) -> Result<(), InjectError> {
        let old_text = self.last_partial_text.clone();
        if new_text.is_empty() && old_text.is_empty() {
            return Ok(());
        }
        let common_bytes = common_prefix_bytes(&old_text, new_text);
        let common_text = &old_text[..common_bytes];
        let to_delete_chars = old_text[common_bytes..].chars().count();
        let to_insert = &new_text[common_bytes..];
        let backend = self.select_backend();

        if to_delete_chars > 0
            && !self.send_backspaces_backend(to_delete_chars, backend, "partial backspace")
        {
            // Screen still shows old_text — keep tracking so retry can re-diff.
            return Err(InjectError::PartialBackspace);
        }
        if !to_insert.is_empty() {
            if to_insert.len() > MAX_ARGV_PAYLOAD_BYTES {
                // Backspace already applied; track the retained prefix only.
                self.last_partial_text = common_text.to_string();
                self.last_partial_len = common_text.chars().count();
                self.emit_warn("partial type: payload too large for argv injection".into());
                return Err(InjectError::PayloadTooLarge);
            }
            if !self.type_text_backend(to_insert, backend, "partial type") {
                // Screen shows common prefix only.
                self.last_partial_text = common_text.to_string();
                self.last_partial_len = common_text.chars().count();
                return Err(InjectError::PartialType);
            }
        }
        self.last_partial_text = new_text.to_string();
        self.last_partial_len = new_text.chars().count();
        Ok(())
    }

    /// Erase partial text, then inject final text using the resolved mode.
    ///
    /// # Success semantics
    ///
    /// - **Direct mode**: succeeds only when the suffix update (backspace + type)
    ///   fully lands. On failure, tracking reflects on-screen state for retry.
    /// - **Clipboard mode**: succeeds only when partial erase **and**
    ///   (clipboard paste **or** direct fallback) succeed.
    /// - **Clipboard restore**: runs after a successful insert when preservation
    ///   is enabled. Restore failure does **not** fail the commit — it returns
    ///   [`CommitOutcome::CommittedClipboardNotRestored`] so composition can
    ///   latch without retrying (retry would duplicate inserted text).
    ///
    /// # Errors
    ///
    /// Returns [`InjectError`] when text was not confirmed inserted. Those
    /// errors are retryable.
    pub fn commit_final(&mut self, final_text: &str) -> Result<CommitOutcome, InjectError> {
        let final_text = sanitize_final_injection_text(final_text);

        let use_clipboard = match self.cfg.final_injection_mode {
            FinalInjectionMode::Direct => false,
            FinalInjectionMode::Clipboard => true,
            FinalInjectionMode::Auto => {
                let prefer_x = self.prefer_xdotool();
                if prefer_x {
                    info!("Auto mode selected clipboard paste for focused XWayland window.");
                }
                prefer_x || !self.detect_clipboard_watchers()
            }
        };

        if !use_clipboard {
            return self.commit_final_direct(&final_text);
        }

        self.commit_final_clipboard(&final_text)
    }

    fn commit_final_direct(&mut self, final_text: &str) -> Result<CommitOutcome, InjectError> {
        match self.update_partial(final_text) {
            Ok(()) => {
                // Tracking holds the committed text; clear local state only.
                let _ = self.reset();
                Ok(CommitOutcome::Committed)
            }
            Err(InjectError::PartialBackspace) => Err(InjectError::PartialBackspace),
            Err(InjectError::PartialType) => Err(InjectError::DirectType),
            Err(InjectError::PayloadTooLarge) => Err(InjectError::PayloadTooLarge),
            Err(other) => Err(other),
        }
    }

    fn commit_final_clipboard(&mut self, final_text: &str) -> Result<CommitOutcome, InjectError> {
        let prior = if self.cfg.preserve_clipboard {
            self.capture_clipboard()
        } else {
            None
        };

        // Erase must succeed — otherwise we would paste on top of stale partials.
        if !self.backspace_partial() {
            // Partial still on screen; keep tracking for an honest retry.
            return Err(InjectError::PartialBackspace);
        }
        // Erase landed. Clear tracking so a later retry does not re-erase.
        self.last_partial_len = 0;
        self.last_partial_text.clear();

        let oversized = !final_text.is_empty() && final_text.len() > MAX_ARGV_PAYLOAD_BYTES;
        let inserted = final_text.is_empty() || self.paste_via_clipboard(final_text) || {
            self.emit_warn("Clipboard paste failed, falling back to direct typing".into());
            if oversized {
                // Direct path cannot accept this payload either.
                self.emit_warn("direct type: payload too large for argv injection".into());
                false
            } else {
                self.type_direct(final_text)
            }
        };

        // Always attempt restore after the clipboard path (wl-copy may have run
        // even when paste/direct ultimately failed).
        let restore_ok = self.restore_clipboard(prior.as_deref());
        if !restore_ok {
            // Never include prior clipboard contents in the log line.
            self.emit_warn("clipboard restore failed after commit attempt".into());
        }

        if !inserted {
            return Err(if oversized {
                InjectError::PayloadTooLarge
            } else {
                InjectError::ClipboardInjectFailed
            });
        }

        if self.cfg.preserve_clipboard && prior.is_some() && !restore_ok {
            // Text is in the target window. Composition must latch and must not
            // retry — retrying would duplicate the inserted text.
            return Ok(CommitOutcome::CommittedClipboardNotRestored);
        }

        Ok(CommitOutcome::Committed)
    }

    /// Reset tracking state without sending keystrokes.
    ///
    /// Infallible today: local bookkeeping only. Returns `Result` so callers
    /// (and the app `TextInjector` trait) can use a uniform fallible surface.
    pub fn reset(&mut self) -> Result<(), InjectError> {
        self.last_partial_len = 0;
        self.last_partial_text.clear();
        Ok(())
    }
}

/// Python `len`-style: common prefix measured in Unicode scalar values, returned as byte index.
fn common_prefix_bytes(left: &str, right: &str) -> usize {
    let mut bytes = 0;
    for (lc, rc) in left.chars().zip(right.chars()) {
        if lc != rc {
            break;
        }
        bytes += lc.len_utf8();
    }
    bytes
}

fn redact_process_error(err: &ProcessError, program_hint: Option<&str>) -> String {
    match err {
        ProcessError::ExitCode { program, code } => {
            format!("{program} failed with exit code {code}")
        }
        ProcessError::Timeout { program, timeout } => {
            format!("{program} timed out after {timeout:?}")
        }
        ProcessError::NotFound { program } => format!("{program} not found on PATH"),
        ProcessError::OutputTooLarge { program, limit } => {
            format!("{program} output exceeded {limit} bytes")
        }
        ProcessError::Io { program, source } => {
            let prog = if program.is_empty() {
                program_hint.unwrap_or("subprocess")
            } else {
                program.as_str()
            };
            format!("{prog} I/O error: {source}")
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default, clippy::get_first)]

    use super::*;
    use crate::process::{RunOutput, ScriptedRunner};

    fn ok_runner() -> ScriptedRunner {
        let r = ScriptedRunner::new();
        r.set_dynamic(|_| {
            Ok(RunOutput {
                status_code: Some(0),
                stdout: Vec::new(),
                stderr: Vec::new(),
                success: true,
            })
        });
        r
    }

    fn fail_program(name: &str) -> ProcessError {
        ProcessError::ExitCode {
            program: name.into(),
            code: 1,
        }
    }

    fn typer_wtype(cfg: TyperConfig) -> (StreamingTyper, ScriptedRunner, RecordingSleeper) {
        let r = ok_runner();
        let sleeper = RecordingSleeper::new();
        let mut t =
            StreamingTyper::new_with_sleeper(cfg, Arc::new(r.clone()), Arc::new(sleeper.clone()));
        t.set_tool_availability(false, false);
        t.force_xwayland = Some(false);
        (t, r, sleeper)
    }

    // --- retries ---

    #[test]
    fn run_retries_until_success() {
        let r = ScriptedRunner::new();
        let state = Arc::new(Mutex::new(0u32));
        let s2 = Arc::clone(&state);
        r.set_dynamic(move |_| {
            let mut n = s2.lock().unwrap();
            *n += 1;
            if *n < 3 {
                Err(fail_program("wtype"))
            } else {
                Ok(RunOutput {
                    status_code: Some(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    success: true,
                })
            }
        });
        let cfg = TyperConfig {
            retry_attempts: 3,
            retry_delay: Duration::ZERO,
            ..TyperConfig::default()
        };
        let mut typer = StreamingTyper::new(cfg, Arc::new(r));
        typer.set_tool_availability(false, false);
        typer.force_xwayland = Some(false);
        assert!(typer.type_direct("hello"));
        assert_eq!(*state.lock().unwrap(), 3);
    }

    // --- partial / common prefix / same backend ---

    #[test]
    fn update_partial_batches_backspaces_wtype() {
        let (mut typer, r, _) = typer_wtype(TyperConfig {
            retry_attempts: 1,
            retry_delay: Duration::ZERO,
            ..TyperConfig::default()
        });
        typer.last_partial_text = "x".repeat(120);
        typer.last_partial_len = 120;
        typer.update_partial("abc").unwrap();
        let calls = r.calls();
        assert!(calls.len() >= 4);
        assert_eq!(calls[0].iter().filter(|a| *a == "BackSpace").count(), 50);
        assert_eq!(calls[1].iter().filter(|a| *a == "BackSpace").count(), 50);
        assert_eq!(calls[2].iter().filter(|a| *a == "BackSpace").count(), 20);
        assert_eq!(calls[3], argv(["wtype", "--", "abc"]));
        assert_eq!(typer.last_partial_len, 3);
        assert_eq!(typer.last_partial_text, "abc");
    }

    #[test]
    fn update_partial_common_prefix_small_suffix() {
        let (mut typer, r, _) = typer_wtype(TyperConfig {
            retry_attempts: 1,
            retry_delay: Duration::ZERO,
            ..TyperConfig::default()
        });
        typer.last_partial_text = "hello world".into();
        typer.last_partial_len = 11;
        typer.update_partial("hello there").unwrap();
        let calls = r.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].iter().filter(|a| *a == "BackSpace").count(), 5);
        assert_eq!(calls[1], argv(["wtype", "--", "there"]));
    }

    #[test]
    fn update_partial_same_backend_xdotool() {
        let r = ok_runner();
        let mut typer = StreamingTyper::new(TyperConfig::default(), Arc::new(r.clone()));
        typer.set_tool_availability(true, false);
        typer.force_xwayland = Some(true);
        typer.force_xdotool_window_id = Some("99".into());
        typer.last_partial_text = "hello world".into();
        typer.last_partial_len = 11;
        typer.update_partial("hello there").unwrap();
        let calls = r.calls();
        assert!(
            calls
                .iter()
                .all(|c| c.first().map(String::as_str) == Some("xdotool")),
            "mixed backends: {calls:?}"
        );
        let type_call = calls
            .iter()
            .find(|c| c.get(1).map(String::as_str) == Some("type"))
            .expect("type call");
        assert!(type_call.iter().any(|a| a == "--args"));
        assert_eq!(type_call.last().map(String::as_str), Some("there"));
    }

    // --- commit_final policy matrix ---

    #[test]
    fn commit_final_auto_watchers_use_direct_no_clipboard() {
        let (mut typer, r, _) = typer_wtype(TyperConfig {
            final_injection_mode: FinalInjectionMode::Auto,
            preserve_clipboard: true,
            ..TyperConfig::default()
        });
        typer.set_watchers_detected(true);
        typer.last_partial_text = "partial".into();
        typer.last_partial_len = 7;
        assert_eq!(
            typer.commit_final("hello").unwrap(),
            CommitOutcome::Committed
        );
        let calls = r.calls();
        assert!(
            calls
                .iter()
                .any(|c| c.first().map(String::as_str) == Some("wtype"))
        );
        assert!(
            calls
                .iter()
                .all(|c| c.first().map(String::as_str) != Some("wl-copy"))
        );
        assert!(
            calls
                .iter()
                .all(|c| c.first().map(String::as_str) != Some("wl-paste"))
        );
        assert_eq!(typer.last_partial_len, 0);
        assert!(typer.last_partial_text.is_empty());
    }

    #[test]
    fn commit_final_auto_no_watchers_uses_clipboard() {
        let (mut typer, r, _) = typer_wtype(TyperConfig {
            final_injection_mode: FinalInjectionMode::Auto,
            preserve_clipboard: false,
            clipboard_settle_delay: Duration::ZERO,
            ..TyperConfig::default()
        });
        typer.set_watchers_detected(false);
        assert_eq!(
            typer.commit_final("hello").unwrap(),
            CommitOutcome::Committed
        );
        let calls = r.calls();
        assert!(
            calls
                .iter()
                .any(|c| c.first().map(String::as_str) == Some("wl-copy"))
        );
        // Ctrl+V via wtype
        assert!(calls.iter().any(|c| {
            c.first().map(String::as_str) == Some("wtype")
                && c.iter().any(|a| a == "ctrl" || a == "v")
        }));
    }

    #[test]
    fn commit_final_auto_xwayland_prefers_clipboard_even_with_watchers() {
        let r = ok_runner();
        let sleeper = RecordingSleeper::new();
        let mut typer = StreamingTyper::new_with_sleeper(
            TyperConfig {
                final_injection_mode: FinalInjectionMode::Auto,
                preserve_clipboard: false,
                clipboard_settle_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
            Arc::new(sleeper),
        );
        typer.set_tool_availability(true, false);
        typer.force_xwayland = Some(true);
        typer.force_xdotool_window_id = Some("123".into());
        typer.set_watchers_detected(true); // would force direct on Wayland
        assert_eq!(
            typer.commit_final("hello").unwrap(),
            CommitOutcome::Committed
        );
        let calls = r.calls();
        assert!(
            calls
                .iter()
                .any(|c| c.first().map(String::as_str) == Some("wl-copy")),
            "expected clipboard path on XWayland: {calls:?}"
        );
    }

    #[test]
    fn commit_final_clipboard_sanitizes_newlines() {
        let (mut typer, r, _) = typer_wtype(TyperConfig {
            final_injection_mode: FinalInjectionMode::Clipboard,
            clipboard_settle_delay: Duration::ZERO,
            ..TyperConfig::default()
        });
        assert_eq!(
            typer.commit_final("fix this\n").unwrap(),
            CommitOutcome::Committed
        );
        let payloads = r.stdin_payloads();
        assert!(
            payloads
                .iter()
                .any(|p| p.as_ref().is_some_and(|b| b == b"fix this")),
            "stdin payloads: {payloads:?}"
        );
    }

    #[test]
    fn commit_final_direct_sanitizes_newlines() {
        let (mut typer, r, _) = typer_wtype(TyperConfig {
            final_injection_mode: FinalInjectionMode::Direct,
            retry_attempts: 1,
            retry_delay: Duration::ZERO,
            ..TyperConfig::default()
        });
        assert_eq!(
            typer.commit_final("hello\r\nworld").unwrap(),
            CommitOutcome::Committed
        );
        let calls = r.calls();
        assert!(
            calls
                .iter()
                .any(|c| c == &argv(["wtype", "--", "hello world"])),
            "{calls:?}"
        );
        assert_eq!(typer.last_partial_len, 0);
        assert!(typer.last_partial_text.is_empty());
    }

    #[test]
    fn commit_final_clipboard_fallback_sanitizes_and_types_direct() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|argv| {
            if argv.first().map(String::as_str) == Some("wl-copy") {
                Err(fail_program("wl-copy"))
            } else {
                Ok(RunOutput {
                    status_code: Some(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    success: true,
                })
            }
        });
        let mut typer = StreamingTyper::new(
            TyperConfig {
                final_injection_mode: FinalInjectionMode::Clipboard,
                preserve_clipboard: false,
                clipboard_settle_delay: Duration::ZERO,
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
        );
        typer.set_tool_availability(false, false);
        typer.force_xwayland = Some(false);
        assert_eq!(
            typer.commit_final("line one\rline two").unwrap(),
            CommitOutcome::Committed
        );
        let calls = r.calls();
        assert!(
            calls
                .iter()
                .any(|c| c == &argv(["wtype", "--", "line one line two"])),
            "fallback direct missing: {calls:?}"
        );
    }

    #[test]
    fn commit_final_paste_fail_restores_prior_clipboard() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|argv| {
            match argv.first().map(String::as_str) {
                Some("wl-paste") => Ok(RunOutput {
                    status_code: Some(0),
                    stdout: b"orig".to_vec(),
                    stderr: Vec::new(),
                    success: true,
                }),
                Some("wl-copy") if argv.iter().any(|a| a == "--clear") => Ok(RunOutput {
                    status_code: Some(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    success: true,
                }),
                Some("wl-copy") => {
                    // First wl-copy is set (paste path) — fail it; restore uses stdin too
                    // Distinguish by whether stdin would be used: ScriptedRunner still sees argv.
                    // Fail only the first non-restore? Simpler: fail all wl-copy without clear
                    // until we've seen paste attempt, then succeed restore.
                    // Use call counting:
                    Err(fail_program("wl-copy"))
                }
                _ => Ok(RunOutput {
                    status_code: Some(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    success: true,
                }),
            }
        });
        // Need more careful dynamic: fail paste set, succeed restore
        let r = ScriptedRunner::new();
        let n = Arc::new(Mutex::new(0u32));
        let n2 = Arc::clone(&n);
        r.set_dynamic(move |argv| {
            if argv.first().map(String::as_str) == Some("wl-paste") {
                return Ok(RunOutput {
                    status_code: Some(0),
                    stdout: b"orig".to_vec(),
                    stderr: Vec::new(),
                    success: true,
                });
            }
            if argv.first().map(String::as_str) == Some("wl-copy") {
                let mut c = n2.lock().unwrap();
                *c += 1;
                if *c == 1 {
                    // paste set fails
                    return Err(fail_program("wl-copy"));
                }
                // restore succeeds
                return Ok(RunOutput {
                    status_code: Some(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    success: true,
                });
            }
            Ok(RunOutput {
                status_code: Some(0),
                stdout: Vec::new(),
                stderr: Vec::new(),
                success: true,
            })
        });
        let mut typer = StreamingTyper::new(
            TyperConfig {
                final_injection_mode: FinalInjectionMode::Clipboard,
                preserve_clipboard: true,
                clipboard_settle_delay: Duration::ZERO,
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
        );
        typer.set_tool_availability(false, false);
        typer.force_xwayland = Some(false);
        assert_eq!(
            typer.commit_final("hello").unwrap(),
            CommitOutcome::Committed
        );
        // restore should have pushed orig via stdin
        let payloads = r.stdin_payloads();
        assert!(
            payloads
                .iter()
                .any(|p| p.as_ref().is_some_and(|b| b == b"orig")),
            "restore stdin missing: {payloads:?} calls={:?}",
            r.calls()
        );
        assert_eq!(typer.last_partial_len, 0);
    }

    // --- settle delay ---

    #[test]
    fn paste_via_clipboard_applies_settle_delay() {
        let r = ok_runner();
        let sleeper = RecordingSleeper::new();
        let mut typer = StreamingTyper::new_with_sleeper(
            TyperConfig {
                clipboard_settle_delay: Duration::from_millis(40),
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
            Arc::new(sleeper.clone()),
        );
        typer.set_tool_availability(false, false);
        typer.force_xwayland = Some(false);
        assert!(typer.paste_via_clipboard("hello"));
        let sleeps = sleeper.sleeps();
        assert_eq!(sleeps, vec![Duration::from_millis(40)]);
        let calls = r.calls();
        assert_eq!(calls[0], argv(["wl-copy", "--"]));
        assert_eq!(
            calls[1],
            argv(["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"])
        );
    }

    // --- backend preference / failure order ---

    #[test]
    fn type_direct_prefers_ydotool_on_xwayland() {
        let r = ok_runner();
        let mut typer = StreamingTyper::new(
            TyperConfig {
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
        );
        typer.set_tool_availability(false, true);
        typer.force_xwayland = Some(true);
        assert!(typer.type_direct("hello"));
        assert_eq!(
            r.calls()[0],
            argv([
                "ydotool",
                "type",
                "--key-delay",
                "0",
                "--key-hold",
                "0",
                "--",
                "hello"
            ])
        );
    }

    #[test]
    fn type_direct_falls_back_to_ydotool_when_wtype_fails() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|argv| {
            if argv.first().map(String::as_str) == Some("wtype") {
                Err(fail_program("wtype"))
            } else {
                Ok(RunOutput {
                    status_code: Some(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    success: true,
                })
            }
        });
        let mut typer = StreamingTyper::new(
            TyperConfig {
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
        );
        typer.set_tool_availability(false, true);
        typer.force_xwayland = Some(false); // ydotool not preferred, but available as fallback
        assert!(typer.type_direct("hello"));
        let calls = r.calls();
        assert_eq!(calls[0], argv(["wtype", "--", "hello"]));
        assert_eq!(
            calls[1],
            argv([
                "ydotool",
                "type",
                "--key-delay",
                "0",
                "--key-hold",
                "0",
                "--",
                "hello"
            ])
        );
    }

    #[test]
    fn type_direct_prefers_xdotool_on_xwayland() {
        let r = ok_runner();
        let mut typer = StreamingTyper::new(
            TyperConfig {
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
        );
        typer.set_tool_availability(true, false);
        typer.force_xwayland = Some(true);
        typer.force_xdotool_window_id = Some("123".into());
        assert!(typer.type_direct("hello"));
        let call = &r.calls()[0];
        assert_eq!(call[0], "xdotool");
        assert_eq!(call[1], "type");
        assert!(call.iter().any(|a| a == "--args"));
        assert_eq!(call.last().map(String::as_str), Some("hello"));
        assert!(call.iter().any(|a| a == "123"));
    }

    #[test]
    fn paste_prefers_xdotool_on_xwayland() {
        let r = ok_runner();
        let sleeper = RecordingSleeper::new();
        let mut typer = StreamingTyper::new_with_sleeper(
            TyperConfig {
                clipboard_settle_delay: Duration::from_millis(40),
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
            Arc::new(sleeper.clone()),
        );
        typer.set_tool_availability(true, false);
        typer.force_xwayland = Some(true);
        typer.force_xdotool_window_id = Some("123".into());
        assert!(typer.paste_via_clipboard("hello"));
        assert_eq!(sleeper.sleeps(), vec![Duration::from_millis(40)]);
        let calls = r.calls();
        assert_eq!(calls[0], argv(["wl-copy", "--"]));
        let key = &calls[1];
        assert_eq!(key[0], "xdotool");
        assert_eq!(key[1], "key");
        assert!(key.iter().any(|a| a == "--"));
        assert_eq!(key.last().map(String::as_str), Some("ctrl+v"));
    }

    #[test]
    fn paste_prefers_ydotool_on_xwayland() {
        let r = ok_runner();
        let sleeper = RecordingSleeper::new();
        let mut typer = StreamingTyper::new_with_sleeper(
            TyperConfig {
                clipboard_settle_delay: Duration::from_millis(40),
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
            Arc::new(sleeper.clone()),
        );
        typer.set_tool_availability(false, true);
        typer.force_xwayland = Some(true);
        assert!(typer.paste_via_clipboard("hello"));
        assert_eq!(sleeper.sleeps(), vec![Duration::from_millis(40)]);
        assert_eq!(
            r.calls()[1],
            argv(["ydotool", "key", "-d", "0", "29:1", "47:1", "47:0", "29:0"])
        );
    }

    #[test]
    fn send_backspaces_prefers_xdotool_on_xwayland() {
        let r = ok_runner();
        let mut typer = StreamingTyper::new(
            TyperConfig {
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
        );
        typer.set_tool_availability(true, false);
        typer.force_xwayland = Some(true);
        typer.force_xdotool_window_id = Some("123".into());
        assert!(typer.send_backspaces(55, "backspace"));
        let calls = r.calls();
        assert_eq!(calls.len(), 2);
        // batch 50 then 5 with --repeat
        assert!(calls[0].iter().any(|a| a == "--repeat"));
        assert!(calls[0].iter().any(|a| a == "50"));
        assert_eq!(calls[0].last().map(String::as_str), Some("BackSpace"));
        assert!(calls[1].iter().any(|a| a == "5"));
    }

    #[test]
    fn active_window_reports_xwayland_from_hyprctl() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|args| {
            assert_eq!(args, &["hyprctl", "activewindow", "-j"]);
            Ok(RunOutput {
                status_code: Some(0),
                stdout: br#"{"xwayland": true}"#.to_vec(),
                stderr: Vec::new(),
                success: true,
            })
        });
        let mut typer = StreamingTyper::new(
            TyperConfig {
                retry_attempts: 1,
                ..TyperConfig::default()
            },
            Arc::new(r),
        );
        // no force_xwayland — read from runner
        assert!(typer.active_window_is_xwayland());
    }

    // --- clipboard preserve / restore ---

    #[test]
    fn restore_clipboard_clear_when_empty() {
        let r = ok_runner();
        let mut typer = StreamingTyper::new(
            TyperConfig {
                preserve_clipboard: true,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
        );
        assert!(typer.restore_clipboard(Some("")));
        assert!(r.calls().iter().any(|c| c == &argv(["wl-copy", "--clear"])));
    }

    #[test]
    fn restore_clipboard_restores_nonempty_via_stdin() {
        let r = ok_runner();
        let mut typer = StreamingTyper::new(
            TyperConfig {
                preserve_clipboard: true,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
        );
        assert!(typer.restore_clipboard(Some("prior text")));
        assert_eq!(r.calls()[0], argv(["wl-copy", "--"]));
        assert_eq!(
            r.stdin_payloads()[0].as_deref(),
            Some(b"prior text".as_slice())
        );
    }

    #[test]
    fn restore_skips_when_capture_absent() {
        let r = ok_runner();
        let mut typer = StreamingTyper::new(
            TyperConfig {
                preserve_clipboard: true,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
        );
        assert!(typer.restore_clipboard(None));
        assert!(r.calls().is_empty());
    }

    #[test]
    fn commit_final_empty_prior_clears_after_success() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|argv| {
            if argv.first().map(String::as_str) == Some("wl-paste") {
                // empty clipboard: non-zero exit
                Ok(RunOutput {
                    status_code: Some(1),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    success: false,
                })
            } else {
                Ok(RunOutput {
                    status_code: Some(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    success: true,
                })
            }
        });
        let mut typer = StreamingTyper::new(
            TyperConfig {
                final_injection_mode: FinalInjectionMode::Clipboard,
                preserve_clipboard: true,
                clipboard_settle_delay: Duration::ZERO,
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
        );
        typer.set_tool_availability(false, false);
        typer.force_xwayland = Some(false);
        assert_eq!(typer.commit_final("hi").unwrap(), CommitOutcome::Committed);
        assert!(r.calls().iter().any(|c| c == &argv(["wl-copy", "--clear"])));
    }

    // --- reset ---

    #[test]
    fn reset_clears_partial_state() {
        let mut typer = StreamingTyper::with_defaults();
        typer.last_partial_len = 99;
        typer.last_partial_text = "abc".into();
        typer.reset().unwrap();
        assert_eq!(typer.last_partial_len, 0);
        assert!(typer.last_partial_text.is_empty());
    }

    // --- security ---

    #[test]
    fn xdotool_type_safe_with_dash_prefix() {
        let args = StreamingTyper::xdotool_type_args("-display evil\nsecret", Some("1"));
        assert!(args.iter().any(|a| a == "--args"));
        assert_eq!(
            args.last().map(String::as_str),
            Some("-display evil\nsecret")
        );
    }

    #[test]
    fn long_payload_fail_closed_on_xdotool() {
        let r = ok_runner();
        let logs = Arc::new(Mutex::new(Vec::new()));
        let mut typer = StreamingTyper::new(
            TyperConfig {
                retry_attempts: 1,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
        );
        typer.set_tool_availability(true, false);
        typer.force_xwayland = Some(true);
        typer.force_xdotool_window_id = Some("1".into());
        typer.set_log_capture(Arc::clone(&logs));
        let huge = "x".repeat(MAX_ARGV_PAYLOAD_BYTES + 1);
        assert!(!typer.type_direct(&huge));
        // Should not have invoked xdotool type with huge argv
        assert!(
            r.calls()
                .iter()
                .filter(|c| c.get(1).map(String::as_str) == Some("type"))
                .all(|c| c.last().map(|s| s.len()).unwrap_or(0) <= MAX_ARGV_PAYLOAD_BYTES)
        );
        let captured = logs.lock().unwrap();
        assert!(captured.iter().any(|m| m.contains("payload too large")));
        assert!(
            r.calls().is_empty(),
            "must not spawn tools for oversized payload"
        );
        assert!(captured.iter().all(|m| !m.contains(&huge)));
    }

    #[test]
    fn logs_never_contain_payload_on_failure() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|_| Err(fail_program("wtype")));
        let logs = Arc::new(Mutex::new(Vec::new()));
        let mut typer = StreamingTyper::new(
            TyperConfig {
                retry_attempts: 2,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r),
        );
        typer.set_tool_availability(false, false);
        typer.force_xwayland = Some(false);
        typer.set_log_capture(Arc::clone(&logs));
        let secret = "SECRET_PASSWORD";
        assert!(!typer.type_direct(secret));
        let captured = logs.lock().unwrap();
        assert!(!captured.is_empty());
        for line in captured.iter() {
            assert!(!line.contains(secret), "payload leaked into log: {line}");
            assert!(!line.contains("--"));
        }
        assert!(
            captured
                .iter()
                .any(|m| m.contains("direct type failed") || m.contains("failed after"))
        );
    }

    #[test]
    fn redact_process_error_omits_args() {
        let err = ProcessError::ExitCode {
            program: "wtype".into(),
            code: 1,
        };
        let msg = redact_process_error(&err, Some("wtype"));
        assert!(msg.contains("wtype"));
        assert!(!msg.contains("SECRET"));
    }

    #[test]
    fn commit_final_clipboard_uses_stdin_not_argv_payload() {
        let (mut typer, r, _) = typer_wtype(TyperConfig {
            final_injection_mode: FinalInjectionMode::Clipboard,
            clipboard_settle_delay: Duration::ZERO,
            ..TyperConfig::default()
        });
        assert_eq!(
            typer.commit_final("secret-transcript").unwrap(),
            CommitOutcome::Committed
        );
        let copy_calls: Vec<_> = r
            .calls()
            .into_iter()
            .filter(|c| c.first().map(String::as_str) == Some("wl-copy"))
            .collect();
        assert!(!copy_calls.is_empty());
        for c in &copy_calls {
            assert!(
                !c.iter().any(|a| a == "secret-transcript"),
                "payload in argv: {c:?}"
            );
        }
        assert!(
            r.stdin_payloads()
                .iter()
                .any(|p| p.as_ref().is_some_and(|b| b == b"secret-transcript"))
        );
    }

    // --- honest Result seam: failure / retry / privacy ---

    #[test]
    fn update_partial_backspace_failure_keeps_prior_state() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|argv| {
            if argv.iter().any(|a| a == "BackSpace") {
                Err(fail_program("wtype"))
            } else {
                Ok(RunOutput {
                    status_code: Some(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    success: true,
                })
            }
        });
        let mut typer = StreamingTyper::new(
            TyperConfig {
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r),
        );
        typer.set_tool_availability(false, false);
        typer.force_xwayland = Some(false);
        typer.last_partial_text = "hello world".into();
        typer.last_partial_len = 11;

        let err = typer.update_partial("hello there").unwrap_err();
        assert_eq!(err, InjectError::PartialBackspace);
        // Screen still has old text — tracking must not lie.
        assert_eq!(typer.last_partial_text, "hello world");
        assert_eq!(typer.last_partial_len, 11);
    }

    #[test]
    fn update_partial_type_failure_tracks_common_prefix() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|argv| {
            // Allow backspaces; fail type (wtype -- <text>).
            if argv.get(1).map(String::as_str) == Some("--") {
                Err(fail_program("wtype"))
            } else {
                Ok(RunOutput {
                    status_code: Some(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    success: true,
                })
            }
        });
        let mut typer = StreamingTyper::new(
            TyperConfig {
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r),
        );
        typer.set_tool_availability(false, false);
        typer.force_xwayland = Some(false);
        typer.last_partial_text = "hello world".into();
        typer.last_partial_len = 11;

        let err = typer.update_partial("hello there").unwrap_err();
        assert_eq!(err, InjectError::PartialType);
        // Backspace of "world" succeeded; "hello " remains.
        assert_eq!(typer.last_partial_text, "hello ");
        assert_eq!(typer.last_partial_len, 6);
    }

    #[test]
    fn update_partial_retry_after_type_failure_succeeds() {
        let r = ScriptedRunner::new();
        let n = Arc::new(Mutex::new(0u32));
        let n2 = Arc::clone(&n);
        r.set_dynamic(move |argv| {
            if argv.get(1).map(String::as_str) == Some("--") {
                let mut c = n2.lock().unwrap();
                *c += 1;
                if *c == 1 {
                    return Err(fail_program("wtype"));
                }
            }
            Ok(RunOutput {
                status_code: Some(0),
                stdout: Vec::new(),
                stderr: Vec::new(),
                success: true,
            })
        });
        let mut typer = StreamingTyper::new(
            TyperConfig {
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
        );
        typer.set_tool_availability(false, false);
        typer.force_xwayland = Some(false);
        typer.last_partial_text = "hello world".into();
        typer.last_partial_len = 11;

        assert_eq!(
            typer.update_partial("hello there").unwrap_err(),
            InjectError::PartialType
        );
        // Retry with honest common-prefix state types only the missing suffix.
        typer.update_partial("hello there").unwrap();
        assert_eq!(typer.last_partial_text, "hello there");
        let type_calls: Vec<_> = r
            .calls()
            .into_iter()
            .filter(|c| c.get(1).map(String::as_str) == Some("--"))
            .collect();
        // First attempt typed "there" (failed); retry typed "there" again (no re-backspace of "hello ").
        assert!(
            type_calls
                .iter()
                .any(|c| c == &argv(["wtype", "--", "there"])),
            "expected suffix-only retry type: {type_calls:?}"
        );
    }

    #[test]
    fn commit_final_direct_surfaces_type_failure() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|_| Err(fail_program("wtype")));
        let mut typer = StreamingTyper::new(
            TyperConfig {
                final_injection_mode: FinalInjectionMode::Direct,
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r),
        );
        typer.set_tool_availability(false, false);
        typer.force_xwayland = Some(false);

        let err = typer.commit_final("hello").unwrap_err();
        assert_eq!(err, InjectError::DirectType);
        // Nothing landed — tracking stays empty (no prior partial).
        assert_eq!(typer.last_partial_len, 0);
        assert!(typer.last_partial_text.is_empty());
    }

    #[test]
    fn commit_final_direct_backspace_failure_preserves_partial() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|argv| {
            if argv.iter().any(|a| a == "BackSpace") {
                Err(fail_program("wtype"))
            } else {
                Ok(RunOutput {
                    status_code: Some(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    success: true,
                })
            }
        });
        let mut typer = StreamingTyper::new(
            TyperConfig {
                final_injection_mode: FinalInjectionMode::Direct,
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r),
        );
        typer.set_tool_availability(false, false);
        typer.force_xwayland = Some(false);
        typer.last_partial_text = "old".into();
        typer.last_partial_len = 3;

        let err = typer.commit_final("new").unwrap_err();
        assert_eq!(err, InjectError::PartialBackspace);
        assert_eq!(typer.last_partial_text, "old");
        assert_eq!(typer.last_partial_len, 3);
    }

    #[test]
    fn commit_final_clipboard_erase_failure_is_err_and_keeps_partial() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|argv| {
            if argv.iter().any(|a| a == "BackSpace") {
                Err(fail_program("wtype"))
            } else {
                Ok(RunOutput {
                    status_code: Some(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    success: true,
                })
            }
        });
        let mut typer = StreamingTyper::new(
            TyperConfig {
                final_injection_mode: FinalInjectionMode::Clipboard,
                clipboard_settle_delay: Duration::ZERO,
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r.clone()),
        );
        typer.set_tool_availability(false, false);
        typer.force_xwayland = Some(false);
        typer.last_partial_text = "partial".into();
        typer.last_partial_len = 7;

        let err = typer.commit_final("final").unwrap_err();
        assert_eq!(err, InjectError::PartialBackspace);
        assert_eq!(typer.last_partial_text, "partial");
        // Must not have attempted wl-copy after erase failure.
        assert!(
            r.calls()
                .iter()
                .all(|c| c.first().map(String::as_str) != Some("wl-copy")),
            "wl-copy after erase failure: {:?}",
            r.calls()
        );
    }

    #[test]
    fn commit_final_clipboard_and_direct_both_fail() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|argv| match argv.first().map(String::as_str) {
            Some("wl-copy") | Some("wtype") => Err(fail_program(
                argv.first().map(String::as_str).unwrap_or("tool"),
            )),
            _ => Ok(RunOutput {
                status_code: Some(0),
                stdout: Vec::new(),
                stderr: Vec::new(),
                success: true,
            }),
        });
        let mut typer = StreamingTyper::new(
            TyperConfig {
                final_injection_mode: FinalInjectionMode::Clipboard,
                preserve_clipboard: false,
                clipboard_settle_delay: Duration::ZERO,
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r),
        );
        typer.set_tool_availability(false, false);
        typer.force_xwayland = Some(false);

        let err = typer.commit_final("hello").unwrap_err();
        assert_eq!(err, InjectError::ClipboardInjectFailed);
        assert!(err.is_retryable());
        // Erase (noop) cleared tracking; safe to retry insert without re-erase.
        assert_eq!(typer.last_partial_len, 0);
    }

    #[test]
    fn commit_final_restore_failure_after_success_is_committed_not_err() {
        let r = ScriptedRunner::new();
        let n = Arc::new(Mutex::new(0u32));
        let n2 = Arc::clone(&n);
        r.set_dynamic(move |argv| {
            if argv.first().map(String::as_str) == Some("wl-paste") {
                return Ok(RunOutput {
                    status_code: Some(0),
                    stdout: b"PRIOR_SECRET_CLIP".to_vec(),
                    stderr: Vec::new(),
                    success: true,
                });
            }
            if argv.first().map(String::as_str) == Some("wl-copy") {
                let mut c = n2.lock().unwrap();
                *c += 1;
                // 1 = set for paste (ok), 2 = restore (fail)
                if *c >= 2 {
                    return Err(fail_program("wl-copy"));
                }
            }
            Ok(RunOutput {
                status_code: Some(0),
                stdout: Vec::new(),
                stderr: Vec::new(),
                success: true,
            })
        });
        let logs = Arc::new(Mutex::new(Vec::new()));
        let mut typer = StreamingTyper::new(
            TyperConfig {
                final_injection_mode: FinalInjectionMode::Clipboard,
                preserve_clipboard: true,
                clipboard_settle_delay: Duration::ZERO,
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r),
        );
        typer.set_tool_availability(false, false);
        typer.force_xwayland = Some(false);
        typer.set_log_capture(Arc::clone(&logs));

        let outcome = typer.commit_final("hello").unwrap();
        assert_eq!(outcome, CommitOutcome::CommittedClipboardNotRestored);
        assert!(outcome.text_inserted());
        assert!(!outcome.clipboard_restored());
        assert!(outcome.needs_clipboard_warning());
        assert_eq!(typer.last_partial_len, 0);

        let captured = logs.lock().unwrap();
        assert!(
            captured
                .iter()
                .any(|m| m.contains("clipboard restore failed")),
            "expected restore warning: {captured:?}"
        );
        // Privacy: prior clipboard payload must never appear in logs.
        for line in captured.iter() {
            assert!(
                !line.contains("PRIOR_SECRET_CLIP"),
                "clipboard payload leaked into log: {line}"
            );
            assert!(
                !line.contains("hello"),
                "transcript leaked into log: {line}"
            );
        }
    }

    #[test]
    fn commit_final_error_display_and_debug_omit_transcript() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|_| Err(fail_program("wtype")));
        let logs = Arc::new(Mutex::new(Vec::new()));
        let mut typer = StreamingTyper::new(
            TyperConfig {
                final_injection_mode: FinalInjectionMode::Direct,
                retry_attempts: 1,
                retry_delay: Duration::ZERO,
                ..TyperConfig::default()
            },
            Arc::new(r),
        );
        typer.set_tool_availability(false, false);
        typer.force_xwayland = Some(false);
        typer.set_log_capture(Arc::clone(&logs));

        let secret = "TOP_SECRET_TRANSCRIPT_XYZ";
        let err = typer.commit_final(secret).unwrap_err();
        let display = format!("{err}");
        let debug = format!("{err:?}");
        assert!(!display.contains(secret), "Display leaked: {display}");
        assert!(!debug.contains(secret), "Debug leaked: {debug}");
        assert!(!debug.contains('{'), "Debug should be field-less: {debug}");

        let captured = logs.lock().unwrap();
        for line in captured.iter() {
            assert!(!line.contains(secret), "log leaked transcript: {line}");
        }
    }

    #[test]
    fn commit_final_payload_too_large_is_distinct_error() {
        let (mut typer, r, _) = typer_wtype(TyperConfig {
            final_injection_mode: FinalInjectionMode::Direct,
            retry_attempts: 1,
            retry_delay: Duration::ZERO,
            ..TyperConfig::default()
        });
        let huge = "x".repeat(MAX_ARGV_PAYLOAD_BYTES + 1);
        let err = typer.commit_final(&huge).unwrap_err();
        assert_eq!(err, InjectError::PayloadTooLarge);
        assert!(
            r.calls().is_empty(),
            "must not spawn tools for oversized direct payload: {:?}",
            r.calls()
        );
    }

    #[test]
    fn reset_is_ok_and_clears_state() {
        let mut typer = StreamingTyper::with_defaults();
        typer.last_partial_len = 3;
        typer.last_partial_text = "abc".into();
        assert!(typer.reset().is_ok());
        assert_eq!(typer.last_partial_len, 0);
        assert!(typer.last_partial_text.is_empty());
        // Second reset still Ok (idempotent).
        assert!(typer.reset().is_ok());
    }
}
