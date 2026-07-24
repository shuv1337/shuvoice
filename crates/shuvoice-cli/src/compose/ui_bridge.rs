//! Compose-layer UI bridge: session ↔ UI protocol mappings and host helpers.
//!
//! # Integration notes
//!
//! Expected crate deps / features (declared by the integration owner):
//! - `shuvoice-app` (session types, `OverlaySink`)
//! - `shuvoice-ui` (protocol/channel/VMs); feature `ui` enables GTK host types
//! - Direct `gtk4` + `glib` deps on the CLI crate if using [`run_gtk_main_host`]
//! - `tokio` (already in `shuvoice-cli`)
//!
//! Wire from `compose/mod.rs` once the root declares the module.
//!
//! # Design
//!
//! - Pure maps: [`UiEvent`] → [`SessionCommand`], [`SessionEvent`] → ordered
//!   [`UiCmd`] list.
//! - Caption dual-drive: prefer [`paired_overlay_sink_path`] so
//!   [`UiCmdOverlaySink`] + [`SessionToUiMapper`] always share
//!   [`CaptionDriveMode::OverlaySink`] (no double-apply).
//! - Error toasts allocate monotonic flash tokens for `CaptionFlashError`.
//! - [`SessionEvent::TtsError`] messages are re-redacted before UI.
//! - UI → session control uses [`UiToSessionCommandBridge`] so TtsStop/Pause
//!   etc. coalesce under enqueue backpressure instead of silent drop.
//! - TTS show / idle auto-hide is expressed as `TtsSetState` (GTK host arms
//!   its own auto-hide timer); explicit hide is emitted when useful.
//! - Essential-first merge task drains the reliable event lane before
//!   best-effort partials and surfaces [`ShutdownComplete`] out-of-band.
//! - Optional GTK main-thread bootstrap never moves GTK objects across threads.
//! - GTK loop exit uses an all-`Send` quit channel (not `UiCmd::Quit`) polled on
//!   the GLib thread; lifecycle teardown is signaled back on a `Send` endpoint.

#![allow(clippy::collapsible_if)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::let_and_return)]
#![allow(clippy::double_must_use)]
#![allow(clippy::result_unit_err)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use shuvoice_app::traits::OverlaySink;
use shuvoice_app::{
    OverlayState, SessionCommand, SessionEvent, TtsPlayerState, event_is_essential,
};
use shuvoice_core::ERROR_TOAST_SECONDS;
use shuvoice_ui::channel::{UiCmdReceiver, UiCmdSender, UiEventReceiver, UiEventSender};
use shuvoice_ui::protocol::{UiCmd, UiEvent};
use shuvoice_ui::tts_overlay::TtsOverlayState;
use shuvoice_ui::{CaptionVm, TtsVm};
use tokio::sync::{mpsc, oneshot};

/// Default toast duration forwarded to the caption flash command.
pub const DEFAULT_ERROR_FLASH_SECS: u32 = ERROR_TOAST_SECONDS as u32;

// ── Caption drive mode ────────────────────────────────────────────────────

/// Who owns STT caption mutation.
///
/// Session currently calls [`OverlaySink`] *and* emits overlay events. Pick
/// exactly one caption driver to avoid duplicate application.
///
/// Prefer constructors that pair the mode correctly:
/// - [`paired_overlay_sink_path`] — production default (sink + matching mapper)
/// - [`SessionToUiMapper::for_session_events`] — events-only / headless tests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaptionDriveMode {
    /// Caption is driven by [`UiCmdOverlaySink`] (session sink calls).
    /// Overlay* session events are not re-applied; [`SessionEvent::ErrorToast`]
    /// is also skipped (sink already showed error; session hides later).
    #[default]
    OverlaySink,
    /// Caption is driven solely by session events (use a no-op / ignore sink).
    /// [`SessionEvent::ErrorToast`] becomes a tokenized flash command.
    SessionEvents,
}

/// Production caption path: sink + mapper locked to [`CaptionDriveMode::OverlaySink`].
#[derive(Debug)]
pub struct PairedOverlaySinkPath {
    pub sink: UiCmdOverlaySink,
    pub mapper: SessionToUiMapper,
}

impl PairedOverlaySinkPath {
    #[must_use]
    pub fn new(tx: UiCmdSender) -> Self {
        Self {
            sink: UiCmdOverlaySink::new(tx),
            mapper: SessionToUiMapper::for_overlay_sink(),
        }
    }

    #[must_use]
    pub fn with_flash_secs(mut self, secs: u32) -> Self {
        self.sink = self.sink.with_flash_secs(secs);
        self.mapper = self.mapper.with_flash_secs(secs);
        self
    }

    #[must_use]
    pub fn mode(&self) -> CaptionDriveMode {
        CaptionDriveMode::OverlaySink
    }
}

/// Build a sink + mapper pair that cannot disagree on caption drive mode.
#[must_use]
pub fn paired_overlay_sink_path(tx: UiCmdSender) -> PairedOverlaySinkPath {
    PairedOverlaySinkPath::new(tx)
}

// ── UiEvent → SessionCommand ──────────────────────────────────────────────

/// Map an interactive UI event into a session command.
///
/// Wizard events return `None` — the welcome wizard remains a separate host.
#[must_use]
pub fn map_ui_event_to_session_command(event: &UiEvent) -> Option<SessionCommand> {
    match event {
        UiEvent::TtsPause => Some(SessionCommand::TtsPause),
        UiEvent::TtsResume => Some(SessionCommand::TtsResume),
        UiEvent::TtsRestart => Some(SessionCommand::TtsRestart),
        UiEvent::TtsStop => Some(SessionCommand::TtsStop),
        UiEvent::TtsVoiceSelected { voice_id } => {
            Some(SessionCommand::TtsSelectVoice(voice_id.clone()))
        }
        UiEvent::TtsSpeedChanged { speed } => Some(SessionCommand::TtsSetSpeed(*speed)),
        UiEvent::WizardPageChanged { .. }
        | UiEvent::WizardBack
        | UiEvent::WizardNext
        | UiEvent::WizardFinishRequested
        | UiEvent::WizardLaunch
        | UiEvent::WizardCancelDownload
        | UiEvent::WizardClosed { .. } => None,
    }
}

// ── SessionEvent → UiCmd ──────────────────────────────────────────────────

/// Mutable mapper state (flash tokens + ErrorToast dedupe).
#[derive(Debug, Clone)]
pub struct SessionToUiMapper {
    mode: CaptionDriveMode,
    flash_token: u64,
    /// When true, the next Error-state OverlayShow from the same flash is skipped.
    suppress_next_error_overlay_show: bool,
    flash_secs: u32,
}

impl Default for SessionToUiMapper {
    fn default() -> Self {
        // Safe production default: OverlaySink mode (pair with UiCmdOverlaySink).
        Self::for_overlay_sink()
    }
}

impl SessionToUiMapper {
    /// Low-level constructor. Prefer [`for_overlay_sink`] / [`for_session_events`]
    /// or [`paired_overlay_sink_path`] so mode pairing stays correct.
    #[must_use]
    pub fn new(mode: CaptionDriveMode) -> Self {
        Self {
            mode,
            flash_token: 0,
            suppress_next_error_overlay_show: false,
            flash_secs: DEFAULT_ERROR_FLASH_SECS,
        }
    }

    /// Mapper for use with [`UiCmdOverlaySink`] / [`paired_overlay_sink_path`].
    #[must_use]
    pub fn for_overlay_sink() -> Self {
        Self::new(CaptionDriveMode::OverlaySink)
    }

    /// Mapper when captions are driven only by session events (no live sink).
    #[must_use]
    pub fn for_session_events() -> Self {
        Self::new(CaptionDriveMode::SessionEvents)
    }

    #[must_use]
    pub fn with_flash_secs(mut self, secs: u32) -> Self {
        self.flash_secs = secs.max(1);
        self
    }

    #[must_use]
    pub fn mode(&self) -> CaptionDriveMode {
        self.mode
    }

    #[must_use]
    pub fn last_flash_token(&self) -> u64 {
        self.flash_token
    }

    /// Map one session event into an ordered list of UI commands.
    #[must_use]
    pub fn map_event(&mut self, event: &SessionEvent) -> Vec<UiCmd> {
        match event {
            SessionEvent::OverlayShow { state, text } => self.map_overlay_show(*state, text),
            SessionEvent::OverlayUpdate { state, text } => self.map_overlay_update(state, text),
            SessionEvent::OverlayHide => self.map_overlay_hide(),
            SessionEvent::ErrorToast { text } => self.map_error_toast(text),
            SessionEvent::TtsState {
                state,
                preview_text,
            } => map_tts_state(state, preview_text, None),
            SessionEvent::TtsError { message } => {
                // Defensive re-redact: never trust upstream error strings for UI.
                let safe = redact_user_visible_text(message);
                map_tts_state(&TtsPlayerState::Error, "", Some(safe.as_str()))
            }
            // Non-UI / handled elsewhere.
            SessionEvent::Status(_)
            | SessionEvent::PartialTranscript { .. }
            | SessionEvent::FinalTranscript { .. }
            | SessionEvent::InjectFinal { .. }
            | SessionEvent::InjectPartial { .. }
            | SessionEvent::AsrDisabled { .. }
            | SessionEvent::AsrRecovered
            | SessionEvent::AsrThreadDead
            | SessionEvent::CudaFallbackApplied { .. }
            | SessionEvent::AudioOverflow { .. }
            | SessionEvent::ShutdownComplete => Vec::new(),
        }
    }

    fn map_overlay_show(&mut self, state: OverlayState, text: &str) -> Vec<UiCmd> {
        if self.mode == CaptionDriveMode::OverlaySink {
            return Vec::new();
        }
        if state == OverlayState::Error && self.suppress_next_error_overlay_show {
            self.suppress_next_error_overlay_show = false;
            return Vec::new();
        }
        vec![
            UiCmd::CaptionSetState { state },
            UiCmd::CaptionSetText {
                text: text.to_string(),
            },
            UiCmd::CaptionShow,
        ]
    }

    fn map_overlay_update(
        &mut self,
        state: &Option<OverlayState>,
        text: &Option<String>,
    ) -> Vec<UiCmd> {
        if self.mode == CaptionDriveMode::OverlaySink {
            return Vec::new();
        }
        let mut cmds = Vec::with_capacity(2);
        if let Some(state) = state {
            cmds.push(UiCmd::CaptionSetState { state: *state });
        }
        if let Some(text) = text {
            cmds.push(UiCmd::CaptionSetText { text: text.clone() });
        }
        cmds
    }

    fn map_overlay_hide(&mut self) -> Vec<UiCmd> {
        if self.mode == CaptionDriveMode::OverlaySink {
            return Vec::new();
        }
        self.suppress_next_error_overlay_show = false;
        vec![UiCmd::CaptionHide]
    }

    fn map_error_toast(&mut self, text: &str) -> Vec<UiCmd> {
        match self.mode {
            CaptionDriveMode::OverlaySink => {
                // Sink already showed the error; session will hide via OverlaySink.
                Vec::new()
            }
            CaptionDriveMode::SessionEvents => {
                self.flash_token = self.flash_token.saturating_add(1);
                self.suppress_next_error_overlay_show = true;
                vec![UiCmd::CaptionFlashError {
                    // Defensive: toast text must never carry paths/URLs/transcripts.
                    text: redact_user_visible_text(text),
                    token: self.flash_token,
                    secs: self.flash_secs,
                }]
            }
        }
    }
}

/// Convenience free function with a temporary mapper (defaults to OverlaySink mode).
#[must_use]
pub fn map_session_event_to_ui_cmds(event: &SessionEvent) -> Vec<UiCmd> {
    SessionToUiMapper::default().map_event(event)
}

/// Lossless app TTS state → UI overlay state.
#[must_use]
pub fn map_tts_player_state(state: &TtsPlayerState) -> TtsOverlayState {
    match state {
        TtsPlayerState::Idle => TtsOverlayState::Idle,
        TtsPlayerState::Synthesizing => TtsOverlayState::Synthesizing,
        TtsPlayerState::Playing => TtsOverlayState::Playing,
        TtsPlayerState::Paused => TtsOverlayState::Paused,
        TtsPlayerState::Error => TtsOverlayState::Error,
    }
}

/// Lossless UI overlay state → app TTS state.
#[must_use]
pub fn map_tts_overlay_state(state: TtsOverlayState) -> TtsPlayerState {
    match state {
        TtsOverlayState::Idle => TtsPlayerState::Idle,
        TtsOverlayState::Synthesizing => TtsPlayerState::Synthesizing,
        TtsOverlayState::Playing => TtsPlayerState::Playing,
        TtsOverlayState::Paused => TtsPlayerState::Paused,
        TtsOverlayState::Error => TtsPlayerState::Error,
    }
}

/// Build TTS UI commands for a player state transition.
///
/// Idle uses `TtsSetState(Idle)` so the GTK host can show briefly then auto-hide.
/// Active/error states show immediately via `TtsSetState` (host calls show).
#[must_use]
pub fn map_tts_state(
    state: &TtsPlayerState,
    preview_text: &str,
    error_message: Option<&str>,
) -> Vec<UiCmd> {
    let overlay = map_tts_player_state(state);
    let error_message = match overlay {
        TtsOverlayState::Error => {
            let raw = error_message.map(str::trim).filter(|s| !s.is_empty());
            Some(match raw {
                Some(msg) => redact_user_visible_text(msg),
                None => "TTS error".to_string(),
            })
        }
        _ => None,
    };
    vec![UiCmd::TtsSetState {
        state: overlay,
        preview_text: preview_text.to_string(),
        error_message,
    }]
}

// ── Payload-safe text for UI surfaces ─────────────────────────────────────

/// Defensive redaction for user-visible error strings (URLs, sensitive paths).
///
/// Local to the UI bridge so caption/TTS error surfaces never depend on the
/// optional `tts` feature. Caps length and collapses whitespace.
///
/// Path heuristic is **narrow**: only `~/…` and absolute paths under known
/// roots (`/home`, `/tmp`, `/var`, …). Bare fractions like `1/10` are kept.
#[must_use]
pub fn redact_user_visible_text(message: &str) -> String {
    let mut out = String::with_capacity(message.len().min(256));
    let chars: Vec<char> = message.chars().collect();
    let mut idx = 0usize;
    while idx < chars.len() {
        let c = chars[idx];
        // http:// or https://
        if c == 'h' && matches_at(&chars, idx, "http://") {
            out.push_str("[redacted-url]");
            idx += "http://".chars().count();
            while idx < chars.len() && !chars[idx].is_whitespace() {
                idx += 1;
            }
            continue;
        }
        if c == 'h' && matches_at(&chars, idx, "https://") {
            out.push_str("[redacted-url]");
            idx += "https://".chars().count();
            while idx < chars.len() && !chars[idx].is_whitespace() {
                idx += 1;
            }
            continue;
        }
        // ~/path
        if c == '~' && idx + 1 < chars.len() && chars[idx + 1] == '/' {
            out.push_str("[redacted-path]");
            idx += 2;
            while idx < chars.len() && is_path_body_char(chars[idx]) {
                idx += 1;
            }
            continue;
        }
        // Absolute paths under known sensitive roots only (not "1/10").
        if c == '/' && is_sensitive_abs_path_at(&chars, idx) {
            out.push_str("[redacted-path]");
            idx += 1;
            while idx < chars.len() && is_path_body_char(chars[idx]) {
                idx += 1;
            }
            continue;
        }
        out.push(c);
        idx += 1;
    }
    let collapsed = out.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX: usize = 240;
    if collapsed.chars().count() > MAX {
        let trimmed: String = collapsed.chars().take(MAX.saturating_sub(1)).collect();
        format!("{trimmed}…")
    } else {
        collapsed
    }
}

fn is_path_body_char(c: char) -> bool {
    !(c.is_whitespace() || c == '\'' || c == '"' || c == ',' || c == ')' || c == ']' || c == ';')
}

/// True when `chars[idx] == '/'` begins a sensitive absolute filesystem path.
fn is_sensitive_abs_path_at(chars: &[char], idx: usize) -> bool {
    if idx >= chars.len() || chars[idx] != '/' {
        return false;
    }
    // First path segment after '/'.
    let mut end = idx + 1;
    while end < chars.len() && is_path_body_char(chars[end]) && chars[end] != '/' {
        end += 1;
    }
    if end == idx + 1 {
        return false; // lone '/'
    }
    let segment: String = chars[idx + 1..end].iter().collect();
    matches!(
        segment.as_str(),
        "home"
            | "tmp"
            | "var"
            | "usr"
            | "etc"
            | "opt"
            | "root"
            | "Users"
            | "private"
            | "mnt"
            | "media"
            | "run"
            | "data"
            | "workspaces"
            | "nix"
            | "proc"
            | "sys"
            | "boot"
            | "srv"
            | "org"
            | "Applications"
    )
}

fn matches_at(chars: &[char], idx: usize, pat: &str) -> bool {
    let p: Vec<char> = pat.chars().collect();
    if idx + p.len() > chars.len() {
        return false;
    }
    chars[idx..idx + p.len()] == p[..]
}

// ── OverlaySink via UiCmd ─────────────────────────────────────────────────

/// [`OverlaySink`] that forwards caption mutations as [`UiCmd`] values.
///
/// Pair with [`CaptionDriveMode::OverlaySink`] so the event mapper does not
/// re-apply the same caption transitions.
#[derive(Debug, Clone)]
pub struct UiCmdOverlaySink {
    tx: UiCmdSender,
    /// Optional shared counter if compose wants sink-originated flash tokens.
    flash_tokens: Arc<AtomicU64>,
    flash_secs: u32,
}

impl UiCmdOverlaySink {
    #[must_use]
    pub fn new(tx: UiCmdSender) -> Self {
        Self {
            tx,
            flash_tokens: Arc::new(AtomicU64::new(0)),
            flash_secs: DEFAULT_ERROR_FLASH_SECS,
        }
    }

    #[must_use]
    pub fn with_flash_secs(mut self, secs: u32) -> Self {
        self.flash_secs = secs.max(1);
        self
    }

    /// Emit a tokenized error flash (optional path; session usually uses show+timer).
    pub fn flash_error(&mut self, text: &str) -> u64 {
        let token = self.flash_tokens.fetch_add(1, Ordering::Relaxed) + 1;
        let _ = self.tx.send(UiCmd::CaptionFlashError {
            text: redact_user_visible_text(text),
            token,
            secs: self.flash_secs,
        });
        token
    }

    fn emit(&self, cmd: UiCmd) {
        // Never log SendError/UiCmd: Debug includes caption/TTS text payloads.
        if self.tx.send(cmd).is_err() {
            tracing::warn!("UI cmd channel closed while applying OverlaySink");
        }
    }
}

impl OverlaySink for UiCmdOverlaySink {
    fn show(&mut self, state: OverlayState, text: &str) {
        self.emit(UiCmd::CaptionSetState { state });
        self.emit(UiCmd::CaptionSetText {
            text: text.to_string(),
        });
        self.emit(UiCmd::CaptionShow);
    }

    fn set_state(&mut self, state: OverlayState) {
        self.emit(UiCmd::CaptionSetState { state });
    }

    fn set_text(&mut self, text: &str) {
        self.emit(UiCmd::CaptionSetText {
            text: text.to_string(),
        });
    }

    fn hide(&mut self) {
        self.emit(UiCmd::CaptionHide);
    }

    fn set_debug_text(&mut self, text: &str) {
        self.emit(UiCmd::CaptionSetDebug {
            text: text.to_string(),
        });
    }
}

// ── UI event pump → session commands (coalescing under backpressure) ──────

/// Default bound for the UI→session control coalesce slot (single latest).
pub const DEFAULT_UI_SESSION_BRIDGE_CAPACITY: usize = 1;

/// Outcome of forwarding one UI-originated session command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiSessionEnqueue {
    /// Accepted by the integrator enqueue callback.
    Queued,
    /// Enqueue returned full/false; stored in latest-state coalesce slot.
    Coalesced,
}

/// Stats from one [`pump_ui_events_to_session`] drain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UiPumpStats {
    /// UI events that mapped to a session command.
    pub mapped: usize,
    /// Commands accepted by enqueue immediately.
    pub queued: usize,
    /// Commands retained in the coalesce slot (enqueue full).
    pub coalesced: usize,
    /// Non-command UI events (wizard, etc.).
    pub ignored: usize,
}

/// Whether a session command from the UI must not be silently dropped.
#[must_use]
pub fn is_ui_control_command(cmd: &SessionCommand) -> bool {
    matches!(
        cmd,
        SessionCommand::TtsPause
            | SessionCommand::TtsResume
            | SessionCommand::TtsTogglePause
            | SessionCommand::TtsRestart
            | SessionCommand::TtsStop
            | SessionCommand::TtsSetSpeed(_)
            | SessionCommand::TtsSelectVoice(_)
    )
}

/// Bounded latest-state bridge for UI → session control commands.
///
/// When the session command queue is full, TTS control (Stop/Pause/…) is kept
/// in a single newest-wins slot instead of being dropped. Integration should
/// periodically [`UiToSessionCommandBridge::drain_into`] after/during pumps.
///
/// ## Stop sticky policy
///
/// An **unsent** [`SessionCommand::TtsStop`] stays sticky against every other
/// control — including [`SessionCommand::TtsRestart`] — until it is successfully
/// enqueued. Restart may be sent later on a subsequent pump once Stop has been
/// delivered; it must not erase a pending Stop under backpressure.
///
/// ## Locking
///
/// Coalesce mutex is never held across the integrator `enqueue` callback
/// (take → unlock → enqueue → put-back/merge on failure) so re-entrant
/// `try_forward`/`drain_into` from `enqueue` cannot deadlock.
///
/// ```text
/// loop {
///     pump_ui_events_to_session(&ui_rx, &bridge, |cmd| session.try_enqueue(cmd).is_ok());
///     bridge.drain_into(|cmd| session.try_enqueue(cmd).is_ok());
/// }
/// ```
#[derive(Clone, Debug, Default)]
pub struct UiToSessionCommandBridge {
    coalesce: Arc<Mutex<Option<SessionCommand>>>,
    coalesced_total: Arc<AtomicU64>,
    coalesce_overwrites: Arc<AtomicU64>,
    /// Coalesced commands that were Stop (terminal control intent).
    stop_coalesced: Arc<AtomicU64>,
}

impl UiToSessionCommandBridge {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many times a command was stored due to enqueue pressure.
    #[must_use]
    pub fn coalesced_total(&self) -> u64 {
        self.coalesced_total.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn coalesce_overwrites(&self) -> u64 {
        self.coalesce_overwrites.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn stop_coalesced(&self) -> u64 {
        self.stop_coalesced.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn has_coalesced(&self) -> bool {
        self.coalesce.lock().expect("ui session bridge").is_some()
    }

    /// Take the latest coalesced command, if any.
    pub fn take_coalesced(&self) -> Option<SessionCommand> {
        self.coalesce.lock().expect("ui session bridge").take()
    }

    /// Forward one command: try enqueue, else coalesce (never silent-drop controls).
    pub fn try_forward<F>(&self, cmd: SessionCommand, enqueue: &mut F) -> UiSessionEnqueue
    where
        F: FnMut(SessionCommand) -> bool,
    {
        // Best-effort: push any prior coalesce first so Stop isn't stuck behind.
        let _ = self.flush_coalesce(enqueue);

        if enqueue(cmd.clone()) {
            UiSessionEnqueue::Queued
        } else {
            self.store_coalesced(cmd);
            UiSessionEnqueue::Coalesced
        }
    }

    /// Drain coalesce slot into enqueue (call after pump / on idle ticks).
    ///
    /// Returns `true` if a command was delivered.
    pub fn drain_into<F>(&self, enqueue: &mut F) -> bool
    where
        F: FnMut(SessionCommand) -> bool,
    {
        self.flush_coalesce(enqueue)
    }

    /// Take under mutex, enqueue **outside** the lock, put-back/merge on failure.
    fn flush_coalesce<F>(&self, enqueue: &mut F) -> bool
    where
        F: FnMut(SessionCommand) -> bool,
    {
        let pending = {
            let mut slot = self.coalesce.lock().expect("ui session bridge");
            slot.take()
        };
        let Some(pending) = pending else {
            return false;
        };
        if enqueue(pending.clone()) {
            true
        } else {
            // Still full — restore with Stop-sticky merge if a newer cmd arrived.
            self.restore_coalesced(pending);
            false
        }
    }

    fn store_coalesced(&self, cmd: SessionCommand) {
        let is_stop = matches!(cmd, SessionCommand::TtsStop);
        let mut slot = self.coalesce.lock().expect("ui session bridge");
        if slot.is_some() {
            self.coalesce_overwrites.fetch_add(1, Ordering::Relaxed);
        }
        // Unsent Stop is sticky against *all* other controls (including Restart).
        if let Some(prev) = slot.as_ref() {
            if matches!(prev, SessionCommand::TtsStop) && !matches!(cmd, SessionCommand::TtsStop) {
                self.coalesced_total.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        *slot = Some(cmd);
        self.coalesced_total.fetch_add(1, Ordering::Relaxed);
        if is_stop {
            self.stop_coalesced.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Put a failed deliver back; if a newer command arrived while unlocked,
    /// merge with Stop-sticky policy (unsent Stop always wins).
    fn restore_coalesced(&self, failed: SessionCommand) {
        let mut slot = self.coalesce.lock().expect("ui session bridge");
        match slot.take() {
            None => {
                *slot = Some(failed);
            }
            Some(newer) => {
                *slot = Some(merge_ui_coalesce(failed, newer));
                self.coalesce_overwrites.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Merge two coalesced UI controls: unsent Stop beats everything; else newer wins.
fn merge_ui_coalesce(
    older_failed: SessionCommand,
    arrived_while_unlocked: SessionCommand,
) -> SessionCommand {
    if matches!(older_failed, SessionCommand::TtsStop)
        || matches!(arrived_while_unlocked, SessionCommand::TtsStop)
    {
        SessionCommand::TtsStop
    } else {
        arrived_while_unlocked
    }
}

/// Drain UI events into session commands via a coalescing bridge.
///
/// `enqueue` must be non-blocking (`try_send` / `try_enqueue`) and return
/// `false` when the session command queue is full — never block the UI pump.
///
/// Control commands that cannot be queued are retained on `bridge` (newest wins,
/// with unsent Stop sticky against all other controls) until [`UiToSessionCommandBridge::drain_into`].
pub fn pump_ui_events_to_session<F>(
    event_rx: &UiEventReceiver,
    bridge: &UiToSessionCommandBridge,
    mut enqueue: F,
) -> UiPumpStats
where
    F: FnMut(SessionCommand) -> bool,
{
    let mut stats = UiPumpStats::default();
    loop {
        match event_rx.try_recv() {
            Ok(ev) => match map_ui_event_to_session_command(&ev) {
                Some(cmd) => {
                    stats.mapped += 1;
                    match bridge.try_forward(cmd, &mut enqueue) {
                        UiSessionEnqueue::Queued => stats.queued += 1,
                        UiSessionEnqueue::Coalesced => stats.coalesced += 1,
                    }
                }
                None => stats.ignored += 1,
            },
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
        }
    }
    stats
}

/// Convenience: pump then immediately attempt one coalesce drain.
pub fn pump_and_drain_ui_events_to_session<F>(
    event_rx: &UiEventReceiver,
    bridge: &UiToSessionCommandBridge,
    mut enqueue: F,
) -> UiPumpStats
where
    F: FnMut(SessionCommand) -> bool,
{
    let mut stats = pump_ui_events_to_session(event_rx, bridge, &mut enqueue);
    if bridge.drain_into(&mut enqueue) {
        // Coalesce delivered — count as queued for observability.
        stats.queued = stats.queued.saturating_add(1);
        stats.coalesced = stats.coalesced.saturating_sub(1);
    }
    stats
}

// ── Essential-first merge task ────────────────────────────────────────────

/// Out-of-band signal from the merge task (not a [`UiCmd`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiBridgeSignal {
    /// Session finished shutdown; UI host should quit the main loop.
    ShutdownComplete,
}

/// Run an essential-first merge of session event lanes into UI commands.
///
/// Ordering guarantees:
/// - When both lanes are ready, **essentials are always preferred** (`biased`).
/// - Within a lane, FIFO order is preserved.
/// - [`SessionEvent::ShutdownComplete`] emits [`UiBridgeSignal::ShutdownComplete`]
///   and ends the task (after mapping — currently no UiCmds).
///
/// `cmd_tx` failures (UI gone) end the task cleanly.
pub async fn run_essential_first_ui_merge(
    mut essential_rx: mpsc::Receiver<SessionEvent>,
    mut partial_rx: mpsc::Receiver<SessionEvent>,
    cmd_tx: UiCmdSender,
    mut mapper: SessionToUiMapper,
    signal_tx: Option<oneshot::Sender<UiBridgeSignal>>,
) {
    let mut signal_tx = signal_tx;
    let mut essentials_open = true;
    let mut partials_open = true;

    while essentials_open || partials_open {
        tokio::select! {
            biased;

            ev = essential_rx.recv(), if essentials_open => {
                match ev {
                    Some(event) => {
                        let shutdown = matches!(event, SessionEvent::ShutdownComplete);
                        // Payload-free assert: never format SessionEvent (may hold transcripts).
                        debug_assert!(
                            event_is_essential(&event)
                                || matches!(event, SessionEvent::ShutdownComplete),
                            "essential lane received non-essential event"
                        );
                        if !emit_mapped(&cmd_tx, &mut mapper, &event) {
                            return;
                        }
                        if shutdown {
                            if let Some(tx) = signal_tx.take() {
                                let _ = tx.send(UiBridgeSignal::ShutdownComplete);
                            }
                            return;
                        }
                    }
                    None => essentials_open = false,
                }
            }

            ev = partial_rx.recv(), if partials_open => {
                match ev {
                    Some(event) => {
                        if !emit_mapped(&cmd_tx, &mut mapper, &event) {
                            return;
                        }
                    }
                    None => partials_open = false,
                }
            }
        }
    }
}

fn emit_mapped(cmd_tx: &UiCmdSender, mapper: &mut SessionToUiMapper, event: &SessionEvent) -> bool {
    for cmd in mapper.map_event(event) {
        // is_err() only — never log SendError (Debug embeds UiCmd text payloads).
        if cmd_tx.send(cmd).is_err() {
            tracing::debug!("UI cmd channel closed; merge task stopping");
            return false;
        }
    }
    true
}

// ── GTK main-thread host bootstrap ────────────────────────────────────────

/// Control messages toward the GTK main loop (all `Send`, no GTK types).
///
/// Not a [`UiCmd`] — keeps quit out of the UI protocol crate ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GtkHostControl {
    /// Request `Application::quit` on the GLib main thread.
    Quit,
}

/// Lifecycle notices from the GTK host back to integration (all `Send`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GtkHostLifecycle {
    /// GTK main loop is exiting (quit requested and/or application shutdown).
    Exiting,
}

/// Bounded capacity for the quit control channel.
pub const DEFAULT_GTK_QUIT_CHANNEL_CAPACITY: usize = 4;

/// Cloneable, non-blocking quit requester (background tasks / merge signal).
#[derive(Clone, Debug)]
pub struct GtkQuitSender(std::sync::mpsc::SyncSender<GtkHostControl>);

/// GLib-thread receiver for quit requests.
#[derive(Debug)]
pub struct GtkQuitReceiver(std::sync::mpsc::Receiver<GtkHostControl>);

/// Cloneable lifecycle notifier (GTK thread → integration).
#[derive(Clone, Debug)]
pub struct GtkLifecycleSender(std::sync::mpsc::Sender<GtkHostLifecycle>);

/// Integration receiver for host lifecycle events.
#[derive(Debug)]
pub struct GtkLifecycleReceiver(std::sync::mpsc::Receiver<GtkHostLifecycle>);

/// Why a quit request could not be delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GtkQuitSendError {
    Full,
    Disconnected,
}

impl GtkQuitSender {
    /// Non-blocking quit request (safe from any thread / async task).
    pub fn try_request_quit(&self) -> Result<(), GtkQuitSendError> {
        self.0
            .try_send(GtkHostControl::Quit)
            .map_err(|err| match err {
                std::sync::mpsc::TrySendError::Full(_) => GtkQuitSendError::Full,
                std::sync::mpsc::TrySendError::Disconnected(_) => GtkQuitSendError::Disconnected,
            })
    }

    /// Blocking quit request (prefer [`try_request_quit`] from async contexts).
    pub fn request_quit(&self) -> Result<(), GtkQuitSendError> {
        self.0
            .send(GtkHostControl::Quit)
            .map_err(|_| GtkQuitSendError::Disconnected)
    }
}

impl GtkQuitReceiver {
    /// Non-blocking recv of one control message.
    pub fn try_recv(&self) -> Result<GtkHostControl, std::sync::mpsc::TryRecvError> {
        self.0.try_recv()
    }
}

impl GtkLifecycleSender {
    /// Non-blocking lifecycle notify (payload-free).
    pub fn try_send(&self, event: GtkHostLifecycle) -> Result<(), ()> {
        self.0.send(event).map_err(|_| ())
    }
}

impl GtkLifecycleReceiver {
    pub fn try_recv(&self) -> Result<GtkHostLifecycle, std::sync::mpsc::TryRecvError> {
        self.0.try_recv()
    }

    pub fn recv(&self) -> Result<GtkHostLifecycle, std::sync::mpsc::RecvError> {
        self.0.recv()
    }
}

/// Build a bounded quit channel (sender cloneable across threads).
#[must_use]
pub fn gtk_quit_channel() -> (GtkQuitSender, GtkQuitReceiver) {
    gtk_quit_channel_with_capacity(DEFAULT_GTK_QUIT_CHANNEL_CAPACITY)
}

/// Build a quit channel with an explicit bound (min 1).
#[must_use]
pub fn gtk_quit_channel_with_capacity(capacity: usize) -> (GtkQuitSender, GtkQuitReceiver) {
    let (tx, rx) = std::sync::mpsc::sync_channel(capacity.max(1));
    (GtkQuitSender(tx), GtkQuitReceiver(rx))
}

/// Build an unbounded lifecycle notify channel.
#[must_use]
pub fn gtk_lifecycle_channel() -> (GtkLifecycleSender, GtkLifecycleReceiver) {
    let (tx, rx) = std::sync::mpsc::channel();
    (GtkLifecycleSender(tx), GtkLifecycleReceiver(rx))
}

/// Map a merge-task signal into a GTK host control message.
#[must_use]
pub fn gtk_control_from_bridge_signal(signal: UiBridgeSignal) -> Option<GtkHostControl> {
    match signal {
        UiBridgeSignal::ShutdownComplete => Some(GtkHostControl::Quit),
    }
}

/// Forward [`UiBridgeSignal::ShutdownComplete`] → quit (non-blocking).
///
/// Returns whether a quit request was successfully enqueued.
pub fn forward_bridge_signal_to_quit(
    signal: UiBridgeSignal,
    quit: &GtkQuitSender,
) -> Result<bool, GtkQuitSendError> {
    match gtk_control_from_bridge_signal(signal) {
        Some(GtkHostControl::Quit) => quit.try_request_quit().map(|()| true),
        None => Ok(false),
    }
}

/// Drain pending quit requests (non-blocking).
///
/// Returns `true` if at least one [`GtkHostControl::Quit`] was observed.
/// Does not treat sender-disconnect as quit (explicit Quit only).
#[must_use]
pub fn drain_quit_requests(rx: &GtkQuitReceiver) -> bool {
    let mut quit = false;
    loop {
        match rx.try_recv() {
            Ok(GtkHostControl::Quit) => quit = true,
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
        }
    }
    quit
}

/// Inputs for the long-lived GTK host (all `Send`; no GTK objects).
#[derive(Debug)]
pub struct GtkMainHostBootstrap {
    pub caption_vm: CaptionVm,
    pub tts_vm: Option<TtsVm>,
    pub cmd_rx: UiCmdReceiver,
    pub event_tx: UiEventSender,
    /// Background → GLib quit requests ([`GtkQuitSender`] held by integration).
    pub quit_rx: GtkQuitReceiver,
    /// Optional GLib → integration lifecycle notices (e.g. loop exiting).
    pub lifecycle_tx: Option<GtkLifecycleSender>,
}

/// Run the main-service GTK application on the **current** thread.
///
/// # Threading contract
///
/// - Must be invoked on the thread that will own the GLib main context.
/// - Only channel endpoints and view-models cross thread boundaries.
/// - GTK widgets are created and mutated exclusively inside `activate` /
///   GLib timeouts on this thread — never moved or shared across threads.
/// - Quit is requested via [`GtkQuitSender`] (any thread); polled here on GLib.
/// - Application shutdown notifies [`GtkLifecycleSender`] with
///   [`GtkHostLifecycle::Exiting`] (still no GTK types on that channel).
/// - The welcome wizard is **not** hosted here (use the wizard binary/entry).
/// - Does **not** add `UiCmd::Quit` — quit stays on the compose control channel.
///
/// # Integration wiring
///
/// ```text
/// let (quit_tx, quit_rx) = gtk_quit_channel();
/// let (life_tx, life_rx) = gtk_lifecycle_channel();
/// // background:
/// //   on UiBridgeSignal::ShutdownComplete => quit_tx.try_request_quit();
/// // main/UI thread:
/// //   run_gtk_main_host(GtkMainHostBootstrap { quit_rx, lifecycle_tx: Some(life_tx), ... });
/// //   // after return, or concurrently: life_rx.recv() == Exiting
/// ```
///
/// Returns the GTK application exit code.
#[cfg(feature = "ui")]
pub fn run_gtk_main_host(boot: GtkMainHostBootstrap) -> i32 {
    use std::time::Duration;

    use glib::ControlFlow;
    use gtk4::prelude::{ApplicationExt, ApplicationExtManual, ObjectExt};
    use shuvoice_ui::{APP_APPLICATION_ID, UiHost};

    let GtkMainHostBootstrap {
        caption_vm,
        tts_vm,
        cmd_rx,
        event_tx,
        quit_rx,
        lifecycle_tx,
    } = boot;

    let app = gtk4::Application::builder()
        .application_id(APP_APPLICATION_ID)
        .build();

    // Notify integration when the application object shuts down (user close or quit).
    if let Some(life_tx) = lifecycle_tx.clone() {
        app.connect_shutdown(move |_| {
            let _ = life_tx.try_send(GtkHostLifecycle::Exiting);
        });
    }

    // Option-take so activate runs once without cloning GTK-bound state.
    // `connect_activate` requires `Fn` (not `FnMut`), so use a cell.
    let once = std::cell::RefCell::new(Some((caption_vm, tts_vm, cmd_rx, event_tx, quit_rx)));
    app.connect_activate(move |app| {
        let Some((caption_vm, tts_vm, cmd_rx, event_tx, quit_rx)) = once.borrow_mut().take() else {
            return;
        };
        let host = UiHost::new(app, caption_vm, tts_vm, event_tx);
        // Host is moved into the GLib timeout pump; stays on this thread.
        let _cmd_pump = host.attach_cmd_pump(cmd_rx);

        // Quit pump: poll Send-safe channel on the GLib thread, then Application::quit.
        // Holds only a Downgrade'd app ref — never moves GTK widgets across threads.
        let app_weak = app.downgrade();
        let _quit_pump = glib::timeout_add_local(Duration::from_millis(16), move || {
            if drain_quit_requests(&quit_rx) {
                if let Some(app) = app_weak.upgrade() {
                    app.quit();
                }
                return ControlFlow::Break;
            }
            // Keep polling while the app lives. If quit senders disconnect without
            // an explicit Quit, continue (explicit control only).
            ControlFlow::Continue
        });
    });

    let code = app.run();
    // Best-effort second notify if shutdown signal raced.
    if let Some(life_tx) = lifecycle_tx {
        let _ = life_tx.try_send(GtkHostLifecycle::Exiting);
    }
    exit_code_to_i32(code)
}

#[cfg(feature = "ui")]
fn exit_code_to_i32(code: glib::ExitCode) -> i32 {
    i32::from(code)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shuvoice_app::types::RecordingStatus;
    use shuvoice_ui::UiBus;
    use shuvoice_ui::wizard::WizardPageId;

    #[test]
    fn ui_event_to_session_command_table() {
        let rows: Vec<(UiEvent, Option<SessionCommand>)> = vec![
            (UiEvent::TtsPause, Some(SessionCommand::TtsPause)),
            (UiEvent::TtsResume, Some(SessionCommand::TtsResume)),
            (UiEvent::TtsRestart, Some(SessionCommand::TtsRestart)),
            (UiEvent::TtsStop, Some(SessionCommand::TtsStop)),
            (
                UiEvent::TtsVoiceSelected {
                    voice_id: "af_heart".into(),
                },
                Some(SessionCommand::TtsSelectVoice("af_heart".into())),
            ),
            (
                UiEvent::TtsSpeedChanged { speed: 1.25 },
                Some(SessionCommand::TtsSetSpeed(1.25)),
            ),
            (UiEvent::WizardBack, None),
            (UiEvent::WizardNext, None),
            (UiEvent::WizardFinishRequested, None),
            (UiEvent::WizardLaunch, None),
            (UiEvent::WizardCancelDownload, None),
            (UiEvent::WizardClosed { completed: true }, None),
            (
                UiEvent::WizardPageChanged {
                    page: WizardPageId::Welcome,
                },
                None,
            ),
        ];

        for (ev, expect) in rows {
            let got = map_ui_event_to_session_command(&ev);
            match (got, expect) {
                (None, None) => {}
                (Some(SessionCommand::TtsPause), Some(SessionCommand::TtsPause))
                | (Some(SessionCommand::TtsResume), Some(SessionCommand::TtsResume))
                | (Some(SessionCommand::TtsRestart), Some(SessionCommand::TtsRestart))
                | (Some(SessionCommand::TtsStop), Some(SessionCommand::TtsStop)) => {}
                (
                    Some(SessionCommand::TtsSelectVoice(a)),
                    Some(SessionCommand::TtsSelectVoice(b)),
                ) => assert_eq!(a, b),
                (Some(SessionCommand::TtsSetSpeed(a)), Some(SessionCommand::TtsSetSpeed(b))) => {
                    assert!((a - b).abs() < 1e-9);
                }
                (g, e) => panic!("mismatch for {ev:?}: got={g:?} expect={e:?}"),
            }
        }
    }

    #[test]
    fn tts_state_roundtrip_table() {
        let rows = [
            TtsPlayerState::Idle,
            TtsPlayerState::Synthesizing,
            TtsPlayerState::Playing,
            TtsPlayerState::Paused,
            TtsPlayerState::Error,
        ];
        for st in rows {
            let overlay = map_tts_player_state(&st);
            assert_eq!(map_tts_overlay_state(overlay), st);
            assert_eq!(overlay.as_str(), st.as_str());
        }
    }

    #[test]
    fn session_event_mapping_overlay_sink_mode_skips_caption_events() {
        let mut mapper = SessionToUiMapper::new(CaptionDriveMode::OverlaySink);
        let events = [
            SessionEvent::OverlayShow {
                state: OverlayState::Listening,
                text: "Listening…".into(),
            },
            SessionEvent::OverlayUpdate {
                state: Some(OverlayState::Processing),
                text: None,
            },
            SessionEvent::OverlayHide,
            SessionEvent::ErrorToast {
                text: "⚠ ASR error".into(),
            },
        ];
        for ev in &events {
            assert!(
                mapper.map_event(ev).is_empty(),
                "OverlaySink mode must not re-apply {ev:?}"
            );
        }
    }

    #[test]
    fn session_event_mapping_events_mode_error_flash_dedupes_overlay_show() {
        let mut mapper = SessionToUiMapper::new(CaptionDriveMode::SessionEvents).with_flash_secs(5);

        // flash_error order: ErrorToast then OverlayShow(Error)
        let cmds = mapper.map_event(&SessionEvent::ErrorToast {
            text: "⚠ ASR error (1/10) — see logs".into(),
        });
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            UiCmd::CaptionFlashError { text, token, secs } => {
                assert!(text.contains("ASR error"));
                assert_eq!(*token, 1);
                assert_eq!(*secs, 5);
            }
            other => panic!("expected flash, got {other:?}"),
        }

        let dup = mapper.map_event(&SessionEvent::OverlayShow {
            state: OverlayState::Error,
            text: "⚠ ASR error (1/10) — see logs".into(),
        });
        assert!(dup.is_empty(), "paired OverlayShow must be suppressed");

        // Non-error show still maps.
        let show = mapper.map_event(&SessionEvent::OverlayShow {
            state: OverlayState::Listening,
            text: "Listening…".into(),
        });
        assert_eq!(
            show,
            vec![
                UiCmd::CaptionSetState {
                    state: OverlayState::Listening
                },
                UiCmd::CaptionSetText {
                    text: "Listening…".into()
                },
                UiCmd::CaptionShow,
            ]
        );
    }

    #[test]
    fn session_event_mapping_tts_and_shutdown_table() {
        let mut mapper = SessionToUiMapper::new(CaptionDriveMode::OverlaySink);

        let rows: Vec<(SessionEvent, usize)> = vec![
            (
                SessionEvent::TtsState {
                    state: TtsPlayerState::Playing,
                    preview_text: "hello".into(),
                },
                1,
            ),
            (
                SessionEvent::TtsState {
                    state: TtsPlayerState::Idle,
                    preview_text: "hello".into(),
                },
                1,
            ),
            (
                SessionEvent::TtsError {
                    message: "network".into(),
                },
                1,
            ),
            (SessionEvent::ShutdownComplete, 0),
            (SessionEvent::Status(RecordingStatus::Idle), 0),
            (SessionEvent::PartialTranscript { text: "x".into() }, 0),
            (SessionEvent::FinalTranscript { text: "x".into() }, 0),
            (SessionEvent::AsrRecovered, 0),
            (
                SessionEvent::CudaFallbackApplied {
                    detail: "oom".into(),
                },
                0,
            ),
        ];

        for (ev, n) in rows {
            let cmds = mapper.map_event(&ev);
            assert_eq!(cmds.len(), n, "event {ev:?}");
            if let SessionEvent::TtsState {
                state: TtsPlayerState::Idle,
                ..
            } = &ev
            {
                match &cmds[0] {
                    UiCmd::TtsSetState {
                        state: TtsOverlayState::Idle,
                        ..
                    } => {}
                    other => panic!("idle should set TTS idle for autohide: {other:?}"),
                }
            }
            if let SessionEvent::TtsError { .. } = &ev {
                match &cmds[0] {
                    UiCmd::TtsSetState {
                        state: TtsOverlayState::Error,
                        error_message: Some(msg),
                        ..
                    } => assert_eq!(msg, "network"),
                    other => panic!("expected TTS error state: {other:?}"),
                }
            }
        }
    }

    #[test]
    fn overlay_sink_emits_caption_cmds_without_duplicate_semantics() {
        let bus = UiBus::new();
        let (cmd_tx, cmd_rx, _etx, _erx) = bus.split();
        let mut sink = UiCmdOverlaySink::new(cmd_tx);

        sink.show(OverlayState::Listening, "Listening…");
        sink.set_text("partial");
        sink.set_state(OverlayState::Processing);
        sink.set_debug_text("dbg");
        sink.hide();

        let mut got = Vec::new();
        while let Ok(cmd) = cmd_rx.try_recv() {
            got.push(cmd);
        }
        assert_eq!(
            got,
            vec![
                UiCmd::CaptionSetState {
                    state: OverlayState::Listening
                },
                UiCmd::CaptionSetText {
                    text: "Listening…".into()
                },
                UiCmd::CaptionShow,
                UiCmd::CaptionSetText {
                    text: "partial".into()
                },
                UiCmd::CaptionSetState {
                    state: OverlayState::Processing
                },
                UiCmd::CaptionSetDebug { text: "dbg".into() },
                UiCmd::CaptionHide,
            ]
        );
    }

    #[test]
    fn overlay_sink_flash_error_token_monotonic() {
        let bus = UiBus::new();
        let (cmd_tx, cmd_rx, _, _) = bus.split();
        let mut sink = UiCmdOverlaySink::new(cmd_tx).with_flash_secs(3);
        let t1 = sink.flash_error("a");
        let t2 = sink.flash_error("b");
        assert_eq!(t1, 1);
        assert_eq!(t2, 2);
        let mut tokens = Vec::new();
        while let Ok(cmd) = cmd_rx.try_recv() {
            if let UiCmd::CaptionFlashError { token, secs, .. } = cmd {
                tokens.push((token, secs));
            }
        }
        assert_eq!(tokens, vec![(1, 3), (2, 3)]);
    }

    #[tokio::test]
    async fn essential_first_merge_prefers_essentials_and_signals_shutdown() {
        let (ess_tx, ess_rx) = mpsc::channel::<SessionEvent>(8);
        let (part_tx, part_rx) = mpsc::channel::<SessionEvent>(8);
        let bus = UiBus::new();
        let (cmd_tx, cmd_rx, _, _) = bus.split();
        let (sig_tx, sig_rx) = oneshot::channel();

        let merge = tokio::spawn(run_essential_first_ui_merge(
            ess_rx,
            part_rx,
            cmd_tx,
            SessionToUiMapper::new(CaptionDriveMode::SessionEvents),
            Some(sig_tx),
        ));

        // Enqueue partial first, then essential — merge must still deliver essential mapping.
        part_tx
            .send(SessionEvent::OverlayUpdate {
                state: Some(OverlayState::Processing),
                text: Some("p".into()),
            })
            .await
            .unwrap();
        ess_tx
            .send(SessionEvent::OverlayShow {
                state: OverlayState::Listening,
                text: "Listening…".into(),
            })
            .await
            .unwrap();
        ess_tx.send(SessionEvent::ShutdownComplete).await.unwrap();

        // Allow merge to run.
        let sig = tokio::time::timeout(std::time::Duration::from_secs(2), sig_rx)
            .await
            .expect("timeout")
            .expect("signal");
        assert_eq!(sig, UiBridgeSignal::ShutdownComplete);
        merge.await.unwrap();

        let mut cmds = Vec::new();
        while let Ok(cmd) = cmd_rx.try_recv() {
            cmds.push(cmd);
        }
        // Essential show commands should be present; partial may also appear.
        assert!(
            cmds.iter().any(|c| matches!(
                c,
                UiCmd::CaptionShow
                    | UiCmd::CaptionSetState {
                        state: OverlayState::Listening
                    }
            )),
            "expected essential caption cmds, got {cmds:?}"
        );
    }

    #[test]
    fn pump_ui_events_skips_wizard_and_enqueues_tts() {
        let bus = UiBus::new();
        let (_ctx, _crx, etx, erx) = bus.split();
        etx.send(UiEvent::WizardBack).unwrap();
        etx.send(UiEvent::TtsPause).unwrap();
        etx.send(UiEvent::TtsSpeedChanged { speed: 1.5 }).unwrap();

        let bridge = UiToSessionCommandBridge::new();
        let mut got = Vec::new();
        let stats = pump_ui_events_to_session(&erx, &bridge, |cmd| {
            got.push(cmd);
            true
        });
        assert_eq!(stats.mapped, 2);
        assert_eq!(stats.queued, 2);
        assert_eq!(stats.ignored, 1);
        assert_eq!(stats.coalesced, 0);
        assert!(matches!(got[0], SessionCommand::TtsPause));
        assert!(matches!(got[1], SessionCommand::TtsSetSpeed(s) if (s - 1.5).abs() < 1e-9));
    }

    #[test]
    fn ui_control_coalesces_stop_when_enqueue_full() {
        let bus = UiBus::new();
        let (_c, _cr, etx, erx) = bus.split();
        etx.send(UiEvent::TtsPause).unwrap();
        etx.send(UiEvent::TtsStop).unwrap();

        let bridge = UiToSessionCommandBridge::new();
        let mut accepted = 0u32;
        // First enqueue succeeds, rest fail (simulate capacity 1 session queue).
        let stats = pump_ui_events_to_session(&erx, &bridge, |_cmd| {
            if accepted == 0 {
                accepted += 1;
                true
            } else {
                false
            }
        });
        assert_eq!(stats.queued, 1);
        assert_eq!(stats.coalesced, 1);
        assert!(bridge.has_coalesced());
        // Latest control should be Stop (second event).
        match bridge.take_coalesced() {
            Some(SessionCommand::TtsStop) => {}
            other => panic!("expected coalesced Stop, got {other:?}"),
        }
        assert!(bridge.stop_coalesced() >= 1 || bridge.coalesced_total() >= 1);
    }

    #[test]
    fn ui_control_stop_sticky_against_weaker_coalesce() {
        let bridge = UiToSessionCommandBridge::new();
        // Force coalesce Stop.
        assert_eq!(
            bridge.try_forward(SessionCommand::TtsStop, &mut |_| false),
            UiSessionEnqueue::Coalesced
        );
        // Weaker Pause must not overwrite Stop.
        assert_eq!(
            bridge.try_forward(SessionCommand::TtsPause, &mut |_| false),
            UiSessionEnqueue::Coalesced
        );
        match bridge.take_coalesced() {
            Some(SessionCommand::TtsStop) => {}
            other => panic!("Stop must remain sticky: {other:?}"),
        }
    }

    #[test]
    fn ui_control_unsent_stop_sticky_against_restart() {
        let bridge = UiToSessionCommandBridge::new();
        assert_eq!(
            bridge.try_forward(SessionCommand::TtsStop, &mut |_| false),
            UiSessionEnqueue::Coalesced
        );
        // Restart must not erase an unsent Stop.
        assert_eq!(
            bridge.try_forward(SessionCommand::TtsRestart, &mut |_| false),
            UiSessionEnqueue::Coalesced
        );
        match bridge.take_coalesced() {
            Some(SessionCommand::TtsStop) => {}
            other => panic!("unsent Stop must beat Restart: {other:?}"),
        }
    }

    #[test]
    fn ui_control_flush_put_back_outside_lock_and_preserves_stop() {
        let bridge = UiToSessionCommandBridge::new();
        assert_eq!(
            bridge.try_forward(SessionCommand::TtsStop, &mut |_| false),
            UiSessionEnqueue::Coalesced
        );
        // Failed drain puts Stop back.
        assert!(!bridge.drain_into(&mut |_| false));
        match bridge.take_coalesced() {
            Some(SessionCommand::TtsStop) => {}
            other => panic!("Stop restored after failed drain: {other:?}"),
        }
    }

    #[test]
    fn ui_control_reentrant_enqueue_does_not_deadlock() {
        let bridge = UiToSessionCommandBridge::new();
        let bridge2 = bridge.clone();
        // Enqueue callback re-enters try_forward — must not deadlock on mutex.
        let mut n = 0;
        let _ = bridge.try_forward(SessionCommand::TtsPause, &mut |_cmd| {
            n += 1;
            if n == 1 {
                // Re-enter while outer enqueue is in progress (lock must be free).
                let _ = bridge2.try_forward(SessionCommand::TtsStop, &mut |_| false);
            }
            // Reject so Pause is stored or merged; Stop should win sticky.
            false
        });
        // After reentrancy: Stop sticky should be present.
        match bridge.take_coalesced() {
            Some(SessionCommand::TtsStop) => {}
            other => panic!("expected Stop after reentrant coalesce: {other:?}"),
        }
    }

    #[test]
    fn tts_error_event_is_reredacted_for_ui() {
        let mut mapper = SessionToUiMapper::for_session_events();
        let cmds = mapper.map_event(&SessionEvent::TtsError {
            message: "fail https://api.example/v1/x path /tmp/secret.onnx".into(),
        });
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            UiCmd::TtsSetState {
                state: TtsOverlayState::Error,
                error_message: Some(msg),
                ..
            } => {
                assert!(!msg.contains("https://"), "{msg}");
                assert!(!msg.contains("/tmp/"), "{msg}");
                assert!(
                    msg.contains("[redacted-url]") || msg.contains("[redacted-path]"),
                    "{msg}"
                );
            }
            other => panic!("expected redacted TTS error cmd: {other:?}"),
        }
    }

    #[test]
    fn redact_user_visible_text_strips_url_and_path() {
        let r = redact_user_visible_text("boom https://x.test/a /home/u/secret.bin ok");
        assert!(!r.contains("https://"));
        assert!(!r.contains("/home/"));
        assert!(r.contains("[redacted-url]") || r.contains("[redacted-path]"));
        assert!(r.contains("ok") || r.contains("boom"));
    }

    #[test]
    fn redact_user_visible_text_keeps_fractions_and_redacts_tmp() {
        let r = redact_user_visible_text("⚠ ASR error (1/10) — see /tmp/secret.log");
        assert!(r.contains("1/10"), "fraction must survive: {r}");
        assert!(!r.contains("/tmp/"));
        assert!(r.contains("[redacted-path]"), "{r}");
    }

    #[test]
    fn error_toast_is_redacted_in_session_events_mode() {
        let mut mapper = SessionToUiMapper::for_session_events();
        let cmds = mapper.map_event(&SessionEvent::ErrorToast {
            text: "fail https://api.example/x /home/u/a.bin".into(),
        });
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            UiCmd::CaptionFlashError { text, .. } => {
                assert!(!text.contains("https://"), "{text}");
                assert!(!text.contains("/home/"), "{text}");
            }
            other => panic!("expected flash: {other:?}"),
        }
    }

    #[test]
    fn paired_overlay_sink_path_locks_mode() {
        let bus = UiBus::new();
        let (tx, rx, _, _) = bus.split();
        let path = paired_overlay_sink_path(tx);
        assert_eq!(path.mode(), CaptionDriveMode::OverlaySink);
        assert_eq!(path.mapper.mode(), CaptionDriveMode::OverlaySink);
        let PairedOverlaySinkPath {
            mut sink,
            mut mapper,
        } = path;
        // Sink applies captions…
        sink.show(OverlayState::Listening, "hi");
        // …mapper must not duplicate OverlayShow.
        let cmds = mapper.map_event(&SessionEvent::OverlayShow {
            state: OverlayState::Listening,
            text: "hi".into(),
        });
        assert!(
            cmds.is_empty(),
            "paired path must not double-apply captions"
        );
        // And the sink did emit something.
        assert!(rx.try_recv().is_ok());
    }

    #[test]
    fn no_duplicate_caption_when_overlay_sink_mode() {
        let bus = UiBus::new();
        let (tx, rx, _, _) = bus.split();
        let path = paired_overlay_sink_path(tx).with_flash_secs(5);
        let PairedOverlaySinkPath {
            mut sink,
            mut mapper,
        } = path;
        sink.show(OverlayState::Error, "⚠ ASR");
        // ErrorToast + OverlayShow Error must not add caption cmds in OverlaySink mode.
        let c1 = mapper.map_event(&SessionEvent::ErrorToast {
            text: "⚠ ASR".into(),
        });
        let c2 = mapper.map_event(&SessionEvent::OverlayShow {
            state: OverlayState::Error,
            text: "⚠ ASR".into(),
        });
        assert!(c1.is_empty());
        assert!(c2.is_empty());
        // Sink side produced caption cmds only.
        let mut n = 0;
        while rx.try_recv().is_ok() {
            n += 1;
        }
        assert!(n >= 3, "sink should have set state/text/show");
    }

    #[test]
    fn overlay_sink_closed_channel_swallows_send_without_panic() {
        // Drop receiver so send fails; emit path must not panic or require
        // formatting SendError (which would embed caption text in Debug).
        let bus = UiBus::new();
        let (cmd_tx, cmd_rx, _, _) = bus.split();
        drop(cmd_rx);
        let mut sink = UiCmdOverlaySink::new(cmd_tx);
        sink.show(
            OverlayState::Listening,
            "secret transcript must not be logged via SendError Debug",
        );
        sink.set_text("another secret");
        sink.hide();
    }

    #[test]
    fn gtk_quit_channel_try_request_and_drain() {
        let (tx, rx) = gtk_quit_channel_with_capacity(1);
        assert!(!drain_quit_requests(&rx));
        tx.try_request_quit().unwrap();
        // Coalesce multiple quits while full: second may be Full.
        let second = tx.try_request_quit();
        assert!(matches!(second, Ok(()) | Err(GtkQuitSendError::Full)));
        assert!(drain_quit_requests(&rx));
        // Drained — empty again.
        assert!(!drain_quit_requests(&rx));
    }

    #[test]
    fn gtk_quit_disconnected_is_explicit_error() {
        let (tx, rx) = gtk_quit_channel();
        drop(rx);
        assert_eq!(tx.try_request_quit(), Err(GtkQuitSendError::Disconnected));
    }

    #[test]
    fn bridge_signal_maps_to_quit_control() {
        assert_eq!(
            gtk_control_from_bridge_signal(UiBridgeSignal::ShutdownComplete),
            Some(GtkHostControl::Quit)
        );
        let (tx, rx) = gtk_quit_channel();
        assert_eq!(
            forward_bridge_signal_to_quit(UiBridgeSignal::ShutdownComplete, &tx),
            Ok(true)
        );
        assert!(drain_quit_requests(&rx));
    }

    #[test]
    fn gtk_lifecycle_channel_exiting_roundtrip() {
        let (tx, rx) = gtk_lifecycle_channel();
        tx.try_send(GtkHostLifecycle::Exiting).unwrap();
        assert_eq!(rx.try_recv().unwrap(), GtkHostLifecycle::Exiting);
    }

    #[test]
    fn drain_quit_does_not_treat_disconnect_as_quit() {
        let (_tx, rx) = gtk_quit_channel();
        drop(_tx);
        // No explicit Quit was sent — disconnect alone must not imply quit.
        assert!(!drain_quit_requests(&rx));
    }
}
