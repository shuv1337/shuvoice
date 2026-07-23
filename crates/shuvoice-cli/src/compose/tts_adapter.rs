//! Compose-layer TTS adapter: `Config` → backend/player → app [`TtsEngine`].
//!
//! # Integration notes
//!
//! Expected crate deps / features (declared by the integration owner):
//! - `shuvoice-app` (`TtsEngine`, `SessionCommand`, `TtsPlayerState`)
//! - `shuvoice-core` (`Config`, `expand_user_path`, `TtsBackendKind`, …)
//! - `shuvoice-tts` with features `cpal-output` + `worker-proto` (Melo)
//! - CLI features: `tts`, `tts-worker` (required for Melo; fail-closed otherwise)
//! - `tokio` (already in `shuvoice-cli`)
//!
//! Wire from `compose/mod.rs` once the root declares the module.
//!
//! # Design
//!
//! - Settings mapping is pure and complete for Kokoro / OpenAI / ElevenLabs /
//!   Piper (`local`) / MeloTTS.
//! - Melo is **worker-proto only**: legacy helper is never selected. Building a
//!   Melo backend without CLI feature `tts-worker` fails closed.
//! - Playback device indices are resolved to **enumerated output device names**
//!   before reaching CPAL (no bare numeric selector).
//! - Player `on_event` uses a bounded non-blocking bridge that yields
//!   [`SessionCommand::TtsPlayerUpdate`] without a `SessionHandle`. Under
//!   backpressure, updates coalesce into a latest-state slot so terminal
//!   `Error` / `Idle` cannot be silently lost.
//!
//! # Bridge drain contract (integration)
//!
//! Prefer fallible drain so terminal/latest state stays sticky under queue pressure:
//!
//! ```text
//! loop {
//!     // FIFO first, then sticky coalesce. Only removes an item when enqueue returns true.
//!     bridge.drain_with_try(&mut rx, |cmd| session.try_enqueue(cmd).is_ok());
//! }
//! ```
//!
//! The player callback never blocks. Coalesced terminal state is sticky across
//! consumer `Full` failures via [`TtsPlayerUpdateBridge::drain_with_try`] /
//! [`TtsPlayerUpdateBridge::try_deliver_coalesced`] (peek/ack). Avoid bare
//! [`take_coalesced`] unless the caller can guarantee enqueue success or put-back.

#![allow(clippy::collapsible_if)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::let_and_return)]
#![allow(clippy::double_must_use)]
#![allow(clippy::result_unit_err)]
#![allow(dead_code)] // public helpers for composition/tests not all referenced yet
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
#[cfg(feature = "tts-worker")]
use std::time::Duration;

use shuvoice_app::traits::TtsEngine;
use shuvoice_app::{SessionCommand, TtsPlayerState};
use shuvoice_core::{
    Config, DeviceRef, TtsBackendKind, TtsCapabilities, VoiceInfo as CoreVoiceInfo,
    expand_user_path as core_expand_user_path,
};
use shuvoice_tts::player::AudioOutputFactory;
#[cfg(feature = "tts-worker")]
use shuvoice_tts::types::DEFAULT_MELOTTS_VOICE_ID;
use shuvoice_tts::types::{
    BackendId, Capabilities, DEFAULT_ELEVENLABS_TTS_BASE_URL, DEFAULT_OPENAI_TTS_BASE_URL,
    PlayerEvent, PlayerState, VoiceInfo as TtsVoiceInfo,
};
use shuvoice_tts::{
    CpalAudioOutputFactory, CpalOutputConfig, SharedBackend, TtsBackendSettings, TtsError,
    TtsPlayer, create_tts_backend, parse_backend_name, redact_for_ui,
};
use tokio::sync::mpsc;

/// Default bound for the player → session command bridge FIFO.
pub const DEFAULT_TTS_PLAYER_BRIDGE_CAPACITY: usize = 64;

// ── Config → settings ─────────────────────────────────────────────────────

/// Map core backend kind → TTS crate [`BackendId`].
#[must_use]
pub fn backend_id_from_kind(kind: TtsBackendKind) -> BackendId {
    match kind {
        TtsBackendKind::Elevenlabs => BackendId::ElevenLabs,
        TtsBackendKind::Openai => BackendId::OpenAi,
        TtsBackendKind::Local => BackendId::Local,
        TtsBackendKind::Melotts => BackendId::MeloTts,
        TtsBackendKind::Kokoro => BackendId::Kokoro,
    }
}

/// Inverse of [`backend_id_from_kind`] (lossless for known ids).
#[must_use]
pub fn kind_from_backend_id(id: BackendId) -> TtsBackendKind {
    match id {
        BackendId::ElevenLabs => TtsBackendKind::Elevenlabs,
        BackendId::OpenAi => TtsBackendKind::Openai,
        BackendId::Local => TtsBackendKind::Local,
        BackendId::MeloTts => TtsBackendKind::Melotts,
        BackendId::Kokoro => TtsBackendKind::Kokoro,
    }
}

/// Whether the CLI build includes Melo worker-proto support (`tts-worker`).
#[must_use]
pub const fn tts_worker_feature_enabled() -> bool {
    cfg!(feature = "tts-worker")
}

/// Build complete [`TtsBackendSettings`] from validated runtime [`Config`].
///
/// Always clears the legacy Melo helper path. Paths use
/// [`shuvoice_core::expand_user_path`].
///
/// Prefer [`create_shared_backend`] (not raw `create_tts_backend`) so Melo
/// fails closed without `tts-worker`.
#[must_use]
pub fn tts_backend_settings_from_config(cfg: &Config) -> TtsBackendSettings {
    let backend = backend_id_from_kind(cfg.tts_backend);
    TtsBackendSettings {
        backend,
        api_key_env: cfg.tts_api_key_env.clone(),
        output_format: cfg.tts_output_format.clone(),
        max_chars: cfg.tts_max_chars.max(1) as usize,
        request_timeout_sec: cfg.tts_request_timeout_sec,
        default_voice_id: cfg.tts_default_voice_id.clone(),
        model_id: cfg.tts_model_id.clone(),
        local_model_path: cfg
            .tts_local_model_path
            .as_deref()
            .map(expand_config_path)
            .filter(|p| !p.as_os_str().is_empty()),
        local_voice: cfg.tts_local_voice.clone(),
        piper_binary: None,
        melotts_venv_path: cfg
            .tts_melotts_venv_path
            .as_deref()
            .map(expand_config_path)
            .filter(|p| !p.as_os_str().is_empty()),
        melotts_device: cfg.tts_melotts_device.as_str().to_string(),
        // Worker-proto only — never wire the legacy Python helper.
        melotts_helper_script: None,
        melotts_worker_root: None,
        melotts_python_binary: None,
        melotts_worker_spawn: None,
        melotts_worker_command: None,
        melotts_worker_env: Vec::new(),
        kokoro_base_url: cfg.tts_kokoro_base_url.clone(),
        elevenlabs_base_url: DEFAULT_ELEVENLABS_TTS_BASE_URL.to_string(),
        openai_base_url: DEFAULT_OPENAI_TTS_BASE_URL.to_string(),
    }
}

/// Resolve backend name string (config/CLI) → [`BackendId`].
pub fn parse_tts_backend_name(name: &str) -> Result<BackendId, TtsError> {
    parse_backend_name(name)
}

/// Construct a shared backend with compose-layer policy.
///
/// - Non-Melo: delegates to [`create_tts_backend`].
/// - Melo: **worker-proto only**. Requires CLI feature `tts-worker`; never
///   selects [`shuvoice_tts::MeloWireMode::LegacyHelper`].
pub fn create_shared_backend(settings: &TtsBackendSettings) -> Result<SharedBackend, TtsError> {
    // Belt-and-suspenders: ignore any helper path that sneaks in.
    let mut settings = settings.clone();
    settings.melotts_helper_script = None;
    // Never carry a legacy helper; worker overrides stay available for tests.

    if settings.backend == BackendId::MeloTts {
        return create_melotts_worker_only(&settings);
    }
    create_tts_backend(&settings)
}

fn create_melotts_worker_only(settings: &TtsBackendSettings) -> Result<SharedBackend, TtsError> {
    if !tts_worker_feature_enabled() {
        return Err(TtsError::config(
            "MeloTTS requires the CLI feature `tts-worker` (shuvoice-tts worker-proto). \
             Rebuild with `--features tts-worker`. The legacy melo_helper.py path is not supported.",
        ));
    }
    create_melotts_worker_only_inner(settings)
}

#[cfg(feature = "tts-worker")]
fn create_melotts_worker_only_inner(
    settings: &TtsBackendSettings,
) -> Result<SharedBackend, TtsError> {
    use shuvoice_tts::{MeloTtsBackend, MeloTtsConfig, MeloWireMode};

    let timeout = if settings.request_timeout_sec.is_finite() && settings.request_timeout_sec > 0.0
    {
        Duration::from_secs_f64(settings.request_timeout_sec)
    } else {
        Duration::from_secs(30)
    };

    let mut cfg = MeloTtsConfig {
        device: settings.melotts_device.clone(),
        max_chars: settings.max_chars,
        request_timeout: timeout,
        default_voice_id: if settings.default_voice_id.is_empty() {
            DEFAULT_MELOTTS_VOICE_ID.into()
        } else {
            settings.default_voice_id.clone()
        },
        helper_script: None,
        python_binary: settings.melotts_python_binary.clone(),
        wire_mode: MeloWireMode::WorkerProto,
        worker_root: settings.melotts_worker_root.clone(),
        worker_spawn: settings.melotts_worker_spawn.clone(),
        worker_command: settings.melotts_worker_command.clone(),
        worker_env: settings.melotts_worker_env.clone(),
        ..MeloTtsConfig::default()
    };
    // Re-assert after default() in case Default ever changes.
    cfg.wire_mode = MeloWireMode::WorkerProto;
    cfg.helper_script = None;

    if let Some(venv) = &settings.melotts_venv_path {
        cfg.venv_path = venv.clone();
    }

    debug_assert_eq!(cfg.wire_mode, MeloWireMode::WorkerProto);
    debug_assert!(cfg.helper_script.is_none());
    Ok(Arc::new(MeloTtsBackend::new(cfg)))
}

#[cfg(not(feature = "tts-worker"))]
fn create_melotts_worker_only_inner(
    _settings: &TtsBackendSettings,
) -> Result<SharedBackend, TtsError> {
    // Unreachable when tts_worker_feature_enabled() is checked first; keep a
    // second hard error for defense in depth.
    Err(TtsError::config(
        "MeloTTS worker-proto support was not compiled into this binary (`tts-worker` missing)",
    ))
}

// ── Lossless maps ─────────────────────────────────────────────────────────

/// Player crate state → app session state (lossless).
#[must_use]
pub fn map_player_state(state: PlayerState) -> TtsPlayerState {
    match state {
        PlayerState::Idle => TtsPlayerState::Idle,
        PlayerState::Synthesizing => TtsPlayerState::Synthesizing,
        PlayerState::Playing => TtsPlayerState::Playing,
        PlayerState::Paused => TtsPlayerState::Paused,
        PlayerState::Error => TtsPlayerState::Error,
    }
}

/// App session state → player crate state (lossless).
#[must_use]
pub fn map_app_player_state(state: &TtsPlayerState) -> PlayerState {
    match state {
        TtsPlayerState::Idle => PlayerState::Idle,
        TtsPlayerState::Synthesizing => PlayerState::Synthesizing,
        TtsPlayerState::Playing => PlayerState::Playing,
        TtsPlayerState::Paused => PlayerState::Paused,
        TtsPlayerState::Error => PlayerState::Error,
    }
}

/// Whether a player/app state is terminal for UI/session lifecycle.
#[must_use]
pub fn is_terminal_player_state(state: &TtsPlayerState) -> bool {
    matches!(state, TtsPlayerState::Idle | TtsPlayerState::Error)
}

/// TTS crate capabilities → core/UI capabilities (lossless field map).
#[must_use]
pub fn map_capabilities(caps: &Capabilities) -> TtsCapabilities {
    TtsCapabilities {
        supports_streaming: caps.supports_streaming,
        supports_voice_list: caps.supports_voice_list,
        requires_api_key: caps.requires_api_key,
        supports_speed_control: caps.supports_speed_control,
        speed_min: caps.speed_min,
        speed_max: caps.speed_max,
    }
}

/// Core capabilities → TTS crate capabilities (lossless field map).
#[must_use]
pub fn map_capabilities_to_tts(caps: &TtsCapabilities) -> Capabilities {
    Capabilities {
        supports_streaming: caps.supports_streaming,
        supports_voice_list: caps.supports_voice_list,
        requires_api_key: caps.requires_api_key,
        supports_speed_control: caps.supports_speed_control,
        speed_min: caps.speed_min,
        speed_max: caps.speed_max,
    }
}

/// TTS voice entry → core/UI voice entry (lossless).
#[must_use]
pub fn map_voice_to_core(voice: &TtsVoiceInfo) -> CoreVoiceInfo {
    CoreVoiceInfo {
        id: voice.id.clone(),
        name: voice.name.clone(),
        description: voice.description.clone(),
    }
}

/// Core/UI voice entry → TTS voice entry (lossless).
#[must_use]
pub fn map_voice_from_core(voice: &CoreVoiceInfo) -> TtsVoiceInfo {
    TtsVoiceInfo {
        id: voice.id.clone(),
        name: voice.name.clone(),
        description: voice.description.clone(),
    }
}

/// Redact a failure string for UI / session surfaces.
#[must_use]
pub fn redact_tts_error_message(message: &str) -> String {
    redact_for_ui(message)
}

// ── Player update bridge (no SessionHandle) ───────────────────────────────

/// Outcome of a non-blocking bridge send.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsBridgeEnqueue {
    /// Accepted into the bounded FIFO channel.
    Queued,
    /// FIFO was full; stored in the latest-state coalesce slot (replaced prior).
    Coalesced,
}

/// Why a bridge send failed hard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsBridgeSendError {
    /// Receiver dropped; player should stop emitting.
    Closed,
    /// A non-[`SessionCommand::TtsPlayerUpdate`] was offered while the FIFO was full.
    ///
    /// The command is not stored (this bridge only carries player updates).
    UnexpectedCommand,
}

/// Snapshot of bridge pressure counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TtsBridgeDiagnostics {
    pub queued: u64,
    pub coalesced: u64,
    /// Coalesce overwrites that displaced a previous coalesced update.
    pub coalesce_overwrites: u64,
    /// Coalesced updates that were terminal (`Idle` / `Error`).
    pub terminal_coalesced: u64,
}

/// Stats from one [`TtsPlayerUpdateBridge::drain_with_try`] pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TtsBridgeDrainStats {
    /// Commands successfully delivered to the consumer callback.
    pub delivered: usize,
    /// `true` when a deliver attempt failed and at least one update remains pending
    /// (undelivered FIFO head and/or sticky coalesce).
    pub pending_remaining: bool,
}

#[derive(Debug, Clone)]
struct CoalescedUpdate {
    state: TtsPlayerState,
    error_message: Option<String>,
}

impl CoalescedUpdate {
    fn into_command(self) -> SessionCommand {
        SessionCommand::TtsPlayerUpdate {
            state: self.state,
            error_message: self.error_message,
        }
    }

    fn from_command(cmd: SessionCommand) -> Option<Self> {
        match cmd {
            SessionCommand::TtsPlayerUpdate {
                state,
                error_message,
            } => Some(Self {
                state,
                error_message,
            }),
            _ => None,
        }
    }
}

/// Bounded, non-blocking bridge from player callbacks into session commands.
///
/// # Backpressure semantics
///
/// 1. `try_send` into a bounded FIFO (`capacity`).
/// 2. If **full**, the update is written to a single **latest-state slot**
///    (replaceable / newest-wins). The callback never blocks and never
///    silently drops the newest state — including terminal `Error` / `Idle`.
/// 3. Integration must drain with [`drain_with_try`] (FIFO then sticky coalesce).
///    Failed consumer enqueues **do not** drop coalesced terminal/latest state.
///
/// Holds only channel + sticky slots — never a `SessionHandle`.
#[derive(Clone, Debug)]
pub struct TtsPlayerUpdateBridge {
    tx: mpsc::Sender<SessionCommand>,
    /// Single undelivered FIFO head retained when consumer enqueue fails mid-drain.
    undelivered_front: Arc<Mutex<Option<SessionCommand>>>,
    coalesce: Arc<Mutex<Option<CoalescedUpdate>>>,
    queued: Arc<AtomicU64>,
    coalesced: Arc<AtomicU64>,
    coalesce_overwrites: Arc<AtomicU64>,
    terminal_coalesced: Arc<AtomicU64>,
}

impl TtsPlayerUpdateBridge {
    /// Create a bridge + FIFO receiver pair with the given capacity (min 1).
    #[must_use]
    pub fn new(capacity: usize) -> (Self, mpsc::Receiver<SessionCommand>) {
        let (tx, rx) = mpsc::channel(capacity.max(1));
        (Self::from_sender(tx), rx)
    }

    /// Wrap an existing session command sender (e.g. actor enqueue lane).
    #[must_use]
    pub fn from_sender(tx: mpsc::Sender<SessionCommand>) -> Self {
        Self {
            tx,
            undelivered_front: Arc::new(Mutex::new(None)),
            coalesce: Arc::new(Mutex::new(None)),
            queued: Arc::new(AtomicU64::new(0)),
            coalesced: Arc::new(AtomicU64::new(0)),
            coalesce_overwrites: Arc::new(AtomicU64::new(0)),
            terminal_coalesced: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Access the underlying FIFO sender.
    #[must_use]
    pub fn sender(&self) -> mpsc::Sender<SessionCommand> {
        self.tx.clone()
    }

    /// Diagnostics counters (queued / coalesced / overwrites).
    #[must_use]
    pub fn diagnostics(&self) -> TtsBridgeDiagnostics {
        TtsBridgeDiagnostics {
            queued: self.queued.load(Ordering::Relaxed),
            coalesced: self.coalesced.load(Ordering::Relaxed),
            coalesce_overwrites: self.coalesce_overwrites.load(Ordering::Relaxed),
            terminal_coalesced: self.terminal_coalesced.load(Ordering::Relaxed),
        }
    }

    /// Non-blocking enqueue of a player event as [`SessionCommand::TtsPlayerUpdate`].
    ///
    /// On FIFO pressure, coalesces into the latest-state slot instead of
    /// dropping — so terminal `Error`/`Idle` remain observable via
    /// [`try_deliver_coalesced`] / [`drain_with_try`].
    pub fn try_send_player_event(
        &self,
        event: &PlayerEvent,
    ) -> Result<TtsBridgeEnqueue, TtsBridgeSendError> {
        let cmd = session_command_from_player_event(event);
        self.try_send_command(cmd)
    }

    /// Non-blocking enqueue of an already-built command (tests / adapters).
    pub fn try_send_command(
        &self,
        cmd: SessionCommand,
    ) -> Result<TtsBridgeEnqueue, TtsBridgeSendError> {
        // Prefer moving any prior coalesce into FIFO first (best-effort).
        self.flush_coalesce_into_fifo();

        match self.tx.try_send(cmd) {
            Ok(()) => {
                self.queued.fetch_add(1, Ordering::Relaxed);
                Ok(TtsBridgeEnqueue::Queued)
            }
            Err(mpsc::error::TrySendError::Closed(cmd)) => {
                // Preserve the update in coalesce so a late drain can still see
                // terminal state if the integration inspects before drop.
                if let SessionCommand::TtsPlayerUpdate {
                    state,
                    error_message,
                } = cmd
                {
                    self.store_coalesced(state, error_message);
                }
                Err(TtsBridgeSendError::Closed)
            }
            Err(mpsc::error::TrySendError::Full(cmd)) => match cmd {
                SessionCommand::TtsPlayerUpdate {
                    state,
                    error_message,
                } => {
                    self.store_coalesced(state, error_message);
                    Ok(TtsBridgeEnqueue::Coalesced)
                }
                other => {
                    // Non-update commands shouldn't appear on this bridge.
                    // Never Debug-format SessionCommand (may hold speak text).
                    let _ = other;
                    tracing::warn!("unexpected non-TtsPlayerUpdate on TTS bridge while full");
                    Err(TtsBridgeSendError::UnexpectedCommand)
                }
            },
        }
    }

    /// Peek a clone of the sticky coalesced update without clearing it.
    #[must_use]
    pub fn peek_coalesced(&self) -> Option<SessionCommand> {
        self.coalesce
            .lock()
            .expect("tts bridge coalesce")
            .as_ref()
            .map(|u| SessionCommand::TtsPlayerUpdate {
                state: u.state.clone(),
                error_message: u.error_message.clone(),
            })
    }

    /// Peek whether a coalesced update is waiting (does not clear).
    #[must_use]
    pub fn has_coalesced(&self) -> bool {
        self.coalesce.lock().expect("tts bridge coalesce").is_some()
    }

    /// `true` when undelivered FIFO head and/or coalesce still hold work.
    #[must_use]
    pub fn has_pending(&self) -> bool {
        self.undelivered_front
            .lock()
            .expect("tts bridge front")
            .is_some()
            || self.has_coalesced()
    }

    /// Take the latest coalesced update (if any) as a session command.
    ///
    /// **Prefer** [`try_deliver_coalesced`] / [`drain_with_try`]: this method
    /// removes the sticky slot even if the caller later fails to enqueue.
    pub fn take_coalesced(&self) -> Option<SessionCommand> {
        self.coalesce
            .lock()
            .expect("tts bridge coalesce")
            .take()
            .map(CoalescedUpdate::into_command)
    }

    /// Peek/ack deliver for the sticky coalesce slot.
    ///
    /// Calls `enqueue` **outside** the coalesce mutex. On `false` / when the
    /// slot was empty, the previous sticky value is restored (or kept). Returns
    /// `true` only when a command was accepted by `enqueue`.
    pub fn try_deliver_coalesced<F>(&self, enqueue: &mut F) -> bool
    where
        F: FnMut(SessionCommand) -> bool,
    {
        let pending = {
            let mut slot = self.coalesce.lock().expect("tts bridge coalesce");
            slot.take()
        };
        let Some(pending) = pending else {
            return false;
        };
        let cmd = pending.clone().into_command();
        if enqueue(cmd) {
            true
        } else {
            // Restore sticky — merge if a newer update arrived while unlocked.
            self.restore_coalesced(pending);
            false
        }
    }

    /// Drain helper (infallible consumer): FIFO then coalesced.
    ///
    /// Prefer [`drain_with_try`] when enqueue can fail — this variant always
    /// acknowledges delivered items (legacy / tests with infallible sinks).
    pub fn drain_into<F>(&self, rx: &mut mpsc::Receiver<SessionCommand>, mut f: F) -> usize
    where
        F: FnMut(SessionCommand),
    {
        self.drain_with_try(rx, |cmd| {
            f(cmd);
            true
        })
        .delivered
    }

    /// Fallible drain: FIFO (plus undelivered head) first, then sticky coalesce.
    ///
    /// `enqueue` must be non-blocking and return `false` on full/reject.
    /// On `false`, the failed command is retained (FIFO head slot or sticky
    /// coalesce) and draining stops so order is preserved for the next pass.
    pub fn drain_with_try<F>(
        &self,
        rx: &mut mpsc::Receiver<SessionCommand>,
        mut enqueue: F,
    ) -> TtsBridgeDrainStats
    where
        F: FnMut(SessionCommand) -> bool,
    {
        let mut delivered = 0usize;

        // 1) Previously failed FIFO head (ordered before new recv).
        {
            let front = {
                let mut g = self.undelivered_front.lock().expect("tts bridge front");
                g.take()
            };
            if let Some(cmd) = front {
                if enqueue(cmd.clone()) {
                    delivered += 1;
                } else {
                    *self.undelivered_front.lock().expect("tts bridge front") = Some(cmd);
                    return TtsBridgeDrainStats {
                        delivered,
                        pending_remaining: true,
                    };
                }
            }
        }

        // 2) FIFO channel.
        loop {
            match rx.try_recv() {
                Ok(cmd) => {
                    if enqueue(cmd.clone()) {
                        delivered += 1;
                    } else {
                        // Retain head without clobbering newer sticky coalesce.
                        *self.undelivered_front.lock().expect("tts bridge front") = Some(cmd);
                        return TtsBridgeDrainStats {
                            delivered,
                            pending_remaining: true,
                        };
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => break,
            }
        }

        // 3) Sticky coalesce last (newest-wins terminal).
        if self.try_deliver_coalesced(&mut enqueue) {
            delivered += 1;
        }

        TtsBridgeDrainStats {
            delivered,
            pending_remaining: self.has_pending(),
        }
    }

    /// Build an `on_event` callback suitable for [`TtsPlayer`] / builder.
    ///
    /// Never blocks. Full FIFO → coalesce latest state (including terminal).
    #[must_use]
    pub fn on_event_callback(&self) -> impl Fn(PlayerEvent) + Send + Sync + 'static {
        let bridge = self.clone();
        move |event: PlayerEvent| match bridge.try_send_player_event(&event) {
            Ok(TtsBridgeEnqueue::Queued) => {}
            Ok(TtsBridgeEnqueue::Coalesced) => {
                // state is an enum label only — never log EventInfo/message/text.
                tracing::debug!(
                    state = event.state.as_str(),
                    terminal = is_terminal_player_state(&map_player_state(event.state)),
                    "TTS player update bridge full; coalesced latest state"
                );
            }
            Err(TtsBridgeSendError::Closed) => {
                tracing::debug!(
                    state = event.state.as_str(),
                    "TTS player update bridge closed"
                );
            }
            Err(TtsBridgeSendError::UnexpectedCommand) => {
                tracing::warn!("TTS player update bridge rejected unexpected command");
            }
        }
    }

    fn store_coalesced(&self, state: TtsPlayerState, error_message: Option<String>) {
        let terminal = is_terminal_player_state(&state);
        let mut slot = self.coalesce.lock().expect("tts bridge coalesce");
        if slot.is_some() {
            self.coalesce_overwrites.fetch_add(1, Ordering::Relaxed);
        }
        *slot = Some(CoalescedUpdate {
            state,
            error_message,
        });
        self.coalesced.fetch_add(1, Ordering::Relaxed);
        if terminal {
            self.terminal_coalesced.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn restore_coalesced(&self, failed: CoalescedUpdate) {
        let mut slot = self.coalesce.lock().expect("tts bridge coalesce");
        match slot.take() {
            None => {
                *slot = Some(failed);
            }
            Some(newer) => {
                // A newer update landed while enqueue ran — keep newest-wins.
                // (failed was older; drop it.)
                let _ = failed;
                *slot = Some(newer);
            }
        }
    }

    fn flush_coalesce_into_fifo(&self) {
        let mut slot = self.coalesce.lock().expect("tts bridge coalesce");
        let Some(pending) = slot.take() else {
            return;
        };
        let cmd = pending.clone().into_command();
        match self.tx.try_send(cmd) {
            Ok(()) => {
                self.queued.fetch_add(1, Ordering::Relaxed);
            }
            Err(mpsc::error::TrySendError::Full(_)) | Err(mpsc::error::TrySendError::Closed(_)) => {
                // Put it back — newest still wins if a newer update arrives.
                *slot = Some(pending);
            }
        }
    }
}

/// Convert a player event into the session actor re-entry command.
#[must_use]
pub fn session_command_from_player_event(event: &PlayerEvent) -> SessionCommand {
    let error_message = event
        .info
        .message
        .as_ref()
        .map(|m| redact_tts_error_message(m));
    SessionCommand::TtsPlayerUpdate {
        state: map_player_state(event.state),
        // Session keeps spoken preview text and emits TtsState from it.
        error_message,
    }
}

// ── Output device resolution ──────────────────────────────────────────────

/// Resolve a configured playback [`DeviceRef`] into a CPAL **output device name**.
///
/// - `None` / empty name → host default (`None`)
/// - `Name` → trimmed exact name
/// - `Index` → **deterministic name** from `CpalAudioOutputFactory::list_output_devices()`
///   ordering (not a bare numeric selector — avoids name/`"0"` ambiguity)
///
/// Mirrors the input-side index resolution in `audio_bridge`, but for outputs.
pub fn resolve_output_device_selector(
    device: Option<&DeviceRef>,
) -> Result<Option<String>, TtsError> {
    match device {
        None => Ok(None),
        Some(DeviceRef::Name(name)) => {
            let trimmed = name.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
        Some(DeviceRef::Index(idx)) => resolve_output_device_index(*idx).map(Some),
    }
}

/// Resolve a single device ref to a non-empty selector string.
pub fn device_ref_to_selector(device: &DeviceRef) -> Result<String, TtsError> {
    resolve_output_device_selector(Some(device))?.ok_or_else(|| {
        TtsError::config("tts_playback_device resolved to empty selector (use default device)")
    })
}

fn resolve_output_device_index(idx: i64) -> Result<String, TtsError> {
    if idx < 0 {
        return Err(TtsError::config(format!(
            "tts_playback_device index {idx} is negative"
        )));
    }
    let i = idx as usize;
    let names = CpalAudioOutputFactory::list_output_devices().map_err(|_err| {
        // Static label only — do not forward provider error strings (may include paths).
        TtsError::audio(format!(
            "failed to enumerate TTS output devices while resolving index {idx}"
        ))
    })?;
    names.get(i).cloned().ok_or_else(|| {
        TtsError::config(format!(
            "tts_playback_device index {idx} out of range ({} output devices)",
            names.len()
        ))
    })
}

/// Build a CPAL output factory from config playback device selection.
///
/// Indices are resolved to real device names before construction.
/// Requires `shuvoice-tts` built with `cpal-output`.
pub fn cpal_output_factory_from_config(cfg: &Config) -> Result<CpalAudioOutputFactory, TtsError> {
    let device = resolve_output_device_selector(cfg.tts_playback_device.as_ref())?;
    Ok(CpalAudioOutputFactory::new(CpalOutputConfig {
        device,
        ..CpalOutputConfig::default()
    }))
}

// ── Backend readiness ─────────────────────────────────────────────────────

/// Collect backend dependency errors (paths/URLs already backend-owned labels).
///
/// Empty means the backend is ready enough to construct a player. Does **not**
/// discover or inject Melo `worker_root` — central wiring must set that on
/// [`TtsBackendSettings`] before [`create_shared_backend`].
#[must_use]
pub fn backend_dependency_errors(backend: &SharedBackend) -> Vec<String> {
    backend.dependency_errors()
}

/// Fail closed when the backend reports any dependency error.
///
/// Used by [`PlayerTtsEngine::from_config`] / [`PlayerTtsEngine::from_backend`]
/// so missing Melo root/python surfaces at compose time (exit 78) rather than
/// first-speak.
pub fn ensure_backend_ready(backend: &SharedBackend) -> Result<(), TtsError> {
    let errors = backend_dependency_errors(backend);
    if errors.is_empty() {
        return Ok(());
    }
    // Join static-ish dependency labels only — backends must not embed transcripts.
    Err(TtsError::config(errors.join("; ")))
}

// ── PlayerTtsEngine ───────────────────────────────────────────────────────

/// App-facing [`TtsEngine`] backed by [`TtsPlayer`].
pub struct PlayerTtsEngine {
    player: TtsPlayer,
    caps: Capabilities,
}

impl PlayerTtsEngine {
    /// Wrap an existing player + its backend capability snapshot.
    ///
    /// Does **not** re-check dependency errors (caller already validated).
    #[must_use]
    pub fn new(player: TtsPlayer, caps: Capabilities) -> Self {
        Self { player, caps }
    }

    /// Construct backend + CPAL player + event bridge from config.
    ///
    /// Uses the current Tokio handle when `runtime` is `None`.
    /// Default playback speed comes from `cfg.tts_playback_speed`.
    ///
    /// Melo requires `tts-worker` (fail closed via [`create_shared_backend`]).
    /// Does **not** invent `melotts_worker_root` — when unset, readiness fails
    /// closed via [`ensure_backend_ready`] after construction.
    pub fn from_config(
        cfg: &Config,
        bridge: &TtsPlayerUpdateBridge,
        runtime: Option<tokio::runtime::Handle>,
    ) -> Result<Self, TtsError> {
        let settings = tts_backend_settings_from_config(cfg);
        let backend = create_shared_backend(&settings)?;
        ensure_backend_ready(&backend)?;
        let factory: Arc<dyn AudioOutputFactory> = Arc::new(cpal_output_factory_from_config(cfg)?);
        Self::from_backend(
            backend,
            factory,
            cfg.tts_playback_speed,
            bridge,
            runtime.unwrap_or_else(tokio::runtime::Handle::current),
        )
    }

    /// Construct from an already-built backend and output factory.
    ///
    /// Validates [`ensure_backend_ready`] before building the player so central
    /// paths that inject Melo root still fail closed on missing python/root.
    pub fn from_backend(
        backend: SharedBackend,
        output_factory: Arc<dyn AudioOutputFactory>,
        default_speed: f64,
        bridge: &TtsPlayerUpdateBridge,
        runtime: tokio::runtime::Handle,
    ) -> Result<Self, TtsError> {
        ensure_backend_ready(&backend)?;
        let caps = backend.capabilities();
        let player = TtsPlayer::builder(backend, output_factory)
            .playback_speed(default_speed)
            .runtime_handle(runtime)
            .on_event(bridge.on_event_callback())
            .build();
        Ok(Self { player, caps })
    }

    /// Borrow the inner player (status payload / advanced control).
    #[must_use]
    pub fn player(&self) -> &TtsPlayer {
        &self.player
    }

    /// Backend capabilities snapshot taken at construction.
    #[must_use]
    pub fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    /// Core-shaped capabilities for UI / wizard surfaces.
    #[must_use]
    pub fn core_capabilities(&self) -> TtsCapabilities {
        map_capabilities(&self.caps)
    }
}

impl TtsEngine for PlayerTtsEngine {
    fn state(&self) -> TtsPlayerState {
        map_player_state(self.player.state())
    }

    fn supports_speed_control(&self) -> bool {
        self.caps.supports_speed_control
    }

    fn speed_bounds(&self) -> Option<(f64, f64)> {
        self.caps.speed_bounds()
    }

    fn speak(&mut self, text: &str, voice_id: &str, model_id: &str) -> Result<bool, String> {
        self.player
            .speak(text, voice_id, model_id)
            .map_err(|err| redact_tts_error_message(&err.to_string()))
    }

    fn pause(&mut self) -> bool {
        self.player.pause()
    }

    fn resume(&mut self) -> bool {
        self.player.resume()
    }

    fn toggle_pause(&mut self) -> bool {
        self.player.toggle_pause()
    }

    fn restart(&mut self) -> bool {
        self.player.restart()
    }

    fn stop(&mut self) -> bool {
        self.player.stop()
    }

    fn set_playback_speed(&mut self, speed: f64) -> f64 {
        self.player.set_playback_speed(speed)
    }
}

// ── Path helpers ──────────────────────────────────────────────────────────

/// Expand `~` using the shared core helper; empty/whitespace → empty path.
fn expand_config_path(raw: &str) -> PathBuf {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return PathBuf::new();
    }
    core_expand_user_path(trimmed)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shuvoice_tts::types::EventInfo;
    use std::path::Path;

    fn playing_ev() -> PlayerEvent {
        PlayerEvent {
            state: PlayerState::Playing,
            info: EventInfo::default(),
        }
    }

    fn idle_ev() -> PlayerEvent {
        PlayerEvent {
            state: PlayerState::Idle,
            info: EventInfo::default(),
        }
    }

    fn error_ev(msg: &str) -> PlayerEvent {
        PlayerEvent {
            state: PlayerState::Error,
            info: EventInfo {
                message: Some(msg.into()),
                error_class: Some("Http".into()),
                ..EventInfo::default()
            },
        }
    }

    #[test]
    fn backend_kind_roundtrip_table() {
        let rows = [
            (
                TtsBackendKind::Elevenlabs,
                BackendId::ElevenLabs,
                "elevenlabs",
            ),
            (TtsBackendKind::Openai, BackendId::OpenAi, "openai"),
            (TtsBackendKind::Local, BackendId::Local, "local"),
            (TtsBackendKind::Melotts, BackendId::MeloTts, "melotts"),
            (TtsBackendKind::Kokoro, BackendId::Kokoro, "kokoro"),
        ];
        for (kind, id, name) in rows {
            assert_eq!(backend_id_from_kind(kind), id, "{name}");
            assert_eq!(kind_from_backend_id(id), kind, "{name}");
            assert_eq!(parse_tts_backend_name(name).unwrap(), id, "{name}");
            assert_eq!(
                parse_tts_backend_name(&name.to_ascii_uppercase()).unwrap(),
                id,
                "{name}"
            );
        }
    }

    #[test]
    fn player_state_roundtrip_table() {
        let rows = [
            (PlayerState::Idle, TtsPlayerState::Idle),
            (PlayerState::Synthesizing, TtsPlayerState::Synthesizing),
            (PlayerState::Playing, TtsPlayerState::Playing),
            (PlayerState::Paused, TtsPlayerState::Paused),
            (PlayerState::Error, TtsPlayerState::Error),
        ];
        for (player, app) in rows {
            assert_eq!(map_player_state(player), app);
            assert_eq!(map_app_player_state(&app), player);
            assert_eq!(map_player_state(player).as_str(), player.as_str());
        }
        assert!(is_terminal_player_state(&TtsPlayerState::Idle));
        assert!(is_terminal_player_state(&TtsPlayerState::Error));
        assert!(!is_terminal_player_state(&TtsPlayerState::Playing));
    }

    #[test]
    fn capabilities_roundtrip_lossless() {
        let caps = Capabilities {
            supports_streaming: true,
            supports_voice_list: false,
            requires_api_key: true,
            supports_speed_control: true,
            speed_min: Some(0.5),
            speed_max: Some(2.0),
        };
        let core = map_capabilities(&caps);
        let back = map_capabilities_to_tts(&core);
        assert_eq!(back, caps);
        assert_eq!(core.speed_bounds(), caps.speed_bounds());
    }

    #[test]
    fn voice_roundtrip_lossless() {
        let v = TtsVoiceInfo::with_description("af_heart", "Heart", "warm");
        let core = map_voice_to_core(&v);
        let back = map_voice_from_core(&core);
        assert_eq!(back, v);
    }

    #[test]
    fn settings_from_config_table() {
        let rows = [
            (TtsBackendKind::Kokoro, "af_heart", "kokoro"),
            (TtsBackendKind::Openai, "onyx", "gpt-4o-mini-tts"),
            (
                TtsBackendKind::Elevenlabs,
                "zNsotODqUhvbJ5wMG7Ei",
                "eleven_flash_v2_5",
            ),
            (TtsBackendKind::Melotts, "EN-US", "melotts"),
            (TtsBackendKind::Local, "en_US-amy-medium", "piper"),
        ];

        for (kind, voice, model) in rows {
            let cfg = Config::try_with(|c| {
                c.tts_backend = kind;
                c.tts_default_voice_id = voice.into();
                c.tts_model_id = model.into();
                c.tts_playback_speed = 1.25;
                c.tts_max_chars = 1234;
                c.tts_request_timeout_sec = 12.5;
                c.tts_output_format = "pcm_24000".into();
                c.tts_api_key_env = "TEST_KEY".into();
                c.tts_kokoro_base_url = "http://127.0.0.1:9999/v1".into();
                c.tts_local_model_path = Some("~/models/piper".into());
                c.tts_local_voice = Some("en_US-amy-medium".into());
                c.tts_melotts_venv_path = Some("~/melotts-venv".into());
                c.tts_melotts_device = shuvoice_core::MeloTtsDevice::Cpu;
            })
            .unwrap_or_else(|e| panic!("config for {voice}/{model}: {e}"));

            let settings = tts_backend_settings_from_config(&cfg);
            assert_eq!(settings.backend, backend_id_from_kind(kind));
            assert_eq!(settings.default_voice_id, voice);
            assert_eq!(settings.model_id, model);
            assert_eq!(settings.max_chars, 1234);
            assert!((settings.request_timeout_sec - 12.5).abs() < 1e-9);
            assert_eq!(settings.api_key_env, "TEST_KEY");
            assert!(settings.melotts_helper_script.is_none());
            assert_eq!(settings.melotts_device, "cpu");
            // Paths use core expand_user_path — must match core exactly.
            let expect_local = core_expand_user_path("~/models/piper");
            assert_eq!(
                settings.local_model_path.as_deref(),
                Some(expect_local.as_path())
            );
            let expect_venv = core_expand_user_path("~/melotts-venv");
            assert_eq!(
                settings.melotts_venv_path.as_deref(),
                Some(expect_venv.as_path())
            );
        }
    }

    #[test]
    fn expand_config_path_matches_core() {
        assert_eq!(
            expand_config_path("~/models/x"),
            core_expand_user_path("~/models/x")
        );
        assert_eq!(expand_config_path("~"), core_expand_user_path("~"));
        assert_eq!(expand_config_path("/abs"), PathBuf::from("/abs"));
        assert!(expand_config_path("  ").as_os_str().is_empty());
        // Core does not trim; we trim before calling core — document that.
        assert_eq!(expand_config_path("  ~/a  "), core_expand_user_path("~/a"));
    }

    #[test]
    fn melotts_settings_never_set_legacy_helper() {
        let cfg = Config::try_with(|c| {
            c.tts_backend = TtsBackendKind::Melotts;
        })
        .unwrap();
        let settings = tts_backend_settings_from_config(&cfg);
        assert_eq!(settings.backend, BackendId::MeloTts);
        assert!(settings.melotts_helper_script.is_none());
    }

    #[test]
    #[cfg(feature = "tts-worker")]
    fn melotts_create_shared_backend_uses_worker_proto_only() {
        let settings = TtsBackendSettings {
            backend: BackendId::MeloTts,
            // Must be ignored / stripped — never legacy helper.
            melotts_helper_script: Some(PathBuf::from("/tmp/should-be-ignored.py")),
            ..TtsBackendSettings::default()
        };
        let backend = create_shared_backend(&settings).expect("worker melo backend");
        assert_eq!(backend.id(), BackendId::MeloTts);
        // Dependency errors may mention missing worker binary, never legacy helper setup.
        for err in backend.dependency_errors() {
            assert!(
                !err.contains("melo_helper.py"),
                "must not reference legacy helper: {err}"
            );
        }
    }

    #[test]
    #[cfg(not(feature = "tts-worker"))]
    fn melotts_create_shared_backend_fails_closed_without_tts_worker() {
        let settings = TtsBackendSettings {
            backend: BackendId::MeloTts,
            melotts_helper_script: Some(PathBuf::from("/tmp/legacy-helper.py")),
            ..TtsBackendSettings::default()
        };
        let err = match create_shared_backend(&settings) {
            Ok(_) => panic!("Melo must fail closed without tts-worker"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("tts-worker") || msg.contains("worker-proto"),
            "unexpected error: {msg}"
        );
        assert!(
            msg.contains("not supported") || msg.contains("not compiled"),
            "must refuse legacy helper path: {msg}"
        );
    }

    #[test]
    fn create_shared_backend_strips_helper_even_if_set() {
        // For non-melo, helper field is irrelevant; for melo we force None.
        let mut settings = TtsBackendSettings {
            backend: BackendId::Kokoro,
            melotts_helper_script: Some(PathBuf::from("/evil/helper.py")),
            kokoro_base_url: "http://127.0.0.1:9/v1".into(),
            ..TtsBackendSettings::default()
        };
        // Kokoro construction only needs a client — should succeed.
        let backend = create_shared_backend(&settings).expect("kokoro");
        assert_eq!(backend.id(), BackendId::Kokoro);
        // And settings input with helper still works after internal clear path
        // when switching to melo fail-closed / worker.
        settings.backend = BackendId::MeloTts;
        let _ = create_shared_backend(&settings); // ok or fail-closed; must not panic
    }

    #[test]
    fn resolve_output_device_name_passthrough_and_empty() {
        assert_eq!(
            resolve_output_device_selector(Some(&DeviceRef::Name("  Speakers  ".into()))).unwrap(),
            Some("Speakers".into())
        );
        assert_eq!(
            resolve_output_device_selector(Some(&DeviceRef::Name("   ".into()))).unwrap(),
            None
        );
        assert_eq!(resolve_output_device_selector(None).unwrap(), None);
    }

    #[test]
    fn resolve_output_device_negative_index_errors() {
        let err = resolve_output_device_selector(Some(&DeviceRef::Index(-1))).unwrap_err();
        assert!(err.to_string().contains("negative"), "{err}");
    }

    #[test]
    fn resolve_output_device_index_to_real_name_or_range_error() {
        // Index 0: either resolves to a concrete name (not "0") or reports OOR.
        match resolve_output_device_selector(Some(&DeviceRef::Index(0))) {
            Ok(Some(name)) => {
                assert!(!name.is_empty());
                // Must be a real enumerated name, not a numeric selector string
                // that happens to equal the index (unless a device is literally
                // named "0" — still a name from the list).
                let listed = CpalAudioOutputFactory::list_output_devices().unwrap_or_default();
                assert!(
                    listed.iter().any(|n| n == &name),
                    "resolved name {name:?} not in listed devices {listed:?}"
                );
            }
            Ok(None) => panic!("index must not resolve to default None"),
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("out of range") || msg.contains("enumerate"),
                    "{msg}"
                );
            }
        }

        // Huge index → out of range (or enumerate failure).
        let err = resolve_output_device_selector(Some(&DeviceRef::Index(9_999_999))).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("out of range") || msg.contains("enumerate"),
            "{msg}"
        );
    }

    #[test]
    fn device_ref_to_selector_name() {
        let s = device_ref_to_selector(&DeviceRef::Name("PipeWire".into())).unwrap();
        assert_eq!(s, "PipeWire");
    }

    #[test]
    fn player_event_maps_to_tts_player_update_with_redacted_error() {
        let event = error_ev("boom at https://api.example/v1/x path /tmp/secret.onnx key=abc");
        match session_command_from_player_event(&event) {
            SessionCommand::TtsPlayerUpdate {
                state,
                error_message,
            } => {
                assert_eq!(state, TtsPlayerState::Error);
                let msg = error_message.expect("error");
                assert!(!msg.contains("https://"));
                assert!(!msg.contains("/tmp/"));
                assert!(msg.contains("[redacted-url]") || msg.contains("[redacted-path]"));
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[tokio::test]
    async fn bridge_coalesces_terminal_when_fifo_full() {
        let (bridge, mut rx) = TtsPlayerUpdateBridge::new(1);

        // Fill FIFO with a non-terminal update.
        assert_eq!(
            bridge.try_send_player_event(&playing_ev()).unwrap(),
            TtsBridgeEnqueue::Queued
        );

        // Terminal Idle must not be lost under pressure.
        assert_eq!(
            bridge.try_send_player_event(&idle_ev()).unwrap(),
            TtsBridgeEnqueue::Coalesced
        );
        assert!(bridge.has_coalesced());
        let diag = bridge.diagnostics();
        assert_eq!(diag.coalesced, 1);
        assert_eq!(diag.terminal_coalesced, 1);

        // Another terminal Error overwrites coalesce (latest wins).
        assert_eq!(
            bridge
                .try_send_player_event(&error_ev("https://x/fail /tmp/a"))
                .unwrap(),
            TtsBridgeEnqueue::Coalesced
        );
        assert_eq!(bridge.diagnostics().coalesce_overwrites, 1);
        assert_eq!(bridge.diagnostics().terminal_coalesced, 2);

        // Drain contract: FIFO first, then coalesced terminal.
        let mut got = Vec::new();
        let stats = bridge.drain_with_try(&mut rx, |cmd| {
            got.push(cmd);
            true
        });
        assert_eq!(stats.delivered, 2);
        assert!(!stats.pending_remaining);
        match &got[0] {
            SessionCommand::TtsPlayerUpdate {
                state: TtsPlayerState::Playing,
                ..
            } => {}
            other => panic!("fifo first: {other:?}"),
        }
        match &got[1] {
            SessionCommand::TtsPlayerUpdate {
                state: TtsPlayerState::Error,
                error_message: Some(msg),
            } => {
                assert!(!msg.contains("https://"));
            }
            other => panic!("coalesced terminal last: {other:?}"),
        }
        assert!(!bridge.has_coalesced());
    }

    #[tokio::test]
    async fn bridge_callback_never_blocks_and_preserves_idle() {
        let (bridge, mut rx) = TtsPlayerUpdateBridge::new(1);
        let cb = bridge.on_event_callback();
        cb(playing_ev());
        cb(playing_ev()); // coalesced
        cb(idle_ev()); // coalesced latest = Idle

        let mut states = Vec::new();
        bridge.drain_with_try(&mut rx, |cmd| {
            if let SessionCommand::TtsPlayerUpdate { state, .. } = cmd {
                states.push(state);
            }
            true
        });
        assert_eq!(states.first(), Some(&TtsPlayerState::Playing));
        assert_eq!(states.last(), Some(&TtsPlayerState::Idle));
    }

    #[tokio::test]
    async fn bridge_sticky_terminal_survives_enqueue_full_then_retries() {
        let (bridge, mut rx) = TtsPlayerUpdateBridge::new(1);
        assert_eq!(
            bridge.try_send_player_event(&playing_ev()).unwrap(),
            TtsBridgeEnqueue::Queued
        );
        assert_eq!(
            bridge
                .try_send_player_event(&error_ev("boom /tmp/x"))
                .unwrap(),
            TtsBridgeEnqueue::Coalesced
        );

        // Consumer rejects everything — nothing may be lost.
        let stats = bridge.drain_with_try(&mut rx, |_| false);
        assert_eq!(stats.delivered, 0);
        assert!(stats.pending_remaining);
        assert!(bridge.has_pending());

        // Peek must still show terminal Error in coalesce (FIFO head held separately).
        // After failed FIFO Playing, undelivered_front holds Playing; coalesce keeps Error.
        assert!(bridge.has_coalesced());
        match bridge.peek_coalesced() {
            Some(SessionCommand::TtsPlayerUpdate {
                state: TtsPlayerState::Error,
                ..
            }) => {}
            other => panic!("sticky error peek: {other:?}"),
        }

        // Retry with capacity: FIFO Playing then Error.
        let mut got = Vec::new();
        let stats = bridge.drain_with_try(&mut rx, |cmd| {
            got.push(cmd);
            true
        });
        assert_eq!(stats.delivered, 2);
        assert!(!stats.pending_remaining);
        assert!(matches!(
            &got[0],
            SessionCommand::TtsPlayerUpdate {
                state: TtsPlayerState::Playing,
                ..
            }
        ));
        assert!(matches!(
            &got[1],
            SessionCommand::TtsPlayerUpdate {
                state: TtsPlayerState::Error,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn bridge_try_deliver_coalesced_put_back_on_fail() {
        let (bridge, _rx) = TtsPlayerUpdateBridge::new(1);
        // Force coalesce only (fill then overflow).
        assert!(bridge.try_send_player_event(&playing_ev()).is_ok());
        assert_eq!(
            bridge.try_send_player_event(&idle_ev()).unwrap(),
            TtsBridgeEnqueue::Coalesced
        );
        assert!(!bridge.try_deliver_coalesced(&mut |_| false));
        assert!(bridge.has_coalesced());
        match bridge.peek_coalesced() {
            Some(SessionCommand::TtsPlayerUpdate {
                state: TtsPlayerState::Idle,
                ..
            }) => {}
            other => panic!("expected sticky idle: {other:?}"),
        }
        assert!(bridge.try_deliver_coalesced(&mut |_| true));
        assert!(!bridge.has_coalesced());
    }

    #[test]
    fn bridge_full_unexpected_command_is_distinct_error() {
        let (bridge, _rx) = TtsPlayerUpdateBridge::new(1);
        assert!(
            bridge
                .try_send_command(SessionCommand::TtsPlayerUpdate {
                    state: TtsPlayerState::Playing,
                    error_message: None,
                })
                .is_ok()
        );
        let err = bridge
            .try_send_command(SessionCommand::TtsStop)
            .expect_err("non-update must error");
        assert_eq!(err, TtsBridgeSendError::UnexpectedCommand);
    }

    #[test]
    fn ensure_backend_ready_fails_closed_for_melo_without_root() {
        // Construction may succeed; readiness must fail without worker_root.
        let settings = TtsBackendSettings {
            backend: BackendId::MeloTts,
            ..TtsBackendSettings::default()
        };
        match create_shared_backend(&settings) {
            Ok(backend) => {
                let err = ensure_backend_ready(&backend).expect_err("melo without root");
                let msg = err.to_string();
                assert!(
                    msg.to_lowercase().contains("worker_root")
                        || msg.to_lowercase().contains("melotts")
                        || msg.to_lowercase().contains("tts-worker")
                        || msg.to_lowercase().contains("worker-proto"),
                    "{msg}"
                );
            }
            Err(err) => {
                // Fail-closed at create is also acceptable (no tts-worker).
                let msg = err.to_string();
                assert!(
                    msg.contains("tts-worker") || msg.contains("worker"),
                    "{msg}"
                );
            }
        }
    }

    #[test]
    fn redact_strips_urls_and_paths() {
        let r = redact_tts_error_message("fail https://x.test/a /home/u/secret.onnx");
        assert!(!r.contains("https://"));
        assert!(!r.contains("/home/"));
    }

    #[test]
    fn tts_worker_feature_flag_is_explicit() {
        // Compiles both ways; value matches cfg.
        assert_eq!(tts_worker_feature_enabled(), cfg!(feature = "tts-worker"));
    }

    #[test]
    fn path_helper_uses_core_for_tilde() {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/"));
        assert_eq!(expand_config_path("~/z"), home.join("z"));
        assert_eq!(expand_config_path("~"), home);
        // Absolute passthrough identity with core.
        assert_eq!(
            expand_config_path("/var/lib/shuvoice"),
            Path::new("/var/lib/shuvoice").to_path_buf()
        );
    }
}
