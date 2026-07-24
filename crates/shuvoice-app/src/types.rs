//! Session commands/events and core/ASR type re-exports.

use serde::{Deserialize, Serialize};
pub use shuvoice_core::{
    ASR_MAX_FAILURES, AsrCapabilities, Config, DEPENDENCY_EXIT_CODE, ERROR_TOAST_SECONDS,
    ExpectedChunking, FinalizationMode, OutputMode, OverlayState, PTT_REARM_GRACE, RecordingStatus,
    TypingTextCase,
};

/// Seconds form of core `PTT_REARM_GRACE`.
pub const PTT_REARM_GRACE_SEC: f64 = 0.35;
/// Seconds form of core cooldown.
pub const ASR_CIRCUIT_COOLDOWN_SEC: f64 = 30.0;

pub const DEFAULT_COMMAND_CAPACITY: usize = 64;
pub const DEFAULT_EVENT_CAPACITY: usize = 256;
pub const DEFAULT_AUDIO_CAPACITY: usize = 64;

/// Default timeout for a single ASR mailbox op (chunk/utterance/reset).
pub const DEFAULT_ASR_OP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// Max time TTS will wait for STT finalization after stop.
pub const TTS_AWAIT_FINALIZE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
/// Max time test/direct helpers will *poll* for an in-flight effect job to finish.
/// Not an outer timeout around the effect itself — the injector/selection adapter
/// owns its finite process timeout. The actor never wraps effects in this bound.
pub const EFFECT_PUMP_BOUND: std::time::Duration = std::time::Duration::from_secs(30);

/// UTF-8-safe truncation by Unicode scalar count.
pub fn truncate_chars(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    match text.char_indices().nth(max_chars) {
        None => text.to_string(),
        Some((idx, _)) => text[..idx].to_string(),
    }
}

/// Derive the capture/runtime sample rate from live ASR capabilities.
///
/// `preferred_sample_rate` from the loaded backend wins (e.g. OpenAI 24 kHz);
/// otherwise fall back to `config.sample_rate` (typically 16 kHz).
#[must_use]
pub fn effective_audio_sample_rate(
    config_sample_rate: u32,
    preferred_sample_rate: Option<u32>,
) -> u32 {
    preferred_sample_rate.unwrap_or(config_sample_rate).max(1)
}

/// Chunk length in samples at the effective capture rate (`rate * chunk_ms / 1000`).
#[must_use]
pub fn effective_audio_chunk_samples(sample_rate: u32, chunk_ms: u32) -> usize {
    sample_rate as usize * chunk_ms as usize / 1000
}

pub type SessionConfig = Config;

/// TTS capture source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsSource {
    Selection,
    Clipboard,
    Explicit,
}

/// Owned TTS player state (interior-mutability friendly).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TtsPlayerState {
    #[default]
    Idle,
    Synthesizing,
    Playing,
    Paused,
    Error,
}

impl TtsPlayerState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Synthesizing => "synthesizing",
            Self::Playing => "playing",
            Self::Paused => "paused",
            Self::Error => "error",
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Synthesizing | Self::Playing | Self::Paused)
    }
}

impl std::fmt::Display for TtsPlayerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Commands accepted by the session actor (enqueue-only from control).
#[derive(Debug, Clone)]
pub enum SessionCommand {
    Start,
    Stop,
    Toggle,
    Shutdown,
    TtsSpeak {
        text: String,
        source: TtsSource,
    },
    TtsSpeakSelection,
    TtsSpeakClipboard,
    TtsPause,
    TtsResume,
    TtsTogglePause,
    TtsRestart,
    TtsStop,
    TtsSetSpeed(f64),
    TtsSelectVoice(String),
    /// Non-blocking re-entry from a TTS player callback (state machine update).
    ///
    /// Player threads/tasks must never call into `Session` directly — enqueue
    /// this command instead so the actor applies the transition ordered with
    /// other control work. Uses the session's current preview text for
    /// `TtsState` emission (callback does not need to re-supply it).
    TtsPlayerUpdate {
        state: TtsPlayerState,
        error_message: Option<String>,
    },
}

/// Events for UI/metrics observers.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionEvent {
    Status(RecordingStatus),
    OverlayShow {
        state: OverlayState,
        text: String,
    },
    OverlayUpdate {
        state: Option<OverlayState>,
        text: Option<String>,
    },
    OverlayHide,
    ErrorToast {
        text: String,
    },
    PartialTranscript {
        text: String,
    },
    FinalTranscript {
        text: String,
    },
    InjectFinal {
        text: String,
    },
    InjectPartial {
        text: String,
    },
    AsrDisabled {
        reason: String,
    },
    AsrRecovered,
    AsrThreadDead,
    CudaFallbackApplied {
        detail: String,
    },
    TtsState {
        state: TtsPlayerState,
        preview_text: String,
    },
    TtsError {
        message: String,
    },
    AudioOverflow {
        dropped: u64,
    },
    ShutdownComplete,
}

/// Whether an event is essential and must not be dropped for partial spam.
///
/// Lifecycle-critical overlay transitions, TTS player state, and control-plane
/// status are essential. High-frequency partials/diagnostics are not.
pub fn event_is_essential(event: &SessionEvent) -> bool {
    event_is_replaceable_essential(event) || event_is_critical_essential(event)
}

/// Replaceable essentials: newest-wins coalesce under backpressure.
///
/// These may be evicted or coalesced to free capacity. They must be drained
/// **before** any critical essential is touched.
pub fn event_is_replaceable_essential(event: &SessionEvent) -> bool {
    matches!(
        event,
        SessionEvent::Status(_)
            | SessionEvent::AsrDisabled { .. }
            | SessionEvent::AsrRecovered
            | SessionEvent::AsrThreadDead
            | SessionEvent::CudaFallbackApplied { .. }
            | SessionEvent::ErrorToast { .. }
            | SessionEvent::OverlayShow { .. }
            | SessionEvent::OverlayHide
            | SessionEvent::TtsState { .. }
            | SessionEvent::TtsError { .. }
            | SessionEvent::AudioOverflow { .. }
    )
}

/// Critical essentials: never silently evicted while replaceable essentials remain.
///
/// Under true saturation the bus rejects new events and sets
/// `reliable_delivery_degraded` rather than pretending delivery is reliable.
pub fn event_is_critical_essential(event: &SessionEvent) -> bool {
    matches!(
        event,
        SessionEvent::FinalTranscript { .. }
            | SessionEvent::InjectFinal { .. }
            | SessionEvent::ShutdownComplete
    )
}

/// Droppable under backpressure (partials / high-frequency diagnostics).
pub fn event_is_partial(event: &SessionEvent) -> bool {
    matches!(
        event,
        SessionEvent::PartialTranscript { .. }
            | SessionEvent::InjectPartial { .. }
            | SessionEvent::OverlayUpdate { .. }
    )
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusSnapshot {
    pub status: String,
    pub recording: bool,
    pub processing: bool,
    pub asr_disabled: bool,
    pub asr_thread_alive: bool,
}

/// Cached control-plane view (lock-free reads for handlers).
///
/// STT recording status and TTS player status are **distinct** fields so
/// `tts_status` control never confuses player state with PTT/recording state.
#[derive(Debug, Clone)]
pub struct RuntimeView {
    status: std::sync::Arc<parking_lot::RwLock<String>>,
    tts_status: std::sync::Arc<parking_lot::RwLock<String>>,
    metrics_json: std::sync::Arc<parking_lot::RwLock<String>>,
    debug_json: std::sync::Arc<parking_lot::RwLock<String>>,
}

impl Default for RuntimeView {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeView {
    pub fn new() -> Self {
        Self {
            status: std::sync::Arc::new(parking_lot::RwLock::new("idle".into())),
            tts_status: std::sync::Arc::new(parking_lot::RwLock::new(
                TtsPlayerState::Idle.as_str().into(),
            )),
            metrics_json: std::sync::Arc::new(parking_lot::RwLock::new("{\"counters\":{}}".into())),
            debug_json: std::sync::Arc::new(parking_lot::RwLock::new("{}".into())),
        }
    }

    /// STT / recording status (`idle` / `recording` / `processing` / …).
    pub fn status(&self) -> String {
        self.status.read().clone()
    }

    /// TTS player status (`idle` / `synthesizing` / `playing` / …).
    pub fn tts_status(&self) -> String {
        self.tts_status.read().clone()
    }

    pub fn metrics_json(&self) -> String {
        self.metrics_json.read().clone()
    }

    pub fn debug_json(&self) -> String {
        self.debug_json.read().clone()
    }

    pub fn publish_status(&self, status: impl Into<String>) {
        *self.status.write() = status.into();
    }

    pub fn publish_tts_status(&self, status: impl Into<String>) {
        *self.tts_status.write() = status.into();
    }

    pub fn publish_metrics(&self, json: impl Into<String>) {
        *self.metrics_json.write() = json.into();
    }

    pub fn publish_debug(&self, json: impl Into<String>) {
        *self.debug_json.write() = json.into();
    }
}

/// Generation token for ASR work (stale completion guard).
pub type UtteranceGen = u64;
