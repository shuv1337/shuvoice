//! Headless session actor and application orchestration.
//!
//! # Actor / task / event architecture
//!
//! ```text
//! CPAL ──try_push──► AudioRing ──drain──► Session actor (select)
//!                                            │
//! control ──enqueue──► cmd/reply mailboxes ──┤
//!                                            ├─ spawn finalize job (gen-tagged)
//!                                            ├─ spawn chunk job (≤1 in flight)
//!                                            ◄─ job_rx results
//!                                            │
//!                                            ├─ EventBus essential (reliable)
//!                                            └─ EventBus partial (drop-ok)
//!
//! AsrOwner task: exclusive Box<dyn AsrBackend>, HOL on long ops;
//! caller timeouts abandon wait; shutdown + join_timeout can abort.
//! ```

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod asr_owner;
pub mod audio;
pub mod error;
pub mod events;
pub mod fakes;
pub mod finalize;
pub mod runtime;
pub mod session;
pub mod traits;
pub mod types;

pub use asr_owner::{AsrOwnerHandle, AsrOwnerInfo, AsrOwnerJoin, spawn_asr_owner};
pub use audio::{AudioIngress, AudioRing};
pub use error::{AppError, AppResult};
pub use events::{EventBus, EventBusRx, push_event_log};
pub use runtime::{
    ControlHandlerSurface, EnqueueControlAdapter, SESSION_SHUTDOWN_GRACE, SessionHandle,
    SessionRuntime, TestHarness, spawn_session_runtime, spawn_test_runtime,
};
pub use session::{Session, SessionDeps, TEST_HANG_ACTOR_ON_SHUTDOWN};
pub use types::{
    ASR_CIRCUIT_COOLDOWN_SEC, ASR_MAX_FAILURES, AsrCapabilities, Config, DEFAULT_AUDIO_CAPACITY,
    DEFAULT_COMMAND_CAPACITY, DEFAULT_EVENT_CAPACITY, EFFECT_PUMP_BOUND, FinalizationMode,
    OutputMode, OverlayState, PTT_REARM_GRACE_SEC, RecordingStatus, RuntimeView, SessionCommand,
    SessionConfig, SessionEvent, StatusSnapshot, TTS_AWAIT_FINALIZE_TIMEOUT, TtsPlayerState,
    TtsSource, TypingTextCase, effective_audio_chunk_samples, effective_audio_sample_rate,
    event_is_critical_essential, event_is_essential, event_is_partial,
    event_is_replaceable_essential, truncate_chars,
};

pub mod core {
    pub use shuvoice_core::{
        CircuitBreaker, DEPENDENCY_EXIT_CODE, MetricsCollector, PTT_REARM_GRACE, STOP_TAIL_GRACE,
        UtteranceState, apply_utterance_gain, audio_rms, looks_like_cuda_oom_error,
        metrics_to_json, prefer_transcript, render_transcript_text, sanitize_final_injection_text,
    };
}
