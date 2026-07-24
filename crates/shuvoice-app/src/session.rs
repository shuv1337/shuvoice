//! Responsive async session orchestrator.
#![allow(clippy::collapsible_if)]
//!
//! # Architecture
//!
//! - **Actor task** (`tick` / `handle_command`) never awaits ASR backend round-trips.
//!   Long work is generation-tagged **child tasks** (`finalize`, chunk, reset, fallback).
//! - **Result lane** (`job_rx`) delivers completions; actor applies them if utt_gen
//!   is still current.
//! - **Stop grace**: after stop, audio continues into `late_audio_tx` for
//!   [`STOP_TAIL_GRACE`] before finalize consumes it.
//! - **Events**: [`EventBus`] essential lane is reliable; partials are best-effort.
//! - **Start during finalize**: cancel prior job, reset injector, then start
//!   (no stale partials / silent loss).

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use tracing::{debug, info, warn};

use shuvoice_asr::{AsrError, FallbackOutcome};
use shuvoice_core::{
    ASR_MAX_FAILURES, BeginUtteranceParams, BreakerAction, CircuitBreaker, Config,
    ERROR_TOAST_SECONDS, FinalizationMode, MetricsCollector, OutputMode, OverlayState,
    RecordingStatus, RenderOptions, STOP_TAIL_GRACE, StartGate, UtteranceState,
    apply_utterance_gain, audio_rms, begin_utterance, capture_preroll, compile_text_replacements,
    evaluate_start_gate, metrics_to_json, ms_to_samples, observe_recording_chunk,
    prefer_transcript, recording_status as core_recording_status, render_transcript_text,
    sanitize_final_injection_text, update_noise_floor,
};
use tokio::sync::mpsc;

use crate::asr_owner::{AsrOwnerHandle, gen_is_current};
use crate::audio::AudioRing;
use crate::error::{AppError, AppResult};
use crate::events::{EventBus, push_event_log};
use crate::finalize::{
    FinalizeOutcome, JobResult, LifecyclePurpose, TtsCaptureKind, default_grace, spawn_chunk_job,
    spawn_fallback_job, spawn_finalize_job, spawn_reset_job,
};
use crate::traits::{Clock, FeedbackSink, OverlaySink, SelectionCapture, TextInjector, TtsEngine};
use crate::types::{
    EFFECT_PUMP_BOUND, RuntimeView, SessionCommand, SessionEvent, StatusSnapshot,
    TTS_AWAIT_FINALIZE_TIMEOUT, TtsPlayerState, TtsSource, UtteranceGen,
    effective_audio_chunk_samples, effective_audio_sample_rate, truncate_chars,
};

const MAX_TAIL_FLUSH_ACTOR_STALL: Duration = Duration::from_millis(5);

/// Test-only: when set, [`Session::shutdown`] awaits forever (async-cancelable) so
/// `SessionRuntime::shutdown` join-grace/abort can be validated. Never set in prod.
pub static TEST_HANG_ACTOR_ON_SHUTDOWN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub struct SessionDeps<I, T, S, O, F, C> {
    pub asr: AsrOwnerHandle,
    pub audio: Arc<AudioRing>,
    pub injector: Arc<I>,
    pub tts: Option<T>,
    pub selection: Arc<S>,
    pub overlay: O,
    pub feedback: F,
    pub clock: Arc<C>,
    pub metrics: Arc<MetricsCollector>,
    pub view: RuntimeView,
    pub events: Option<EventBus>,
}

impl<I, T, S, O, F, C> SessionDeps<I, T, S, O, F, C> {
    pub fn new(
        asr: AsrOwnerHandle,
        audio: Arc<AudioRing>,
        injector: Arc<I>,
        selection: Arc<S>,
        overlay: O,
        feedback: F,
        clock: Arc<C>,
    ) -> Self {
        Self {
            asr,
            audio,
            injector,
            tts: None,
            selection,
            overlay,
            feedback,
            clock,
            metrics: Arc::new(MetricsCollector::new()),
            view: RuntimeView::new(),
            events: None,
        }
    }
    pub fn with_tts(mut self, tts: T) -> Self {
        self.tts = Some(tts);
        self
    }
    pub fn with_metrics(mut self, m: Arc<MetricsCollector>) -> Self {
        self.metrics = m;
        self
    }
    pub fn with_view(mut self, v: RuntimeView) -> Self {
        self.view = v;
        self
    }
    pub fn with_events(mut self, e: EventBus) -> Self {
        self.events = Some(e);
        self
    }
}

struct ActiveFinalize {
    utt_gen: UtteranceGen,
    cancel: Arc<AtomicBool>,
    join: tokio::task::JoinHandle<()>,
    late_tx: mpsc::UnboundedSender<Vec<f32>>,
    grace_until: Instant,
}

struct ActiveChunk {
    utt_gen: UtteranceGen,
    join: tokio::task::JoinHandle<()>,
}

/// Tracked ASR lifecycle job (reset / fallback) — never awaited on the actor.
struct ActiveLifecycle {
    purpose: LifecyclePurpose,
    join: tokio::task::JoinHandle<()>,
}

#[allow(dead_code)]
struct ActiveInjectCommit {
    utt_gen: UtteranceGen,
    #[allow(dead_code)]
    text: String,
    /// Definitive-failure retries only (never used for ambiguous/timeout outcomes).
    attempt: u32,
    join: tokio::task::JoinHandle<()>,
}

#[allow(dead_code)]
struct ActiveInjectPartial {
    utt_gen: UtteranceGen,
    join: tokio::task::JoinHandle<()>,
}

struct ActiveEffectJob {
    join: tokio::task::JoinHandle<()>,
}

/// TTS intent waiting on finalize and/or selection capture (actor never awaits).
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum TtsIntent {
    Selection,
    Clipboard,
    Explicit { text: String, source: TtsSource }, // source retained for metrics/debug
}

#[derive(Debug)]
enum PendingTts {
    /// Stopped STT; waiting for finalize job (or silence) before capture/speak.
    WaitFinalize {
        intent: TtsIntent,
        deadline: Instant,
    },
    /// Selection/clipboard capture job in flight.
    Capturing { kind: TtsCaptureKind },
}

/// Responsive session core.
pub struct Session<I, T, S, O, F, C> {
    pub config: Config,
    deps: SessionDeps<I, T, S, O, F, C>,
    render: RenderOptions,

    recording: bool,
    processing: bool,
    running: bool,
    asr_thread_alive: bool,
    model_load_failed: bool,

    circuit: CircuitBreaker,
    last_stop_at: Option<Instant>,
    utterance_gen: UtteranceGen,
    committed_gen: Option<UtteranceGen>,
    cuda_recovered_skip_reset: bool,

    speech_rms_threshold: f32,
    speech_rms_multiplier: f32,
    min_speech_samples: usize,
    auto_gain_target_peak: f32,
    auto_gain_max: f32,
    auto_gain_settle_chunks: usize,
    noise_floor_rms: f32,
    sample_rate: u32,

    preroll: Vec<Vec<f32>>,
    state: UtteranceState,
    was_recording: bool,

    finalize: Option<ActiveFinalize>,
    chunk_job: Option<ActiveChunk>,
    lifecycle_job: Option<ActiveLifecycle>,
    inject_commit: Option<ActiveInjectCommit>,
    inject_partial: Option<ActiveInjectPartial>,
    inject_reset: Option<ActiveEffectJob>,
    selection_job: Option<ActiveEffectJob>,
    /// Coalesce target while a partial inject job is in flight.
    pending_partial_text: Option<String>,
    /// Reset requested while partial/commit still running — start when idle.
    pending_inject_reset: bool,
    /// Commit waiting because a reset job was already running.
    pending_inject_commit: Option<(UtteranceGen, String, u32)>,
    /// Suppress new partials until in-flight reset completes.
    inject_barrier: bool,
    pending_tts: Option<PendingTts>,
    /// Last TTS request error (for direct/test pump helpers).
    tts_last_error: Option<String>,
    /// External effect jobs in flight (inject/selection) for pump helpers.
    inflight_effects: u32,
    /// True after Start is accepted until reset succeeds, fails, or is cancelled.
    start_pending: bool,
    job_tx: mpsc::UnboundedSender<JobResult>,
    job_rx: mpsc::UnboundedReceiver<JobResult>,

    error_toast_until: Option<Instant>,
    last_reported_drops: u64,

    debug_current_transcript: String,
    debug_last_final_transcript: String,

    tts_voice_id: String,
    tts_playback_speed: f64,
    tts_last_preview_text: String,

    /// Diagnostic/event log for `take_events` (tests + diagnostics).
    /// Never used as the EventBus delivery outbox — flush must not pop this.
    event_log: VecDeque<SessionEvent>,
    /// Delivery outbox flushed into the EventBus ingress (ordered, bounded).
    /// Distinct from `event_log` so successful queueing cannot erase diagnostics.
    bus_outbox: VecDeque<SessionEvent>,
    event_capacity: usize,
    /// Mirrored TTS player state (also published on RuntimeView.tts_status).
    tts_player_state: TtsPlayerState,
}

impl<I, T, S, O, F, C> Session<I, T, S, O, F, C>
where
    I: TextInjector + 'static,
    T: TtsEngine,
    S: SelectionCapture + 'static,
    O: OverlaySink,
    F: FeedbackSink,
    C: Clock,
{
    pub fn new(config: Config, deps: SessionDeps<I, T, S, O, F, C>) -> Self {
        // Prefer live ASR owner caps (post-load) over config — OpenAI must stay 24 kHz.
        let sample_rate = effective_audio_sample_rate(
            config.sample_rate,
            deps.asr.capabilities().preferred_sample_rate,
        );
        let render = RenderOptions {
            text_case: config.typing_text_case,
            auto_capitalize: config.auto_capitalize,
            replacements: compile_text_replacements(&config.text_replacements),
        };
        let (job_tx, job_rx) = mpsc::unbounded_channel();
        let mut s = Self {
            speech_rms_threshold: config.silence_rms_threshold.max(0.0) as f32,
            speech_rms_multiplier: config.silence_rms_multiplier.max(1.0) as f32,
            min_speech_samples: ms_to_samples(sample_rate, config.min_speech_ms),
            auto_gain_target_peak: config.auto_gain_target_peak.max(1e-4) as f32,
            auto_gain_max: config.auto_gain_max.max(1.0) as f32,
            auto_gain_settle_chunks: config.auto_gain_settle_chunks.max(1) as usize,
            noise_floor_rms: 0.0,
            sample_rate,
            render,
            tts_voice_id: config.tts_default_voice_id.clone(),
            tts_playback_speed: config.tts_playback_speed,
            tts_last_preview_text: String::new(),
            config,
            deps,
            recording: false,
            processing: false,
            running: true,
            asr_thread_alive: true,
            model_load_failed: false,
            circuit: CircuitBreaker::new(),
            last_stop_at: None,
            utterance_gen: 0,
            committed_gen: None,
            cuda_recovered_skip_reset: false,
            preroll: Vec::new(),
            state: UtteranceState::new(),
            was_recording: false,
            finalize: None,
            chunk_job: None,
            lifecycle_job: None,
            inject_commit: None,
            inject_partial: None,
            inject_reset: None,
            selection_job: None,
            pending_partial_text: None,
            pending_inject_reset: false,
            pending_inject_commit: None,
            inject_barrier: false,
            pending_tts: None,
            tts_last_error: None,
            inflight_effects: 0,
            start_pending: false,
            job_tx,
            job_rx,
            error_toast_until: None,
            last_reported_drops: 0,
            debug_current_transcript: String::new(),
            debug_last_final_transcript: String::new(),
            event_log: VecDeque::new(),
            bus_outbox: VecDeque::new(),
            event_capacity: 256,
            tts_player_state: TtsPlayerState::Idle,
        };
        // Seed mirror from engine so pre-set player state is visible to tts_status.
        if let Some(tts) = s.deps.tts.as_ref() {
            s.tts_player_state = tts.state();
        }
        s.publish_view();
        s
    }

    pub fn set_model_load_failed(&mut self, failed: bool) {
        self.model_load_failed = failed;
        self.publish_view();
    }

    /// Re-read live ASR caps and refresh sample-rate-dependent sizing.
    ///
    /// Call after backend `load` (or CPU fallback) when caps/manifest may have
    /// changed preferred sample rate or when composing capture outside the session.
    pub fn sync_audio_params_from_asr(&mut self) {
        let rate = effective_audio_sample_rate(
            self.config.sample_rate,
            self.deps.asr.capabilities().preferred_sample_rate,
        );
        if rate != self.sample_rate {
            info!(
                old = self.sample_rate,
                new = rate,
                "effective audio sample rate updated from ASR caps"
            );
        }
        self.sample_rate = rate;
        self.min_speech_samples = ms_to_samples(rate, self.config.min_speech_ms);
        self.publish_view();
    }

    /// Effective capture/runtime sample rate (ASR preferred or config fallback).
    #[must_use]
    pub fn effective_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Capture chunk length in samples at the effective rate.
    #[must_use]
    pub fn audio_chunk_samples(&self) -> usize {
        effective_audio_chunk_samples(self.sample_rate, self.config.chunk_ms)
    }

    /// Minimum speech samples gate (derived from effective rate × min_speech_ms).
    #[must_use]
    pub fn min_speech_samples(&self) -> usize {
        self.min_speech_samples
    }

    /// Config sample rate before ASR override (for diagnostics).
    #[must_use]
    pub fn config_sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// ASR-advertised preferred rate, if any.
    #[must_use]
    pub fn asr_preferred_sample_rate(&self) -> Option<u32> {
        self.deps.asr.capabilities().preferred_sample_rate
    }

    pub fn view(&self) -> RuntimeView {
        self.deps.view.clone()
    }
    pub fn metrics(&self) -> Arc<MetricsCollector> {
        Arc::clone(&self.deps.metrics)
    }
    pub fn asr(&self) -> &AsrOwnerHandle {
        &self.deps.asr
    }
    pub fn audio(&self) -> Arc<AudioRing> {
        Arc::clone(&self.deps.audio)
    }
    pub fn injector(&self) -> &I {
        self.deps.injector.as_ref()
    }
    pub fn overlay(&self) -> &O {
        &self.deps.overlay
    }
    pub fn overlay_mut(&mut self) -> &mut O {
        &mut self.deps.overlay
    }
    pub fn tts(&self) -> Option<&T> {
        self.deps.tts.as_ref()
    }
    pub fn tts_mut(&mut self) -> Option<&mut T> {
        self.deps.tts.as_mut()
    }
    pub fn utterance_state(&self) -> &UtteranceState {
        &self.state
    }
    pub fn consecutive_failures(&self) -> u32 {
        self.circuit.consecutive_failures()
    }
    pub fn is_recording(&self) -> bool {
        self.recording
    }
    pub fn is_processing(&self) -> bool {
        self.processing
    }
    pub fn is_running(&self) -> bool {
        self.running
    }
    pub fn is_finalizing(&self) -> bool {
        self.finalize.is_some()
    }
    pub fn is_starting(&self) -> bool {
        self.start_pending
    }
    /// In-flight inject/selection effect jobs (for tests/diagnostics).
    pub fn effect_jobs_in_flight(&self) -> u32 {
        self.inflight_effects
    }
    pub fn debug_current_transcript(&self) -> &str {
        &self.debug_current_transcript
    }
    pub fn debug_last_final_transcript(&self) -> &str {
        &self.debug_last_final_transcript
    }
    pub fn take_events(&mut self) -> Vec<SessionEvent> {
        self.event_log.drain(..).collect()
    }

    /// Current mirrored TTS player state (distinct from STT recording status).
    pub fn tts_player_state(&self) -> TtsPlayerState {
        self.tts_player_state.clone()
    }
    pub fn job_receiver(&mut self) -> &mut mpsc::UnboundedReceiver<JobResult> {
        &mut self.job_rx
    }

    pub fn is_offline_instant_mode(&self) -> bool {
        matches!(
            self.deps.asr.finalization_mode(),
            FinalizationMode::OfflineInstant
        )
    }
    pub fn is_remote_manual_commit_mode(&self) -> bool {
        matches!(
            self.deps.asr.finalization_mode(),
            FinalizationMode::RemoteManualCommit
        )
    }

    pub fn recording_status(&self) -> RecordingStatus {
        core_recording_status(
            self.circuit.is_disabled(),
            self.asr_thread_alive && self.deps.asr.is_alive(),
            self.recording,
            self.processing,
        )
    }

    pub fn status_snapshot(&self) -> StatusSnapshot {
        let status = self.recording_status();
        StatusSnapshot {
            status: status.as_str().to_string(),
            recording: self.recording,
            processing: self.processing,
            asr_disabled: self.circuit.is_disabled(),
            asr_thread_alive: self.asr_thread_alive && self.deps.asr.is_alive(),
        }
    }

    fn now(&self) -> Instant {
        self.deps.clock.now()
    }

    fn emit(&mut self, event: SessionEvent) {
        // Deliberate dual-buffer push:
        // - event_log: diagnostic mirror for take_events (never popped by flush)
        // - bus_outbox: delivery queue drained into EventBus ingress
        if let Some(bus) = &self.deps.events {
            bus.emit_now(
                event,
                &mut self.event_log,
                &mut self.bus_outbox,
                self.event_capacity,
            );
        } else {
            push_event_log(&mut self.event_log, self.event_capacity, event);
        }
    }

    fn flush_bus(&mut self) {
        if let Some(bus) = &self.deps.events {
            bus.flush_pending(&mut self.bus_outbox);
        }
    }

    /// Test-only: emit a session event through the production bus path.
    pub fn emit_for_test(&mut self, event: SessionEvent) {
        self.emit(event);
    }

    fn publish_view(&self) {
        self.deps
            .view
            .publish_status(self.recording_status().as_str());
        self.deps
            .view
            .publish_tts_status(self.tts_player_state.as_str());
        self.deps
            .view
            .publish_metrics(metrics_to_json(&self.deps.metrics.snapshot()));
        self.deps.view.publish_debug(self.debug_status_json());
    }

    fn flash_error(&mut self, text: &str) {
        self.deps.overlay.show(OverlayState::Error, text);
        self.error_toast_until = Some(self.now() + Duration::from_secs(ERROR_TOAST_SECONDS));
        self.emit(SessionEvent::ErrorToast {
            text: text.to_string(),
        });
        self.emit(SessionEvent::OverlayShow {
            state: OverlayState::Error,
            text: text.to_string(),
        });
    }

    fn poll_error_toast(&mut self) {
        let Some(until) = self.error_toast_until else {
            return;
        };
        if self.now() < until {
            return;
        }
        self.error_toast_until = None;
        if self.circuit.is_disabled() || self.recording || self.finalize.is_some() {
            return;
        }
        self.deps.overlay.hide();
        self.emit(SessionEvent::OverlayHide);
    }

    fn show_error(&mut self, text: &str) {
        self.deps.overlay.show(OverlayState::Error, text);
        self.emit(SessionEvent::OverlayShow {
            state: OverlayState::Error,
            text: text.to_string(),
        });
    }

    fn disable_asr(&mut self, reason: &str) {
        self.circuit.force_open(self.now());
        self.recording = false;
        self.processing = false;
        warn!(%reason, "ASR disabled");
        self.show_error("⚠ ASR error — will retry in 30s");
        self.emit(SessionEvent::AsrDisabled {
            reason: reason.to_string(),
        });
        self.emit(SessionEvent::Status(self.recording_status()));
        self.publish_view();
    }

    fn note_asr_failure(&mut self, context: &str) {
        match self.circuit.on_failure(self.now()) {
            BreakerAction::Opened => self.disable_asr(context),
            BreakerAction::Counted { failures } => {
                self.flash_error(&format!(
                    "⚠ ASR error ({failures}/{ASR_MAX_FAILURES}) — see logs"
                ));
            }
            BreakerAction::Ignored | BreakerAction::ClosedClear => {}
        }
        self.publish_view();
    }

    fn note_asr_success(&mut self) {
        let _ = self.circuit.on_success();
    }

    fn recover_asr_after_failure(&mut self, context: &str) {
        if self.circuit.is_disabled() {
            return;
        }
        if self.cuda_recovered_skip_reset {
            self.cuda_recovered_skip_reset = false;
            debug!(%context, "skipping recovery reset after CUDA CPU fallback");
            return;
        }
        self.spawn_lifecycle(LifecyclePurpose::RecoverAfterFailure {
            context: context.to_string(),
        });
    }

    /// Queue CUDA fallback as a tracked job (never await backend on actor).
    /// Returns true if the error looked like CUDA OOM and a job was spawned (or
    /// one is already in flight).
    fn handle_asr_runtime_error(&mut self, err: &AsrError) -> bool {
        let msg = err.to_string();
        if !shuvoice_core::looks_like_cuda_oom_error(&msg) && !matches!(err, AsrError::CudaOom(_)) {
            return false;
        }
        self.spawn_fallback_lifecycle();
        true
    }

    fn spawn_lifecycle(&mut self, purpose: LifecyclePurpose) {
        // Replace any prior lifecycle job — ASR owner serializes ops anyway.
        self.abort_lifecycle_job();
        let join = spawn_reset_job(self.deps.asr.clone(), purpose.clone(), self.job_tx.clone());
        self.lifecycle_job = Some(ActiveLifecycle { purpose, join });
    }

    fn spawn_fallback_lifecycle(&mut self) {
        self.abort_lifecycle_job();
        let join = spawn_fallback_job(self.deps.asr.clone(), self.job_tx.clone());
        self.lifecycle_job = Some(ActiveLifecycle {
            purpose: LifecyclePurpose::RecoverAfterFailure {
                context: "cuda_fallback".into(),
            },
            join,
        });
    }

    fn abort_lifecycle_job(&mut self) {
        if let Some(life) = self.lifecycle_job.take() {
            life.join.abort();
        }
    }

    // ── Recording control (never awaits ASR backend) ───────────────────

    /// Non-blocking start: enqueue reset job, enter `start_pending`.
    ///
    /// Direct callers (tests) should use [`Self::start_recording`] which pumps
    /// job results until the start settles. The actor `handle_command` path
    /// only calls [`Self::request_start_recording`] so select stays live.
    pub fn request_start_recording(&mut self) {
        if let Some(tts) = self.deps.tts.as_mut()
            && tts.is_active()
        {
            info!("Stopping TTS playback before recording start");
            let _ = tts.stop();
            self.emit_tts_state(TtsPlayerState::Idle);
        }

        if !self.deps.asr.is_alive() {
            self.asr_thread_alive = false;
            self.show_error("⚠ ASR thread crashed — restart ShuVoice");
            self.emit(SessionEvent::AsrThreadDead);
            self.publish_view();
            return;
        }

        if self.start_pending || self.recording {
            debug!("Recording already active/starting; ignoring start");
            return;
        }

        let since_stop = self
            .last_stop_at
            .map(|t| self.now().saturating_duration_since(t))
            .unwrap_or(Duration::MAX);

        match evaluate_start_gate(
            self.recording,
            self.asr_thread_alive && self.deps.asr.is_alive(),
            self.processing && !self.start_pending,
            since_stop,
            self.circuit.is_disabled(),
        ) {
            StartGate::AlreadyRecording => {
                debug!("Recording already active; ignoring start");
                return;
            }
            StartGate::AsrThreadDead => {
                self.show_error("⚠ ASR thread crashed — restart ShuVoice");
                return;
            }
            StartGate::RearmGrace => {
                debug!(?since_stop, "Ignoring start during processing rearm window");
                return;
            }
            StartGate::AsrDisabledNeedsRecovery => {
                warn!("ASR disabled; queueing one-shot reset on recording start");
                // Cancel prior finalize without awaiting ASR.
                if self.finalize.is_some() || self.processing || self.was_recording {
                    self.cancel_finalize_and_reset_injector();
                }
                self.capture_recording_preroll();
                self.start_pending = true;
                self.processing = true;
                self.emit(SessionEvent::Status(self.recording_status()));
                self.publish_view();
                self.spawn_lifecycle(LifecyclePurpose::DisabledRecovery);
                return;
            }
            StartGate::Allow => {}
        }

        // Stop→rearm→Start: cancel prior finalize without awaiting ASR.
        if self.finalize.is_some() || self.processing || self.was_recording {
            self.cancel_finalize_and_reset_injector();
        }

        // Preroll before reset; second preroll applied when reset completes.
        self.capture_recording_preroll();
        self.start_pending = true;
        self.processing = true;
        self.emit(SessionEvent::Status(self.recording_status()));
        self.publish_view();
        self.spawn_lifecycle(LifecyclePurpose::StartRecording);
    }

    /// Test/direct API: request start and pump job results until settled.
    pub async fn start_recording(&mut self) {
        self.request_start_recording();
        self.pump_jobs_while(
            |s| s.start_pending || s.lifecycle_job.is_some(),
            Duration::from_secs(5),
        )
        .await;
    }

    /// Pump `job_rx` until `pred` is false or timeout (no ASR awaits).
    pub async fn pump_jobs_while(
        &mut self,
        mut pred: impl FnMut(&Self) -> bool,
        timeout: Duration,
    ) {
        let start = tokio::time::Instant::now();
        while pred(self) {
            // Always advance deferred inject/TTS state machines so pumps cannot
            // stall forever on pending_inject_reset with zero in-flight jobs.
            self.drive_pending_tts();
            self.pump_inject_pipeline();
            self.poll_jobs().await;
            self.drive_pending_tts();
            self.pump_inject_pipeline();
            if !pred(self) {
                break;
            }
            if start.elapsed() > timeout {
                warn!("pump_jobs_while timed out");
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    fn complete_start_recording(&mut self, utt_gen: UtteranceGen) {
        self.utterance_gen = utt_gen;
        self.capture_recording_preroll();
        self.committed_gen = None;
        self.was_recording = false;
        self.start_pending = false;
        self.processing = false;
        self.recording = true;
        // Reset injector for the new utterance after prior commit/partial drain.
        self.request_inject_reset();
        self.deps.metrics.recording_started();
        info!("Recording started");
        self.deps
            .overlay
            .show(OverlayState::Listening, "Listening…");
        self.emit(SessionEvent::OverlayShow {
            state: OverlayState::Listening,
            text: "Listening…".into(),
        });
        self.emit(SessionEvent::Status(RecordingStatus::Recording));
        if self.config.audio_feedback {
            self.deps.feedback.play_start();
        }
        self.publish_view();
    }

    fn fail_start_recording(&mut self, err: &AsrError) {
        self.start_pending = false;
        self.processing = false;
        warn!(%err, "ASR reset failed on recording start");
        match self.circuit.on_failure(self.now()) {
            BreakerAction::Opened => {
                self.disable_asr("ASR disabled after repeated reset failures");
            }
            _ => self.show_error("⚠ ASR error — restart ShuVoice"),
        }
        self.emit(SessionEvent::Status(self.recording_status()));
        self.publish_view();
    }

    pub fn stop_recording(&mut self) {
        // Cancel in-flight start reset without waiting for ASR.
        if self.start_pending {
            debug!("Cancelling pending start on stop");
            self.start_pending = false;
            self.abort_lifecycle_job();
            self.processing = false;
            self.emit(SessionEvent::Status(self.recording_status()));
            self.publish_view();
            if !self.recording {
                return;
            }
        }
        if !self.recording {
            debug!("Recording already stopped; ignoring stop");
            return;
        }
        info!("Recording stopped");
        self.recording = false;
        self.processing = true;
        self.last_stop_at = Some(self.now());
        self.deps.metrics.recording_stopped();
        if self.config.audio_feedback {
            self.deps.feedback.play_stop();
        }
        self.deps.overlay.set_state(OverlayState::Processing);
        self.emit(SessionEvent::OverlayUpdate {
            state: Some(OverlayState::Processing),
            text: None,
        });
        self.emit(SessionEvent::Status(RecordingStatus::Processing));
        self.publish_view();
    }

    pub async fn toggle_recording(&mut self) {
        if self.recording || self.start_pending {
            self.stop_recording();
        } else {
            self.request_start_recording();
        }
    }

    fn capture_recording_preroll(&mut self) {
        let mut chunks = self.deps.audio.drain();
        let mut combined = std::mem::take(&mut self.preroll);
        combined.append(&mut chunks);
        let max_samples = ms_to_samples(self.sample_rate, self.config.recording_preroll_ms);
        self.preroll = capture_preroll(&combined, max_samples);
    }

    /// Begin utterance: threshold + preroll only (reset already done on start).
    pub async fn begin_utterance(&mut self) {
        let preroll = std::mem::take(&mut self.preroll);
        begin_utterance(
            &mut self.state,
            BeginUtteranceParams {
                noise_floor_rms: self.noise_floor_rms,
                speech_rms_threshold: self.speech_rms_threshold,
                speech_rms_multiplier: self.speech_rms_multiplier,
                preroll_chunks: &preroll,
                wants_raw_audio: self.deps.asr.wants_raw_audio(),
                auto_gain_settle_chunks: self.auto_gain_settle_chunks,
                auto_gain_target_peak: self.auto_gain_target_peak,
                auto_gain_max: self.auto_gain_max,
            },
        );
        self.committed_gen = None;
    }

    pub fn append_recording_chunk(&mut self, chunk: &[f32]) {
        observe_recording_chunk(
            &mut self.state,
            chunk,
            self.deps.asr.wants_raw_audio(),
            self.auto_gain_settle_chunks,
            self.auto_gain_target_peak,
            self.auto_gain_max,
        );
    }

    // ── Finalize coordination ───────────────────────────────────────────

    fn start_finalize_job(&mut self) {
        if self.finalize.is_some() {
            return;
        }
        // No concurrent chunk job while finalize owns the generation.
        self.abort_chunk_job();
        // Incorporate all buffered audio at stop edge (never drop key-up audio).
        self.drain_and_buffer();
        let utt_gen = self.utterance_gen;
        let state = std::mem::replace(&mut self.state, UtteranceState::new());
        let (late_tx, late_rx) = mpsc::unbounded_channel::<Vec<f32>>();
        let cancel = Arc::new(AtomicBool::new(false));
        let finish_timeout = {
            let secs = self.config.openai_realtime_commit_timeout_sec;
            if secs > 0.0 {
                Some(Duration::from_secs_f64(secs))
            } else {
                None
            }
        };
        let join = spawn_finalize_job(
            self.deps.asr.clone(),
            utt_gen,
            state,
            late_rx,
            default_grace(),
            Arc::clone(&cancel),
            self.min_speech_samples,
            self.speech_rms_threshold,
            self.job_tx.clone(),
            finish_timeout,
        );
        self.finalize = Some(ActiveFinalize {
            utt_gen,
            cancel,
            join,
            late_tx,
            grace_until: self.now() + STOP_TAIL_GRACE,
        });
        self.processing = true;
    }

    /// Cancel finalize/chunk without awaiting ASR. Injection is **non-cancellable**:
    /// reset is deferred until in-flight partial/commit resolve; new partials are
    /// suppressed until that reset completes. Recording/control proceed immediately.
    fn cancel_finalize_and_reset_injector(&mut self) {
        if let Some(f) = self.finalize.take() {
            f.cancel.store(true, Ordering::Release);
            let _ = self.deps.asr.bump_gen();
            f.join.abort();
        }
        self.abort_chunk_job();
        // Do **not** abort inject partial/commit — side effects keep running.
        self.pending_partial_text = None;
        self.processing = false;
        self.was_recording = false;
        self.state.reset(self.speech_rms_threshold);
        self.debug_current_transcript.clear();
        self.request_inject_reset();
    }

    /// Selection is read-only: safe to abandon (abort handle). Track join until done.
    fn abandon_selection_job(&mut self) {
        if let Some(s) = self.selection_job.take() {
            s.join.abort();
            self.effect_end();
        }
        if matches!(self.pending_tts, Some(PendingTts::Capturing { .. })) {
            self.pending_tts = None;
        }
    }

    /// Shutdown: detach inject jobs (do not abort — non-cancellable side effects),
    /// abandon selection. Late inject results are ignored when `!running`.
    fn release_effect_jobs_on_shutdown(&mut self) {
        // Detach inject handles without abort() so we don't pretend work stopped.
        let _ = self.inject_commit.take();
        let _ = self.inject_partial.take();
        let _ = self.inject_reset.take();
        self.pending_partial_text = None;
        self.pending_inject_reset = false;
        self.pending_inject_commit = None;
        self.inject_barrier = false;
        self.abandon_selection_job();
        self.pending_tts = None;
        // inflight_effects drained by late JobResults' effect_end, or zeroed:
        self.inflight_effects = 0;
    }

    fn effect_begin(&mut self) {
        self.inflight_effects = self.inflight_effects.saturating_add(1);
    }

    fn effect_end(&mut self) {
        self.inflight_effects = self.inflight_effects.saturating_sub(1);
    }

    /// True while any non-cancellable inject op is outstanding.
    #[allow(dead_code)]
    fn inject_pipeline_busy(&self) -> bool {
        self.inject_commit.is_some() || self.inject_partial.is_some() || self.inject_reset.is_some()
    }

    /// Request injector reset. Deferred until partial/commit finish; suppresses
    /// new partials until the reset job completes (`inject_barrier`).
    fn request_inject_reset(&mut self) {
        self.pending_partial_text = None;
        self.inject_barrier = true;
        if self.inject_commit.is_some() || self.inject_partial.is_some() {
            self.pending_inject_reset = true;
            debug!("deferring inject reset until in-flight partial/commit resolve");
            return;
        }
        if self.inject_reset.is_some() {
            // Already resetting; barrier stays up until it completes.
            self.pending_inject_reset = false;
            return;
        }
        self.start_inject_reset_job();
    }

    fn start_inject_reset_job(&mut self) {
        self.pending_inject_reset = false;
        self.inject_barrier = true;
        self.effect_begin();
        let injector = Arc::clone(&self.deps.injector);
        let tx = self.job_tx.clone();
        let join = tokio::spawn(async move {
            let result = injector.reset().await;
            let _ = tx.send(JobResult::InjectReset { result });
        });
        self.inject_reset = Some(ActiveEffectJob { join });
    }

    /// Advance deferred reset when the pipeline is idle.
    fn pump_inject_pipeline(&mut self) {
        if self.inject_commit.is_some()
            || self.inject_partial.is_some()
            || self.inject_reset.is_some()
        {
            return;
        }
        if let Some((utt_gen, text, attempt)) = self.pending_inject_commit.take() {
            self.spawn_inject_commit(utt_gen, text, attempt);
            return;
        }
        if self.pending_inject_reset {
            self.start_inject_reset_job();
        }
    }

    fn spawn_inject_commit(&mut self, utt_gen: UtteranceGen, text: String, attempt: u32) {
        if self.committed_gen == Some(utt_gen) {
            return;
        }
        if self
            .inject_commit
            .as_ref()
            .is_some_and(|c| c.utt_gen == utt_gen)
        {
            return;
        }
        // If a reset job is already running, queue commit until it finishes.
        // A *deferred* reset (pending_inject_reset) yields to commit.
        if self.inject_reset.is_some() {
            debug!(utt_gen, "queue commit until in-flight reset resolves");
            self.pending_inject_commit = Some((utt_gen, text, attempt));
            return;
        }
        self.effect_begin();
        let injector = Arc::clone(&self.deps.injector);
        let tx = self.job_tx.clone();
        let text_job = text.clone();
        let join = tokio::spawn(async move {
            let result = injector.commit_final(&text_job).await;
            let _ = tx.send(JobResult::InjectCommit {
                utt_gen,
                text: text_job,
                attempt,
                result,
            });
        });
        self.inject_commit = Some(ActiveInjectCommit {
            utt_gen,
            text,
            attempt,
            join,
        });
    }

    fn spawn_inject_partial(&mut self, utt_gen: UtteranceGen, text: String) {
        // Barrier: no new partials until deferred reset completes.
        if self.inject_barrier || self.pending_inject_reset || self.inject_reset.is_some() {
            return;
        }
        // Don't interleave with commit.
        if self.inject_commit.is_some() {
            return;
        }
        if self.inject_partial.is_some() {
            // Coalesce latest text for the active partial job's generation only.
            self.pending_partial_text = Some(text);
            return;
        }
        self.effect_begin();
        let injector = Arc::clone(&self.deps.injector);
        let tx = self.job_tx.clone();
        let text_job = text.clone();
        let join = tokio::spawn(async move {
            let result = injector.update_partial(&text_job).await;
            let _ = tx.send(JobResult::InjectPartial {
                utt_gen,
                text: text_job,
                result,
            });
        });
        self.inject_partial = Some(ActiveInjectPartial { utt_gen, join });
    }

    fn spawn_selection_capture(&mut self, kind: TtsCaptureKind) {
        // Selection is read-only — abandon prior capture safely.
        self.abandon_selection_job();
        self.effect_begin();
        let selection = Arc::clone(&self.deps.selection);
        let tx = self.job_tx.clone();
        let join = tokio::spawn(async move {
            let result = match kind {
                TtsCaptureKind::Selection => selection.capture_selection().await,
                TtsCaptureKind::Clipboard => selection.capture_clipboard().await,
            };
            let _ = tx.send(JobResult::SelectionCapture { kind, result });
        });
        self.selection_job = Some(ActiveEffectJob { join });
        self.pending_tts = Some(PendingTts::Capturing { kind });
    }

    fn abort_chunk_job(&mut self) {
        if let Some(c) = self.chunk_job.take() {
            c.join.abort();
        }
    }

    /// Wait for in-flight finalize (or cancel after timeout). Real coordination.
    pub async fn await_or_cancel_finalization(&mut self, timeout: Duration) -> AppResult<()> {
        // Pending stop edge not yet turned into a job?
        if self.was_recording && !self.recording && self.finalize.is_none() {
            self.start_finalize_job();
            self.was_recording = false;
        }
        if self.finalize.is_none() {
            self.processing = false;
            return Ok(());
        }
        let start = self.now();
        loop {
            self.poll_jobs().await;
            if self.finalize.is_none() {
                // Final inject may still be in flight — poll until it completes.
                // Never abort/retry an in-flight commit (exactly-once).
                if self.inject_commit.is_some() {
                    // Extend wait for the single in-flight commit (adapter-bounded).
                    if self.now().saturating_duration_since(start) > timeout + EFFECT_PUMP_BOUND {
                        // Still do not abort — leave job tracked; surface timeout to caller.
                        self.processing = false;
                        return Err(AppError::SttProcessingTimeout);
                    }
                    tokio::time::sleep(Duration::from_millis(2)).await;
                    continue;
                }
                if !self.recording && !self.start_pending {
                    self.processing = false;
                }
                return Ok(());
            }
            if self.now().saturating_duration_since(start) > timeout {
                self.cancel_finalize_and_reset_injector();
                return Err(AppError::SttProcessingTimeout);
            }
            // Feed grace audio while waiting.
            self.feed_finalize_grace_audio();
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn feed_finalize_grace_audio(&mut self) {
        let Some(f) = self.finalize.as_ref() else {
            return;
        };
        if self.now() > f.grace_until {
            return;
        }
        let chunks = self.deps.audio.drain();
        for chunk in chunks {
            let _ = f.late_tx.send(chunk);
        }
    }

    async fn poll_jobs(&mut self) {
        while let Ok(msg) = self.job_rx.try_recv() {
            self.apply_job_result(msg).await;
        }
    }

    async fn apply_job_result(&mut self, msg: JobResult) {
        match msg {
            JobResult::Chunk { utt_gen, text } => {
                // Clear tracking only for the matching in-flight job.
                if self
                    .chunk_job
                    .as_ref()
                    .is_some_and(|c| c.utt_gen == utt_gen)
                {
                    self.chunk_job = None;
                }
                if utt_gen != self.utterance_gen || !gen_is_current(&self.deps.asr, utt_gen) {
                    return;
                }
                if self.finalize.is_some() {
                    // Finalize owns the generation now — ignore late chunks.
                    return;
                }
                match text {
                    Ok(t) => {
                        self.note_asr_success();
                        let merged = prefer_transcript(&self.state.last_text, t.as_str());
                        if merged != self.state.last_text {
                            self.state.last_text = merged;
                            self.on_transcript_update();
                        }
                    }
                    Err(err) => {
                        if self.handle_asr_runtime_error(&err) {
                            // fallback job queued
                        } else if err.counts_for_breaker() {
                            // Timeouts count toward breaker (HOL / stuck backend signal).
                            self.note_asr_failure("ASR disabled after repeated chunk failures");
                            self.recover_asr_after_failure("ASR chunk failed");
                        }
                    }
                }
            }
            JobResult::Finalize { utt_gen, outcome } => {
                // Stale/cancelled completions must never mutate session or commit.
                let Some(f) = self.finalize.as_ref() else {
                    debug!(utt_gen, "ignoring finalize result with no active job");
                    return;
                };
                if f.utt_gen != utt_gen {
                    debug!(
                        utt_gen,
                        active = f.utt_gen,
                        "ignoring finalize result for non-active generation"
                    );
                    return;
                }
                // Join the tracked task (should already be finished).
                if let Some(f) = self.finalize.take() {
                    if !f.join.is_finished() {
                        // Should be rare; don't block the actor on a wedged job.
                        f.cancel.store(true, Ordering::Release);
                        f.join.abort();
                    }
                    let _ = f.join.await;
                }

                let gen_live =
                    gen_is_current(&self.deps.asr, utt_gen) && self.committed_gen != Some(utt_gen);

                match outcome {
                    FinalizeOutcome::Silent => {
                        self.debug_current_transcript.clear();
                        self.deps.overlay.hide();
                        self.emit(SessionEvent::OverlayHide);
                        self.request_inject_reset();
                    }
                    FinalizeOutcome::Ready { text } | FinalizeOutcome::Committed { text } => {
                        if !gen_live {
                            debug!(utt_gen, "skipping stale finalize commit");
                            self.request_inject_reset();
                        } else {
                            let text = if text.is_empty() {
                                text
                            } else {
                                let rendered = self.render_transcript_text(&text);
                                sanitize_final_injection_text(&rendered)
                            };
                            if text.is_empty() {
                                // Nothing to type — still clear any streaming partial residue.
                                debug!(utt_gen, "finalize ready but empty after render/sanitize");
                                self.request_inject_reset();
                            } else {
                                self.apply_commit(utt_gen, text);
                            }
                        }
                    }
                    FinalizeOutcome::Cancelled => {
                        debug!(utt_gen, "finalize cancelled");
                        self.request_inject_reset();
                    }
                    FinalizeOutcome::Failed {
                        err,
                        count_breaker,
                        cuda_recovered,
                    } => {
                        if cuda_recovered {
                            self.cuda_recovered_skip_reset = true;
                            self.sync_audio_params_from_asr();
                            self.flash_error("⚠ GPU busy — switched ASR to CPU for this session");
                            self.emit(SessionEvent::CudaFallbackApplied {
                                detail: err.clone(),
                            });
                        } else if count_breaker && gen_live {
                            self.note_asr_failure(&err);
                            self.recover_asr_after_failure(&err);
                        }
                        // Always clear streaming partial residue after a failed finalize.
                        self.request_inject_reset();
                    }
                }
                // Keep processing true while final inject commit is in flight so
                // observers/tests don't see "idle" before text is committed.
                let waiting_inject = self.inject_commit.is_some();
                if !waiting_inject {
                    self.processing = false;
                }
                self.state.reset(self.speech_rms_threshold);
                self.debug_current_transcript.clear();
                if !waiting_inject {
                    self.deps.overlay.hide();
                    self.emit(SessionEvent::OverlayHide);
                }
                self.emit(SessionEvent::Status(self.recording_status()));
                self.publish_view();
            }
            JobResult::Reset { purpose, result } => {
                if self.lifecycle_job.as_ref().is_some_and(|l| {
                    std::mem::discriminant(&l.purpose) == std::mem::discriminant(&purpose)
                }) {
                    if let Some(life) = self.lifecycle_job.take() {
                        // Task should be finished; abort if not to avoid HOL.
                        if !life.join.is_finished() {
                            life.join.abort();
                        }
                    }
                }
                match purpose {
                    LifecyclePurpose::StartRecording => {
                        if !self.start_pending {
                            debug!("start reset completed after cancel; ignoring");
                            return;
                        }
                        match result {
                            Ok(g) => self.complete_start_recording(g),
                            Err(err) => self.fail_start_recording(&err),
                        }
                    }
                    LifecyclePurpose::DisabledRecovery => {
                        if !self.start_pending {
                            debug!("disabled-recovery reset completed after cancel; ignoring");
                            return;
                        }
                        match result {
                            Ok(g) => {
                                self.utterance_gen = g;
                                self.circuit.close_after_recovery();
                                // Continue into normal start reset.
                                self.spawn_lifecycle(LifecyclePurpose::StartRecording);
                            }
                            Err(err) => {
                                warn!(%err, "ASR recovery reset failed; still disabled");
                                self.start_pending = false;
                                self.processing = false;
                                self.show_error("⚠ ASR error — restart ShuVoice");
                                self.publish_view();
                            }
                        }
                    }
                    LifecyclePurpose::RecoverAfterFailure { context } => match result {
                        Ok(g) => {
                            self.utterance_gen = g;
                            self.deps.metrics.observe_recovery_reset();
                            debug!(%context, "recovery reset ok");
                        }
                        Err(err) => {
                            warn!(%context, %err, "ASR reset failed after error");
                            self.note_asr_failure(&context);
                        }
                    },
                    LifecyclePurpose::CircuitRecovery => match result {
                        Ok(g) => {
                            self.utterance_gen = g;
                            self.circuit.close_after_recovery();
                            self.deps.overlay.hide();
                            self.emit(SessionEvent::AsrRecovered);
                            self.emit(SessionEvent::OverlayHide);
                            self.publish_view();
                        }
                        Err(err) => {
                            warn!(%err, "ASR circuit recovery reset failed");
                            self.circuit.bump_open_timestamp(self.now());
                        }
                    },
                }
            }
            JobResult::Fallback { result } => {
                if let Some(life) = self.lifecycle_job.take() {
                    if !life.join.is_finished() {
                        life.join.abort();
                    }
                }
                match result {
                    Ok(FallbackOutcome::Applied { detail }) => {
                        warn!(%detail, "CUDA OOM; CPU fallback applied");
                        let _ = self.circuit.on_recovered_cuda_fallback();
                        self.cuda_recovered_skip_reset = true;
                        self.sync_audio_params_from_asr();
                        self.flash_error("⚠ GPU busy — switched ASR to CPU for this session");
                        self.emit(SessionEvent::CudaFallbackApplied { detail });
                    }
                    Ok(other) => {
                        debug!(detail = %other.detail(), "CUDA OOM fallback not applied");
                    }
                    Err(e) => {
                        warn!(%e, "fallback call failed");
                    }
                }
            }
            JobResult::InjectCommit {
                utt_gen,
                text,
                attempt,
                result,
            } => {
                if !self.running {
                    // Shutdown already dropped the op; ignore late completion (no retry).
                    self.effect_end();
                    let _ = self.inject_commit.take();
                    return;
                }
                self.on_inject_commit_result(utt_gen, text, attempt, result);
            }
            JobResult::InjectPartial {
                utt_gen,
                text,
                result,
            } => {
                self.effect_end();
                // Clear slot if this completion matches the tracked job.
                if self
                    .inject_partial
                    .as_ref()
                    .is_some_and(|p| p.utt_gen == utt_gen)
                {
                    let _ = self.inject_partial.take();
                }
                if !self.running {
                    self.pending_partial_text = None;
                    self.pump_inject_pipeline();
                    return;
                }
                // Generation-tag: stale partials (or barrier) must not affect UI/injector order.
                let stale = utt_gen != self.utterance_gen
                    || self.inject_barrier
                    || self.pending_inject_reset
                    || self.inject_reset.is_some();
                if !stale && result.is_ok() {
                    self.deps.metrics.observe_partial_update();
                    self.emit(SessionEvent::InjectPartial { text });
                } else if stale {
                    debug!(
                        utt_gen,
                        current = self.utterance_gen,
                        "suppressing stale/barrier partial result"
                    );
                    // Drop coalesced text from old gen.
                    self.pending_partial_text = None;
                }
                // Drain coalesced partial only if still current and not barred.
                if !stale {
                    if let Some(pending) = self.pending_partial_text.take() {
                        let g = self.utterance_gen;
                        self.spawn_inject_partial(g, pending);
                    }
                }
                self.pump_inject_pipeline();
            }
            JobResult::InjectReset { result } => {
                self.effect_end();
                let _ = self.inject_reset.take();
                match result {
                    Ok(()) => {
                        // Barrier lifts only after a *successful* reset.
                        self.inject_barrier = false;
                        self.pending_inject_reset = false;
                    }
                    Err(e) => {
                        // Keep barrier up — dirty injector must not accept new partials.
                        warn!(%e, "injector reset failed; keeping inject_barrier");
                        self.deps.metrics.observe_commit_failure();
                        self.flash_error(&format!("⚠ inject reset failed: {e}"));
                        // Do not clear inject_barrier. A later request_inject_reset
                        // (start/cancel) will try again once the slot is free.
                        self.pending_inject_reset = false;
                    }
                }
                self.pump_inject_pipeline();
            }
            JobResult::SelectionCapture { kind, result } => {
                // If we still own the job slot, this is the live completion.
                let owned = self.selection_job.take().is_some();
                if owned {
                    self.effect_end();
                } else {
                    // Abandoned earlier (already effect_end'd) — ignore late result entirely.
                    return;
                }
                if !self.running {
                    self.pending_tts = None;
                    return;
                }
                match self.pending_tts.take() {
                    Some(PendingTts::Capturing { kind: k }) if k == kind => {}
                    other => {
                        // Stale capture kind mismatch.
                        self.pending_tts = other;
                        return;
                    }
                }
                match result {
                    Ok(text) => {
                        self.tts_last_error = None;
                        if let Err(e) = self.tts_speak_text(text) {
                            self.tts_last_error = Some(e.to_string());
                        }
                    }
                    Err(e) => {
                        self.deps.metrics.observe_tts_selection_failure();
                        self.tts_last_error = Some(e.clone());
                        self.emit(SessionEvent::TtsError { message: e });
                        self.emit_tts_state(TtsPlayerState::Error);
                    }
                }
                self.publish_view();
            }
        }
    }

    fn apply_commit(&mut self, utt_gen: UtteranceGen, text: String) {
        // Never await injector on the actor — one non-cancellable attempt per gen.
        self.spawn_inject_commit(utt_gen, text, 1);
    }

    fn on_inject_commit_result(
        &mut self,
        utt_gen: UtteranceGen,
        text: String,
        _attempt: u32,
        result: Result<(), String>,
    ) {
        self.effect_end();
        // Clear tracking if this completion matches in-flight job (never abort).
        if self
            .inject_commit
            .as_ref()
            .is_some_and(|c| c.utt_gen == utt_gen)
        {
            let _ = self.inject_commit.take();
        }
        if self.committed_gen == Some(utt_gen) {
            // Already latched — ignore duplicate/late result.
            self.pump_inject_pipeline();
            return;
        }
        match result {
            Ok(()) => {
                // One attempt per generation: latch only on definitive Ok.
                self.committed_gen = Some(utt_gen);
                self.debug_last_final_transcript = text.clone();
                self.debug_current_transcript.clear();
                self.deps.overlay.set_text(&text);
                self.deps.metrics.observe_final_commit();
                self.emit(SessionEvent::FinalTranscript { text: text.clone() });
                self.emit(SessionEvent::InjectFinal { text });
                self.note_asr_success();
                if !self.recording && !self.start_pending {
                    self.processing = false;
                    self.deps.overlay.hide();
                    self.emit(SessionEvent::OverlayHide);
                    self.emit(SessionEvent::Status(self.recording_status()));
                }
                self.publish_view();
            }
            Err(e) => {
                // Untyped Err is treated as Unknown: may have side-effected already
                // (e.g. subprocess timeout after paste). NEVER retry — avoids duplicates.
                warn!(%e, utt_gen, "inject commit failed (no retry; unknown side-effect)");
                self.deps.metrics.observe_commit_failure();
                // Latch generation as spent so we never re-issue commit for it.
                self.committed_gen = Some(utt_gen);
                self.flash_error(&format!("⚠ inject failed: {e}"));
                if !self.recording && !self.start_pending {
                    self.processing = false;
                    self.emit(SessionEvent::Status(self.recording_status()));
                }
                self.publish_view();
            }
        }
        self.pump_inject_pipeline();
    }

    // ── Test-facing helpers (await finalize to completion) ──────────────

    /// Drain audio into utterance buffer.
    pub fn drain_and_buffer(&mut self) {
        for chunk in self.deps.audio.drain() {
            self.append_recording_chunk(&chunk);
        }
    }

    /// Start finalize and wait (test / direct API). Actor tick uses non-blocking path.
    pub async fn handle_recording_stop(&mut self) {
        self.deps.overlay.set_state(OverlayState::Processing);
        if self.finalize.is_none() {
            // Ensure stop edge state
            if self.recording {
                self.stop_recording();
            }
            self.start_finalize_job();
            self.was_recording = false;
        }
        let _ = self
            .await_or_cancel_finalization(Duration::from_secs(60))
            .await;
        self.pump_jobs_while(
            |s| s.inject_commit.is_some() || s.inflight_effects > 0,
            EFFECT_PUMP_BOUND * 4,
        )
        .await;
    }

    pub async fn begin_utterance_public(&mut self) {
        self.begin_utterance().await;
    }

    pub async fn process_recording_chunks(&mut self) {
        if self.is_offline_instant_mode() || self.circuit.is_disabled() {
            return;
        }
        // Synchronous helper for tests: process one native chunk inline via job+await.
        while self.recording && self.state.total >= self.deps.asr.native_chunk_samples() {
            if self.chunk_job.is_some() {
                // wait for prior
                loop {
                    self.poll_jobs().await;
                    if self.chunk_job.is_none() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            }
            let native = self.deps.asr.native_chunk_samples();
            let (pcm, _) = match self.state.consume_native_chunk(native) {
                Ok(v) => v,
                Err(_) => break,
            };
            if pcm.is_empty() {
                break;
            }
            let audio = if self.deps.asr.wants_raw_audio() {
                pcm
            } else {
                apply_utterance_gain(&pcm, self.state.utterance_gain)
            };
            let utt_gen = self.utterance_gen;
            let join = spawn_chunk_job(self.deps.asr.clone(), utt_gen, audio, self.job_tx.clone());
            self.chunk_job = Some(ActiveChunk { utt_gen, join });
            // await completion for test helper
            loop {
                self.poll_jobs().await;
                if self.chunk_job.is_none() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
        self.pump_jobs_while(
            |s| s.inject_partial.is_some() || s.inflight_effects > 0,
            EFFECT_PUMP_BOUND * 2,
        )
        .await;
    }

    pub async fn decode_offline_utterance(&mut self) {
        // Test helper: one-shot offline via owner call on a child task + pump
        // (never blocks the actor select loop if used from handle_command later).
        if self.state.total == 0 || self.circuit.is_disabled() {
            return;
        }
        let mut audio = self.state.concatenated();
        if !self.deps.asr.wants_raw_audio() && self.state.utterance_gain > 1.05 {
            audio = apply_utterance_gain(&audio, self.state.utterance_gain);
        }
        let utt_gen = self.utterance_gen;
        let asr = self.deps.asr.clone();
        let tx = self.job_tx.clone();
        // Reuse Chunk result lane with a dedicated offline path via Finalize-like job:
        // spawn utterance as a one-shot task reporting through a custom channel via Chunk.
        let join = tokio::spawn(async move {
            let text = asr.process_utterance(utt_gen, audio).await.map(|(_, t)| t);
            let _ = tx.send(crate::finalize::JobResult::Chunk { utt_gen, text });
        });
        // Track as chunk job so poll_jobs clears it.
        self.chunk_job = Some(ActiveChunk { utt_gen, join });
        self.pump_jobs_while(
            |s| s.chunk_job.is_some() || s.lifecycle_job.is_some(),
            Duration::from_secs(35),
        )
        .await;
    }

    pub async fn commit_utterance(&mut self) {
        let utt_gen = self.utterance_gen;
        let text = self.state.last_text.trim();
        if text.is_empty() {
            return;
        }
        let text = self.render_transcript_text(text);
        let text = sanitize_final_injection_text(&text);
        if text.is_empty() {
            return;
        }
        self.apply_commit(utt_gen, text);
        self.pump_jobs_while(
            |s| s.inject_commit.is_some() || s.inflight_effects > 0,
            EFFECT_PUMP_BOUND * 4,
        )
        .await;
    }

    pub async fn flush_tail_silence(&mut self) {
        // No-op on actor: tail flush runs inside finalize job.
        // Kept for API compatibility with tests that call it while recording.
        let _ = MAX_TAIL_FLUSH_ACTOR_STALL;
    }

    pub fn render_transcript_text(&self, text: &str) -> String {
        render_transcript_text(text, &self.render)
    }

    fn on_transcript_update(&mut self) {
        let rendered = self.render_transcript_text(&self.state.last_text);
        self.debug_current_transcript = rendered.clone();
        self.deps.overlay.set_text(&rendered);
        self.emit(SessionEvent::PartialTranscript {
            text: rendered.clone(),
        });
        self.emit(SessionEvent::OverlayUpdate {
            state: None,
            text: Some(rendered.clone()),
        });
        if self.config.output_mode == OutputMode::StreamingPartial
            && !self.is_offline_instant_mode()
        {
            let utt_gen = self.utterance_gen;
            self.spawn_inject_partial(utt_gen, rendered);
        }
        self.publish_view();
    }

    // ── Tick (responsive) ───────────────────────────────────────────────

    pub async fn tick(&mut self) -> bool {
        if !self.running {
            return false;
        }
        self.poll_error_toast();
        // Non-blocking ordered flush under observer backpressure.
        self.flush_bus();
        self.poll_jobs().await;
        self.pump_inject_pipeline();
        self.drive_pending_tts();
        // Capture/speak may have completed via jobs; poll again.
        self.poll_jobs().await;
        self.pump_inject_pipeline();

        if !self.deps.asr.is_alive() && self.asr_thread_alive {
            self.asr_thread_alive = false;
            self.recording = false;
            self.show_error("⚠ ASR thread crashed — restart ShuVoice");
            self.emit(SessionEvent::AsrThreadDead);
            self.publish_view();
        }
        if self.circuit.is_disabled() {
            let _ = self.try_circuit_recovery();
        }

        let chunks = self.deps.audio.drain();
        let dropped = self.deps.audio.dropped();
        if dropped > self.last_reported_drops {
            self.emit(SessionEvent::AudioOverflow { dropped });
            self.last_reported_drops = dropped;
        }

        // Late audio during grace goes to finalize job, not noise floor.
        if self.finalize.is_some() {
            let f_grace = self
                .finalize
                .as_ref()
                .map(|f| self.now() <= f.grace_until)
                .unwrap_or(false);
            if f_grace {
                if let Some(f) = self.finalize.as_ref() {
                    for chunk in chunks {
                        let _ = f.late_tx.send(chunk);
                    }
                }
            } else {
                // Grace over — discard stray audio (or noise floor).
                for chunk in chunks {
                    self.noise_floor_rms =
                        update_noise_floor(self.noise_floor_rms, audio_rms(&chunk));
                }
            }
        } else {
            let is_recording = self.recording;
            if is_recording && !self.was_recording {
                self.begin_utterance().await;
            }
            // Stop-edge window: recording just ended but finalize hasn't
            // started yet (possibly deferred across ticks while an in-flight
            // chunk job lands). Audio drained here is still tail-of-utterance
            // — it must reach the finalize buffer, not the noise floor.
            let capture_tail = self.was_recording && !is_recording;
            for chunk in chunks {
                if is_recording || capture_tail {
                    self.append_recording_chunk(&chunk);
                } else {
                    self.noise_floor_rms =
                        update_noise_floor(self.noise_floor_rms, audio_rms(&chunk));
                }
            }
            // Non-blocking streaming: at most one chunk job.
            if is_recording
                && !self.circuit.is_disabled()
                && !self.is_offline_instant_mode()
                && self.chunk_job.is_none()
                && self.state.total >= self.deps.asr.native_chunk_samples()
            {
                let native = self.deps.asr.native_chunk_samples();
                if let Ok((pcm, _)) = self.state.consume_native_chunk(native) {
                    if !pcm.is_empty() {
                        let audio = if self.deps.asr.wants_raw_audio() {
                            pcm
                        } else {
                            apply_utterance_gain(&pcm, self.state.utterance_gain)
                        };
                        let utt_gen = self.utterance_gen;
                        let join = spawn_chunk_job(
                            self.deps.asr.clone(),
                            utt_gen,
                            audio,
                            self.job_tx.clone(),
                        );
                        self.chunk_job = Some(ActiveChunk { utt_gen, join });
                    }
                }
            }

            if self.was_recording && !is_recording {
                // Falling edge: wait until in-flight chunk job lands so last_text
                // / remainder stay coherent, then spawn finalize (still non-blocking
                // w.r.t. multi-second ASR — only polls results).
                if self.chunk_job.is_some() {
                    // Keep was_recording true so we retry finalize next tick.
                    self.publish_view();
                } else {
                    self.start_finalize_job();
                    self.was_recording = false;
                    self.emit(SessionEvent::Status(self.recording_status()));
                    self.publish_view();
                }
            } else {
                self.was_recording = is_recording;
            }
        }

        true
    }

    fn try_circuit_recovery(&mut self) -> bool {
        let now = self.now();
        if !self.circuit.can_attempt_recovery(now) {
            return false;
        }
        if self.lifecycle_job.is_some() {
            return false;
        }
        info!("ASR circuit breaker cooldown elapsed; queueing recovery reset");
        self.spawn_lifecycle(LifecyclePurpose::CircuitRecovery);
        true
    }

    pub async fn shutdown(&mut self) {
        if TEST_HANG_ACTOR_ON_SHUTDOWN.load(std::sync::atomic::Ordering::SeqCst) {
            // Async hang — abortable by JoinHandle::abort (unlike thread::sleep).
            std::future::pending::<()>().await;
        }
        self.running = false;
        self.recording = false;
        self.processing = false;
        self.start_pending = false;
        self.pending_tts = None;
        self.abort_lifecycle_job();
        // Tear down finalize/chunk without spawning a new inject reset.
        if let Some(f) = self.finalize.take() {
            f.cancel.store(true, Ordering::Release);
            let _ = self.deps.asr.bump_gen();
            f.join.abort();
        }
        self.abort_chunk_job();
        self.release_effect_jobs_on_shutdown();
        let _ = self.deps.asr.bump_gen();
        if let Some(tts) = self.deps.tts.as_mut() {
            let _ = tts.stop();
            self.emit_tts_state(TtsPlayerState::Idle);
        }
        // Non-blocking: SessionRuntime + AsrOwnerJoin own graceful/abort bounds.
        self.deps.asr.request_shutdown();
        self.flush_bus();
        self.emit(SessionEvent::ShutdownComplete);
        self.flush_bus();
        self.publish_view();
    }

    // ── TTS ─────────────────────────────────────────────────────────────

    fn emit_tts_state(&mut self, state: TtsPlayerState) {
        self.tts_player_state = state.clone();
        self.deps.view.publish_tts_status(state.as_str());
        self.emit(SessionEvent::TtsState {
            state,
            preview_text: self.tts_last_preview_text.clone(),
        });
    }

    /// Apply a non-blocking TTS player callback update (via SessionCommand).
    ///
    /// Emits essential `TtsState` using the current preview text, and optional
    /// essential `TtsError` when `error_message` is present. Mirrors state onto
    /// `RuntimeView.tts_status` for enqueue-only control reads.
    pub fn apply_tts_player_update(
        &mut self,
        state: TtsPlayerState,
        error_message: Option<String>,
    ) {
        if let Some(message) = error_message {
            self.emit(SessionEvent::TtsError { message });
        }
        self.emit_tts_state(state);
    }

    fn tts_runtime_ready(&self) -> bool {
        self.config.tts_enabled && self.deps.tts.is_some()
    }

    /// Non-blocking TTS request for the actor command path.
    fn request_tts(&mut self, intent: TtsIntent) {
        self.tts_last_error = None;
        if !self.tts_runtime_ready() {
            self.tts_last_error = Some(AppError::TtsNotAvailable.to_string());
            return;
        }
        // Stop STT if active (sync cancel of start_pending / recording).
        if self.recording || self.start_pending {
            self.stop_recording();
        }
        // Need finalize if stop edge pending or finalize in flight / processing.
        let needs_finalize = self.finalize.is_some()
            || self.inject_commit.is_some()
            || (self.was_recording && !self.recording)
            || (self.processing && !self.start_pending);
        if needs_finalize {
            if self.was_recording && !self.recording && self.finalize.is_none() {
                self.start_finalize_job();
                self.was_recording = false;
            }
            let deadline = self.now() + TTS_AWAIT_FINALIZE_TIMEOUT;
            self.pending_tts = Some(PendingTts::WaitFinalize { intent, deadline });
            return;
        }
        self.begin_tts_intent(intent);
    }

    fn begin_tts_intent(&mut self, intent: TtsIntent) {
        match intent {
            TtsIntent::Selection => self.spawn_selection_capture(TtsCaptureKind::Selection),
            TtsIntent::Clipboard => self.spawn_selection_capture(TtsCaptureKind::Clipboard),
            TtsIntent::Explicit { text, source: _ } => {
                if let Err(e) = self.tts_speak_text(text) {
                    self.tts_last_error = Some(e.to_string());
                }
            }
        }
    }

    /// Drive pending TTS state machine (called from tick; never awaits effects).
    fn drive_pending_tts(&mut self) {
        let Some(pending) = self.pending_tts.as_ref() else {
            return;
        };
        match pending {
            PendingTts::WaitFinalize { intent, deadline } => {
                let deadline = *deadline;
                let intent = intent.clone();
                let timed_out = self.now() >= deadline;
                let finalize_done =
                    self.finalize.is_none() && self.inject_commit.is_none() && !self.was_recording;
                if timed_out {
                    // Bounded wait — cancel finalize and proceed (or fail capture).
                    self.cancel_finalize_and_reset_injector();
                    self.processing = false;
                    self.pending_tts = None;
                    self.begin_tts_intent(intent);
                } else if finalize_done {
                    self.pending_tts = None;
                    self.begin_tts_intent(intent);
                }
            }
            PendingTts::Capturing { .. } => {
                // Wait for SelectionCapture JobResult.
            }
        }
    }

    /// Sync engine speak (no I/O await).
    fn tts_speak_text(&mut self, mut text: String) -> AppResult<()> {
        if !self.tts_runtime_ready() {
            return Err(AppError::TtsNotAvailable);
        }
        let max_chars = self.config.tts_max_chars.max(1) as usize;
        if text.chars().count() > max_chars {
            text = truncate_chars(&text, max_chars);
        }
        let tts = self.deps.tts.as_mut().ok_or(AppError::TtsNotAvailable)?;
        match tts.speak(&text, &self.tts_voice_id, &self.config.tts_model_id) {
            Ok(interrupted) => {
                if interrupted {
                    self.deps.metrics.observe_tts_interrupt();
                }
                self.deps.metrics.observe_tts_speak();
                self.tts_last_preview_text = text;
                self.emit_tts_state(TtsPlayerState::Synthesizing);
                Ok(())
            }
            Err(e) => {
                self.emit(SessionEvent::TtsError { message: e.clone() });
                self.emit_tts_state(TtsPlayerState::Error);
                Err(AppError::message(e))
            }
        }
    }

    pub async fn tts_prepare(&mut self) -> AppResult<()> {
        // Direct/test helper: queue empty finalize wait via request path.
        if !self.tts_runtime_ready() {
            return Err(AppError::TtsNotAvailable);
        }
        if self.recording || self.start_pending {
            self.stop_recording();
        }
        // Pump finalize only (no selection).
        if self.was_recording && !self.recording && self.finalize.is_none() {
            self.start_finalize_job();
            self.was_recording = false;
        }
        self.pump_jobs_while(
            |s| s.finalize.is_some() || s.inject_commit.is_some(),
            TTS_AWAIT_FINALIZE_TIMEOUT,
        )
        .await;
        if self.finalize.is_some() {
            self.cancel_finalize_and_reset_injector();
            return Err(AppError::SttProcessingTimeout);
        }
        Ok(())
    }

    pub async fn tts_speak(&mut self, text: String, source: TtsSource) -> AppResult<()> {
        self.request_tts(TtsIntent::Explicit { text, source });
        self.pump_tts_settled().await
    }

    pub async fn tts_speak_selection(&mut self) -> AppResult<()> {
        self.request_tts(TtsIntent::Selection);
        self.pump_tts_settled().await
    }

    pub async fn tts_speak_clipboard(&mut self) -> AppResult<()> {
        self.request_tts(TtsIntent::Clipboard);
        self.pump_tts_settled().await
    }

    async fn pump_tts_settled(&mut self) -> AppResult<()> {
        let bound = TTS_AWAIT_FINALIZE_TIMEOUT + EFFECT_PUMP_BOUND + Duration::from_millis(500);
        let start = tokio::time::Instant::now();
        while self.pending_tts.is_some()
            || self.finalize.is_some()
            || self.inject_commit.is_some()
            || self.lifecycle_job.is_some()
            || self.inflight_effects > 0
        {
            // Advance TTS state machine then drain job results.
            self.drive_pending_tts();
            self.poll_jobs().await;
            if !(self.pending_tts.is_some()
                || self.finalize.is_some()
                || self.inject_commit.is_some()
                || self.lifecycle_job.is_some()
                || self.inflight_effects > 0)
            {
                break;
            }
            if start.elapsed() > bound {
                warn!("pump_tts_settled timed out");
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        self.drive_pending_tts();
        self.poll_jobs().await;
        if let Some(err) = self.tts_last_error.clone() {
            if err.contains("selection") || err.contains("capture") || err.contains("timed out") {
                return Err(AppError::Selection(err));
            }
            if err.contains("not available") || err.contains("disabled") {
                return Err(AppError::TtsNotAvailable);
            }
            return Err(AppError::message(err));
        }
        if self.pending_tts.is_some() {
            return Err(AppError::SttProcessingTimeout);
        }
        Ok(())
    }

    pub fn tts_pause(&mut self) -> AppResult<bool> {
        if !self.tts_runtime_ready() {
            return Err(AppError::TtsNotAvailable);
        }
        let ok = self.deps.tts.as_mut().unwrap().pause();
        if ok {
            self.deps.metrics.observe_tts_pause();
            self.emit_tts_state(TtsPlayerState::Paused);
        }
        Ok(ok)
    }

    pub fn tts_resume(&mut self) -> AppResult<bool> {
        if !self.tts_runtime_ready() {
            return Err(AppError::TtsNotAvailable);
        }
        let ok = self.deps.tts.as_mut().unwrap().resume();
        if ok {
            self.emit_tts_state(TtsPlayerState::Playing);
        }
        Ok(ok)
    }

    pub fn tts_toggle_pause(&mut self) -> AppResult<&'static str> {
        if !self.tts_runtime_ready() {
            return Err(AppError::TtsNotAvailable);
        }
        let tts = self.deps.tts.as_mut().unwrap();
        if !tts.toggle_pause() {
            return Err(AppError::message("tts not playing"));
        }
        let st = tts.state();
        self.emit_tts_state(st.clone());
        if st == TtsPlayerState::Paused {
            self.deps.metrics.observe_tts_pause();
            Ok("paused")
        } else {
            Ok("resumed")
        }
    }

    pub fn tts_restart(&mut self) -> AppResult<bool> {
        if !self.tts_runtime_ready() {
            return Err(AppError::TtsNotAvailable);
        }
        let ok = self.deps.tts.as_mut().unwrap().restart();
        if ok {
            self.emit_tts_state(TtsPlayerState::Synthesizing);
        }
        Ok(ok)
    }

    pub fn tts_stop(&mut self) -> AppResult<bool> {
        if !self.tts_runtime_ready() {
            return Err(AppError::TtsNotAvailable);
        }
        let ok = self.deps.tts.as_mut().unwrap().stop();
        self.emit_tts_state(TtsPlayerState::Idle);
        Ok(ok)
    }

    pub fn tts_set_playback_speed(&mut self, speed: f64) -> f64 {
        let mut normalized = shuvoice_core::normalize_tts_playback_speed_lossy(speed);
        let previous = self.tts_playback_speed;
        let Some(tts) = self.deps.tts.as_mut() else {
            return previous;
        };
        if !tts.supports_speed_control() {
            self.deps.metrics.observe_tts_speed_unsupported();
            return previous;
        }
        if let Some((lo, hi)) = tts.speed_bounds() {
            normalized = normalized.clamp(lo, hi);
        }
        normalized = tts.set_playback_speed(normalized);
        self.tts_playback_speed = normalized;
        if (normalized - previous).abs() < 1e-6 {
            return normalized;
        }
        self.deps.metrics.observe_tts_speed_change();
        let restarted = tts.is_active() && tts.restart();
        if restarted {
            self.deps.metrics.observe_tts_speed_restart();
            self.emit_tts_state(TtsPlayerState::Synthesizing);
        }
        normalized
    }

    pub fn tts_status(&self) -> AppResult<String> {
        if !self.config.tts_enabled {
            return Err(AppError::TtsDisabled);
        }
        if !self.tts_runtime_ready() {
            return Err(AppError::TtsNotAvailable);
        }
        // Mirrored player state — never STT recording status.
        Ok(self.tts_player_state.as_str().to_string())
    }

    pub fn metrics_json(&self) -> String {
        metrics_to_json(&self.deps.metrics.snapshot())
    }

    pub fn debug_status_json(&self) -> String {
        serde_json::json!({
            "app": {
                "asr_backend": self.config.asr_backend.as_str(),
                "tts_backend": self.config.tts_backend.as_str(),
                "recording": self.recording,
                "processing": self.processing,
                "finalizing": self.finalize.is_some(),
                "start_pending": self.start_pending,
                "lifecycle_in_flight": self.lifecycle_job.is_some(),
                "inject_commit_in_flight": self.inject_commit.is_some(),
                "pending_tts": self.pending_tts.is_some(),
                "tts_player_state": self.tts_player_state.as_str(),
                "asr_disabled": self.circuit.is_disabled(),
                "asr_thread_alive": self.asr_thread_alive && self.deps.asr.is_alive(),
                "model_load_failed": self.model_load_failed,
                "utterance_gen": self.utterance_gen,
            },
            "audio": {
                "queue_depth": self.deps.audio.depth(),
                "queue_max": self.deps.audio.capacity(),
                "dropped": self.deps.audio.dropped(),
                "contention_drops": self.deps.audio.contention_drops(),
                "noise_floor_rms": self.noise_floor_rms,
                "asr_op_in_flight": self.deps.asr.op_in_flight(),
                "effective_sample_rate": self.sample_rate,
                "config_sample_rate": self.config.sample_rate,
                "asr_preferred_sample_rate": self.deps.asr.capabilities().preferred_sample_rate,
                "chunk_ms": self.config.chunk_ms,
                "audio_chunk_samples": self.audio_chunk_samples(),
                "min_speech_samples": self.min_speech_samples,
                "preroll_ms": self.config.recording_preroll_ms,
            },
            "asr": {
                "native_chunk_samples": self.deps.asr.native_chunk_samples(),
                "wants_raw_audio": self.deps.asr.wants_raw_audio(),
                "consecutive_failures": self.circuit.consecutive_failures(),
                "current_transcript": if self.config.overlay_debug_mode {
                    self.debug_current_transcript.clone()
                } else {
                    "[redacted]".into()
                },
                "last_final_transcript": if self.config.overlay_debug_mode {
                    self.debug_last_final_transcript.clone()
                } else {
                    "[redacted]".into()
                },
                "finalization_mode": format!("{:?}", self.deps.asr.finalization_mode()),
            },
            "metrics": self.deps.metrics.snapshot(),
        })
        .to_string()
    }

    pub async fn handle_command(&mut self, cmd: SessionCommand) -> AppResult<String> {
        use SessionCommand::*;
        let res = match cmd {
            Start => {
                // Enqueue-only: never await ASR reset on the actor task.
                self.request_start_recording();
                Ok("OK started".into())
            }
            Stop => {
                self.stop_recording();
                Ok("OK stopped".into())
            }
            Toggle => {
                self.toggle_recording().await;
                Ok("OK toggled".into())
            }
            Shutdown => {
                self.shutdown().await;
                Ok("OK shutdown".into())
            }
            TtsSpeak { text, source } => {
                self.request_tts(TtsIntent::Explicit { text, source });
                Ok("OK tts speaking".into())
            }
            TtsSpeakSelection => {
                self.request_tts(TtsIntent::Selection);
                Ok("OK tts speaking".into())
            }
            TtsSpeakClipboard => {
                self.request_tts(TtsIntent::Clipboard);
                Ok("OK tts speaking".into())
            }
            TtsPause => {
                if self.tts_pause()? {
                    Ok("OK tts paused".into())
                } else {
                    Err(AppError::message("tts not playing"))
                }
            }
            TtsResume => {
                if self.tts_resume()? {
                    Ok("OK tts resumed".into())
                } else {
                    Err(AppError::message("tts not paused"))
                }
            }
            TtsTogglePause => {
                let state = self.tts_toggle_pause()?;
                Ok(format!("OK tts {state}"))
            }
            TtsRestart => {
                if self.tts_restart()? {
                    Ok("OK tts restarted".into())
                } else {
                    Err(AppError::message("tts no previous text"))
                }
            }
            TtsStop => {
                if self.tts_stop()? {
                    Ok("OK tts stopped".into())
                } else {
                    Ok("OK tts already idle".into())
                }
            }
            TtsSetSpeed(speed) => {
                let v = self.tts_set_playback_speed(speed);
                Ok(format!("OK tts speed {v}"))
            }
            TtsSelectVoice(id) => {
                let selected = id.trim();
                if !selected.is_empty() {
                    self.tts_voice_id = selected.to_string();
                }
                Ok("OK".into())
            }
            TtsPlayerUpdate {
                state,
                error_message,
            } => {
                self.apply_tts_player_update(state, error_message);
                Ok("OK tts player updated".into())
            }
        };
        self.publish_view();
        res
    }
}
