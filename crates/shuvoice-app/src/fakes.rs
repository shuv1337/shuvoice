//! Test fakes: scripted `AsrBackend` with shared counters, sinks, clocks.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;
use shuvoice_asr::{AsrBackend, AsrBackendKind, AsrError, AsrResult, FallbackOutcome, ProgressFn};
use shuvoice_core::{AsrCapabilities, FinalizationMode};

use crate::traits::{FeedbackSink, OverlaySink, SelectionCapture, TextInjector, TtsEngine};
use crate::types::{OverlayState, TtsPlayerState};

pub use crate::traits::FakeClock;

#[derive(Debug)]
pub struct ScriptedInner {
    pub loaded: bool,
    pub reset_calls: u32,
    pub chunk_calls: u32,
    pub utterance_calls: u32,
    pub finish_calls: u32,
    pub texts: VecDeque<String>,
    pub utterance_text: String,
    pub finish_text: String,
    pub fail_chunk_with: Option<String>,
    pub fail_utterance_with: Option<String>,
    pub fail_finish_with: Option<String>,
    pub fail_reset_with: Option<String>,
    pub fallback: FallbackOutcome,
    pub step: u64,
    pub last_chunk: Vec<f32>,
    pub last_utterance: Vec<f32>,
    pub delay: Duration,
    pub cpu_fallback_applied: bool,
}

impl Default for ScriptedInner {
    fn default() -> Self {
        Self {
            loaded: false,
            reset_calls: 0,
            chunk_calls: 0,
            utterance_calls: 0,
            finish_calls: 0,
            texts: VecDeque::new(),
            utterance_text: String::new(),
            finish_text: String::new(),
            fail_chunk_with: None,
            fail_utterance_with: None,
            fail_finish_with: None,
            fail_reset_with: None,
            fallback: FallbackOutcome::NotApplicable {
                detail: "n/a".into(),
            },
            step: 0,
            last_chunk: Vec::new(),
            last_utterance: Vec::new(),
            delay: Duration::ZERO,
            cpu_fallback_applied: false,
        }
    }
}

/// Controllable async ASR backend. Mutable counters live in `shared` so tests
/// can inspect them after move into the ASR owner.
#[derive(Clone)]
pub struct ScriptedAsrBackend {
    caps: AsrCapabilities,
    kind: AsrBackendKind,
    native_chunk_samples: usize,
    pub shared: Arc<Mutex<ScriptedInner>>,
}

impl Default for ScriptedAsrBackend {
    fn default() -> Self {
        Self::local_streaming(4)
    }
}

impl ScriptedAsrBackend {
    pub fn local_streaming(native: usize) -> Self {
        Self {
            caps: AsrCapabilities {
                finalization_mode: FinalizationMode::LocalStreaming,
                preferred_sample_rate: Some(16_000),
                ..AsrCapabilities::default()
            },
            kind: AsrBackendKind::Sherpa,
            native_chunk_samples: native.max(1),
            shared: Arc::new(Mutex::new(ScriptedInner::default())),
        }
    }

    pub fn offline_instant(native: usize, text: &str) -> Self {
        let mut s = Self::local_streaming(native);
        s.caps.finalization_mode = FinalizationMode::OfflineInstant;
        s.caps.emits_partials = false;
        s.shared.lock().utterance_text = text.into();
        s
    }

    pub fn remote_manual(native: usize, finish: &str) -> Self {
        let mut s = Self::local_streaming(native);
        s.caps.wants_raw_audio = true;
        s.caps.finalization_mode = FinalizationMode::RemoteManualCommit;
        s.caps.preferred_sample_rate = Some(24_000);
        s.kind = AsrBackendKind::OpenaiRealtime;
        s.shared.lock().finish_text = finish.into();
        s
    }

    pub fn shared(&self) -> Arc<Mutex<ScriptedInner>> {
        Arc::clone(&self.shared)
    }
}

pub fn offline_asr(text: &str) -> ScriptedAsrBackend {
    ScriptedAsrBackend::offline_instant(4, text)
}

pub fn remote_asr(finish: &str) -> ScriptedAsrBackend {
    ScriptedAsrBackend::remote_manual(16, finish)
}

#[async_trait]
impl AsrBackend for ScriptedAsrBackend {
    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }
    fn backend_id(&self) -> AsrBackendKind {
        self.kind
    }
    fn native_chunk_samples(&self) -> usize {
        self.native_chunk_samples
    }
    fn debug_step(&self) -> Option<u64> {
        Some(self.shared.lock().step)
    }
    fn cpu_fallback_applied(&self) -> bool {
        self.shared.lock().cpu_fallback_applied
    }

    async fn load(&mut self, progress: &mut ProgressFn<'_>) -> AsrResult<()> {
        let delay = self.shared.lock().delay;
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        progress(Some(1.0), "scripted ready");
        self.shared.lock().loaded = true;
        Ok(())
    }

    async fn reset(&mut self) -> AsrResult<()> {
        let delay = self.shared.lock().delay;
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        let mut g = self.shared.lock();
        g.reset_calls += 1;
        if let Some(err) = g.fail_reset_with.clone() {
            return Err(AsrError::decode(err));
        }
        g.loaded = true;
        Ok(())
    }

    async fn process_chunk(&mut self, pcm: &[f32]) -> AsrResult<String> {
        let delay = self.shared.lock().delay;
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        let mut g = self.shared.lock();
        g.chunk_calls += 1;
        g.step += 1;
        g.last_chunk = pcm.to_vec();
        if let Some(err) = g.fail_chunk_with.clone() {
            return Err(AsrError::from_runtime_message(err));
        }
        Ok(g.texts.pop_front().unwrap_or_default())
    }

    async fn process_utterance(&mut self, pcm: &[f32]) -> AsrResult<String> {
        let delay = self.shared.lock().delay;
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        let mut g = self.shared.lock();
        g.utterance_calls += 1;
        g.last_utterance = pcm.to_vec();
        if let Some(err) = g.fail_utterance_with.clone() {
            return Err(AsrError::from_runtime_message(err));
        }
        Ok(g.utterance_text.clone())
    }

    async fn finish_utterance(&mut self, _timeout: Option<Duration>) -> AsrResult<String> {
        let delay = self.shared.lock().delay;
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        let mut g = self.shared.lock();
        g.finish_calls += 1;
        if let Some(err) = g.fail_finish_with.clone() {
            return Err(AsrError::from_runtime_message(err));
        }
        Ok(g.finish_text.clone())
    }

    async fn try_fallback_to_cpu(&mut self) -> AsrResult<FallbackOutcome> {
        let delay = self.shared.lock().delay;
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        let mut g = self.shared.lock();
        if g.fallback.applied() {
            g.cpu_fallback_applied = true;
        }
        Ok(g.fallback.clone())
    }
}

// ── Sinks ───────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct InjectorInner {
    partials: Vec<String>,
    finals: Vec<String>,
    resets: u32,
    fail_commits_remaining: u32,
    fail_partials_remaining: u32,
    /// Artificial delay for HOL tests (per op).
    delay: Duration,
    /// Total commit_final entries (including failures before push).
    commit_calls: u32,
    /// Ordered effect log for sequencing tests: "partial", "commit", "reset".
    op_log: Vec<String>,
    /// If true, commit_final appends to finals then returns Err (unknown outcome).
    commit_side_effect_then_err: bool,
    fail_resets_remaining: u32,
    /// Per-op delay override: partial/commit/reset independently (ms); 0 = use `delay`.
    partial_delay: Duration,
    commit_delay: Duration,
    reset_delay: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct FakeInjector {
    inner: Arc<Mutex<InjectorInner>>,
}

impl FakeInjector {
    pub fn partials(&self) -> Vec<String> {
        self.inner.lock().partials.clone()
    }
    pub fn finals(&self) -> Vec<String> {
        self.inner.lock().finals.clone()
    }
    pub fn resets(&self) -> u32 {
        self.inner.lock().resets
    }
    pub fn set_fail_commits(&self, n: u32) {
        self.inner.lock().fail_commits_remaining = n;
    }
    pub fn set_delay(&self, d: Duration) {
        self.inner.lock().delay = d;
    }
    pub fn commit_calls(&self) -> u32 {
        self.inner.lock().commit_calls
    }
    pub fn op_log(&self) -> Vec<String> {
        self.inner.lock().op_log.clone()
    }
    pub fn set_commit_side_effect_then_err(&self, v: bool) {
        self.inner.lock().commit_side_effect_then_err = v;
    }
    pub fn set_partial_delay(&self, d: Duration) {
        self.inner.lock().partial_delay = d;
    }
    pub fn set_commit_delay(&self, d: Duration) {
        self.inner.lock().commit_delay = d;
    }
    pub fn set_reset_delay(&self, d: Duration) {
        self.inner.lock().reset_delay = d;
    }
    pub fn set_fail_resets(&self, n: u32) {
        self.inner.lock().fail_resets_remaining = n;
    }
}
#[async_trait]
impl TextInjector for FakeInjector {
    async fn update_partial(&self, text: &str) -> Result<(), String> {
        let delay = {
            let g = self.inner.lock();
            if !g.partial_delay.is_zero() {
                g.partial_delay
            } else {
                g.delay
            }
        };
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        let mut g = self.inner.lock();
        if g.fail_partials_remaining > 0 {
            g.fail_partials_remaining -= 1;
            return Err("partial inject failed".into());
        }
        g.op_log.push(format!("partial:{text}"));
        g.partials.push(text.to_string());
        Ok(())
    }
    async fn commit_final(&self, text: &str) -> Result<(), String> {
        let delay = {
            let g = self.inner.lock();
            if !g.commit_delay.is_zero() {
                g.commit_delay
            } else {
                g.delay
            }
        };
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        let mut g = self.inner.lock();
        g.commit_calls += 1;
        if g.fail_commits_remaining > 0 {
            g.fail_commits_remaining -= 1;
            g.op_log.push("commit:fail".into());
            return Err("commit inject failed".into());
        }
        // Side effect first (paste landed), then optional Err (subprocess timeout/kill).
        g.op_log.push(format!("commit:{text}"));
        g.finals.push(text.to_string());
        if g.commit_side_effect_then_err {
            g.op_log.push("commit:err-after-side-effect".into());
            return Err("inject subprocess timed out after paste".into());
        }
        Ok(())
    }
    async fn reset(&self) -> Result<(), String> {
        let delay = {
            let g = self.inner.lock();
            if !g.reset_delay.is_zero() {
                g.reset_delay
            } else {
                g.delay
            }
        };
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        let mut g = self.inner.lock();
        g.resets += 1;
        if g.fail_resets_remaining > 0 {
            g.fail_resets_remaining -= 1;
            g.op_log.push("reset:fail".into());
            return Err("injector reset failed".into());
        }
        g.partials.clear();
        g.op_log.push("reset".into());
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FakeSelection {
    selection: Arc<Mutex<Result<String, String>>>,
    clipboard: Arc<Mutex<Result<String, String>>>,
    delay: Arc<Mutex<Duration>>,
}

impl Default for FakeSelection {
    fn default() -> Self {
        Self {
            selection: Arc::new(Mutex::new(Ok(String::new()))),
            clipboard: Arc::new(Mutex::new(Ok(String::new()))),
            delay: Arc::new(Mutex::new(Duration::ZERO)),
        }
    }
}

impl FakeSelection {
    pub fn with(selection: Result<String, String>, clipboard: Result<String, String>) -> Self {
        Self {
            selection: Arc::new(Mutex::new(selection)),
            clipboard: Arc::new(Mutex::new(clipboard)),
            delay: Arc::new(Mutex::new(Duration::ZERO)),
        }
    }
    pub fn set_delay(&self, d: Duration) {
        *self.delay.lock() = d;
    }
}

#[async_trait]
impl SelectionCapture for FakeSelection {
    async fn capture_selection(&self) -> Result<String, String> {
        let delay = *self.delay.lock();
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        self.selection.lock().clone()
    }
    async fn capture_clipboard(&self) -> Result<String, String> {
        let delay = *self.delay.lock();
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        self.clipboard.lock().clone()
    }
}

#[derive(Debug, Default)]
pub struct FakeTts {
    pub active: bool,
    pub state_name: TtsPlayerState,
    pub speak_calls: Vec<(String, String, String)>,
    pub stop_calls: u32,
    pub pause_ok: bool,
    pub resume_ok: bool,
    pub restart_ok: bool,
    pub speed: f64,
    pub supports_speed: bool,
}

impl FakeTts {
    pub fn new() -> Self {
        Self {
            state_name: TtsPlayerState::Idle,
            speed: 1.0,
            ..Default::default()
        }
    }
}

impl TtsEngine for FakeTts {
    fn state(&self) -> TtsPlayerState {
        self.state_name.clone()
    }
    fn supports_speed_control(&self) -> bool {
        self.supports_speed
    }
    fn speed_bounds(&self) -> Option<(f64, f64)> {
        self.supports_speed.then_some((0.5, 2.0))
    }
    fn speak(&mut self, text: &str, voice_id: &str, model_id: &str) -> Result<bool, String> {
        let interrupted = self.active;
        self.speak_calls
            .push((text.to_string(), voice_id.to_string(), model_id.to_string()));
        self.active = true;
        self.state_name = TtsPlayerState::Synthesizing;
        Ok(interrupted)
    }
    fn pause(&mut self) -> bool {
        if self.pause_ok {
            self.state_name = TtsPlayerState::Paused;
        }
        self.pause_ok
    }
    fn resume(&mut self) -> bool {
        if self.resume_ok {
            self.state_name = TtsPlayerState::Playing;
        }
        self.resume_ok
    }
    fn toggle_pause(&mut self) -> bool {
        match self.state_name {
            TtsPlayerState::Playing => {
                self.state_name = TtsPlayerState::Paused;
                true
            }
            TtsPlayerState::Paused => {
                self.state_name = TtsPlayerState::Playing;
                true
            }
            _ => false,
        }
    }
    fn restart(&mut self) -> bool {
        self.restart_ok
    }
    fn stop(&mut self) -> bool {
        self.stop_calls += 1;
        let was = self.active || self.state_name != TtsPlayerState::Idle;
        self.active = false;
        self.state_name = TtsPlayerState::Idle;
        was
    }
    fn set_playback_speed(&mut self, speed: f64) -> f64 {
        self.speed = speed.clamp(0.5, 2.0);
        self.speed
    }
}

#[derive(Debug, Default, Clone)]
pub struct OverlayCall {
    pub kind: String,
    pub state: Option<OverlayState>,
    pub text: Option<String>,
}

#[derive(Debug, Default)]
pub struct FakeOverlay {
    pub calls: Vec<OverlayCall>,
    pub visible: bool,
    pub state: Option<OverlayState>,
    pub text: String,
}

impl OverlaySink for FakeOverlay {
    fn show(&mut self, state: OverlayState, text: &str) {
        self.visible = true;
        self.state = Some(state);
        self.text = text.to_string();
        self.calls.push(OverlayCall {
            kind: "show".into(),
            state: Some(state),
            text: Some(text.to_string()),
        });
    }
    fn set_state(&mut self, state: OverlayState) {
        self.state = Some(state);
        self.calls.push(OverlayCall {
            kind: "set_state".into(),
            state: Some(state),
            text: None,
        });
    }
    fn set_text(&mut self, text: &str) {
        self.text = text.to_string();
        self.calls.push(OverlayCall {
            kind: "set_text".into(),
            state: None,
            text: Some(text.to_string()),
        });
    }
    fn hide(&mut self) {
        self.visible = false;
        self.calls.push(OverlayCall {
            kind: "hide".into(),
            state: None,
            text: None,
        });
    }
}

#[derive(Debug, Default)]
pub struct FakeFeedback {
    pub starts: AtomicU32,
    pub stops: AtomicU32,
}

impl FeedbackSink for FakeFeedback {
    fn play_start(&mut self) {
        self.starts.fetch_add(1, Ordering::Relaxed);
    }
    fn play_stop(&mut self) {
        self.stops.fetch_add(1, Ordering::Relaxed);
    }
}
