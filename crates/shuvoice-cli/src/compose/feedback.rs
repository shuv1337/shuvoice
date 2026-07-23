//! Bounded single-worker audio feedback tones for PTT start/stop.
//!
//! # Design
//!
//! - **One** dedicated OS tone worker (never a thread-per-tone).
//! - Tones synthesized with [`shuvoice_core::generate_tone`] and played through
//!   a shared [`AudioOutputFactory`] (typically CPAL; tests use the fake).
//! - `play_start` / `play_stop` are non-blocking: `try_send` on a small
//!   bounded channel; on flood the tone is **dropped** (never blocks the
//!   session actor).
//! - Disabled config is a pure no-op (no worker thread).
//! - Drop / shutdown joins the worker with a **bounded deadline**. The shutdown
//!   API reports whether the worker was actually joined vs detached; it never
//!   claims a clean release after detach.
//! - Diagnostics expose played / dropped / open-failure / write-failure counters.
//!
//! # Integration notes
//!
//! Expected crate deps / features (declared by the integration owner):
//! - `shuvoice-app` (`FeedbackSink`)
//! - `shuvoice-core` (`generate_tone`, `Config`)
//! - `shuvoice-tts` (`AudioOutputFactory`, `FakeAudioOutputFactory` for tests)
//!   — enable the CLI `tts` feature (and `cpal-output` on `shuvoice-tts` for
//!   real device playback).

#![allow(clippy::collapsible_if)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::let_and_return)]
#![allow(clippy::double_must_use)]
#![allow(clippy::result_unit_err)]
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use shuvoice_app::traits::FeedbackSink;
use shuvoice_core::{Config, generate_tone};
use shuvoice_tts::{AudioOutput, AudioOutputFactory};
use tracing::{debug, warn};

/// Default PCM rate for feedback tones.
pub const DEFAULT_FEEDBACK_SAMPLE_RATE: u32 = 24_000;

/// Bounded tone queue capacity (start/stop flood protection).
pub const DEFAULT_TONE_QUEUE_CAPACITY: usize = 4;

/// Join deadline for worker shutdown.
pub const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

/// Feedback tone parameters (mirrors config `[feedback]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneFeedbackConfig {
    pub enabled: bool,
    pub start_freq: u32,
    pub stop_freq: u32,
    pub duration_ms: u32,
    pub volume: f64,
    pub sample_rate: u32,
    /// Capacity of the bounded tone command queue (try-send / drop on flood).
    pub queue_capacity: usize,
    pub shutdown_timeout: Duration,
}

impl Default for ToneFeedbackConfig {
    fn default() -> Self {
        let cfg = Config::default();
        Self::from_config(&cfg)
    }
}

impl ToneFeedbackConfig {
    #[must_use]
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            enabled: cfg.audio_feedback,
            start_freq: cfg.feedback_start_freq,
            stop_freq: cfg.feedback_stop_freq,
            duration_ms: cfg.feedback_duration_ms,
            volume: cfg.feedback_volume,
            sample_rate: DEFAULT_FEEDBACK_SAMPLE_RATE,
            queue_capacity: DEFAULT_TONE_QUEUE_CAPACITY,
            shutdown_timeout: DEFAULT_SHUTDOWN_TIMEOUT,
        }
    }

    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToneKind {
    Start,
    Stop,
}

enum WorkerMsg {
    Tone(ToneKind),
    Shutdown,
}

/// Diagnostics for the tone worker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FeedbackDiagnostics {
    pub played_start: u64,
    pub played_stop: u64,
    /// Tones dropped because the bounded queue was full or the worker is gone.
    pub dropped: u64,
    /// `AudioOutputFactory::open` failures.
    pub open_errors: u64,
    /// `write_samples` / `close` failures after a successful open.
    pub write_errors: u64,
    pub running: bool,
}

/// Outcome of a bounded shutdown/join attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownOutcome {
    /// Worker exited and was joined within the deadline (or was never started).
    Joined,
    /// Already shut down.
    AlreadyStopped,
    /// Deadline exceeded; join handle detached. **Not** a clean release.
    DetachedAfterTimeout,
    /// Worker joined but panicked.
    JoinedPanicked,
}

impl ShutdownOutcome {
    #[must_use]
    pub const fn joined_cleanly(self) -> bool {
        matches!(self, Self::Joined | Self::AlreadyStopped)
    }

    #[must_use]
    pub const fn detached(self) -> bool {
        matches!(self, Self::DetachedAfterTimeout)
    }
}

struct SharedDiag {
    played_start: AtomicU64,
    played_stop: AtomicU64,
    dropped: AtomicU64,
    open_errors: AtomicU64,
    write_errors: AtomicU64,
    running: AtomicBool,
}

impl Default for SharedDiag {
    fn default() -> Self {
        Self {
            played_start: AtomicU64::new(0),
            played_stop: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            open_errors: AtomicU64::new(0),
            write_errors: AtomicU64::new(0),
            running: AtomicBool::new(false),
        }
    }
}

impl SharedDiag {
    fn snapshot(&self) -> FeedbackDiagnostics {
        FeedbackDiagnostics {
            played_start: self.played_start.load(Ordering::Relaxed),
            played_stop: self.played_stop.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
            open_errors: self.open_errors.load(Ordering::Relaxed),
            write_errors: self.write_errors.load(Ordering::Relaxed),
            running: self.running.load(Ordering::Relaxed),
        }
    }
}

/// Single-worker [`FeedbackSink`] implementation.
pub struct ToneFeedbackSink {
    tx: Option<std::sync::mpsc::SyncSender<WorkerMsg>>,
    join: Option<JoinHandle<()>>,
    done: Arc<AtomicBool>,
    diag: Arc<SharedDiag>,
    shutdown_timeout: Duration,
    enabled: bool,
    shut_down: bool,
}

impl ToneFeedbackSink {
    /// Spawn the tone worker when enabled; otherwise return a no-op sink.
    #[must_use]
    pub fn new(cfg: ToneFeedbackConfig, output_factory: Arc<dyn AudioOutputFactory>) -> Self {
        if !cfg.enabled {
            return Self::disabled();
        }

        let capacity = cfg.queue_capacity.max(1);
        let (tx, rx) = std::sync::mpsc::sync_channel::<WorkerMsg>(capacity);
        let done = Arc::new(AtomicBool::new(false));
        let diag = Arc::new(SharedDiag::default());
        let done_t = Arc::clone(&done);
        let diag_t = Arc::clone(&diag);

        let handle = thread::Builder::new()
            .name("shuvoice-feedback-tone".into())
            .spawn(move || {
                tone_worker_main(rx, cfg, output_factory, done_t, diag_t);
            })
            .ok();

        match handle {
            Some(join) => Self {
                tx: Some(tx),
                join: Some(join),
                done,
                diag,
                shutdown_timeout: cfg.shutdown_timeout,
                enabled: true,
                shut_down: false,
            },
            None => {
                warn!("failed to spawn feedback tone worker; feedback disabled");
                Self::disabled()
            }
        }
    }

    /// Construct from app config + output factory.
    #[must_use]
    pub fn from_config(cfg: &Config, output_factory: Arc<dyn AudioOutputFactory>) -> Self {
        Self::new(ToneFeedbackConfig::from_config(cfg), output_factory)
    }

    /// Explicit disabled sink (no thread).
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            tx: None,
            join: None,
            done: Arc::new(AtomicBool::new(true)),
            diag: Arc::new(SharedDiag::default()),
            shutdown_timeout: DEFAULT_SHUTDOWN_TIMEOUT,
            enabled: false,
            shut_down: true,
        }
    }

    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    #[must_use]
    pub fn diagnostics(&self) -> FeedbackDiagnostics {
        self.diag.snapshot()
    }

    /// Tones dropped on a full queue or disconnected worker.
    #[must_use]
    pub fn dropped_tones(&self) -> u64 {
        self.diag.dropped.load(Ordering::Relaxed)
    }

    /// `AudioOutputFactory::open` failure count.
    #[must_use]
    pub fn open_errors(&self) -> u64 {
        self.diag.open_errors.load(Ordering::Relaxed)
    }

    /// Write/close failure count after a successful open.
    #[must_use]
    pub fn write_errors(&self) -> u64 {
        self.diag.write_errors.load(Ordering::Relaxed)
    }

    /// Bounded shutdown: signal worker, wait up to `shutdown_timeout`, detach if needed.
    ///
    /// Never reports [`ShutdownOutcome::Joined`] when the worker was detached.
    pub fn shutdown(&mut self) -> ShutdownOutcome {
        if self.shut_down && self.join.is_none() {
            return ShutdownOutcome::AlreadyStopped;
        }
        if !self.enabled {
            self.shut_down = true;
            return ShutdownOutcome::AlreadyStopped;
        }

        if let Some(tx) = self.tx.take() {
            // try_send so a full queue cannot block Drop/shutdown.
            let _ = tx.try_send(WorkerMsg::Shutdown);
            // Drop sender so `recv` unblocks if the shutdown message was dropped.
            drop(tx);
        }

        let outcome = join_worker_bounded(self.join.take(), &self.done, self.shutdown_timeout);
        self.shut_down = true;
        if outcome.detached() {
            warn!(
                timeout_ms = self.shutdown_timeout.as_millis() as u64,
                "feedback tone worker join deadline exceeded; thread detached (not released)"
            );
        }
        outcome
    }

    fn try_enqueue(&self, kind: ToneKind) {
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        match tx.try_send(WorkerMsg::Tone(kind)) {
            Ok(()) => {}
            Err(std::sync::mpsc::TrySendError::Full(_)) => {
                self.diag.dropped.fetch_add(1, Ordering::Relaxed);
                debug!(?kind, "feedback tone dropped (queue full)");
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                self.diag.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl FeedbackSink for ToneFeedbackSink {
    fn play_start(&mut self) {
        if self.enabled {
            self.try_enqueue(ToneKind::Start);
        }
    }

    fn play_stop(&mut self) {
        if self.enabled {
            self.try_enqueue(ToneKind::Stop);
        }
    }
}

impl Drop for ToneFeedbackSink {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn join_worker_bounded(
    join: Option<JoinHandle<()>>,
    done: &AtomicBool,
    timeout: Duration,
) -> ShutdownOutcome {
    let Some(handle) = join else {
        return ShutdownOutcome::AlreadyStopped;
    };

    let deadline = Instant::now() + timeout;
    while !done.load(Ordering::SeqCst) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }

    if done.load(Ordering::SeqCst) {
        match handle.join() {
            Ok(()) => ShutdownOutcome::Joined,
            Err(_) => ShutdownOutcome::JoinedPanicked,
        }
    } else {
        drop(handle);
        ShutdownOutcome::DetachedAfterTimeout
    }
}

fn tone_worker_main(
    rx: std::sync::mpsc::Receiver<WorkerMsg>,
    cfg: ToneFeedbackConfig,
    output_factory: Arc<dyn AudioOutputFactory>,
    done: Arc<AtomicBool>,
    diag: Arc<SharedDiag>,
) {
    struct DoneGuard(Arc<AtomicBool>, Arc<SharedDiag>);
    impl Drop for DoneGuard {
        fn drop(&mut self) {
            self.1.running.store(false, Ordering::SeqCst);
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let _guard = DoneGuard(Arc::clone(&done), Arc::clone(&diag));
    diag.running.store(true, Ordering::SeqCst);

    let sample_rate = cfg.sample_rate.max(1);

    while let Ok(msg) = rx.recv() {
        match msg {
            WorkerMsg::Shutdown => break,
            WorkerMsg::Tone(kind) => {
                let (freq, counter): (u32, &AtomicU64) = match kind {
                    ToneKind::Start => (cfg.start_freq, &diag.played_start),
                    ToneKind::Stop => (cfg.stop_freq, &diag.played_stop),
                };
                let tone = generate_tone(
                    f64::from(freq),
                    cfg.duration_ms.max(1),
                    cfg.volume,
                    sample_rate,
                );
                let pcm = f32_slice_to_i16(&tone);
                match play_pcm(output_factory.as_ref(), sample_rate, &pcm, &diag) {
                    Ok(()) => {
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(()) => {
                        // Counters already updated inside play_pcm; static log only.
                        warn!(?kind, "feedback tone playback failed");
                    }
                }
            }
        }
    }
}

fn play_pcm(
    factory: &dyn AudioOutputFactory,
    sample_rate: u32,
    pcm: &[i16],
    diag: &SharedDiag,
) -> Result<(), ()> {
    let out: Arc<dyn AudioOutput> = match factory.open(sample_rate) {
        Ok(o) => o,
        Err(err) => {
            diag.open_errors.fetch_add(1, Ordering::Relaxed);
            // TtsError Display must stay payload-free (no transcript).
            warn!(error = %err, "feedback tone output open failed");
            return Err(());
        }
    };
    if let Err(err) = out.write_samples(pcm) {
        diag.write_errors.fetch_add(1, Ordering::Relaxed);
        warn!(error = %err, "feedback tone write failed");
        let _ = out.close();
        return Err(());
    }
    if let Err(err) = out.close() {
        diag.write_errors.fetch_add(1, Ordering::Relaxed);
        warn!(error = %err, "feedback tone close failed");
        return Err(());
    }
    Ok(())
}

/// Convert mono f32 PCM \([-1, 1]\) to s16le samples.
///
/// Mapping:
/// - `-1.0` → [`i16::MIN`] (`-32768`)
/// - `+1.0` → [`i16::MAX`] (`32767`)
#[must_use]
pub fn f32_slice_to_i16(samples: &[f32]) -> Vec<i16> {
    samples.iter().copied().map(f32_to_i16).collect()
}

/// Single-sample conversion with correct `i16::MIN` endpoint for `-1.0`.
#[must_use]
pub fn f32_to_i16(sample: f32) -> i16 {
    let s = sample.clamp(-1.0, 1.0);
    if s < 0.0 {
        // [-1, 0) → [i16::MIN, 0)
        (f64::from(s) * 32768.0).round() as i16
    } else {
        // [0, 1] → [0, i16::MAX]
        (f64::from(s) * 32767.0).round() as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shuvoice_tts::FakeAudioOutputFactory;
    use shuvoice_tts::TtsError;

    fn wait_until(pred: impl Fn() -> bool, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if pred() {
                return true;
            }
            thread::sleep(Duration::from_millis(5));
        }
        pred()
    }

    #[test]
    fn disabled_is_noop() {
        let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
        let mut sink = ToneFeedbackSink::new(ToneFeedbackConfig::disabled(), factory.clone());
        sink.play_start();
        sink.play_stop();
        assert!(!sink.is_enabled());
        assert_eq!(sink.diagnostics().played_start, 0);
        assert_eq!(sink.diagnostics().played_stop, 0);
        assert_eq!(sink.dropped_tones(), 0);
        assert_eq!(sink.open_errors(), 0);
        assert!(factory.snapshot().is_empty());
        assert_eq!(sink.shutdown(), ShutdownOutcome::AlreadyStopped);
    }

    #[test]
    fn plays_start_and_stop_tones() {
        let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
        let cfg = ToneFeedbackConfig {
            enabled: true,
            start_freq: 880,
            stop_freq: 660,
            duration_ms: 20,
            volume: 0.1,
            sample_rate: 8_000,
            queue_capacity: 4,
            shutdown_timeout: Duration::from_secs(2),
        };
        let mut sink = ToneFeedbackSink::new(cfg, factory.clone());
        sink.play_start();
        sink.play_stop();

        assert!(wait_until(
            || {
                let d = sink.diagnostics();
                d.played_start >= 1 && d.played_stop >= 1
            },
            Duration::from_secs(2)
        ));
        assert!(!factory.snapshot().is_empty());
        assert_eq!(sink.shutdown(), ShutdownOutcome::Joined);
        assert_eq!(sink.shutdown(), ShutdownOutcome::AlreadyStopped);
    }

    #[test]
    fn flood_drops_rather_than_block() {
        let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
        // Slow writes so the queue backs up.
        *factory.write_delay.lock() = Duration::from_millis(80);
        let cfg = ToneFeedbackConfig {
            enabled: true,
            duration_ms: 30,
            sample_rate: 8_000,
            queue_capacity: 1,
            shutdown_timeout: Duration::from_secs(3),
            ..ToneFeedbackConfig::default()
        };
        let mut sink = ToneFeedbackSink::new(cfg, factory);
        for _ in 0..16 {
            sink.play_start();
            sink.play_stop();
        }
        assert!(wait_until(
            || sink.dropped_tones() > 0,
            Duration::from_secs(2)
        ));
        let outcome = sink.shutdown();
        assert!(
            outcome.joined_cleanly() || outcome == ShutdownOutcome::JoinedPanicked,
            "unexpected outcome: {outcome:?}"
        );
    }

    #[test]
    fn f32_to_i16_endpoints() {
        assert_eq!(f32_to_i16(0.0), 0);
        assert_eq!(f32_to_i16(1.0), i16::MAX);
        assert_eq!(f32_to_i16(-1.0), i16::MIN);
        assert_eq!(f32_to_i16(2.0), i16::MAX);
        assert_eq!(f32_to_i16(-2.0), i16::MIN);
        let out = f32_slice_to_i16(&[0.0, 1.0, -1.0]);
        assert_eq!(out, vec![0, i16::MAX, i16::MIN]);
    }

    #[test]
    fn from_config_respects_disabled_flag() {
        let mut cfg = Config::default();
        cfg.audio_feedback = false;
        let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
        let sink = ToneFeedbackSink::from_config(&cfg, factory);
        assert!(!sink.is_enabled());
    }

    #[test]
    fn open_failure_is_counted() {
        /// Factory that always fails open.
        struct FailOpenFactory;
        impl AudioOutputFactory for FailOpenFactory {
            fn open(&self, _sample_rate_hz: u32) -> Result<Arc<dyn AudioOutput>, TtsError> {
                Err(TtsError::audio("synthetic open failure"))
            }
        }

        let cfg = ToneFeedbackConfig {
            enabled: true,
            duration_ms: 10,
            sample_rate: 8_000,
            queue_capacity: 4,
            shutdown_timeout: Duration::from_secs(2),
            ..ToneFeedbackConfig::default()
        };
        let mut sink = ToneFeedbackSink::new(cfg, Arc::new(FailOpenFactory));
        sink.play_start();
        assert!(wait_until(
            || sink.open_errors() >= 1,
            Duration::from_secs(2)
        ));
        assert_eq!(sink.diagnostics().played_start, 0);
        assert_eq!(sink.write_errors(), 0);
        assert!(sink.shutdown().joined_cleanly());
    }

    #[test]
    fn shutdown_outcome_predicates() {
        assert!(ShutdownOutcome::Joined.joined_cleanly());
        assert!(ShutdownOutcome::AlreadyStopped.joined_cleanly());
        assert!(!ShutdownOutcome::DetachedAfterTimeout.joined_cleanly());
        assert!(ShutdownOutcome::DetachedAfterTimeout.detached());
    }
}
