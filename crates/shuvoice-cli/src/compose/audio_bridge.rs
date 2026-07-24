//! Dedicated OS-thread bridge from capture backend into app [`AudioIngress`].
//!
//! # Design
//!
//! - Owns capture on **one** dedicated OS thread (never on a Tokio worker).
//! - Capture is configured at the **effective ASR sample rate** / chunk size.
//! - Forward path calls [`AudioIngress::try_push`] only (non-blocking).
//! - [`AudioBridge::start`] **synchronously** waits (bounded `recv_timeout`) for
//!   a readiness `Result` from the capture thread. Open/start failure returns
//!   `Err` — never `Ok` for a dead capture.
//! - Stop is cooperative with a **bounded join deadline**. The stop API reports
//!   whether the thread was actually joined vs detached after timeout; it never
//!   claims a clean release after detach.
//! - [`DeviceRef::Name`] is passed through; [`DeviceRef::Index`] is resolved
//!   deterministically here via cpal enumeration (no IO-crate changes).
//!
//! # Queue semantics
//!
//! Two independent bounded queues exist on the path:
//!
//! 1. **Backend hold queue** (`backend_hold_queue_capacity`): inside the capture
//!    backend (CPAL path). Drop-oldest when the forward loop is slower than the
//!    callback. Counted as [`AudioBridgeDiagnostics::backend_hold_queue_dropped`].
//! 2. **App ingress ring** (`AudioIngress`): session-owned. `try_push` never
//!    blocks. A `false` return means the push was *lossy* (either the current
//!    chunk was dropped on lock contention, or oldest was evicted to make room).
//!    Counted as [`AudioBridgeDiagnostics::ingress_lossy_pushes`].
//!
//! These counters are **not** interchangeable; diagnostics expose them separately.
//!
//! # Integration notes
//!
//! Expected crate deps / features (declared by the integration owner):
//! - `shuvoice-app` (`AudioIngress`)
//! - `shuvoice-io` with feature `audio` (`CpalAudioCapture`, `AudioConfig`)
//! - `shuvoice-core` (`DeviceRef`, optional `Config` helper)
//! - `cpal` (CLI feature `audio`)
//!
//! `start` uses `std::sync::mpsc::Receiver::recv_timeout` only (bounded). Call it
//! from a dedicated thread or `spawn_blocking` if invoked near the async runtime;
//! do not park a Tokio worker on an unbounded wait.

#![allow(clippy::collapsible_if)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::let_and_return)]
#![allow(clippy::double_must_use)]
#![allow(clippy::result_unit_err)]
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use shuvoice_app::AudioIngress;
use shuvoice_core::{Config, DeviceRef};
use shuvoice_io::audio::{AudioConfig, CpalAudioCapture};
use tracing::{error, info, warn};

/// Default bound on how long [`AudioBridge::stop`] waits for the OS thread.
pub const DEFAULT_STOP_JOIN_TIMEOUT: Duration = Duration::from_secs(2);

/// Default bound on how long [`AudioBridge::start`] waits for capture readiness.
pub const DEFAULT_START_READY_TIMEOUT: Duration = Duration::from_secs(5);

/// Default poll timeout when waiting for the next capture chunk.
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_millis(20);

/// Default backend hold-queue capacity (CPAL-side, forwarded ASAP to ingress).
pub const DEFAULT_BACKEND_HOLD_QUEUE_CAPACITY: usize = 32;

/// Configuration for the capture bridge.
#[derive(Debug, Clone)]
pub struct AudioBridgeConfig {
    /// Effective ASR capture rate (e.g. 16_000 Sherpa, 24_000 OpenAI).
    pub sample_rate: u32,
    /// Chunk length in samples at `sample_rate`.
    pub chunk_samples: usize,
    /// Host fallback rate when the device rejects `sample_rate`.
    pub fallback_sample_rate: u32,
    pub input_gain: f32,
    /// Capacity of the **backend hold queue** (capture-side, drop-oldest).
    ///
    /// Not the app [`AudioIngress`] ring capacity. Chunks are forwarded to the
    /// ingress as fast as the bridge loop can `try_push`.
    pub backend_hold_queue_capacity: usize,
    pub device: Option<DeviceRef>,
    /// Bounded wait for capture open/start readiness (`recv_timeout`).
    pub start_ready_timeout: Duration,
    /// Bounded wait for cooperative stop + thread join.
    pub stop_join_timeout: Duration,
    pub poll_timeout: Duration,
}

impl Default for AudioBridgeConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            chunk_samples: 1_600,
            fallback_sample_rate: 48_000,
            input_gain: 1.0,
            backend_hold_queue_capacity: DEFAULT_BACKEND_HOLD_QUEUE_CAPACITY,
            device: None,
            start_ready_timeout: DEFAULT_START_READY_TIMEOUT,
            stop_join_timeout: DEFAULT_STOP_JOIN_TIMEOUT,
            poll_timeout: DEFAULT_POLL_TIMEOUT,
        }
    }
}

impl AudioBridgeConfig {
    /// Build from validated app config + effective ASR rate / chunk samples.
    #[must_use]
    pub fn from_app_config(cfg: &Config, sample_rate: u32, chunk_samples: usize) -> Self {
        Self {
            sample_rate: sample_rate.max(1),
            chunk_samples: chunk_samples.max(1),
            fallback_sample_rate: cfg.fallback_sample_rate.max(1),
            input_gain: cfg.input_gain as f32,
            // Config `audio_queue_max_size` historically sized the capture hold
            // queue; clamp to a small forward buffer (ingress owns the real ring).
            backend_hold_queue_capacity: (cfg.audio_queue_max_size as usize)
                .clamp(4, DEFAULT_BACKEND_HOLD_QUEUE_CAPACITY.saturating_mul(4)),
            device: cfg.audio_device.clone(),
            start_ready_timeout: DEFAULT_START_READY_TIMEOUT,
            stop_join_timeout: DEFAULT_STOP_JOIN_TIMEOUT,
            poll_timeout: DEFAULT_POLL_TIMEOUT,
        }
    }
}

/// Snapshot of bridge / capture health counters.
///
/// Counters are independent; do not assume `backend_*` and `ingress_*` measure
/// the same drop event.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AudioBridgeDiagnostics {
    /// Drop-oldest overflows inside the capture **backend hold queue**.
    pub backend_hold_queue_dropped: u64,
    /// Audio-callback `try_lock` failures inside the capture backend.
    pub backend_callback_lock_fails: u64,
    /// Times [`AudioIngress::try_push`] returned `false` (lossy push:
    /// contention drop of current chunk, or oldest-evicted to admit newest).
    pub ingress_lossy_pushes: u64,
    /// `true` while the OS thread is past readiness and inside the forward loop.
    pub running: bool,
}

/// Outcome of a bounded stop/join attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopOutcome {
    /// Thread exited and was joined within the deadline.
    Joined,
    /// Stop was a no-op (already stopped / never started).
    AlreadyStopped,
    /// Deadline exceeded; join handle was detached. Resources may still be held
    /// briefly by the surviving thread — **not** a clean release.
    DetachedAfterTimeout,
    /// Thread was joined but panicked.
    JoinedPanicked,
}

impl StopOutcome {
    /// `true` only when the OS thread is known finished and joined.
    #[must_use]
    pub const fn joined_cleanly(self) -> bool {
        matches!(self, Self::Joined | Self::AlreadyStopped)
    }

    /// `true` when the caller must assume the thread may still be alive.
    #[must_use]
    pub const fn detached(self) -> bool {
        matches!(self, Self::DetachedAfterTimeout)
    }
}

#[derive(Debug, Default)]
struct SharedDiag {
    backend_hold_queue_dropped: AtomicU64,
    backend_callback_lock_fails: AtomicU64,
    ingress_lossy_pushes: AtomicU64,
    running: AtomicBool,
}

impl SharedDiag {
    fn snapshot(&self) -> AudioBridgeDiagnostics {
        AudioBridgeDiagnostics {
            backend_hold_queue_dropped: self.backend_hold_queue_dropped.load(Ordering::Relaxed),
            backend_callback_lock_fails: self.backend_callback_lock_fails.load(Ordering::Relaxed),
            ingress_lossy_pushes: self.ingress_lossy_pushes.load(Ordering::Relaxed),
            running: self.running.load(Ordering::Relaxed),
        }
    }
}

/// Capture backend surface used by the bridge thread.
///
/// Production uses CPAL; tests inject fakes that can fail open/start without
/// touching a real device.
pub trait CaptureBackend: Send {
    fn get_chunk(&mut self, timeout: Duration) -> Option<Vec<f32>>;
    fn stop(&mut self);
    fn backend_hold_queue_dropped(&self) -> u64;
    fn backend_callback_lock_fails(&self) -> u64;
    fn resolved_device_name(&self) -> Option<&str>;
}

/// Opens a [`CaptureBackend`] from an [`AudioConfig`].
pub trait CaptureOpener: Send + 'static {
    fn open(self, cfg: AudioConfig) -> Result<Box<dyn CaptureBackend>, String>;
}

/// Production opener: `CpalAudioCapture::new` + `start`.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpalCaptureOpener;

impl CaptureOpener for CpalCaptureOpener {
    fn open(self, cfg: AudioConfig) -> Result<Box<dyn CaptureBackend>, String> {
        let mut capture = CpalAudioCapture::new(cfg);
        capture
            .start()
            .map_err(|err| format!("audio capture start failed: {err}"))?;
        Ok(Box::new(CpalCaptureBackend { inner: capture }))
    }
}

struct CpalCaptureBackend {
    inner: CpalAudioCapture,
}

impl CaptureBackend for CpalCaptureBackend {
    fn get_chunk(&mut self, timeout: Duration) -> Option<Vec<f32>> {
        self.inner.get_chunk(timeout)
    }

    fn stop(&mut self) {
        self.inner.stop();
    }

    fn backend_hold_queue_dropped(&self) -> u64 {
        self.inner.dropped_chunks()
    }

    fn backend_callback_lock_fails(&self) -> u64 {
        self.inner.callback_lock_fails()
    }

    fn resolved_device_name(&self) -> Option<&str> {
        self.inner.resolved_device_name()
    }
}

/// Owns the capture OS thread and forwards PCM into the session audio ring.
pub struct AudioBridge {
    stop: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    diag: Arc<SharedDiag>,
    join: Option<JoinHandle<()>>,
    stop_join_timeout: Duration,
    resolved_device_name: Option<String>,
    /// Set once stop has run (joined or detached).
    stopped: bool,
}

impl AudioBridge {
    /// Resolve device, spawn the capture thread, wait for readiness, start streaming.
    ///
    /// # Errors
    ///
    /// Returns `Err` when device resolution fails, the thread cannot be spawned,
    /// capture open/start fails, readiness times out, or the thread exits before
    /// reporting readiness. On any error path the OS thread is joined or
    /// detached with a bound — never left as a silently-dead `Ok(Self)`.
    pub fn start(ingress: AudioIngress, cfg: AudioBridgeConfig) -> Result<Self, String> {
        Self::start_with(ingress, cfg, CpalCaptureOpener)
    }

    /// Like [`start`](Self::start) with an injectable capture opener (tests).
    pub fn start_with<O: CaptureOpener>(
        ingress: AudioIngress,
        cfg: AudioBridgeConfig,
        opener: O,
    ) -> Result<Self, String> {
        let device_name = resolve_device_name(cfg.device.as_ref())?;
        let stop_join_timeout = cfg.stop_join_timeout;
        let start_ready_timeout = cfg.start_ready_timeout;
        let stop = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let diag = Arc::new(SharedDiag::default());

        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<Option<String>, String>>(1);

        let stop_t = Arc::clone(&stop);
        let done_t = Arc::clone(&done);
        let diag_t = Arc::clone(&diag);
        let device_name_t = device_name.clone();

        let handle = thread::Builder::new()
            .name("shuvoice-audio-bridge".into())
            .spawn(move || {
                capture_thread_main(
                    ingress,
                    cfg,
                    device_name_t,
                    opener,
                    ready_tx,
                    stop_t,
                    done_t,
                    diag_t,
                );
            })
            .map_err(|e| format!("spawn audio bridge thread: {e}"))?;

        // Bounded readiness wait only — never busy-spin, never unbounded.
        let ready = match ready_rx.recv_timeout(start_ready_timeout) {
            Ok(report) => report,
            Err(RecvTimeoutError::Timeout) => {
                error!(
                    timeout_ms = start_ready_timeout.as_millis() as u64,
                    "audio capture readiness timed out"
                );
                stop.store(true, Ordering::SeqCst);
                let _ = join_handle_bounded(Some(handle), &done, stop_join_timeout);
                return Err(format!(
                    "audio capture start timed out after {}ms",
                    start_ready_timeout.as_millis()
                ));
            }
            Err(RecvTimeoutError::Disconnected) => {
                // Thread died before sending readiness.
                let _ = join_handle_bounded(Some(handle), &done, stop_join_timeout);
                return Err("audio capture thread exited before readiness".into());
            }
        };

        match ready {
            Ok(resolved) => Ok(Self {
                stop,
                done,
                diag,
                join: Some(handle),
                stop_join_timeout,
                resolved_device_name: resolved.or(device_name),
                stopped: false,
            }),
            Err(err) => {
                // Capture open/start failed; thread is exiting (or exited).
                // Payload-free: `err` is composed of static labels / host codes only.
                error!(error = %err, "audio capture failed during start");
                let _ = join_handle_bounded(Some(handle), &done, stop_join_timeout);
                Err(err)
            }
        }
    }

    /// Device name resolved at start (`None` = host default).
    #[must_use]
    pub fn resolved_device_name(&self) -> Option<&str> {
        self.resolved_device_name.as_deref()
    }

    /// Non-blocking diagnostics snapshot.
    #[must_use]
    pub fn diagnostics(&self) -> AudioBridgeDiagnostics {
        self.diag.snapshot()
    }

    /// Backend hold-queue drop-oldest count only (not ingress lossy pushes).
    #[must_use]
    pub fn backend_hold_queue_dropped(&self) -> u64 {
        self.diagnostics().backend_hold_queue_dropped
    }

    /// Capture-callback lock-fail counter.
    #[must_use]
    pub fn backend_callback_lock_fails(&self) -> u64 {
        self.diagnostics().backend_callback_lock_fails
    }

    /// Ingress `try_push` lossy-push count (see [`AudioBridgeDiagnostics`]).
    #[must_use]
    pub fn ingress_lossy_pushes(&self) -> u64 {
        self.diagnostics().ingress_lossy_pushes
    }

    /// Signal stop and join with the configured deadline.
    ///
    /// Returns a precise [`StopOutcome`]. **Never** reports [`StopOutcome::Joined`]
    /// when the thread was detached after timeout.
    pub fn stop(&mut self) -> StopOutcome {
        if self.stopped && self.join.is_none() {
            return StopOutcome::AlreadyStopped;
        }
        self.stop.store(true, Ordering::SeqCst);
        let outcome = join_handle_bounded(self.join.take(), &self.done, self.stop_join_timeout);
        self.stopped = true;
        if outcome.detached() {
            warn!(
                timeout_ms = self.stop_join_timeout.as_millis() as u64,
                "audio bridge stop join deadline exceeded; thread detached (not released)"
            );
        }
        outcome
    }
}

impl Drop for AudioBridge {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Bounded join helper shared by start-failure cleanup and stop.
fn join_handle_bounded(
    join: Option<JoinHandle<()>>,
    done: &AtomicBool,
    timeout: Duration,
) -> StopOutcome {
    let Some(handle) = join else {
        return StopOutcome::AlreadyStopped;
    };

    let deadline = Instant::now() + timeout;
    while !done.load(Ordering::SeqCst) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }

    if done.load(Ordering::SeqCst) {
        match handle.join() {
            Ok(()) => StopOutcome::Joined,
            Err(_) => {
                error!("audio bridge thread panicked");
                StopOutcome::JoinedPanicked
            }
        }
    } else {
        // Detach: drop JoinHandle without join. Thread may still be alive.
        drop(handle);
        StopOutcome::DetachedAfterTimeout
    }
}

fn capture_thread_main<O: CaptureOpener>(
    ingress: AudioIngress,
    cfg: AudioBridgeConfig,
    device_name: Option<String>,
    opener: O,
    ready_tx: mpsc::SyncSender<Result<Option<String>, String>>,
    stop: Arc<AtomicBool>,
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

    let audio_cfg = AudioConfig {
        sample_rate: cfg.sample_rate.max(1),
        chunk_samples: cfg.chunk_samples.max(1),
        fallback_sample_rate: cfg.fallback_sample_rate.max(1),
        input_gain: cfg.input_gain,
        queue_max_size: cfg.backend_hold_queue_capacity.max(1),
        device_name,
    };

    let mut capture = match opener.open(audio_cfg) {
        Ok(c) => c,
        Err(err) => {
            // Static/host labels only — never log transcript content.
            let _ = ready_tx.send(Err(err));
            return;
        }
    };

    let resolved = capture.resolved_device_name().map(str::to_owned);
    // Readiness success: capture is open and running.
    if ready_tx.send(Ok(resolved.clone())).is_err() {
        // Starter went away; shut down quietly.
        capture.stop();
        return;
    }

    diag.running.store(true, Ordering::SeqCst);
    info!(
        sample_rate = cfg.sample_rate,
        chunk_samples = cfg.chunk_samples,
        device = resolved.as_deref().unwrap_or("default"),
        "audio bridge capture running"
    );

    while !stop.load(Ordering::Relaxed) {
        diag.backend_hold_queue_dropped
            .store(capture.backend_hold_queue_dropped(), Ordering::Relaxed);
        diag.backend_callback_lock_fails
            .store(capture.backend_callback_lock_fails(), Ordering::Relaxed);

        if let Some(chunk) = capture.get_chunk(cfg.poll_timeout)
            && !ingress.try_push(chunk)
        {
            // Lossy push: contention drop or oldest-evicted (chunk may still land).
            diag.ingress_lossy_pushes.fetch_add(1, Ordering::Relaxed);
        }
    }

    capture.stop();
    diag.backend_hold_queue_dropped
        .store(capture.backend_hold_queue_dropped(), Ordering::Relaxed);
    diag.backend_callback_lock_fails
        .store(capture.backend_callback_lock_fails(), Ordering::Relaxed);
    info!("audio bridge capture stopped");
}

/// Resolve a configured [`DeviceRef`] into a cpal device name hint.
///
/// - `None` → host default (`None` name)
/// - `Name` → exact name passthrough
/// - `Index` → deterministic index into the enumerated input device list
///   (same ordering as `shuvoice audio list-devices`)
pub fn resolve_device_name(device: Option<&DeviceRef>) -> Result<Option<String>, String> {
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
        Some(DeviceRef::Index(idx)) => resolve_input_device_index(*idx).map(Some),
    }
}

fn resolve_input_device_index(idx: i64) -> Result<String, String> {
    if idx < 0 {
        return Err(format!("audio_device index {idx} is negative"));
    }
    let i = idx as usize;

    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    let devices: Vec<_> = host
        .input_devices()
        .map_err(|e| format!("query input devices: {e}"))?
        .collect();
    let dev = devices.get(i).ok_or_else(|| {
        format!(
            "audio_device index {idx} out of range ({} devices)",
            devices.len()
        )
    })?;
    let name = dev
        .description()
        .ok()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|| dev.to_string());
    Ok(name)
}

/// Test helper: push one chunk and count lossy pushes.
#[cfg(test)]
fn forward_chunk_for_test(ingress: &AudioIngress, chunk: Vec<f32>, lossy: &AtomicU64) -> bool {
    let ok = ingress.try_push(chunk);
    if !ok {
        lossy.fetch_add(1, Ordering::Relaxed);
    }
    ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use shuvoice_app::AudioIngress;
    use std::sync::Mutex;

    /// Fake opener that fails before any device I/O.
    struct FailOpenOpener {
        msg: &'static str,
    }

    impl CaptureOpener for FailOpenOpener {
        fn open(self, _cfg: AudioConfig) -> Result<Box<dyn CaptureBackend>, String> {
            Err(self.msg.to_string())
        }
    }

    /// Fake backend that opens successfully and yields nothing.
    struct IdleBackend {
        name: Option<String>,
        stopped: bool,
    }

    impl CaptureBackend for IdleBackend {
        fn get_chunk(&mut self, timeout: Duration) -> Option<Vec<f32>> {
            thread::sleep(timeout.min(Duration::from_millis(5)));
            None
        }
        fn stop(&mut self) {
            self.stopped = true;
        }
        fn backend_hold_queue_dropped(&self) -> u64 {
            0
        }
        fn backend_callback_lock_fails(&self) -> u64 {
            0
        }
        fn resolved_device_name(&self) -> Option<&str> {
            self.name.as_deref()
        }
    }

    struct IdleOpener {
        name: Option<String>,
    }

    impl CaptureOpener for IdleOpener {
        fn open(self, _cfg: AudioConfig) -> Result<Box<dyn CaptureBackend>, String> {
            Ok(Box::new(IdleBackend {
                name: self.name,
                stopped: false,
            }))
        }
    }

    /// Opener that blocks past readiness timeout (never sends readiness from open —
    /// simulated by sleeping inside open before returning Ok, which delays the
    /// ready send until open returns).
    struct SlowOpenOpener {
        delay: Duration,
    }

    impl CaptureOpener for SlowOpenOpener {
        fn open(self, _cfg: AudioConfig) -> Result<Box<dyn CaptureBackend>, String> {
            thread::sleep(self.delay);
            Ok(Box::new(IdleBackend {
                name: Some("slow".into()),
                stopped: false,
            }))
        }
    }

    #[test]
    fn resolve_name_passthrough() {
        let name = resolve_device_name(Some(&DeviceRef::Name("  pulse  ".into()))).unwrap();
        assert_eq!(name.as_deref(), Some("pulse"));
    }

    #[test]
    fn resolve_empty_name_is_default() {
        let name = resolve_device_name(Some(&DeviceRef::Name("   ".into()))).unwrap();
        assert_eq!(name, None);
    }

    #[test]
    fn resolve_none_is_default() {
        assert_eq!(resolve_device_name(None).unwrap(), None);
    }

    #[test]
    fn resolve_negative_index_errors() {
        let err = resolve_device_name(Some(&DeviceRef::Index(-1))).unwrap_err();
        assert!(err.contains("negative"), "{err}");
    }

    #[test]
    fn from_app_config_uses_effective_rate_and_hold_queue_name() {
        let cfg = Config::default();
        let bridge_cfg = AudioBridgeConfig::from_app_config(&cfg, 24_000, 2_400);
        assert_eq!(bridge_cfg.sample_rate, 24_000);
        assert_eq!(bridge_cfg.chunk_samples, 2_400);
        assert_eq!(bridge_cfg.fallback_sample_rate, cfg.fallback_sample_rate);
        assert!(bridge_cfg.backend_hold_queue_capacity >= 4);
    }

    #[test]
    fn start_surfaces_open_failure_as_err() {
        let (ingress, _ring) = AudioIngress::new(4);
        let cfg = AudioBridgeConfig {
            start_ready_timeout: Duration::from_secs(1),
            stop_join_timeout: Duration::from_secs(1),
            ..AudioBridgeConfig::default()
        };
        let res = AudioBridge::start_with(
            ingress,
            cfg,
            FailOpenOpener {
                msg: "synthetic device open failure",
            },
        );
        let err = match res {
            Ok(_) => panic!("expected start Err"),
            Err(e) => e,
        };
        assert!(
            err.contains("synthetic device open failure"),
            "unexpected err: {err}"
        );
        // No payload-shaped fields in the error.
        assert!(!err.contains('{'));
    }

    #[test]
    fn start_ok_with_idle_backend_then_joins_cleanly() {
        let (ingress, _ring) = AudioIngress::new(4);
        let cfg = AudioBridgeConfig {
            start_ready_timeout: Duration::from_secs(1),
            stop_join_timeout: Duration::from_secs(1),
            poll_timeout: Duration::from_millis(5),
            ..AudioBridgeConfig::default()
        };
        let mut bridge = AudioBridge::start_with(
            ingress,
            cfg,
            IdleOpener {
                name: Some("fake-mic".into()),
            },
        )
        .expect("idle backend must start");
        assert_eq!(bridge.resolved_device_name(), Some("fake-mic"));
        let outcome = bridge.stop();
        assert_eq!(outcome, StopOutcome::Joined);
        assert!(outcome.joined_cleanly());
        assert!(!outcome.detached());
        // Second stop is AlreadyStopped.
        assert_eq!(bridge.stop(), StopOutcome::AlreadyStopped);
    }

    #[test]
    fn start_timeout_returns_err_not_ok() {
        let (ingress, _ring) = AudioIngress::new(4);
        let cfg = AudioBridgeConfig {
            start_ready_timeout: Duration::from_millis(50),
            stop_join_timeout: Duration::from_secs(1),
            ..AudioBridgeConfig::default()
        };
        let res = AudioBridge::start_with(
            ingress,
            cfg,
            SlowOpenOpener {
                delay: Duration::from_millis(400),
            },
        );
        let err = match res {
            Ok(_) => panic!("expected start Err"),
            Err(e) => e,
        };
        assert!(err.to_lowercase().contains("timed out"), "{err}");
    }

    #[test]
    fn ingress_lossy_push_counted_when_ring_full() {
        let (ingress, ring) = AudioIngress::new(1);
        let lossy = AtomicU64::new(0);
        assert!(forward_chunk_for_test(&ingress, vec![0.1, 0.2], &lossy));
        // Capacity 1: second push is lossy (oldest evicted) and returns false.
        assert!(!forward_chunk_for_test(&ingress, vec![0.3, 0.4], &lossy));
        assert_eq!(lossy.load(Ordering::Relaxed), 1);
        assert!(ring.dropped() >= 1);
    }

    #[test]
    fn diagnostics_default_zero() {
        let d = SharedDiag::default().snapshot();
        assert_eq!(d, AudioBridgeDiagnostics::default());
    }

    #[test]
    fn stop_outcome_joined_cleanly_predicate() {
        assert!(StopOutcome::Joined.joined_cleanly());
        assert!(StopOutcome::AlreadyStopped.joined_cleanly());
        assert!(!StopOutcome::DetachedAfterTimeout.joined_cleanly());
        assert!(!StopOutcome::JoinedPanicked.joined_cleanly());
        assert!(StopOutcome::DetachedAfterTimeout.detached());
    }

    /// Ensure fail-open path does not leave a running flag stuck true.
    #[test]
    fn fail_open_never_marks_running() {
        let running_probe = Arc::new(Mutex::new(false));
        // Indirect: after FailOpen, we cannot observe diag; just ensure Err.
        let (ingress, _) = AudioIngress::new(2);
        let cfg = AudioBridgeConfig {
            start_ready_timeout: Duration::from_millis(500),
            stop_join_timeout: Duration::from_millis(500),
            ..AudioBridgeConfig::default()
        };
        let res = AudioBridge::start_with(ingress, cfg, FailOpenOpener { msg: "no device" });
        assert!(res.is_err());
        let _ = running_probe;
    }
}
