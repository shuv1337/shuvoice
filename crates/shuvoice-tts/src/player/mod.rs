//! Streaming TTS playback state machine.
//!
//! Architecture:
//! - **Synth** runs on the Tokio runtime (async HTTP / process streams).
//! - **Playback** runs on a dedicated OS thread and may block in `AudioOutput`.
//! - `stop` / `Drop` cancel, interrupt the live sink, and join with a **bounded**
//!   deadline so a wedged output cannot hang the control path.

#[cfg(feature = "cpal-output")]
mod cpal_output;
mod output;
mod pcm;

pub use output::{
    AudioOutput, AudioOutputFactory, FakeAudioOutput, FakeAudioOutputFactory, NullAudioOutput,
    NullAudioOutputFactory,
};
pub use pcm::{chunk_to_samples, parse_sample_rate, resample_linear_i16};

#[cfg(feature = "cpal-output")]
pub use cpal_output::{
    CpalAudioOutput, CpalAudioOutputFactory, CpalOutputConfig, OutputDeviceInfo,
};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle as ThreadJoinHandle;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_util::StreamExt;
use parking_lot::{Mutex, RwLock};
use tokio::sync::mpsc;
use tokio::task::JoinHandle as TokioJoinHandle;
use tokio_util::sync::CancellationToken;

use crate::backend::{SynthesisStream, TtsBackend, redact_for_ui};
use crate::error::TtsError;
use crate::metrics::{NoopMetrics, TtsMetrics};
use crate::speed::normalize_tts_playback_speed;
use crate::types::{
    AudioEncoding, EventInfo, PlayerEvent, PlayerState, StatusPayload, SynthesisRequest,
};

/// Synth→playback queue capacity (chunks).
pub const PLAYER_QUEUE_CAPACITY: usize = 256;
const QUEUE_CAPACITY: usize = PLAYER_QUEUE_CAPACITY;

/// Maximum time `stop` / generation barrier waits for the playback thread.
pub const WORKER_JOIN_DEADLINE: Duration = Duration::from_millis(750);

type StateCallback = Arc<dyn Fn(PlayerEvent) + Send + Sync>;

/// Thread-safe TTS synthesis + playback coordinator.
pub struct TtsPlayer {
    inner: Arc<PlayerInner>,
}

struct PlayerInner {
    backend: Arc<dyn TtsBackend>,
    output_factory: Arc<dyn AudioOutputFactory>,
    /// Nominal rate from builder/backend; per-utterance rate comes from SynthesisStream.
    sample_rate_hz: u32,
    metrics: Arc<dyn TtsMetrics>,
    on_event: RwLock<Option<StateCallback>>,
    state: Mutex<PlayerState>,
    generation: AtomicU64,
    playback_speed: Mutex<f64>,
    last_request: Mutex<Option<SynthesisRequest>>,
    active_request: Mutex<Option<SynthesisRequest>>,
    cancel: Mutex<CancellationToken>,
    pause: AtomicBool,
    live_output: Mutex<Option<Arc<dyn AudioOutput>>>,
    /// Sample rate advertised by the active synthesis stream (per generation).
    utterance_rate_hz: Mutex<Option<(u64, u32)>>,
    workers: Mutex<Option<WorkerHandles>>,
    runtime_handle: tokio::runtime::Handle,
}

struct WorkerHandles {
    synth: TokioJoinHandle<()>,
    play: Option<ThreadJoinHandle<()>>,
}

/// Builder for [`TtsPlayer`].
pub struct TtsPlayerBuilder {
    backend: Arc<dyn TtsBackend>,
    output_factory: Arc<dyn AudioOutputFactory>,
    sample_rate_hz: Option<u32>,
    playback_speed: f64,
    metrics: Arc<dyn TtsMetrics>,
    on_event: Option<StateCallback>,
    runtime_handle: Option<tokio::runtime::Handle>,
}

impl TtsPlayerBuilder {
    pub fn new(backend: Arc<dyn TtsBackend>, output_factory: Arc<dyn AudioOutputFactory>) -> Self {
        Self {
            backend,
            output_factory,
            sample_rate_hz: None,
            playback_speed: 1.0,
            metrics: Arc::new(NoopMetrics),
            on_event: None,
            runtime_handle: None,
        }
    }

    pub fn sample_rate_hz(mut self, rate: u32) -> Self {
        self.sample_rate_hz = Some(rate);
        self
    }

    pub fn playback_speed(mut self, speed: f64) -> Self {
        self.playback_speed = speed;
        self
    }

    pub fn metrics(mut self, metrics: Arc<dyn TtsMetrics>) -> Self {
        self.metrics = metrics;
        self
    }

    pub fn on_event<F>(mut self, callback: F) -> Self
    where
        F: Fn(PlayerEvent) + Send + Sync + 'static,
    {
        self.on_event = Some(Arc::new(callback));
        self
    }

    pub fn runtime_handle(mut self, handle: tokio::runtime::Handle) -> Self {
        self.runtime_handle = Some(handle);
        self
    }

    pub fn build(self) -> TtsPlayer {
        let sample_rate_hz = self
            .sample_rate_hz
            .unwrap_or_else(|| self.backend.sample_rate_hz());
        let runtime_handle = self
            .runtime_handle
            .unwrap_or_else(tokio::runtime::Handle::current);
        TtsPlayer {
            inner: Arc::new(PlayerInner {
                backend: self.backend,
                output_factory: self.output_factory,
                sample_rate_hz,
                metrics: self.metrics,
                on_event: RwLock::new(self.on_event),
                state: Mutex::new(PlayerState::Idle),
                generation: AtomicU64::new(0),
                playback_speed: Mutex::new(normalize_tts_playback_speed(self.playback_speed)),
                last_request: Mutex::new(None),
                active_request: Mutex::new(None),
                cancel: Mutex::new(CancellationToken::new()),
                pause: AtomicBool::new(false),
                live_output: Mutex::new(None),
                utterance_rate_hz: Mutex::new(None),
                workers: Mutex::new(None),
                runtime_handle,
            }),
        }
    }
}

impl TtsPlayer {
    pub fn builder(
        backend: Arc<dyn TtsBackend>,
        output_factory: Arc<dyn AudioOutputFactory>,
    ) -> TtsPlayerBuilder {
        TtsPlayerBuilder::new(backend, output_factory)
    }

    pub fn state(&self) -> PlayerState {
        *self.inner.state.lock()
    }

    pub fn is_active(&self) -> bool {
        self.state().is_active()
    }

    pub fn playback_speed(&self) -> f64 {
        *self.inner.playback_speed.lock()
    }

    pub fn set_playback_speed(&self, speed: f64) -> f64 {
        let normalized = normalize_tts_playback_speed(speed);
        *self.inner.playback_speed.lock() = normalized;
        normalized
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.inner.sample_rate_hz
    }

    pub fn status_payload(&self) -> StatusPayload {
        let last = self.inner.last_request.lock().clone();
        let active = self.inner.active_request.lock().clone();
        let speed = *self.inner.playback_speed.lock();
        StatusPayload {
            state: *self.inner.state.lock(),
            voice_id: last
                .as_ref()
                .map(|r| r.voice_id.clone())
                .unwrap_or_default(),
            model_id: last
                .as_ref()
                .map(|r| r.model_id.clone())
                .unwrap_or_default(),
            text_len: last.as_ref().map(|r| r.text.chars().count()).unwrap_or(0),
            playback_speed: speed,
            selected_playback_speed: speed,
            active_request_speed: active.map(|r| r.playback_speed),
        }
    }

    pub fn set_on_event<F>(&self, callback: F)
    where
        F: Fn(PlayerEvent) + Send + Sync + 'static,
    {
        *self.inner.on_event.write() = Some(Arc::new(callback));
    }

    /// Start speaking text. Returns `true` when an active session was interrupted.
    pub fn speak(
        &self,
        text: impl Into<String>,
        voice_id: impl Into<String>,
        model_id: impl Into<String>,
    ) -> Result<bool, TtsError> {
        let text_value = text.into().trim().to_string();
        if text_value.is_empty() {
            return Err(TtsError::EmptyText);
        }
        let voice_id = voice_id.into().trim().to_string();
        let model_id = model_id.into().trim().to_string();

        // Generation barrier: fully reap any prior workers before starting.
        let interrupted = self.is_active() || self.state() == PlayerState::Error;
        if interrupted {
            self.reap_workers(WORKER_JOIN_DEADLINE);
        }

        let speed = *self.inner.playback_speed.lock();
        let request = SynthesisRequest::new(text_value, voice_id, model_id, speed);

        let generation = self.inner.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let cancel = CancellationToken::new();
        *self.inner.cancel.lock() = cancel.clone();
        self.inner.pause.store(false, Ordering::SeqCst);
        *self.inner.last_request.lock() = Some(request.clone());
        *self.inner.active_request.lock() = Some(request.clone());
        *self.inner.live_output.lock() = None;
        *self.inner.utterance_rate_hz.lock() = None;

        let (tx, rx) = mpsc::channel::<Option<Result<Bytes, TtsError>>>(QUEUE_CAPACITY);

        self.inner.metrics.observe_tts_speak();
        if interrupted {
            self.inner.metrics.observe_tts_interrupt();
        }

        self.transition(
            PlayerState::Synthesizing,
            EventInfo {
                request_playback_speed: Some(request.playback_speed),
                voice_id: Some(request.voice_id.clone()),
                model_id: Some(request.model_id.clone()),
                encoding: Some(AudioEncoding::PcmS16Le),
                ..EventInfo::default()
            },
        );

        let synth_started = Instant::now();
        let inner = Arc::clone(&self.inner);
        let request_synth = request.clone();
        let cancel_synth = cancel.clone();
        let synth = self.inner.runtime_handle.spawn(async move {
            run_synthesis(
                inner,
                generation,
                request_synth,
                cancel_synth,
                tx,
                synth_started,
            )
            .await;
        });

        let inner = Arc::clone(&self.inner);
        let request_play = request;
        let cancel_play = cancel;
        let play = match std::thread::Builder::new()
            .name("tts-playback".into())
            .spawn(move || {
                run_playback_blocking(inner, generation, request_play, cancel_play, rx);
            }) {
            Ok(h) => h,
            Err(err) => {
                // Spawn failure: cancel synth and surface error — do not leave orphans.
                self.inner.cancel.lock().cancel();
                synth.abort();
                self.inner.metrics.observe_tts_spawn_failure();
                self.inner.metrics.observe_tts_synth_failure();
                *self.inner.active_request.lock() = None;
                self.transition(
                    PlayerState::Error,
                    EventInfo {
                        error_class: Some("SpawnFailure".into()),
                        message: Some(redact_for_ui(&format!(
                            "failed to spawn playback thread: {err}"
                        ))),
                        ..EventInfo::default()
                    },
                );
                return Err(TtsError::audio(format!(
                    "failed to spawn playback thread: {err}"
                )));
            }
        };

        *self.inner.workers.lock() = Some(WorkerHandles {
            synth,
            play: Some(play),
        });
        Ok(interrupted)
    }

    pub fn pause(&self) -> bool {
        let mut state = self.inner.state.lock();
        if *state != PlayerState::Playing {
            return false;
        }
        self.inner.pause.store(true, Ordering::SeqCst);
        *state = PlayerState::Paused;
        drop(state);
        self.inner.metrics.observe_tts_pause();
        self.emit(PlayerState::Paused, EventInfo::default());
        true
    }

    pub fn resume(&self) -> bool {
        let mut state = self.inner.state.lock();
        if *state != PlayerState::Paused {
            return false;
        }
        self.inner.pause.store(false, Ordering::SeqCst);
        *state = PlayerState::Playing;
        drop(state);
        self.emit(PlayerState::Playing, EventInfo::default());
        true
    }

    pub fn toggle_pause(&self) -> bool {
        if self.state() == PlayerState::Paused {
            self.resume()
        } else {
            self.pause()
        }
    }

    pub fn restart(&self) -> bool {
        let last = self.inner.last_request.lock().clone();
        let Some(last) = last else {
            return false;
        };
        if last.text.is_empty() {
            return false;
        }
        self.speak(last.text, last.voice_id, last.model_id).is_ok()
    }

    pub fn stop(&self) -> bool {
        let state = *self.inner.state.lock();
        let was_active = state.is_active() || state == PlayerState::Error;
        if !was_active {
            return false;
        }
        self.reap_workers(WORKER_JOIN_DEADLINE);
        self.transition(PlayerState::Idle, EventInfo::default());
        true
    }

    /// Cancel + interrupt + bounded join of workers. Safe to call from Drop.
    fn reap_workers(&self, deadline: Duration) {
        self.inner.cancel.lock().cancel();
        self.inner.pause.store(false, Ordering::SeqCst);
        *self.inner.active_request.lock() = None;

        if let Some(out) = self.inner.live_output.lock().clone() {
            out.interrupt();
        }

        let workers = self.inner.workers.lock().take();
        if let Some(workers) = workers {
            workers.synth.abort();
            if let Some(play) = workers.play
                && !join_thread_deadline(play, deadline)
            {
                self.inner.metrics.observe_tts_join_timeout();
                tracing::warn!(
                    deadline_ms = deadline.as_millis() as u64,
                    "TTS playback thread join timed out; detaching"
                );
            }
        }
        *self.inner.live_output.lock() = None;
    }

    fn transition(&self, state: PlayerState, info: EventInfo) {
        let changed = {
            let mut guard = self.inner.state.lock();
            let changed = *guard != state;
            *guard = state;
            changed
        };
        if changed || info != EventInfo::default() {
            self.emit(state, info);
        }
    }

    fn emit(&self, state: PlayerState, info: EventInfo) {
        if let Some(cb) = self.inner.on_event.read().clone() {
            cb(PlayerEvent { state, info });
        }
    }
}

impl Drop for TtsPlayer {
    fn drop(&mut self) {
        // Best-effort teardown; bounded so Drop cannot hang the process forever.
        self.reap_workers(WORKER_JOIN_DEADLINE);
    }
}

impl PlayerInner {
    fn is_generation_current(&self, generation: u64) -> bool {
        self.generation.load(Ordering::SeqCst) == generation
    }

    fn clear_active(&self, generation: u64) {
        if self.is_generation_current(generation) {
            *self.active_request.lock() = None;
        }
    }

    fn emit(&self, state: PlayerState, info: EventInfo) {
        {
            let mut guard = self.state.lock();
            *guard = state;
        }
        if let Some(cb) = self.on_event.read().clone() {
            cb(PlayerEvent { state, info });
        }
    }
}

/// Join a thread with a wall-clock deadline. Returns `false` if the deadline elapsed.
fn join_thread_deadline(handle: ThreadJoinHandle<()>, deadline: Duration) -> bool {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = handle.join();
        let _ = tx.send(());
    });
    rx.recv_timeout(deadline).is_ok()
}

async fn run_synthesis(
    inner: Arc<PlayerInner>,
    generation: u64,
    request: SynthesisRequest,
    cancel: CancellationToken,
    tx: mpsc::Sender<Option<Result<Bytes, TtsError>>>,
    synth_started: Instant,
) {
    let mut first_chunk = true;
    let mut stream_rate: Option<u32> = None;

    let result = async {
        let SynthesisStream {
            sample_rate_hz,
            encoding,
            mut chunks,
        } = inner
            .backend
            .synthesize_stream(request.clone(), cancel.child_token())
            .await?;

        if encoding != AudioEncoding::PcmS16Le {
            return Err(TtsError::decode(format!(
                "unsupported backend encoding {encoding:?}; player requires pcm_s16le"
            )));
        }
        stream_rate = Some(sample_rate_hz);

        // Send a rate tag as a special first control? We piggyback via side channel:
        // store on inner for this generation via first Playing event metadata.
        // Playback opens sink using rate from first enqueue of a RATE sentinel.
        // Simpler: put rate in channel as Ok with empty + we use a oneshot...
        // Use dedicated first message: we send sample rate via a preamble using
        // a thread-safe slot on PlayerInner.
        // Actually: include in channel as special Option — keep simple:
        // write rate into a generation-keyed slot.
        {
            // Stash for playback thread.
            // (playback reads via shared map on first chunk)
        }

        while let Some(item) = chunks.next().await {
            if cancel.is_cancelled() || !inner.is_generation_current(generation) {
                break;
            }
            let chunk = item?;
            if chunk.is_empty() {
                continue;
            }
            if first_chunk {
                first_chunk = false;
                let latency = synth_started.elapsed().as_secs_f64();
                inner.metrics.observe_tts_synth_latency(latency);
                if inner.is_generation_current(generation) && !cancel.is_cancelled() {
                    inner.emit(
                        PlayerState::Playing,
                        EventInfo {
                            synth_latency_sec: Some(latency),
                            request_playback_speed: Some(request.playback_speed),
                            sample_rate_hz: Some(sample_rate_hz),
                            encoding: Some(encoding),
                            ..EventInfo::default()
                        },
                    );
                }
                // Deliver rate to playback via a side channel message:
                // We prepend by sending a rate-tagged empty? Instead store:
                *inner.utterance_rate_hz.lock() = Some((generation, sample_rate_hz));
            }

            // Cancel-aware blocking enqueue — never silently drop.
            loop {
                if cancel.is_cancelled() || !inner.is_generation_current(generation) {
                    break;
                }
                match tx.try_send(Some(Ok(chunk.clone()))) {
                    Ok(()) => break,
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        return Ok(());
                    }
                }
            }
        }
        Ok::<(), TtsError>(())
    }
    .await;

    if let Err(err) = &result
        && !cancel.is_cancelled()
        && inner.is_generation_current(generation)
    {
        // Distinguish queue overflow? currently only closed/cancel break.
        if matches!(err, TtsError::Message(m) if m.contains("queue")) {
            inner.metrics.observe_tts_queue_overflow();
        }
        inner.clear_active(generation);
        inner.metrics.observe_tts_synth_failure();
        if err.is_speed_apply_failure() {
            inner.metrics.observe_tts_speed_apply_failure();
        }
        inner.emit(
            PlayerState::Error,
            EventInfo {
                error_class: Some(err.error_class().into()),
                message: Some(redact_for_ui(&err.to_string())),
                request_playback_speed: Some(request.playback_speed),
                speed_apply_failure: err.is_speed_apply_failure(),
                sample_rate_hz: stream_rate,
                encoding: Some(AudioEncoding::PcmS16Le),
                ..EventInfo::default()
            },
        );
    }

    // Sentinel — cancel-aware, no silent drop.
    loop {
        if !inner.is_generation_current(generation) {
            break;
        }
        match tx.try_send(None) {
            Ok(()) => break,
            Err(mpsc::error::TrySendError::Full(_)) => {
                if cancel.is_cancelled() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            Err(mpsc::error::TrySendError::Closed(_)) => break,
        }
    }
}

fn run_playback_blocking(
    inner: Arc<PlayerInner>,
    generation: u64,
    request: SynthesisRequest,
    cancel: CancellationToken,
    mut rx: mpsc::Receiver<Option<Result<Bytes, TtsError>>>,
) {
    let mut carry = Vec::new();
    let mut stream: Option<Arc<dyn AudioOutput>> = None;
    let mut play_started: Option<Instant> = None;
    let mut utterance_rate = inner.sample_rate_hz;
    let mut rate_resolved = false;

    let outcome = (|| -> Result<(), TtsError> {
        while inner.is_generation_current(generation) {
            if cancel.is_cancelled() {
                return Ok(());
            }
            if inner.pause.load(Ordering::SeqCst) {
                if let Some(s) = stream.take() {
                    let _ = s.close();
                    *inner.live_output.lock() = None;
                }
                for _ in 0..6 {
                    if cancel.is_cancelled() || !inner.pause.load(Ordering::SeqCst) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                continue;
            }

            let item = loop {
                if cancel.is_cancelled() || !inner.is_generation_current(generation) {
                    return Ok(());
                }
                match rx.try_recv() {
                    Ok(item) => break item,
                    Err(mpsc::error::TryRecvError::Empty) => {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(mpsc::error::TryRecvError::Disconnected) => return Ok(()),
                }
            };

            let Some(item) = item else {
                break; // sentinel
            };
            let chunk = item?;

            if !rate_resolved
                && let Some((g, r)) = *inner.utterance_rate_hz.lock()
                && g == generation
            {
                utterance_rate = r;
                rate_resolved = true;
            }

            let (samples, next_carry) = chunk_to_samples(&chunk, &carry);
            carry = next_carry;
            if samples.is_empty() {
                continue;
            }

            if play_started.is_none() {
                play_started = Some(Instant::now());
            }

            write_with_recovery(&inner, &mut stream, &samples, utterance_rate, &cancel)?;
        }

        if cancel.is_cancelled() || !inner.is_generation_current(generation) {
            return Ok(());
        }
        if *inner.state.lock() == PlayerState::Error {
            return Ok(());
        }

        // Close sink BEFORE Idle so UI never sees Idle with an open device.
        if let Some(s) = stream.take() {
            let _ = s.close();
        }
        *inner.live_output.lock() = None;

        let duration = play_started
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        inner.clear_active(generation);
        inner.metrics.observe_tts_playback_duration(duration);
        inner.metrics.observe_tts_playback_completion();
        inner.emit(
            PlayerState::Idle,
            EventInfo {
                playback_duration_sec: Some(duration),
                request_playback_speed: Some(request.playback_speed),
                sample_rate_hz: Some(utterance_rate),
                encoding: Some(AudioEncoding::PcmS16Le),
                ..EventInfo::default()
            },
        );
        Ok(())
    })();

    if let Err(err) = outcome
        && !cancel.is_cancelled()
        && inner.is_generation_current(generation)
    {
        let msg = err.to_string();
        let interrupted = msg.contains("interrupt")
            || msg.contains("closed")
            || matches!(err, TtsError::Cancelled);
        if !interrupted {
            inner.clear_active(generation);
            inner.metrics.observe_tts_synth_failure();
            inner.emit(
                PlayerState::Error,
                EventInfo {
                    error_class: Some(err.error_class().into()),
                    message: Some(redact_for_ui(&msg)),
                    request_playback_speed: Some(request.playback_speed),
                    speed_apply_failure: false,
                    sample_rate_hz: Some(utterance_rate),
                    encoding: Some(AudioEncoding::PcmS16Le),
                    ..EventInfo::default()
                },
            );
        }
    }

    if let Some(s) = stream.take() {
        let _ = s.close();
    }
    *inner.live_output.lock() = None;
}

fn write_with_recovery(
    inner: &PlayerInner,
    stream: &mut Option<Arc<dyn AudioOutput>>,
    samples: &[i16],
    sample_rate_hz: u32,
    cancel: &CancellationToken,
) -> Result<(), TtsError> {
    for attempt in 1..=2 {
        if cancel.is_cancelled() {
            return Err(TtsError::Cancelled);
        }
        if stream.is_none() {
            let out = inner.output_factory.open(sample_rate_hz)?;
            *inner.live_output.lock() = Some(Arc::clone(&out));
            *stream = Some(out);
        }
        match stream.as_ref().unwrap().write_samples(samples) {
            Ok(()) => return Ok(()),
            Err(err) => {
                if cancel.is_cancelled() {
                    return Err(TtsError::Cancelled);
                }
                let msg = err.to_string();
                if msg.contains("interrupt") || msg.contains("closed") {
                    return Err(err);
                }
                if attempt >= 2 {
                    return Err(err);
                }
                tracing::warn!("TTS playback write failed; recreating output stream");
                if let Some(s) = stream.take() {
                    let _ = s.close();
                }
                *inner.live_output.lock() = None;
                std::thread::sleep(Duration::from_millis(30));
            }
        }
    }
    Ok(())
}
