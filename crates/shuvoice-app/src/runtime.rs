//! Session runtime: responsive actor loop, control enqueue, event bus, joins.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use shuvoice_asr::DynAsrBackend;
use shuvoice_core::Config;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use crate::asr_owner::{AsrOwnerHandle, AsrOwnerJoin, spawn_asr_owner};
use crate::audio::AudioIngress;
use crate::error::{AppError, AppResult};
use crate::events::{EventBus, EventBusRx};
use crate::fakes::{
    FakeFeedback, FakeInjector, FakeOverlay, FakeSelection, FakeTts, ScriptedAsrBackend,
    ScriptedInner,
};
use crate::session::{Session, SessionDeps};
use crate::traits::{
    Clock, FakeClock, FeedbackSink, OverlaySink, SelectionCapture, SystemClock, TextInjector,
    TtsEngine,
};
use crate::types::{
    DEFAULT_ASR_OP_TIMEOUT, DEFAULT_COMMAND_CAPACITY, DEFAULT_EVENT_CAPACITY, RuntimeView,
    SessionCommand, SessionEvent,
};

/// Advertised max wait for the session actor join before abort (real bound).
pub const SESSION_SHUTDOWN_GRACE: Duration = Duration::from_secs(2);
const ASR_SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// Production control surface: enqueue-only + cached snapshot reads.
pub trait ControlHandlerSurface: Send + Sync {
    fn on_start(&self);
    fn on_stop(&self);
    fn on_toggle(&self);
    fn on_status(&self) -> String;
    fn on_metrics(&self) -> String;
    fn on_debug_status(&self) -> String;
    fn on_tts_command(&self, command: &str) -> String;
}

/// Enqueue-only control adapter. Returns honest queue-full errors for TTS cmds.
#[derive(Clone)]
pub struct EnqueueControlAdapter {
    cmd_tx: mpsc::Sender<SessionCommand>,
    view: RuntimeView,
    tts_enabled: bool,
}

impl EnqueueControlAdapter {
    pub fn new(cmd_tx: mpsc::Sender<SessionCommand>, view: RuntimeView, tts_enabled: bool) -> Self {
        Self {
            cmd_tx,
            view,
            tts_enabled,
        }
    }

    fn enqueue(&self, cmd: SessionCommand) -> Result<(), AppError> {
        self.cmd_tx.try_send(cmd).map_err(|err| match err {
            mpsc::error::TrySendError::Full(_) => AppError::CommandQueueFull,
            mpsc::error::TrySendError::Closed(_) => AppError::ShutDown,
        })
    }

    /// Public enqueue for player callbacks (ordered, non-blocking).
    pub fn try_enqueue(&self, cmd: SessionCommand) -> AppResult<()> {
        self.enqueue(cmd)
    }

    /// TTS player → actor re-entry. Enqueue-only; never calls Session directly.
    pub fn enqueue_tts_player_update(
        &self,
        state: crate::types::TtsPlayerState,
        error_message: Option<String>,
    ) -> AppResult<()> {
        self.enqueue(SessionCommand::TtsPlayerUpdate {
            state,
            error_message,
        })
    }
}

impl ControlHandlerSurface for EnqueueControlAdapter {
    fn on_start(&self) {
        if let Err(err) = self.enqueue(SessionCommand::Start) {
            error!(%err, "control start enqueue failed");
        }
    }
    fn on_stop(&self) {
        if let Err(err) = self.enqueue(SessionCommand::Stop) {
            error!(%err, "control stop enqueue failed");
        }
    }
    fn on_toggle(&self) {
        if let Err(err) = self.enqueue(SessionCommand::Toggle) {
            error!(%err, "control toggle enqueue failed");
        }
    }
    fn on_status(&self) -> String {
        self.view.status()
    }
    fn on_metrics(&self) -> String {
        self.view.metrics_json()
    }
    fn on_debug_status(&self) -> String {
        self.view.debug_json()
    }
    fn on_tts_command(&self, command: &str) -> String {
        if !self.tts_enabled {
            return "ERROR tts disabled".into();
        }
        let cmd = match command {
            "tts_speak" => SessionCommand::TtsSpeakSelection,
            "tts_speak_clipboard" => SessionCommand::TtsSpeakClipboard,
            "tts_pause" => SessionCommand::TtsPause,
            "tts_resume" => SessionCommand::TtsResume,
            "tts_toggle_pause" => SessionCommand::TtsTogglePause,
            "tts_restart" => SessionCommand::TtsRestart,
            "tts_stop" => SessionCommand::TtsStop,
            "tts_status" => {
                // Distinct from STT recording status — mirrored player state only.
                return format!("OK {}", self.view.tts_status());
            }
            other => return format!("ERROR unknown tts command: {other}"),
        };
        match self.enqueue(cmd) {
            Ok(()) => match command {
                "tts_speak" | "tts_speak_clipboard" => "OK tts speaking".into(),
                "tts_pause" => "OK tts paused".into(),
                "tts_resume" => "OK tts resumed".into(),
                "tts_toggle_pause" => "OK tts toggled".into(),
                "tts_restart" => "OK tts restarted".into(),
                "tts_stop" => "OK tts stopped".into(),
                _ => "OK".into(),
            },
            Err(AppError::CommandQueueFull) => "ERROR control queue full".into(),
            Err(AppError::ShutDown) => "ERROR session shut down".into(),
            Err(e) => format!("ERROR {e}"),
        }
    }
}

enum ActorMsg {
    Command {
        cmd: SessionCommand,
        reply: Option<tokio::sync::oneshot::Sender<AppResult<String>>>,
    },
}

#[derive(Clone)]
pub struct SessionHandle {
    reply_tx: mpsc::Sender<ActorMsg>,
    view: RuntimeView,
}

impl SessionHandle {
    pub async fn send(&self, cmd: SessionCommand) -> AppResult<String> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.reply_tx
            .send(ActorMsg::Command {
                cmd,
                reply: Some(reply_tx),
            })
            .await
            .map_err(|_| AppError::ShutDown)?;
        match tokio::time::timeout(Duration::from_secs(10), reply_rx).await {
            Ok(Ok(res)) => res,
            Ok(Err(_)) => Err(AppError::ShutDown),
            Err(_) => Err(AppError::message("session command timed out")),
        }
    }

    pub fn try_enqueue(&self, cmd: SessionCommand) -> AppResult<()> {
        self.reply_tx
            .try_send(ActorMsg::Command { cmd, reply: None })
            .map_err(|err| match err {
                mpsc::error::TrySendError::Full(_) => AppError::CommandQueueFull,
                mpsc::error::TrySendError::Closed(_) => AppError::ShutDown,
            })
    }

    /// Enqueue-only TTS player callback re-entry (never blocks on the actor).
    pub fn enqueue_tts_player_update(
        &self,
        state: crate::types::TtsPlayerState,
        error_message: Option<String>,
    ) -> AppResult<()> {
        self.try_enqueue(SessionCommand::TtsPlayerUpdate {
            state,
            error_message,
        })
    }

    pub fn status(&self) -> String {
        self.view.status()
    }
    pub fn tts_status(&self) -> String {
        self.view.tts_status()
    }
    pub fn metrics_json(&self) -> String {
        self.view.metrics_json()
    }
    pub fn debug_json(&self) -> String {
        self.view.debug_json()
    }
    pub fn view(&self) -> RuntimeView {
        self.view.clone()
    }

    pub async fn shutdown(&self) -> AppResult<()> {
        let _ = self.send(SessionCommand::Shutdown).await?;
        Ok(())
    }
}

pub struct SessionRuntime {
    pub handle: SessionHandle,
    pub audio: AudioIngress,
    pub control: EnqueueControlAdapter,
    pub asr: AsrOwnerHandle,
    /// Capture sample rate required by the loaded ASR backend (e.g. 24000 OpenAI).
    pub effective_sample_rate: u32,
    /// Capture period in samples at `effective_sample_rate`.
    pub audio_chunk_samples: usize,
    /// Reliable essential events for production observers.
    pub essential_rx: mpsc::Receiver<SessionEvent>,
    /// Best-effort partials.
    pub partial_rx: mpsc::Receiver<SessionEvent>,
    session_join: JoinHandle<()>,
    /// Single ordered event dispatcher task (never one-thread-per-event).
    dispatcher_join: JoinHandle<()>,
    asr_join: AsrOwnerJoin,
}

impl SessionRuntime {
    /// Ordered shutdown with a **real** upper bound on actor teardown:
    ///
    /// 1. Non-blocking `try_enqueue(Shutdown)` — never `handle.send` / 10s reply wait
    /// 2. Immediately `asr.request_shutdown()` (actor may be wedged mid-command)
    /// 3. Retained `session_join`: wait ≤ [`SESSION_SHUTDOWN_GRACE`] (2s), then abort+await
    /// 4. Abort dispatcher (short await)
    /// 5. ASR `join_timeout` (hard abort if owner ignores shutdown)
    ///
    /// Deliberately does **not** call [`SessionHandle::shutdown`], which can stall
    /// up to 10s on a wedged actor before join grace would even start.
    pub async fn shutdown(self) -> AppResult<()> {
        // Best-effort signal only — must not block on actor progress or reply.
        match self.handle.try_enqueue(SessionCommand::Shutdown) {
            Ok(()) => {}
            Err(AppError::CommandQueueFull) => {
                warn!("shutdown enqueue full; proceeding to join grace/abort");
            }
            Err(AppError::ShutDown) => {}
            Err(err) => {
                warn!(%err, "shutdown enqueue failed; proceeding to join grace/abort");
            }
        }

        // Don't rely on the actor reaching `session.shutdown()` if it is wedged.
        self.asr.request_shutdown();

        let mut session_join = self.session_join;
        tokio::select! {
            res = &mut session_join => {
                if let Err(e) = res
                    && !e.is_cancelled()
                {
                    warn!(%e, "session actor join error");
                }
            }
            _ = tokio::time::sleep(SESSION_SHUTDOWN_GRACE) => {
                warn!(
                    ?SESSION_SHUTDOWN_GRACE,
                    "session actor did not exit in grace; aborting join handle"
                );
                session_join.abort();
                let _ = session_join.await;
            }
        }

        // Retain mut join handle; never detach. Abort then await (with short timeout).
        let mut dispatcher_join = self.dispatcher_join;
        dispatcher_join.abort();
        match tokio::time::timeout(Duration::from_millis(500), &mut dispatcher_join).await {
            Ok(res) => {
                if let Err(e) = res
                    && !e.is_cancelled()
                {
                    warn!(%e, "event dispatcher join error");
                }
            }
            Err(_) => {
                let _ = dispatcher_join.await;
            }
        }

        self.asr_join.join_timeout(ASR_SHUTDOWN_GRACE).await;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn spawn_session_runtime<I, T, S, O, F, C>(
    config: Config,
    backend: Box<DynAsrBackend>,
    injector: Arc<I>,
    selection: Arc<S>,
    overlay: O,
    feedback: F,
    clock: Arc<C>,
    tts: Option<T>,
) -> AppResult<SessionRuntime>
where
    I: TextInjector + 'static,
    T: TtsEngine + 'static,
    S: SelectionCapture + 'static,
    O: OverlaySink + 'static,
    F: FeedbackSink + 'static,
    C: Clock + 'static,
{
    let (asr_handle, asr_join) = spawn_asr_owner(backend, 32, DEFAULT_ASR_OP_TIMEOUT);
    if let Err(e) = asr_handle.load().await {
        // Fail-fast: never construct EventBus/session/dispatcher on load failure.
        // Log only a stable class — raw errors may embed filesystem paths.
        let error_class = format!("{:?}", e.class());
        warn!(%error_class, "ASR model load failed");
        asr_handle.request_shutdown();
        asr_join.join_timeout(ASR_SHUTDOWN_GRACE).await;
        return Err(AppError::message("ASR model load failed"));
    }

    // Post-load caps drive capture rate (must not silently stay on config 16 kHz for OpenAI).
    let effective_sample_rate = crate::types::effective_audio_sample_rate(
        config.sample_rate,
        asr_handle.capabilities().preferred_sample_rate,
    );
    let audio_chunk_samples =
        crate::types::effective_audio_chunk_samples(effective_sample_rate, config.chunk_ms);
    info!(
        effective_sample_rate,
        audio_chunk_samples,
        config_sample_rate = config.sample_rate,
        preferred = ?asr_handle.capabilities().preferred_sample_rate,
        "session capture audio parameters"
    );

    let queue_cap = (config.audio_queue_max_size.max(1)) as usize;
    let (audio_ingress, audio_ring) = AudioIngress::new(queue_cap);
    let view = RuntimeView::new();
    let (
        event_bus,
        EventBusRx {
            essential_rx,
            partial_rx,
            dispatcher_join,
        },
    ) = EventBus::new(DEFAULT_EVENT_CAPACITY * 4, DEFAULT_EVENT_CAPACITY);

    let mut deps = SessionDeps::new(
        asr_handle.clone(),
        audio_ring,
        injector,
        selection,
        overlay,
        feedback,
        clock,
    )
    .with_view(view.clone())
    .with_events(event_bus);
    if let Some(tts) = tts {
        deps = deps.with_tts(tts);
    }

    let (cmd_tx, mut cmd_rx) = mpsc::channel::<SessionCommand>(DEFAULT_COMMAND_CAPACITY);
    let (reply_tx, mut reply_rx) = mpsc::channel::<ActorMsg>(DEFAULT_COMMAND_CAPACITY);

    let mut session = Session::new(config.clone(), deps);
    session.sync_audio_params_from_asr();
    // Load succeeded — honest flag for diagnostics.
    session.set_model_load_failed(false);

    let control = EnqueueControlAdapter::new(cmd_tx, view.clone(), config.tts_enabled);
    let handle = SessionHandle { reply_tx, view };

    let session_join = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_millis(10));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut reply_open = true;
        let mut cmd_open = true;
        loop {
            if !reply_open && !cmd_open {
                let _ = session.tick().await;
                break;
            }
            // Also poll job results frequently via tick.
            tokio::select! {
                biased;
                msg = reply_rx.recv(), if reply_open => {
                    match msg {
                        Some(ActorMsg::Command { cmd, reply }) => {
                            let shutdown = matches!(cmd, SessionCommand::Shutdown);
                            let result = session.handle_command(cmd).await;
                            if let Some(reply) = reply {
                                let _ = reply.send(result);
                            }
                            if shutdown || !session.is_running() {
                                break;
                            }
                        }
                        None => reply_open = false,
                    }
                }
                cmd = cmd_rx.recv(), if cmd_open => {
                    match cmd {
                        Some(cmd) => {
                            let shutdown = matches!(cmd, SessionCommand::Shutdown);
                            let _ = session.handle_command(cmd).await;
                            if shutdown || !session.is_running() {
                                break;
                            }
                        }
                        None => cmd_open = false,
                    }
                }
                _ = tick.tick() => {
                    if !session.tick().await {
                        break;
                    }
                }
            }
        }
        if session.is_running() {
            session.shutdown().await;
        }
        info!("session actor stopped");
    });

    Ok(SessionRuntime {
        handle,
        audio: audio_ingress,
        control,
        asr: asr_handle,
        effective_sample_rate,
        audio_chunk_samples,
        essential_rx,
        partial_rx,
        session_join,
        dispatcher_join,
        asr_join,
    })
}

// ── Test harness ────────────────────────────────────────────────────────

type TestSess = Session<FakeInjector, FakeTts, FakeSelection, FakeOverlay, FakeFeedback, FakeClock>;

pub struct TestHarness {
    pub session: TestSess,
    pub asr_join: AsrOwnerJoin,
    pub audio: AudioIngress,
    pub clock: FakeClock,
    pub scripted: Arc<Mutex<ScriptedInner>>,
    pub injector: Arc<FakeInjector>,
    pub events: EventBus,
    pub essential_rx: mpsc::Receiver<SessionEvent>,
    pub partial_rx: mpsc::Receiver<SessionEvent>,
    dispatcher_join: JoinHandle<()>,
}

impl TestHarness {
    pub async fn new_with(
        scripted: ScriptedAsrBackend,
        tts: FakeTts,
        selection: FakeSelection,
        config: Config,
    ) -> Self {
        let shared = scripted.shared();
        let clock = FakeClock::new();
        let (asr, asr_join) = spawn_asr_owner(Box::new(scripted), 32, DEFAULT_ASR_OP_TIMEOUT);
        match asr.load().await {
            Ok(()) => {}
            Err(e) => panic!("test ASR load failed: {e}"),
        }
        let queue_cap = (config.audio_queue_max_size.max(1)) as usize;
        let (audio_ingress, audio_ring) = AudioIngress::new(queue_cap);
        let injector = Arc::new(FakeInjector::default());
        let (
            events,
            EventBusRx {
                essential_rx,
                partial_rx,
                dispatcher_join,
            },
        ) = EventBus::new(256, 64);
        let deps = SessionDeps::new(
            asr,
            audio_ring,
            Arc::clone(&injector),
            Arc::new(selection),
            FakeOverlay::default(),
            FakeFeedback::default(),
            Arc::new(clock.shared_handle()),
        )
        .with_tts(tts)
        .with_events(events.clone());
        let mut session = Session::new(config, deps);
        session.sync_audio_params_from_asr();
        session.set_model_load_failed(false);
        Self {
            session,
            asr_join,
            audio: audio_ingress,
            clock,
            scripted: shared,
            injector,
            events,
            essential_rx,
            partial_rx,
            dispatcher_join,
        }
    }

    pub async fn basic(scripted: ScriptedAsrBackend, config: Config) -> Self {
        Self::new_with(scripted, FakeTts::new(), FakeSelection::default(), config).await
    }

    pub async fn shutdown(mut self) {
        self.session.shutdown().await;
        self.dispatcher_join.abort();
        let _ = self.dispatcher_join.await;
        self.asr_join.join_timeout(ASR_SHUTDOWN_GRACE).await;
    }
}

pub async fn spawn_test_runtime(
    config: Config,
    scripted: ScriptedAsrBackend,
) -> (SessionRuntime, Arc<Mutex<ScriptedInner>>, Arc<FakeInjector>) {
    let shared = scripted.shared();
    let injector = Arc::new(FakeInjector::default());
    let selection = Arc::new(FakeSelection::default());
    let rt = spawn_session_runtime(
        config,
        Box::new(scripted),
        Arc::clone(&injector),
        selection,
        FakeOverlay::default(),
        FakeFeedback::default(),
        Arc::new(SystemClock),
        Some(FakeTts::new()),
    )
    .await
    .expect("spawn runtime");
    (rt, shared, injector)
}

#[cfg(test)]
mod load_fail_cleanup_tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Duration;

    use async_trait::async_trait;
    use shuvoice_asr::{
        AsrBackend, AsrBackendKind, AsrError, AsrResult, DynAsrBackend, ProgressFn,
    };
    use shuvoice_core::{AsrCapabilities, Config};

    use super::{ASR_SHUTDOWN_GRACE, spawn_session_runtime};
    use crate::fakes::{FakeFeedback, FakeInjector, FakeOverlay, FakeSelection, FakeTts};
    use crate::traits::SystemClock;

    /// Backend whose `load` fails after the owner task has started; counts shutdowns.
    struct FailLoadBackend {
        shutdown_calls: Arc<AtomicU32>,
        dropped: Arc<AtomicBool>,
        caps: AsrCapabilities,
    }

    impl Drop for FailLoadBackend {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl AsrBackend for FailLoadBackend {
        fn capabilities(&self) -> &AsrCapabilities {
            &self.caps
        }
        fn backend_id(&self) -> AsrBackendKind {
            AsrBackendKind::Sherpa
        }
        fn native_chunk_samples(&self) -> usize {
            1600
        }

        async fn load(&mut self, _progress: &mut ProgressFn<'_>) -> AsrResult<()> {
            // Path-bearing detail must not escape into spawn_session_runtime logs/errors.
            Err(AsrError::dependency(
                "missing model file /home/secret/models/parakeet.onnx",
            ))
        }

        async fn reset(&mut self) -> AsrResult<()> {
            Ok(())
        }

        async fn process_chunk(&mut self, _pcm: &[f32]) -> AsrResult<String> {
            Ok(String::new())
        }

        async fn shutdown(&mut self) -> AsrResult<()> {
            self.shutdown_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn load_failure_shuts_down_owner_without_session_or_dispatcher() {
        let shutdown_calls = Arc::new(AtomicU32::new(0));
        let dropped = Arc::new(AtomicBool::new(false));
        let backend: Box<DynAsrBackend> = Box::new(FailLoadBackend {
            shutdown_calls: Arc::clone(&shutdown_calls),
            dropped: Arc::clone(&dropped),
            caps: AsrCapabilities::default(),
        });

        let mut config = Config::default();
        config.tts_enabled = false;

        let result = spawn_session_runtime(
            config,
            backend,
            Arc::new(FakeInjector::default()),
            Arc::new(FakeSelection::default()),
            FakeOverlay::default(),
            FakeFeedback::default(),
            Arc::new(SystemClock),
            None::<FakeTts>,
        )
        .await;
        let Err(err) = result else {
            panic!("load failure must fail spawn");
        };

        let msg = err.to_string();
        assert!(
            msg.contains("ASR model load failed"),
            "generic error expected, got {msg:?}"
        );
        assert!(
            !msg.contains("parakeet") && !msg.contains("/home/"),
            "error must not leak path-bearing detail: {msg:?}"
        );

        // Owner should have been request_shutdown + join_timeout'd.
        // Allow a brief moment for Drop after join.
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            shutdown_calls.load(Ordering::SeqCst) >= 1,
            "backend.shutdown must run via owner cleanup"
        );
        assert!(
            dropped.load(Ordering::SeqCst),
            "backend must be dropped after owner join (no orphan)"
        );

        // ASR_SHUTDOWN_GRACE is the bound used on the failure path (compile-time presence).
        let _ = ASR_SHUTDOWN_GRACE;
    }
}
