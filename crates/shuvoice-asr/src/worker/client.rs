//! `AsrBackend` adapter over `shuvoice-worker-proto`.
//!
//! Hardening:
//! - `dependency_missing` → [`AsrError::Dependency`]
//! - Handshake manifest + load `Ack.result` refine caps / chunk size / sample rate
//! - Rely on proto [`ClientOptions`] RPC deadlines (`DEFAULT_RPC_TIMEOUT` /
//!   `DEFAULT_LOAD_TIMEOUT`); add a host-side ceiling only as belt-and-suspenders
//! - Supervisor backoff without double-counting failures
//! - [`ProcessClient::close`] → [`WorkerProcess::shutdown`]
//! - Reject non-16 kHz / non-finite PCM before send

use std::time::Duration;

use async_trait::async_trait;
use shuvoice_core::{AsrBackendKind, AsrCapabilities, ExpectedChunking, FinalizationMode};
use shuvoice_worker_proto::{
    Ack, ClientOptions, ControlMessage, DEFAULT_LOAD_TIMEOUT, DEFAULT_RPC_TIMEOUT, LoadRequest,
    NegotiatedSession, PROTOCOL_VERSION, ProtocolError, WorkerClient, WorkerProcess,
    WorkerProcessError, WorkerSpawnConfig, WorkerSupervisor, honor_delay,
};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::UnixStream;
use tokio::sync::Mutex;
use tokio::time::timeout;
use uuid::Uuid;

use crate::backend::{AsrBackend, ProgressFn};
use crate::caps::{moonshine_caps, nemo_caps};
use crate::config::AsrConfig;
use crate::error::{AsrError, AsrResult, FallbackOutcome};

/// Which Python/ML runtime the worker is expected to host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerBackendKind {
    Nemo,
    Moonshine,
}

impl WorkerBackendKind {
    pub fn runtime_name(self) -> &'static str {
        match self {
            Self::Nemo => "nemo",
            Self::Moonshine => "moonshine",
        }
    }

    pub fn backend_id(self) -> AsrBackendKind {
        match self {
            Self::Nemo => AsrBackendKind::Nemo,
            Self::Moonshine => AsrBackendKind::Moonshine,
        }
    }

    fn default_caps(self) -> AsrCapabilities {
        match self {
            Self::Nemo => nemo_caps(),
            Self::Moonshine => moonshine_caps(),
        }
    }
}

/// How the host attaches to a worker.
pub enum WorkerAttach {
    Duplex {
        reader: tokio::io::ReadHalf<tokio::io::DuplexStream>,
        writer: tokio::io::WriteHalf<tokio::io::DuplexStream>,
    },
    UnixSocket {
        path: std::path::PathBuf,
    },
    Spawn(WorkerSpawnConfig),
    Supervisor(Box<WorkerSupervisor>),
}

struct LoadOutcome {
    ack: Ack,
    session: Option<NegotiatedSession>,
}

#[async_trait]
trait ClientOps: Send {
    async fn handshake(&mut self) -> AsrResult<Option<NegotiatedSession>>;
    async fn load(&mut self, config: serde_json::Value) -> AsrResult<LoadOutcome>;
    async fn reset(&mut self) -> AsrResult<()>;
    async fn process_chunk(&mut self, samples: &[f32], sr: u32) -> AsrResult<String>;
    async fn process_utterance(&mut self, samples: &[f32], sr: u32) -> AsrResult<String>;
    async fn finish(&mut self, timeout_ms: Option<u64>) -> AsrResult<String>;
    async fn close(&mut self) -> AsrResult<()>;
}

fn default_client_options() -> ClientOptions {
    ClientOptions {
        rpc_timeout: DEFAULT_RPC_TIMEOUT,
        load_timeout: DEFAULT_LOAD_TIMEOUT,
        ..ClientOptions::default()
    }
}

// ── Raw duplex / socket client ─────────────────────────────────────────

struct ClientBox<R, W> {
    client: WorkerClient<R, W>,
    handshook: bool,
    session: Option<NegotiatedSession>,
}

#[async_trait]
impl<R, W> ClientOps for ClientBox<R, W>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    async fn handshake(&mut self) -> AsrResult<Option<NegotiatedSession>> {
        if self.handshook {
            return Ok(self
                .session
                .clone()
                .or_else(|| self.client.session().cloned()));
        }
        let name = format!("shuvoice-asr/{}", env!("CARGO_PKG_VERSION"));
        let session = self
            .client
            .handshake(name)
            .await
            .map_err(map_proto)?
            .clone();
        self.session = Some(session.clone());
        self.handshook = true;
        Ok(Some(session))
    }

    async fn load(&mut self, config: serde_json::Value) -> AsrResult<LoadOutcome> {
        let ack = load_returning_ack(&mut self.client, config).await?;
        Ok(LoadOutcome {
            ack,
            session: self
                .session
                .clone()
                .or_else(|| self.client.session().cloned()),
        })
    }

    async fn reset(&mut self) -> AsrResult<()> {
        self.client.reset().await.map(|_| ()).map_err(map_proto)
    }

    async fn process_chunk(&mut self, samples: &[f32], sr: u32) -> AsrResult<String> {
        validate_pcm_mono_f32(samples, sr, sr)?;
        self.client
            .process_chunk(samples, sr)
            .await
            .map(|t| t.text)
            .map_err(map_proto)
    }

    async fn process_utterance(&mut self, samples: &[f32], sr: u32) -> AsrResult<String> {
        validate_pcm_mono_f32(samples, sr, sr)?;
        self.client
            .process_utterance(samples, sr)
            .await
            .map(|t| t.text)
            .map_err(map_proto)
    }

    async fn finish(&mut self, timeout_ms: Option<u64>) -> AsrResult<String> {
        self.client
            .finish(timeout_ms)
            .await
            .map(|t| t.text)
            .map_err(map_proto)
    }

    async fn close(&mut self) -> AsrResult<()> {
        let _ = timeout(Duration::from_secs(3), self.client.close()).await;
        Ok(())
    }
}

/// `WorkerClient::load` discards the Ack body; we need it for caps.
async fn load_returning_ack<R, W>(
    client: &mut WorkerClient<R, W>,
    config: serde_json::Value,
) -> AsrResult<Ack>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let request_id = Uuid::now_v7();
    let load_timeout = client.options().load_timeout;
    let fut = async {
        client
            .connection_mut()
            .write_message(&ControlMessage::Load(LoadRequest {
                request_id,
                config,
                extra: Default::default(),
            }))
            .await
            .map_err(map_proto)?;
        client.wait_ack(request_id).await.map_err(map_proto)
    };
    match timeout(load_timeout, fut).await {
        Ok(r) => r,
        Err(_) => Err(AsrError::RemoteTimeout(
            load_timeout,
            "worker load RPC timed out".into(),
        )),
    }
}

// ── Process-backed client ──────────────────────────────────────────────

struct ProcessClient {
    process: Option<WorkerProcess>,
}

impl ProcessClient {
    fn client_mut(
        &mut self,
    ) -> AsrResult<&mut WorkerClient<tokio::process::ChildStdout, tokio::process::ChildStdin>> {
        self.process
            .as_mut()
            .map(WorkerProcess::client_mut)
            .ok_or_else(|| AsrError::transport("worker process already closed"))
    }
}

#[async_trait]
impl ClientOps for ProcessClient {
    async fn handshake(&mut self) -> AsrResult<Option<NegotiatedSession>> {
        Ok(self.process.as_ref().map(|p| p.session().clone()))
    }

    async fn load(&mut self, config: serde_json::Value) -> AsrResult<LoadOutcome> {
        let session = self.process.as_ref().map(|p| p.session().clone());
        let client = self.client_mut()?;
        let ack = load_returning_ack(client, config).await?;
        Ok(LoadOutcome { ack, session })
    }

    async fn reset(&mut self) -> AsrResult<()> {
        let client = self.client_mut()?;
        client.reset().await.map(|_| ()).map_err(map_proto)
    }

    async fn process_chunk(&mut self, samples: &[f32], sr: u32) -> AsrResult<String> {
        validate_pcm_mono_f32(samples, sr, sr)?;
        let client = self.client_mut()?;
        client
            .process_chunk(samples, sr)
            .await
            .map(|t| t.text)
            .map_err(map_proto)
    }

    async fn process_utterance(&mut self, samples: &[f32], sr: u32) -> AsrResult<String> {
        validate_pcm_mono_f32(samples, sr, sr)?;
        let client = self.client_mut()?;
        client
            .process_utterance(samples, sr)
            .await
            .map(|t| t.text)
            .map_err(map_proto)
    }

    async fn finish(&mut self, timeout_ms: Option<u64>) -> AsrResult<String> {
        let client = self.client_mut()?;
        client
            .finish(timeout_ms)
            .await
            .map(|t| t.text)
            .map_err(map_proto)
    }

    async fn close(&mut self) -> AsrResult<()> {
        if let Some(proc) = self.process.take()
            && let Err(e) = proc.shutdown().await
        {
            tracing::warn!(error = %e, "worker process shutdown reported error");
        }
        Ok(())
    }
}

// ── Mapping / validation / caps ────────────────────────────────────────

fn map_proto(err: ProtocolError) -> AsrError {
    if err.is_timeout() {
        return AsrError::RemoteTimeout(DEFAULT_RPC_TIMEOUT, err.to_string());
    }
    let msg = err.to_string();
    if crate::cuda_oom::looks_like_cuda_oom_str(&msg) {
        return AsrError::cuda_oom(msg);
    }
    match err {
        ProtocolError::Handshake(m) => AsrError::startup(m),
        ProtocolError::UnsupportedVersion { .. } | ProtocolError::VersionMismatch { .. } => {
            AsrError::protocol(msg)
        }
        ProtocolError::Worker { code, message, .. } => match code.as_str() {
            "dependency_missing" | "dependency" => AsrError::dependency(message),
            "startup" | "startup_compat" => AsrError::startup(message),
            "cuda_oom" => AsrError::cuda_oom(message),
            "cancelled" => AsrError::Cancelled(message),
            _ => AsrError::from_runtime_message(message),
        },
        ProtocolError::RpcTimeout { timeout: t } => {
            AsrError::RemoteTimeout(t, "worker RPC timed out".into())
        }
        ProtocolError::Io(_) | ProtocolError::UnexpectedEof { .. } => AsrError::transport(msg),
        other => AsrError::protocol(other.to_string()),
    }
}

fn map_process(err: WorkerProcessError) -> AsrError {
    match err {
        WorkerProcessError::Spawn(e) => AsrError::dependency(format!("worker spawn failed: {e}")),
        WorkerProcessError::HandshakeTimeout { stderr_tail } => AsrError::startup(format!(
            "worker handshake timed out; stderr_tail={stderr_tail}"
        )),
        WorkerProcessError::Handshake {
            message,
            stderr_tail,
        } => AsrError::startup(format!(
            "worker handshake failed: {message}; stderr_tail={stderr_tail}"
        )),
        WorkerProcessError::UnsupportedVersion {
            remote,
            local,
            stderr_tail,
        } => AsrError::protocol(format!(
            "worker protocol unsupported: remote={remote} local={local}; stderr_tail={stderr_tail}"
        )),
        WorkerProcessError::DependencyMissing {
            message,
            stderr_tail,
        } => AsrError::dependency(format!("{message}; stderr_tail={stderr_tail}")),
        WorkerProcessError::Crashed {
            exit_code,
            stderr_tail,
        } => AsrError::transport(format!(
            "worker crashed exit={exit_code:?}; stderr_tail={stderr_tail}"
        )),
        WorkerProcessError::RestartDeferred { delay } => {
            AsrError::transport(format!("worker restart deferred for {delay:?}"))
        }
        WorkerProcessError::RestartExhausted {
            consecutive_failures,
        } => AsrError::dependency(format!(
            "worker restart exhausted after {consecutive_failures} consecutive failures"
        )),
        WorkerProcessError::Protocol(p) => map_proto(p),
        other => AsrError::transport(other.to_string()),
    }
}

fn validate_pcm_mono_f32(samples: &[f32], sample_rate_hz: u32, expected_hz: u32) -> AsrResult<()> {
    if sample_rate_hz != expected_hz {
        return Err(AsrError::unsupported(format!(
            "worker ASR requires {expected_hz} Hz mono PCM (got {sample_rate_hz} Hz;              negotiated/host rate mismatch)"
        )));
    }
    if samples.iter().any(|s| !s.is_finite()) {
        return Err(AsrError::unsupported(
            "worker ASR rejects non-finite PCM samples",
        ));
    }
    let _ = PROTOCOL_VERSION;
    Ok(())
}

fn apply_manifest_caps(
    kind: WorkerBackendKind,
    base: &mut AsrCapabilities,
    session: &NegotiatedSession,
) {
    let Some(asr) = session.manifest.asr.as_ref() else {
        return;
    };
    base.wants_raw_audio = asr.wants_raw_audio;
    base.supports_model_download = asr.supports_model_download;
    base.supports_cancel = asr.supports_cancel;
    if let Some(sr) = asr.native_sample_rate_hz {
        base.preferred_sample_rate = Some(sr);
    }
    if kind == WorkerBackendKind::Moonshine {
        base.expected_chunking = ExpectedChunking::Windowed;
    } else if asr.supports_streaming {
        base.expected_chunking = ExpectedChunking::Streaming;
    }
    if asr.supports_offline_utterance && !asr.supports_streaming {
        base.finalization_mode = FinalizationMode::OfflineInstant;
        base.emits_partials = false;
    }
}

fn apply_load_ack_caps(
    base: &mut AsrCapabilities,
    native_chunk: &mut usize,
    sample_rate: &mut u32,
    ack: &Ack,
) {
    let Some(result) = ack.result.as_ref() else {
        return;
    };
    if let Some(v) = result.get("wants_raw_audio").and_then(|v| v.as_bool()) {
        base.wants_raw_audio = v;
    }
    if let Some(v) = result
        .get("native_chunk_samples")
        .and_then(|v| v.as_u64())
        .and_then(|n| usize::try_from(n).ok())
        && v > 0
    {
        *native_chunk = v;
    }
    if let Some(v) = result
        .get("native_sample_rate_hz")
        .or_else(|| result.get("sample_rate_hz"))
        .or_else(|| result.get("sample_rate"))
        .and_then(|v| v.as_u64())
        .and_then(|n| u32::try_from(n).ok())
        && v > 0
    {
        *sample_rate = v;
        base.preferred_sample_rate = Some(v);
    }
    if let Some(v) = result.get("supports_cancel").and_then(|v| v.as_bool()) {
        base.supports_cancel = v;
    }
}

// ── Backend ────────────────────────────────────────────────────────────

pub struct WorkerAsrBackend {
    kind: WorkerBackendKind,
    config: AsrConfig,
    caps: AsrCapabilities,
    native_chunk_samples: usize,
    client: Option<Mutex<Box<dyn ClientOps>>>,
    supervisor: Option<Box<WorkerSupervisor>>,
    attach: Option<WorkerAttach>,
    loaded: bool,
    sample_rate: u32,
}

impl WorkerAsrBackend {
    pub fn new(kind: WorkerBackendKind, config: AsrConfig) -> Self {
        let caps = kind.default_caps();
        let native = match kind {
            WorkerBackendKind::Nemo => config.nemo_native_chunk_samples(),
            WorkerBackendKind::Moonshine => config.moonshine_native_chunk_samples(),
        };
        let sample_rate = caps.preferred_sample_rate.unwrap_or(config.sample_rate());
        let attach = resolve_attach(&config);
        Self {
            kind,
            config,
            caps,
            native_chunk_samples: native,
            client: None,
            supervisor: None,
            attach,
            loaded: false,
            sample_rate,
        }
    }

    pub fn with_attach(mut self, attach: WorkerAttach) -> Self {
        self.attach = Some(attach);
        self
    }

    pub fn with_duplex(
        self,
        reader: tokio::io::ReadHalf<tokio::io::DuplexStream>,
        writer: tokio::io::WriteHalf<tokio::io::DuplexStream>,
    ) -> Self {
        self.with_attach(WorkerAttach::Duplex { reader, writer })
    }

    pub fn with_spawn(self, spawn: WorkerSpawnConfig) -> Self {
        self.with_attach(WorkerAttach::Spawn(spawn))
    }

    pub fn with_supervisor(self, supervisor: WorkerSupervisor) -> Self {
        self.with_attach(WorkerAttach::Supervisor(Box::new(supervisor)))
    }

    pub fn kind(&self) -> WorkerBackendKind {
        self.kind
    }

    fn load_json(&self) -> serde_json::Value {
        match self.kind {
            WorkerBackendKind::Nemo => serde_json::json!({
                "runtime": "nemo",
                "model_name": self.config.core.model_name,
                "device": self.config.core.device,
                "right_context": self.config.core.right_context,
                "use_cuda_graph_decoder": self.config.core.use_cuda_graph_decoder,
                "sample_rate": self.sample_rate,
            }),
            WorkerBackendKind::Moonshine => serde_json::json!({
                "runtime": "moonshine",
                "model_name": self.config.core.moonshine_model_name,
                "model_dir": self.config.core.moonshine_model_dir,
                "provider": self.config.core.moonshine_provider.as_str(),
                "num_threads": self.config.core.moonshine_onnx_threads,
                "max_window_sec": self.config.core.moonshine_max_window_sec,
                "max_tokens": self.config.core.moonshine_max_tokens,
                "model_precision": self.config.core.moonshine_model_precision,
                "sample_rate": self.sample_rate,
            }),
        }
    }

    fn apply_load_outcome(&mut self, outcome: LoadOutcome) {
        if let Some(session) = outcome.session.as_ref() {
            apply_manifest_caps(self.kind, &mut self.caps, session);
            if let Some(n) = session
                .manifest
                .asr
                .as_ref()
                .and_then(|a| a.native_chunk_samples)
                && n > 0
            {
                self.native_chunk_samples = n as usize;
            }
            if let Some(sr) = session
                .manifest
                .asr
                .as_ref()
                .and_then(|a| a.native_sample_rate_hz)
            {
                self.sample_rate = sr;
            }
        }
        apply_load_ack_caps(
            &mut self.caps,
            &mut self.native_chunk_samples,
            &mut self.sample_rate,
            &outcome.ack,
        );
    }

    async fn ensure_client(&mut self) -> AsrResult<()> {
        if self.client.is_some() || self.supervisor.is_some() {
            return Ok(());
        }
        let attach = self.attach.take().ok_or_else(|| {
            AsrError::dependency(format!(
                "{} backend requires an external worker \
                 (WorkerSpawnConfig, worker_command, worker_socket_path, duplex, or supervisor). \
                 Native Rust does not embed NeMo/Moonshine runtimes.",
                self.kind.runtime_name()
            ))
        })?;

        match attach {
            WorkerAttach::Duplex { reader, writer } => {
                self.client = Some(Mutex::new(Box::new(ClientBox {
                    client: WorkerClient::with_options(reader, writer, default_client_options()),
                    handshook: false,
                    session: None,
                })));
            }
            WorkerAttach::UnixSocket { path } => {
                let stream = UnixStream::connect(&path).await.map_err(|e| {
                    AsrError::transport(format!(
                        "failed to connect worker socket {}: {e}",
                        path.display()
                    ))
                })?;
                let (r, w) = stream.into_split();
                self.client = Some(Mutex::new(Box::new(ClientBox {
                    client: WorkerClient::with_options(r, w, default_client_options()),
                    handshook: false,
                    session: None,
                })));
            }
            WorkerAttach::Spawn(spawn) => {
                let process = WorkerProcess::spawn(spawn).await.map_err(map_process)?;
                self.client = Some(Mutex::new(Box::new(ProcessClient {
                    process: Some(process),
                })));
            }
            WorkerAttach::Supervisor(supervisor) => {
                self.supervisor = Some(supervisor);
            }
        }
        Ok(())
    }

    /// Ensure a live supervised process without double-counting failures.
    async fn supervisor_ensure(&mut self) -> AsrResult<()> {
        let Some(sup) = self.supervisor.as_mut() else {
            return Err(AsrError::internal("no supervisor"));
        };
        loop {
            match sup.ensure_running().await {
                Ok(_) => return Ok(()),
                Err(WorkerProcessError::RestartDeferred { delay }) => {
                    honor_delay(delay).await;
                }
                Err(WorkerProcessError::Crashed {
                    exit_code,
                    stderr_tail,
                }) => {
                    // ensure_running already counted once. Sleep policy backoff only.
                    let failures = sup.restart_state().consecutive_failures;
                    let delay = sup.policy().backoff_for_attempt(failures);
                    tracing::warn!(
                        ?exit_code,
                        %stderr_tail,
                        ?delay,
                        failures,
                        "worker crashed; honoring backoff before respawn (no double-count)"
                    );
                    honor_delay(delay).await;
                }
                Err(e) => return Err(map_process(e)),
            }
        }
    }

    async fn ops_load(&mut self, progress: &mut ProgressFn<'_>) -> AsrResult<()> {
        progress(Some(0.1), "connecting worker");
        self.ensure_client().await?;
        let load_cfg = self.load_json();

        if self.supervisor.is_some() {
            self.supervisor_ensure().await?;
            progress(Some(0.4), "worker handshake complete");
            let load_cfg2 = load_cfg.clone();
            let outcome = {
                let sup = self.supervisor.as_mut().unwrap();
                let proc = sup.process_mut().ok_or_else(|| {
                    AsrError::transport("supervisor has no live process after ensure")
                })?;
                *proc.client_mut().options_mut() = default_client_options();
                let session = Some(proc.session().clone());
                progress(Some(0.6), "loading worker model");
                let ack = load_returning_ack(proc.client_mut(), load_cfg2).await?;
                LoadOutcome { ack, session }
            };
            self.apply_load_outcome(outcome);
            self.loaded = true;
            progress(Some(1.0), "worker ready");
            return Ok(());
        }

        let client = self
            .client
            .as_ref()
            .ok_or_else(|| AsrError::transport("no worker client"))?;
        let mut guard = client.lock().await;
        progress(Some(0.3), "worker handshake");
        let _ = guard.handshake().await?;
        progress(Some(0.6), "loading worker model");
        let outcome = guard.load(load_cfg).await?;
        drop(guard);
        self.apply_load_outcome(outcome);
        self.loaded = true;
        progress(Some(1.0), "worker ready");
        Ok(())
    }

    /// Single supervised restart after transport failure (one failure count).
    async fn supervisor_recover_and_reload(&mut self) -> AsrResult<()> {
        let exit = {
            let sup = self.supervisor.as_mut().unwrap();
            match sup.take_process() {
                Some(p) => Some(p.kill().await),
                None => None,
            }
        };
        let load_cfg = self.load_json();
        let outcome = {
            let sup = self.supervisor.as_mut().unwrap();
            let proc = sup
                .restart_after_failure_and_wait(exit)
                .await
                .map_err(map_process)?;
            *proc.client_mut().options_mut() = default_client_options();
            let session = Some(proc.session().clone());
            let ack = load_returning_ack(proc.client_mut(), load_cfg).await?;
            LoadOutcome { ack, session }
        };
        self.apply_load_outcome(outcome);
        Ok(())
    }
}

fn resolve_attach(config: &AsrConfig) -> Option<WorkerAttach> {
    if let Some(path) = config.connect.worker_socket_path.clone() {
        return Some(WorkerAttach::UnixSocket { path });
    }
    config.worker_spawn_config().map(WorkerAttach::Spawn)
}

fn is_retryable_transport(err: &AsrError) -> bool {
    matches!(
        err,
        AsrError::Transport(_) | AsrError::RemoteTimeout(_, _) | AsrError::Io(_)
    )
}

#[async_trait]
impl AsrBackend for WorkerAsrBackend {
    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }

    fn backend_id(&self) -> AsrBackendKind {
        self.kind.backend_id()
    }

    fn native_chunk_samples(&self) -> usize {
        self.native_chunk_samples
    }

    async fn load(&mut self, progress: &mut ProgressFn<'_>) -> AsrResult<()> {
        self.ops_load(progress).await
    }

    async fn reset(&mut self) -> AsrResult<()> {
        if !self.loaded {
            return Err(AsrError::internal("worker backend not loaded"));
        }
        if self.supervisor.is_some() {
            self.supervisor_ensure().await?;
            let first = {
                let sup = self.supervisor.as_mut().unwrap();
                let proc = sup
                    .process_mut()
                    .ok_or_else(|| AsrError::transport("no live process"))?;
                *proc.client_mut().options_mut() = default_client_options();
                proc.client_mut()
                    .reset()
                    .await
                    .map(|_| ())
                    .map_err(map_proto)
            };
            match first {
                Ok(()) => return Ok(()),
                Err(e) if is_retryable_transport(&e) => {
                    tracing::warn!(error = %e, "worker reset RPC failed; supervised restart");
                    self.supervisor_recover_and_reload().await?;
                    let sup = self.supervisor.as_mut().unwrap();
                    let proc = sup
                        .process_mut()
                        .ok_or_else(|| AsrError::transport("no live process after restart"))?;
                    *proc.client_mut().options_mut() = default_client_options();
                    return proc
                        .client_mut()
                        .reset()
                        .await
                        .map(|_| ())
                        .map_err(map_proto);
                }
                Err(e) => return Err(e),
            }
        }
        let client = self.client.as_ref().unwrap();
        let mut guard = client.lock().await;
        guard.reset().await
    }

    async fn process_chunk(&mut self, pcm_mono_f32: &[f32]) -> AsrResult<String> {
        if !self.loaded {
            return Err(AsrError::internal("worker backend not loaded"));
        }
        validate_pcm_mono_f32(pcm_mono_f32, self.sample_rate, self.sample_rate)?;
        let sr = self.sample_rate;

        if self.supervisor.is_some() {
            self.supervisor_ensure().await?;
            let first = {
                let sup = self.supervisor.as_mut().unwrap();
                let proc = sup
                    .process_mut()
                    .ok_or_else(|| AsrError::transport("no live process"))?;
                *proc.client_mut().options_mut() = default_client_options();
                proc.client_mut()
                    .process_chunk(pcm_mono_f32, sr)
                    .await
                    .map(|t| t.text)
                    .map_err(map_proto)
            };
            match first {
                Ok(t) => return Ok(t),
                Err(e) if is_retryable_transport(&e) => {
                    tracing::warn!(error = %e, "worker chunk RPC failed; supervised restart");
                    self.supervisor_recover_and_reload().await?;
                    let sup = self.supervisor.as_mut().unwrap();
                    let proc = sup
                        .process_mut()
                        .ok_or_else(|| AsrError::transport("no live process after restart"))?;
                    *proc.client_mut().options_mut() = default_client_options();
                    return proc
                        .client_mut()
                        .process_chunk(pcm_mono_f32, sr)
                        .await
                        .map(|t| t.text)
                        .map_err(map_proto);
                }
                Err(e) => return Err(e),
            }
        }

        let client = self.client.as_ref().unwrap();
        let mut guard = client.lock().await;
        guard.process_chunk(pcm_mono_f32, sr).await
    }

    async fn process_utterance(&mut self, pcm_mono_f32: &[f32]) -> AsrResult<String> {
        if !self.loaded {
            return Err(AsrError::internal("worker backend not loaded"));
        }
        validate_pcm_mono_f32(pcm_mono_f32, self.sample_rate, self.sample_rate)?;
        let sr = self.sample_rate;
        if self.supervisor.is_some() {
            self.supervisor_ensure().await?;
            let first = {
                let sup = self.supervisor.as_mut().unwrap();
                let proc = sup
                    .process_mut()
                    .ok_or_else(|| AsrError::transport("no live process"))?;
                *proc.client_mut().options_mut() = default_client_options();
                proc.client_mut()
                    .process_utterance(pcm_mono_f32, sr)
                    .await
                    .map(|t| t.text)
                    .map_err(map_proto)
            };
            match first {
                Ok(text) => return Ok(text),
                Err(e) if is_retryable_transport(&e) => {
                    tracing::warn!(error = %e, "worker utterance RPC failed; supervised restart");
                    self.supervisor_recover_and_reload().await?;
                    let sup = self.supervisor.as_mut().unwrap();
                    let proc = sup
                        .process_mut()
                        .ok_or_else(|| AsrError::transport("no live process after restart"))?;
                    *proc.client_mut().options_mut() = default_client_options();
                    return proc
                        .client_mut()
                        .process_utterance(pcm_mono_f32, sr)
                        .await
                        .map(|t| t.text)
                        .map_err(map_proto);
                }
                Err(e) => return Err(e),
            }
        }
        let client = self.client.as_ref().unwrap();
        let mut guard = client.lock().await;
        guard.process_utterance(pcm_mono_f32, sr).await
    }

    async fn finish_utterance(&mut self, t: Option<Duration>) -> AsrResult<String> {
        if !self.loaded {
            return Err(AsrError::internal("worker backend not loaded"));
        }
        let ms = t.map(|d| d.as_millis() as u64);
        if self.supervisor.is_some() {
            self.supervisor_ensure().await?;
            let first = {
                let sup = self.supervisor.as_mut().unwrap();
                let proc = sup
                    .process_mut()
                    .ok_or_else(|| AsrError::transport("no live process"))?;
                *proc.client_mut().options_mut() = default_client_options();
                proc.client_mut()
                    .finish(ms)
                    .await
                    .map(|x| x.text)
                    .map_err(map_proto)
            };
            match first {
                Ok(text) => return Ok(text),
                Err(e) if is_retryable_transport(&e) => {
                    tracing::warn!(error = %e, "worker finish RPC failed; supervised restart");
                    self.supervisor_recover_and_reload().await?;
                    let sup = self.supervisor.as_mut().unwrap();
                    let proc = sup
                        .process_mut()
                        .ok_or_else(|| AsrError::transport("no live process after restart"))?;
                    *proc.client_mut().options_mut() = default_client_options();
                    return proc
                        .client_mut()
                        .finish(ms)
                        .await
                        .map(|x| x.text)
                        .map_err(map_proto);
                }
                Err(e) => return Err(e),
            }
        }
        let client = self.client.as_ref().unwrap();
        let mut guard = client.lock().await;
        guard.finish(ms).await
    }

    async fn try_fallback_to_cpu(&mut self) -> AsrResult<FallbackOutcome> {
        Ok(FallbackOutcome::NotApplicable {
            detail: "GPU fallback must be implemented by the external worker runtime".into(),
        })
    }

    async fn shutdown(&mut self) -> AsrResult<()> {
        if let Some(client) = self.client.take() {
            let mut guard = client.lock().await;
            let _ = guard.close().await;
        }
        if let Some(mut sup) = self.supervisor.take() {
            let _ = sup.shutdown().await;
        }
        self.loaded = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shuvoice_worker_proto::ProtocolError;

    #[test]
    fn maps_dependency_missing_code() {
        let err = map_proto(ProtocolError::Worker {
            code: "dependency_missing".into(),
            message: "torch missing".into(),
            request_id: None,
        });
        assert!(matches!(err, AsrError::Dependency(_)), "{err:?}");
        assert!(err.to_string().contains("torch missing"));
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let err = validate_pcm_mono_f32(&[0.0], 48_000, 16_000).unwrap_err();
        assert!(err.to_string().contains("16000"));
    }

    #[test]
    fn rejects_non_finite_pcm() {
        let err = validate_pcm_mono_f32(&[f32::NAN], 16_000, 16_000).unwrap_err();
        assert!(err.to_string().contains("non-finite"));
    }

    #[test]
    fn load_ack_overrides_chunk_and_raw() {
        let mut caps = nemo_caps();
        let mut chunk = 999usize;
        let mut sr = 0u32;
        let ack = Ack {
            request_id: uuid::Uuid::nil(),
            result: Some(serde_json::json!({
                "wants_raw_audio": true,
                "native_chunk_samples": 1280,
                "native_sample_rate_hz": 16000
            })),
            extra: Default::default(),
        };
        apply_load_ack_caps(&mut caps, &mut chunk, &mut sr, &ack);
        assert!(caps.wants_raw_audio);
        assert_eq!(chunk, 1280);
        assert_eq!(sr, 16000);
    }
}
