//! MeloTTS backend — subprocess worker client.
//!
//! Production path (feature `worker-proto`): versioned framed protocol via
//! [`shuvoice_worker_proto::WorkerClient`], spawning the bundled
//! `workers/melotts` package as `python -m melotts --device <…>` with an
//! isolated child environment, explicit `PYTHONPATH` / `current_dir`, bounded
//! RPC timeouts, and cancel-aware kill/reap.
//!
//! Legacy JSON-line helper support exists only behind
//! [`MeloTtsBackend::new_for_test`] for unit/integration tests. Production
//! construction via [`MeloTtsBackend::new`] / [`crate::create_tts_backend`]
//! always forces WorkerProto and never consults `melo_helper.py`.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
#[cfg(feature = "worker-proto")]
use shuvoice_worker_proto::{ClientOptions, PcmEncoding, WorkerClient, redact_stderr_tail};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio_util::sync::CancellationToken;

use super::{SynthesisStream, TtsBackend, ensure_text, redact_for_ui};
use crate::error::TtsError;
use crate::types::{
    AudioEncoding, BackendId, Capabilities, DEFAULT_MELOTTS_VOICE_ID, SynthesisRequest, VoiceInfo,
};

const MELOTTS_SAMPLE_RATE_HZ: u32 = 44_100;
const DEFAULT_MELOTTS_VENV_DIR: &str = "~/.local/share/shuvoice/melotts-venv";
const FRAME_HEADER_SIZE: usize = 4;
/// Hard cap: 120s mono s16le @ 44.1 kHz.
const MAX_MELO_FRAME_BYTES: u32 = 120 * 44_100 * 2;
/// Bounded stderr capture (bytes) for process diagnostics.
const MAX_STDERR_TAIL_BYTES: usize = 8 * 1024;
/// Best-effort protocol `close` budget on cancel/success — never the 120s RPC default.
#[cfg(feature = "worker-proto")]
const CLOSE_BUDGET: Duration = Duration::from_millis(250);
/// Forced kill wait after cancel/timeout.
const KILL_WAIT_BUDGET: Duration = Duration::from_secs(2);
#[cfg(feature = "worker-proto")]
const WORKER_MODULE: &str = "melotts";
#[cfg(feature = "worker-proto")]
const CLIENT_NAME: &str = "shuvoice-tts";

/// Parent environment keys forwarded into MeloTTS worker children after
/// `env_clear`. Explicit spawn / `worker_env` overlays are applied last and win
/// on key conflicts.
///
/// Intentionally **excludes** credentials and agent state:
/// API keys, `SSH_*`, `AWS_*`, `GITHUB_TOKEN` / `GH_TOKEN`, cloud SDKs, etc.
pub const CHILD_ENV_ALLOWLIST: &[&str] = &[
    // Process / user
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "USERNAME",
    "SHELL",
    // Locale / time
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "LC_NUMERIC",
    "LC_TIME",
    "LANGUAGE",
    "TZ",
    // Temp + XDG
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_RUNTIME_DIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    // TLS / CA bundles (not private keys)
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "PIP_CERT",
    // Proxies are NOT auto-forwarded: values often embed credentials
    // (http://user:pass@host). Opt in via explicit spawn/worker_env overlays.
    // CUDA / GPU runtime
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_ROOT",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    // Model/cache locations (paths only — not tokens)
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    // Python runtime knobs (PYTHONPATH is set explicitly by spawn)
    "PYTHONHOME",
    "PYTHONNOUSERSITE",
    "PYTHONSAFEPATH",
    "PYTHONHASHSEED",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONWARNINGS",
];

/// Built-in MeloTTS English voices.
pub fn melotts_voices() -> Vec<VoiceInfo> {
    vec![
        VoiceInfo::with_description(
            "EN-US",
            "American English",
            "MeloTTS EN_V2 — American accent",
        ),
        VoiceInfo::with_description("EN-BR", "British English", "MeloTTS EN_V2 — British accent"),
        VoiceInfo::with_description(
            "EN-INDIA",
            "Indian English",
            "MeloTTS EN_V2 — Indian accent",
        ),
        VoiceInfo::with_description(
            "EN-AU",
            "Australian English",
            "MeloTTS EN_V2 — Australian accent",
        ),
        VoiceInfo::with_description(
            "EN-Newest",
            "Newest English",
            "MeloTTS EN_NEWEST — latest improved voice",
        ),
    ]
}

/// Wire protocol used to talk to the MeloTTS worker process.
///
/// Production construction ([`MeloTtsBackend::new`] / [`crate::create_tts_backend`])
/// always forces [`MeloWireMode::WorkerProto`]. [`MeloWireMode::LegacyHelper`] is
/// retained only for [`MeloTtsBackend::new_for_test`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::manual_non_exhaustive)] // LegacyHelper is intentionally test-reachable, not a stability lock.
pub enum MeloWireMode {
    /// [`shuvoice_worker_proto`] framed protocol (requires feature `worker-proto`).
    #[default]
    WorkerProto,
    /// Legacy Python helper (JSON line + u32le PCM frame).
    ///
    /// **Test-only.** Rejected by [`MeloTtsBackend::new`]; use
    /// [`MeloTtsBackend::new_for_test`] from tests.
    LegacyHelper,
}

/// Explicit, shell-free spawn specification for the MeloTTS worker process.
///
/// Prefer this over ad-hoc stringy commands when the host already knows the
/// full argv/env (tests, custom packaging). Production hosts typically leave
/// this unset and populate [`MeloTtsConfig::worker_root`] +
/// [`MeloTtsConfig::venv_path`] instead.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MeloWorkerSpawn {
    /// Absolute or PATH-resolved executable (typically `venv/bin/python`).
    pub program: PathBuf,
    /// Argument vector (not shell-joined), e.g. `["-m", "melotts", "--device", "cpu"]`.
    pub args: Vec<String>,
    /// Extra environment pairs applied **after** the isolated allowlist.
    pub env: Vec<(String, String)>,
    /// Working directory (typically the `workers/` tree root).
    pub current_dir: Option<PathBuf>,
}

impl MeloWorkerSpawn {
    /// Build a spawn spec for `program` with no args/env.
    #[must_use]
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            env: Vec::new(),
            current_dir: None,
        }
    }

    #[must_use]
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    #[must_use]
    pub fn env_pair(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.env.push((key.into(), val.into()));
        self
    }

    #[must_use]
    pub fn current_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.current_dir = Some(dir.into());
        self
    }
}

/// Runtime configuration for MeloTTS.
#[derive(Debug, Clone)]
pub struct MeloTtsConfig {
    /// Isolated MeloTTS venv root (`…/melotts-venv`). Python defaults to
    /// `{venv_path}/bin/python` unless [`Self::python_binary`] is set.
    pub venv_path: PathBuf,
    /// `auto` | `cpu` | `cuda` (normalized at spawn time).
    pub device: String,
    pub max_chars: usize,
    /// Wall-clock budget for handshake + synthesize RPC (also used as child wait cap).
    pub request_timeout: Duration,
    pub default_voice_id: String,
    /// Legacy helper script path. Cleared by [`MeloTtsBackend::new`].
    #[doc(hidden)]
    pub helper_script: Option<PathBuf>,
    /// Optional explicit interpreter; defaults to `{venv_path}/bin/python`.
    pub python_binary: Option<PathBuf>,
    /// Wire mode. [`MeloTtsBackend::new`] forces [`MeloWireMode::WorkerProto`].
    pub wire_mode: MeloWireMode,
    /// Root of the bundled `workers/` tree (must contain `melotts/` **and**
    /// `shuvoice_worker_proto/`). Required for production WorkerProto spawn when
    /// [`Self::worker_spawn`] is unset.
    pub worker_root: Option<PathBuf>,
    /// Full typed spawn override (program/args/env/current_dir). Wins over
    /// derived WorkerProto / legacy resolution and over [`Self::worker_command`].
    pub worker_spawn: Option<MeloWorkerSpawn>,
    /// Optional simple argv override (`[program, arg…]`). Used only when
    /// [`Self::worker_spawn`] is `None`. Prefer [`Self::worker_spawn`] so env
    /// and `current_dir` stay explicit.
    pub worker_command: Option<Vec<String>>,
    /// Extra child env pairs merged into the derived WorkerProto spawn (and
    /// into [`Self::worker_command`] resolution). Applied last after the
    /// isolated allowlist.
    pub worker_env: Vec<(String, String)>,
}

impl Default for MeloTtsConfig {
    fn default() -> Self {
        Self {
            venv_path: expand_user(Path::new(DEFAULT_MELOTTS_VENV_DIR)),
            device: "auto".into(),
            max_chars: 5000,
            request_timeout: Duration::from_secs(30),
            default_voice_id: DEFAULT_MELOTTS_VOICE_ID.into(),
            helper_script: None,
            python_binary: None,
            wire_mode: MeloWireMode::WorkerProto,
            worker_root: None,
            worker_spawn: None,
            worker_command: None,
            worker_env: Vec::new(),
        }
    }
}

/// MeloTTS backend using a subprocess worker.
pub struct MeloTtsBackend {
    config: MeloTtsConfig,
}

impl MeloTtsBackend {
    /// Production constructor.
    ///
    /// Forces [`MeloWireMode::WorkerProto`] and clears any legacy helper path so
    /// callers cannot accidentally select the JSON-line helper.
    pub fn new(config: MeloTtsConfig) -> Self {
        let mut config = config;
        config.wire_mode = MeloWireMode::WorkerProto;
        config.helper_script = None;
        Self { config }
    }

    /// Test-only constructor that preserves `wire_mode` / `helper_script` as given.
    ///
    /// Allows integration/unit tests to exercise the legacy framed-PCM helpers
    /// without exposing that path through production construction.
    #[doc(hidden)]
    pub fn new_for_test(config: MeloTtsConfig) -> Self {
        Self { config }
    }

    /// Borrow the resolved runtime config.
    #[must_use]
    pub fn config(&self) -> &MeloTtsConfig {
        &self.config
    }

    fn python_bin(&self) -> PathBuf {
        if let Some(bin) = &self.config.python_binary {
            return bin.clone();
        }
        self.config.venv_path.join("bin/python")
    }

    fn helper_path(&self) -> Result<PathBuf, TtsError> {
        if let Some(path) = &self.config.helper_script {
            if path.is_file() {
                return Ok(path.clone());
            }
            return Err(TtsError::process(format!(
                "MeloTTS helper script not found: {}",
                path.display()
            )));
        }
        let managed = expand_user(Path::new("~/.local/share/shuvoice/workers/melo_helper.py"));
        if managed.is_file() {
            return Ok(managed);
        }
        Err(TtsError::process(format!(
            "MeloTTS helper script not found: {}",
            managed.display()
        )))
    }

    /// Encode a legacy synthesis request JSON line (no trailing newline).
    #[doc(hidden)]
    pub fn build_request_json(text: &str, voice_id: &str, speed: f64) -> String {
        serde_json::json!({
            "text": text,
            "voice_id": voice_id,
            "speed": speed,
        })
        .to_string()
    }

    /// Read one legacy framed PCM payload (4-byte LE length + body).
    #[doc(hidden)]
    pub async fn read_framed_pcm<R: AsyncReadExt + Unpin>(
        reader: &mut R,
    ) -> Result<Bytes, TtsError> {
        let mut header = [0u8; FRAME_HEADER_SIZE];
        let n = reader.read(&mut header).await?;
        if n == 0 {
            return Ok(Bytes::new());
        }
        if n < FRAME_HEADER_SIZE {
            return Err(TtsError::process(format!(
                "Incomplete frame header from MeloTTS helper \
                 (expected {FRAME_HEADER_SIZE} bytes, got {n})"
            )));
        }
        let payload_len_u32 = u32::from_le_bytes(header);
        if payload_len_u32 > MAX_MELO_FRAME_BYTES {
            return Err(TtsError::process(format!(
                "MeloTTS frame length {payload_len_u32} exceeds cap {MAX_MELO_FRAME_BYTES}"
            )));
        }
        let payload_len = payload_len_u32 as usize;
        let mut payload = vec![0u8; payload_len];
        reader.read_exact(&mut payload).await.map_err(|err| {
            TtsError::process(format!("Incomplete PCM payload from MeloTTS helper: {err}"))
        })?;
        Ok(Bytes::from(payload))
    }

    /// Resolve the deterministic spawn specification for the configured mode.
    ///
    /// WorkerProto **never** consults [`MeloTtsConfig::helper_script`] or the
    /// managed `melo_helper.py` path.
    pub fn resolve_spawn(&self) -> Result<MeloWorkerSpawn, TtsError> {
        if let Some(spawn) = &self.config.worker_spawn {
            return validate_spawn(spawn);
        }

        if let Some(cmd) = &self.config.worker_command {
            return self.resolve_worker_command(cmd);
        }

        match self.config.wire_mode {
            MeloWireMode::WorkerProto => self.resolve_worker_proto_spawn(),
            MeloWireMode::LegacyHelper => self.resolve_legacy_spawn(),
        }
    }

    fn resolve_worker_command(&self, cmd: &[String]) -> Result<MeloWorkerSpawn, TtsError> {
        let (head, tail) = cmd
            .split_first()
            .ok_or_else(|| TtsError::config("MeloTTS worker_command is empty"))?;
        if head.trim().is_empty() {
            return Err(TtsError::config("MeloTTS worker_command program is empty"));
        }
        let mut spawn = MeloWorkerSpawn {
            program: PathBuf::from(head),
            args: tail.to_vec(),
            env: Vec::new(),
            current_dir: self.config.worker_root.clone(),
        };
        if let Some(root) = &self.config.worker_root {
            spawn.env = self.derived_worker_env(root);
        } else {
            spawn.env = merge_env(Vec::new(), &self.config.worker_env);
        }
        validate_spawn(&spawn)
    }

    fn resolve_worker_proto_spawn(&self) -> Result<MeloWorkerSpawn, TtsError> {
        #[cfg(not(feature = "worker-proto"))]
        {
            Err(TtsError::config(
                "MeloTTS WorkerProto mode requires the `worker-proto` cargo feature",
            ))
        }

        #[cfg(feature = "worker-proto")]
        {
            let worker_root = self.config.worker_root.as_ref().ok_or_else(|| {
                TtsError::config(
                    "MeloTTS worker_root is required for worker-proto mode \
                     (directory containing the melotts package)",
                )
            })?;
            validate_worker_root(worker_root)?;

            let program = self.python_bin();
            if !program.exists() {
                return Err(TtsError::process(format!(
                    "MeloTTS python binary not found: {}. \
                     Run 'shuvoice setup --install-missing' or set melotts python/venv paths.",
                    program.display()
                )));
            }

            let device = normalize_device(&self.config.device);
            let spawn = MeloWorkerSpawn {
                program,
                args: vec!["-m".into(), WORKER_MODULE.into(), "--device".into(), device],
                env: self.derived_worker_env(worker_root),
                current_dir: Some(worker_root.clone()),
            };
            validate_spawn(&spawn)
        }
    }

    fn resolve_legacy_spawn(&self) -> Result<MeloWorkerSpawn, TtsError> {
        let python = self.python_bin();
        let helper = self.helper_path()?;
        let spawn = MeloWorkerSpawn {
            program: python,
            args: vec![
                helper.display().to_string(),
                normalize_device(&self.config.device),
            ],
            env: merge_env(Vec::new(), &self.config.worker_env),
            current_dir: None,
        };
        validate_spawn(&spawn)
    }

    fn derived_worker_env(&self, worker_root: &Path) -> Vec<(String, String)> {
        let mut base = vec![
            (
                "PYTHONPATH".into(),
                worker_root.to_string_lossy().into_owned(),
            ),
            ("PYTHONUNBUFFERED".into(), "1".into()),
            (
                "SHUVOICE_MELOTTS_DEVICE".into(),
                normalize_device(&self.config.device),
            ),
            (
                "SHUVOICE_MELOTTS_VENV".into(),
                self.config.venv_path.to_string_lossy().into_owned(),
            ),
        ];
        base = merge_env(base, &self.config.worker_env);
        base
    }
}

#[async_trait]
impl TtsBackend for MeloTtsBackend {
    fn id(&self) -> BackendId {
        BackendId::MeloTts
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_streaming: false,
            supports_voice_list: true,
            requires_api_key: false,
            supports_speed_control: true,
            speed_min: Some(0.5),
            speed_max: Some(2.0),
        }
    }

    fn sample_rate_hz(&self) -> u32 {
        MELOTTS_SAMPLE_RATE_HZ
    }

    fn dependency_errors(&self) -> Vec<String> {
        dependency_errors_for(&self.config)
    }

    async fn synthesize_stream(
        &self,
        request: SynthesisRequest,
        cancel: CancellationToken,
    ) -> Result<SynthesisStream, TtsError> {
        let text = ensure_text(&request.text, self.config.max_chars)?;
        let voice = if request.voice_id.trim().is_empty() {
            self.config.default_voice_id.clone()
        } else {
            request.voice_id.trim().to_string()
        };

        tracing::info!(
            voice = %voice,
            speed = request.playback_speed,
            text_len = text.len(),
            wire = ?self.config.wire_mode,
            "MeloTTS synthesis"
        );

        if cancel.is_cancelled() {
            return Err(TtsError::Cancelled);
        }

        let request_timeout = self.config.request_timeout.max(Duration::from_millis(1));

        let spawn = self.resolve_spawn()?;
        let mut child = spawn_isolated_child(&spawn).map_err(|err| {
            TtsError::process(format!(
                "Failed to start MeloTTS worker process: {}",
                redact_for_ui(&err.to_string())
            ))
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| TtsError::process("MeloTTS stdin missing"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| TtsError::process("MeloTTS stdout missing"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| TtsError::process("MeloTTS stderr missing"))?;

        let stderr_task = tokio::spawn(async move { read_stderr_tail(stderr).await });

        let wire_mode = self.config.wire_mode;
        let speed = request.playback_speed;

        // Melo is non-streaming at the engine: await full utterance under a
        // single wall-clock + cancel budget, then yield PCM chunks.
        let work = async {
            match wire_mode {
                MeloWireMode::WorkerProto => {
                    synthesize_worker_proto(stdout, stdin, text, voice, speed, request_timeout)
                        .await
                }
                MeloWireMode::LegacyHelper => {
                    let mut stdin = stdin;
                    let mut stdout = stdout;
                    let req_json = MeloTtsBackend::build_request_json(&text, &voice, speed);
                    stdin.write_all(req_json.as_bytes()).await?;
                    stdin.write_all(b"\n").await?;
                    drop(stdin);
                    let pcm = MeloTtsBackend::read_framed_pcm(&mut stdout).await?;
                    if pcm.is_empty() {
                        return Err(TtsError::process("MeloTTS helper returned empty PCM"));
                    }
                    Ok(SynthOutcome {
                        pcm,
                        sample_rate_hz: MELOTTS_SAMPLE_RATE_HZ,
                    })
                }
            }
        };

        let outcome = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                kill_and_reap(&mut child).await;
                let _ = finish_stderr(stderr_task).await;
                return Err(TtsError::Cancelled);
            }
            result = tokio::time::timeout(request_timeout, work) => {
                match result {
                    Ok(inner) => inner,
                    Err(_) => {
                        kill_and_reap(&mut child).await;
                        let stderr_tail = finish_stderr(stderr_task).await;
                        return Err(annotate_stderr(
                            TtsError::timed_out("MeloTTS synthesis timed out"),
                            &stderr_tail,
                        ));
                    }
                }
            }
        };

        // Reap child: protocol close (if any) already used a short budget inside
        // the worker-proto path; force-kill leftovers so we never stall.
        kill_and_reap(&mut child).await;
        let stderr_tail = finish_stderr(stderr_task).await;

        let SynthOutcome {
            pcm,
            sample_rate_hz,
        } = match outcome {
            Ok(out) => out,
            Err(err) => return Err(annotate_stderr(err, &stderr_tail)),
        };

        let cancel_stream = cancel.clone();
        let stream = async_stream::stream! {
            let mut offset = 0;
            while offset < pcm.len() {
                if cancel_stream.is_cancelled() {
                    yield Err(TtsError::Cancelled);
                    return;
                }
                let end = (offset + 4096).min(pcm.len());
                yield Ok(pcm.slice(offset..end));
                offset = end;
            }
        };
        Ok(SynthesisStream {
            sample_rate_hz,
            encoding: AudioEncoding::PcmS16Le,
            chunks: Box::pin(stream),
        })
    }

    async fn list_voices(&self) -> Result<Vec<VoiceInfo>, TtsError> {
        Ok(melotts_voices())
    }
}

struct SynthOutcome {
    pcm: Bytes,
    sample_rate_hz: u32,
}

async fn synthesize_worker_proto(
    stdout: tokio::process::ChildStdout,
    stdin: tokio::process::ChildStdin,
    text: String,
    voice: String,
    speed: f64,
    rpc_timeout: Duration,
) -> Result<SynthOutcome, TtsError> {
    #[cfg(not(feature = "worker-proto"))]
    {
        let _ = (stdout, stdin, text, voice, speed, rpc_timeout);
        Err(TtsError::config(
            "MeloTTS WorkerProto mode requires the `worker-proto` cargo feature",
        ))
    }

    #[cfg(feature = "worker-proto")]
    {
        // Never interpolate `text` into error strings (payload redaction).
        let options = ClientOptions {
            rpc_timeout,
            load_timeout: rpc_timeout,
            ..ClientOptions::default()
        };
        let mut client = WorkerClient::with_options(stdout, stdin, options);

        let session = client
            .handshake(CLIENT_NAME)
            .await
            .map_err(proto_err)?
            .clone();

        let manifest_rate = session
            .manifest
            .tts
            .as_ref()
            .and_then(|tts| tts.default_sample_rate_hz);

        let result = client
            .synthesize(text, Some(voice), Some(speed as f32))
            .await
            .map_err(proto_err)?;

        // Short-budget close only — never the default 120s RPC wait on teardown.
        best_effort_close(&mut client).await;

        let sample_rate_hz = resolve_sample_rate(result.sample_rate_hz, manifest_rate)?;
        let pcm = match result.encoding {
            PcmEncoding::I16Le => result.pcm,
            PcmEncoding::F32Le => Bytes::from(f32_le_to_i16_le(&result.pcm)?),
        };
        if pcm.is_empty() {
            return Err(TtsError::process("MeloTTS worker returned empty PCM"));
        }
        Ok(SynthOutcome {
            pcm,
            sample_rate_hz,
        })
    }
}

#[cfg(feature = "worker-proto")]
async fn best_effort_close<R, W>(client: &mut WorkerClient<R, W>)
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    let _ = tokio::time::timeout(CLOSE_BUDGET, client.close()).await;
}

#[cfg(any(test, feature = "worker-proto"))]
fn resolve_sample_rate(
    result_rate: Option<u32>,
    manifest_rate: Option<u32>,
) -> Result<u32, TtsError> {
    let rate = match (result_rate, manifest_rate) {
        (Some(r), Some(m)) if r > 0 && m > 0 && r != m => {
            return Err(TtsError::process(format!(
                "MeloTTS sample rate mismatch: audio_end={r}Hz manifest={m}Hz"
            )));
        }
        (Some(r), _) if r > 0 => r,
        (_, Some(m)) if m > 0 => m,
        (Some(0), _) | (_, Some(0)) => {
            return Err(TtsError::process(
                "MeloTTS worker advertised a zero sample rate",
            ));
        }
        _ => MELOTTS_SAMPLE_RATE_HZ,
    };
    Ok(rate)
}

#[cfg(feature = "worker-proto")]
fn proto_err(err: shuvoice_worker_proto::ProtocolError) -> TtsError {
    TtsError::process(format!(
        "MeloTTS worker protocol error: {}",
        scrub_text(&err.to_string())
    ))
}

#[cfg(feature = "worker-proto")]
fn f32_le_to_i16_le(bytes: &[u8]) -> Result<Vec<u8>, TtsError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(TtsError::decode(
            "MeloTTS f32 PCM length is not a multiple of 4",
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(4) {
        let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let s = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        out.extend_from_slice(&s.to_le_bytes());
    }
    Ok(out)
}

fn dependency_errors_for(config: &MeloTtsConfig) -> Vec<String> {
    match config.wire_mode {
        MeloWireMode::WorkerProto => worker_proto_dependency_errors(config),
        MeloWireMode::LegacyHelper => legacy_dependency_errors(config),
    }
}

fn worker_proto_dependency_errors(config: &MeloTtsConfig) -> Vec<String> {
    #[cfg(not(feature = "worker-proto"))]
    {
        let _ = config;
        vec![
            "MeloTTS WorkerProto mode requires the `worker-proto` cargo feature of shuvoice-tts"
                .into(),
        ]
    }

    #[cfg(feature = "worker-proto")]
    {
        let mut errors = Vec::new();
        if let Some(spawn) = &config.worker_spawn {
            if spawn.program.as_os_str().is_empty() {
                errors.push("MeloTTS worker_spawn.program is empty".into());
            } else if !program_exists(&spawn.program) {
                errors.push(format!(
                    "MeloTTS worker program not found: {}",
                    spawn.program.display()
                ));
            }
            return errors;
        }

        if let Some(cmd) = &config.worker_command {
            match cmd.split_first() {
                None => errors.push("MeloTTS worker_command is empty".into()),
                Some((head, _)) if head.trim().is_empty() => {
                    errors.push("MeloTTS worker_command program is empty".into());
                }
                Some((head, _)) if !program_exists(Path::new(head)) => {
                    errors.push(format!("MeloTTS worker program not found: {head}"));
                }
                Some(_) => {}
            }
            return errors;
        }

        match &config.worker_root {
            None => errors.push(
                "MeloTTS worker_root is not configured \
                 (expected path to the workers/ tree containing melotts/)"
                    .into(),
            ),
            Some(root) => {
                if let Err(err) = validate_worker_root(root) {
                    errors.push(err.to_string());
                }
            }
        }

        let python = config
            .python_binary
            .clone()
            .unwrap_or_else(|| config.venv_path.join("bin/python"));
        if !config.venv_path.is_dir() && config.python_binary.is_none() {
            errors.push(format!(
                "MeloTTS venv directory does not exist: {}. \
                 Run 'shuvoice setup --install-missing' to create it.",
                config.venv_path.display()
            ));
            return errors;
        }
        if !python.exists() {
            errors.push(format!(
                "MeloTTS venv python binary not found: {}",
                python.display()
            ));
            return errors;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(meta) = python.metadata()
                && meta.permissions().mode() & 0o111 == 0
            {
                errors.push(format!(
                    "MeloTTS venv python is not executable: {}",
                    python.display()
                ));
            }
        }
        errors
    }
}

fn legacy_dependency_errors(config: &MeloTtsConfig) -> Vec<String> {
    let mut errors = Vec::new();
    let venv_dir = &config.venv_path;
    if !venv_dir.is_dir() {
        errors.push(format!(
            "MeloTTS venv directory does not exist: {}. \
             Run 'shuvoice setup --install-missing' to create it.",
            venv_dir.display()
        ));
        return errors;
    }
    let python_bin = config
        .python_binary
        .clone()
        .unwrap_or_else(|| venv_dir.join("bin/python"));
    if !python_bin.exists() {
        errors.push(format!(
            "MeloTTS venv python binary not found: {}",
            python_bin.display()
        ));
        return errors;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = python_bin.metadata()
            && meta.permissions().mode() & 0o111 == 0
        {
            errors.push(format!(
                "MeloTTS venv python is not executable: {}",
                python_bin.display()
            ));
        }
    }
    if let Some(helper) = &config.helper_script
        && !helper.is_file()
    {
        errors.push(format!(
            "MeloTTS helper script not found: {}",
            helper.display()
        ));
    }
    errors
}

#[cfg(feature = "worker-proto")]
fn validate_worker_root(worker_root: &Path) -> Result<(), TtsError> {
    let melotts_main = worker_root.join(WORKER_MODULE).join("__main__.py");
    if !melotts_main.is_file() {
        return Err(TtsError::config(format!(
            "MeloTTS worker package not found under {} (expected {WORKER_MODULE}/__main__.py)",
            worker_root.display()
        )));
    }
    let proto_init = worker_root
        .join("shuvoice_worker_proto")
        .join("__init__.py");
    if !proto_init.is_file() {
        return Err(TtsError::config(format!(
            "MeloTTS worker_root missing shuvoice_worker_proto package under {} \
             (expected shuvoice_worker_proto/__init__.py)",
            worker_root.display()
        )));
    }
    Ok(())
}

fn validate_spawn(spawn: &MeloWorkerSpawn) -> Result<MeloWorkerSpawn, TtsError> {
    if spawn.program.as_os_str().is_empty() {
        return Err(TtsError::config("MeloTTS worker program is empty"));
    }
    let prog = spawn.program.to_string_lossy();
    if prog.contains(' ') || prog.contains('\n') || prog.contains('|') || prog.contains(';') {
        return Err(TtsError::config(
            "MeloTTS worker program must be a single path/name (no shell metacharacters)",
        ));
    }
    Ok(spawn.clone())
}

fn normalize_device(device: &str) -> String {
    match device.trim().to_ascii_lowercase().as_str() {
        "cpu" => "cpu".into(),
        "cuda" | "gpu" => "cuda".into(),
        "auto" | "" => "auto".into(),
        other => {
            if other
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                other.to_string()
            } else {
                "auto".into()
            }
        }
    }
}

fn merge_env(mut base: Vec<(String, String)>, extra: &[(String, String)]) -> Vec<(String, String)> {
    for (k, v) in extra {
        if let Some(slot) = base.iter_mut().find(|(bk, _)| bk == k) {
            slot.1 = v.clone();
        } else {
            base.push((k.clone(), v.clone()));
        }
    }
    base
}

/// Build the isolated child environment: allowlisted parent vars, then explicit overlays.
#[must_use]
pub fn build_isolated_child_env(overlays: &[(String, String)]) -> Vec<(String, String)> {
    let mut base = Vec::new();
    for key in CHILD_ENV_ALLOWLIST {
        if let Ok(val) = std::env::var(key)
            && !val.is_empty()
        {
            base.push(((*key).to_string(), val));
        }
    }
    merge_env(base, overlays)
}

fn spawn_isolated_child(spawn: &MeloWorkerSpawn) -> std::io::Result<Child> {
    let mut command = Command::new(&spawn.program);
    command
        .args(&spawn.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .env_clear();

    if let Some(dir) = &spawn.current_dir {
        command.current_dir(dir);
    }
    for (key, val) in build_isolated_child_env(&spawn.env) {
        command.env(key, val);
    }
    command.spawn()
}

async fn kill_and_reap(child: &mut Child) {
    let _ = child.start_kill();
    let _ = tokio::time::timeout(KILL_WAIT_BUDGET, child.wait()).await;
}

#[cfg(feature = "worker-proto")]
fn program_exists(program: &Path) -> bool {
    if program.is_absolute() || program.components().count() > 1 {
        return program.exists();
    }
    // Bare name: treat as PATH-resolved later.
    true
}

fn expand_user(path: &Path) -> PathBuf {
    let raw = path.as_os_str().to_string_lossy();
    if let Some(rest) = raw.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    if raw == "~"
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home);
    }
    path.to_path_buf()
}

async fn read_stderr_tail(mut stderr: impl AsyncReadExt + Unpin) -> String {
    let mut ring: VecDeque<u8> = VecDeque::new();
    let mut buf = [0u8; 512];
    loop {
        match stderr.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => {
                for &b in &buf[..n] {
                    if ring.len() == MAX_STDERR_TAIL_BYTES {
                        ring.pop_front();
                    }
                    ring.push_back(b);
                }
            }
            Err(_) => break,
        }
    }
    let bytes: Vec<u8> = ring.into_iter().collect();
    scrub_stderr_bytes(&bytes)
}

/// Strong stderr scrubber — prefers worker-proto redactor when linked.
fn scrub_stderr_bytes(bytes: &[u8]) -> String {
    #[cfg(feature = "worker-proto")]
    {
        redact_stderr_tail(bytes)
    }
    #[cfg(not(feature = "worker-proto"))]
    {
        local_redact_stderr_tail(bytes)
    }
}

#[cfg(feature = "worker-proto")]
fn scrub_text(input: &str) -> String {
    #[cfg(feature = "worker-proto")]
    {
        shuvoice_worker_proto::redact_text(input)
    }
    #[cfg(not(feature = "worker-proto"))]
    {
        local_redact_text(input)
    }
}

/// Local high-entropy / key / transcript scrub used when `worker-proto` is off.
#[cfg(not(feature = "worker-proto"))]
fn local_redact_stderr_tail(bytes: &[u8]) -> String {
    local_redact_text(&String::from_utf8_lossy(bytes))
}

#[cfg(not(feature = "worker-proto"))]
fn local_redact_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for line in input.split_inclusive('\n') {
        out.push_str(&local_redact_line(line));
    }
    // Also strip absolute paths / URLs for UI safety.
    redact_for_ui(&out)
}

#[cfg(not(feature = "worker-proto"))]
fn local_redact_line(line: &str) -> String {
    let lower = line.to_ascii_lowercase();
    let hot = [
        "api_key",
        "apikey",
        "authorization",
        "secret",
        "password",
        "token",
        "bearer ",
        "sk-",
        "transcript",
        "synth_text",
        "tts_text",
        "openai_api_key",
        "elevenlabs_api_key",
    ];
    if hot.iter().any(|k| lower.contains(k)) {
        return "[REDACTED]\n".to_string();
    }
    // High-entropy blobs ≥ 20 chars.
    let mut result = String::with_capacity(line.len());
    let mut cur = String::new();
    for ch in line.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '+' | '/' | '=' | '-' | '_') {
            cur.push(ch);
        } else {
            flush_local_token(&mut result, &cur);
            cur.clear();
            result.push(ch);
        }
    }
    flush_local_token(&mut result, &cur);
    result
}

#[cfg(not(feature = "worker-proto"))]
fn flush_local_token(out: &mut String, token: &str) {
    if token.len() >= 20 {
        let has_digit = token.chars().any(|c| c.is_ascii_digit());
        let has_alpha = token.chars().any(|c| c.is_ascii_alphabetic());
        if has_digit && has_alpha || token.len() >= 32 {
            out.push_str("[REDACTED_BLOB]");
            return;
        }
    }
    out.push_str(token);
}

async fn finish_stderr(task: tokio::task::JoinHandle<String>) -> String {
    match tokio::time::timeout(Duration::from_secs(1), task).await {
        Ok(Ok(s)) => s,
        _ => String::new(),
    }
}

fn annotate_stderr(err: TtsError, stderr_tail: &str) -> TtsError {
    if stderr_tail.is_empty() {
        return err;
    }
    // stderr_tail is already scrubbed — never attach raw bytes.
    match err {
        TtsError::Process(msg) if !msg.contains(stderr_tail) => {
            TtsError::process(format!("{msg} ({stderr_tail})"))
        }
        TtsError::TimedOut(msg) if !msg.contains(stderr_tail) => {
            TtsError::timed_out(format!("{msg} ({stderr_tail})"))
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "worker-proto")]
    use std::fs;
    #[cfg(feature = "worker-proto")]
    use std::io::Write;

    #[cfg(feature = "worker-proto")]
    fn temp_workers_tree() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let melotts = dir.path().join("melotts");
        fs::create_dir_all(&melotts).unwrap();
        fs::write(melotts.join("__main__.py"), b"# test\n").unwrap();
        fs::write(melotts.join("__init__.py"), b"").unwrap();
        let proto = dir.path().join("shuvoice_worker_proto");
        fs::create_dir_all(&proto).unwrap();
        fs::write(proto.join("__init__.py"), b"").unwrap();
        dir
    }

    #[cfg(feature = "worker-proto")]
    fn temp_venv_python() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("bin");
        fs::create_dir_all(&bin).unwrap();
        let python = bin.join("python");
        let mut f = fs::File::create(&python).unwrap();
        writeln!(f, "#!/bin/sh\nexit 0").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }
        (dir, python)
    }

    #[test]
    fn normalize_device_accepts_known_tokens() {
        assert_eq!(normalize_device("CPU"), "cpu");
        assert_eq!(normalize_device("cuda"), "cuda");
        assert_eq!(normalize_device("GPU"), "cuda");
        assert_eq!(normalize_device("auto"), "auto");
        assert_eq!(normalize_device("  "), "auto");
    }

    #[test]
    fn production_new_forces_worker_proto_and_clears_helper() {
        let cfg = MeloTtsConfig {
            wire_mode: MeloWireMode::LegacyHelper,
            helper_script: Some(PathBuf::from("/tmp/melo_helper.py")),
            ..MeloTtsConfig::default()
        };
        let backend = MeloTtsBackend::new(cfg);
        assert_eq!(backend.config().wire_mode, MeloWireMode::WorkerProto);
        assert!(backend.config().helper_script.is_none());
    }

    #[test]
    fn test_constructor_preserves_legacy_mode() {
        let cfg = MeloTtsConfig {
            wire_mode: MeloWireMode::LegacyHelper,
            helper_script: Some(PathBuf::from("/tmp/melo_helper.py")),
            ..MeloTtsConfig::default()
        };
        let backend = MeloTtsBackend::new_for_test(cfg);
        assert_eq!(backend.config().wire_mode, MeloWireMode::LegacyHelper);
        assert!(backend.config().helper_script.is_some());
    }

    #[test]
    fn child_env_excludes_api_key_sentinels() {
        // SAFETY: isolated unit test mutates process env and restores it before returning.
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "sk-test-sentinel-openai");
            std::env::set_var("ELEVENLABS_API_KEY", "el-test-sentinel-eleven");
            std::env::set_var("SSH_AUTH_SOCK", "/tmp/ssh-agent.sock");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "aws-secret-sentinel");
            std::env::set_var("GH_TOKEN", "gh-sentinel");
            std::env::set_var(
                "HTTP_PROXY",
                "http://proxy-user:proxy-pass-sentinel@127.0.0.1:8080",
            );
            std::env::set_var(
                "HTTPS_PROXY",
                "http://proxy-user:proxy-pass-sentinel@127.0.0.1:8443",
            );
            std::env::set_var(
                "http_proxy",
                "http://proxy-user:proxy-pass-sentinel@127.0.0.1:8080",
            );
            std::env::set_var(
                "https_proxy",
                "http://proxy-user:proxy-pass-sentinel@127.0.0.1:8443",
            );
            if std::env::var_os("PATH").is_none() {
                std::env::set_var("PATH", "/usr/bin");
            }
        }
        let env = build_isolated_child_env(&[("PYTHONPATH".into(), "/workers".into())]);
        let keys: Vec<&str> = env.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"PATH") || keys.contains(&"HOME") || keys.contains(&"PYTHONPATH"));
        assert!(keys.contains(&"PYTHONPATH"));
        assert!(!keys.contains(&"OPENAI_API_KEY"));
        assert!(!keys.contains(&"ELEVENLABS_API_KEY"));
        assert!(!keys.contains(&"SSH_AUTH_SOCK"));
        assert!(!keys.contains(&"AWS_SECRET_ACCESS_KEY"));
        assert!(!keys.contains(&"GH_TOKEN"));
        assert!(!keys.contains(&"HTTP_PROXY"));
        assert!(!keys.contains(&"HTTPS_PROXY"));
        assert!(!keys.contains(&"http_proxy"));
        assert!(!keys.contains(&"https_proxy"));
        assert!(
            env.iter().all(|(_, v)| !v.contains("proxy-pass-sentinel")),
            "credential-bearing proxy value leaked into child env: {env:?}"
        );
        // Explicit overlay remains the deliberate opt-in path.
        let env_opt_in = build_isolated_child_env(&[(
            "HTTPS_PROXY".into(),
            "http://explicit-only@127.0.0.1:9".into(),
        )]);
        assert!(
            env_opt_in
                .iter()
                .any(|(k, v)| k == "HTTPS_PROXY" && v.contains("explicit-only"))
        );
        // Overlays win / appear.
        assert_eq!(
            env.iter()
                .find(|(k, _)| k == "PYTHONPATH")
                .map(|(_, v)| v.as_str()),
            Some("/workers")
        );
        // SAFETY: restore process env mutated above.
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("ELEVENLABS_API_KEY");
            std::env::remove_var("SSH_AUTH_SOCK");
            std::env::remove_var("AWS_SECRET_ACCESS_KEY");
            std::env::remove_var("GH_TOKEN");
            std::env::remove_var("HTTP_PROXY");
            std::env::remove_var("HTTPS_PROXY");
            std::env::remove_var("http_proxy");
            std::env::remove_var("https_proxy");
        }
    }

    #[test]
    fn stderr_scrub_redacts_keys_and_entropy() {
        let raw = b"info api_key=sk-abc123LIVESECRET\npartial_transcript hello world\npayload a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6\n";
        let scrubbed = scrub_stderr_bytes(raw);
        assert!(!scrubbed.contains("sk-abc"));
        assert!(!scrubbed.contains("hello world"));
        assert!(!scrubbed.contains("a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"));
        assert!(
            scrubbed.contains("REDACTED") || scrubbed.contains("redacted"),
            "got {scrubbed:?}"
        );
    }

    #[test]
    fn resolve_sample_rate_prefers_audio_and_detects_mismatch() {
        assert_eq!(
            resolve_sample_rate(Some(44_100), Some(44_100)).unwrap(),
            44_100
        );
        assert_eq!(resolve_sample_rate(Some(22_050), None).unwrap(), 22_050);
        assert_eq!(resolve_sample_rate(None, Some(16_000)).unwrap(), 16_000);
        assert_eq!(
            resolve_sample_rate(None, None).unwrap(),
            MELOTTS_SAMPLE_RATE_HZ
        );
        assert!(resolve_sample_rate(Some(44_100), Some(22_050)).is_err());
        assert!(resolve_sample_rate(Some(0), None).is_err());
    }

    #[test]
    #[cfg(feature = "worker-proto")]
    fn worker_proto_spawn_is_deterministic_python_module() {
        let workers = temp_workers_tree();
        let (venv, _py) = temp_venv_python();
        let cfg = MeloTtsConfig {
            venv_path: venv.path().to_path_buf(),
            device: "CUDA".into(),
            wire_mode: MeloWireMode::WorkerProto,
            worker_root: Some(workers.path().to_path_buf()),
            helper_script: Some(PathBuf::from("/tmp/should-not-use-melo_helper.py")),
            ..MeloTtsConfig::default()
        };
        // Production constructor clears helper; resolve still WorkerProto.
        let backend = MeloTtsBackend::new(cfg);
        let spawn = backend.resolve_spawn().expect("spawn");
        assert_eq!(spawn.program, venv.path().join("bin/python"));
        assert_eq!(
            spawn.args,
            vec![
                "-m".to_string(),
                "melotts".to_string(),
                "--device".to_string(),
                "cuda".to_string()
            ]
        );
        assert_eq!(spawn.current_dir.as_deref(), Some(workers.path()));
        let pythonpath = spawn
            .env
            .iter()
            .find(|(k, _)| k == "PYTHONPATH")
            .map(|(_, v)| v.as_str())
            .unwrap();
        assert_eq!(Path::new(pythonpath), workers.path());
        assert!(spawn.args.iter().all(|a| !a.contains("melo_helper")));
    }

    #[test]
    #[cfg(feature = "worker-proto")]
    fn worker_root_requires_proto_package() {
        let dir = tempfile::tempdir().unwrap();
        let melotts = dir.path().join("melotts");
        fs::create_dir_all(&melotts).unwrap();
        fs::write(melotts.join("__main__.py"), b"#\n").unwrap();
        // Missing shuvoice_worker_proto/
        let (venv, _) = temp_venv_python();
        let cfg = MeloTtsConfig {
            venv_path: venv.path().to_path_buf(),
            worker_root: Some(dir.path().to_path_buf()),
            ..MeloTtsConfig::default()
        };
        let err = MeloTtsBackend::new(cfg)
            .resolve_spawn()
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("shuvoice_worker_proto"),
            "expected proto package error, got {err}"
        );
    }

    #[test]
    #[cfg(feature = "worker-proto")]
    fn worker_proto_never_falls_back_to_legacy_helper_when_root_missing() {
        let (venv, _) = temp_venv_python();
        let cfg = MeloTtsConfig {
            venv_path: venv.path().to_path_buf(),
            wire_mode: MeloWireMode::WorkerProto,
            worker_root: None,
            helper_script: Some(PathBuf::from("/tmp/melo_helper.py")),
            ..MeloTtsConfig::default()
        };
        // new_for_test keeps helper_script but WorkerProto still requires root.
        let backend = MeloTtsBackend::new_for_test(cfg);
        let err = backend.resolve_spawn().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("worker_root"),
            "expected worker_root error, got {msg}"
        );
        assert!(!msg.contains("melo_helper"));
    }

    #[test]
    #[cfg(feature = "worker-proto")]
    fn worker_spawn_override_wins() {
        let spawn = MeloWorkerSpawn::new("/usr/bin/python3")
            .args(["-m", "melotts", "--fake", "--device", "cpu"])
            .env_pair("PYTHONPATH", "/workers")
            .env_pair("SHUVOICE_WORKER_FAKE", "1")
            .current_dir("/workers");
        let cfg = MeloTtsConfig {
            wire_mode: MeloWireMode::WorkerProto,
            worker_spawn: Some(spawn.clone()),
            worker_root: None,
            ..MeloTtsConfig::default()
        };
        let backend = MeloTtsBackend::new(cfg);
        assert_eq!(backend.resolve_spawn().unwrap(), spawn);
    }

    #[test]
    fn reject_shell_metacharacters_in_program() {
        let cfg = MeloTtsConfig {
            worker_spawn: Some(MeloWorkerSpawn::new("python -c 'import os'")),
            wire_mode: MeloWireMode::WorkerProto,
            ..MeloTtsConfig::default()
        };
        let err = MeloTtsBackend::new(cfg).resolve_spawn().unwrap_err();
        assert!(err.to_string().contains("shell") || err.to_string().contains("single path"));
    }

    #[test]
    #[cfg(feature = "worker-proto")]
    fn worker_proto_dependency_errors_report_missing_root() {
        let cfg = MeloTtsConfig {
            venv_path: PathBuf::from("/no/such/melotts-venv"),
            wire_mode: MeloWireMode::WorkerProto,
            worker_root: None,
            ..MeloTtsConfig::default()
        };
        let errors = MeloTtsBackend::new(cfg).dependency_errors();
        assert!(
            errors.iter().any(|e| e.contains("worker_root")),
            "{errors:?}"
        );
    }

    #[test]
    fn legacy_dependency_errors_missing_venv() {
        let cfg = MeloTtsConfig {
            venv_path: PathBuf::from("/no/such/melotts-venv"),
            wire_mode: MeloWireMode::LegacyHelper,
            ..MeloTtsConfig::default()
        };
        let errors = MeloTtsBackend::new_for_test(cfg).dependency_errors();
        assert!(!errors.is_empty());
        assert!(errors[0].contains("does not exist"));
    }

    #[test]
    fn merge_env_overrides_by_key() {
        let base = vec![("A".into(), "1".into()), ("B".into(), "2".into())];
        let out = merge_env(base, &[("B".into(), "9".into()), ("C".into(), "3".into())]);
        assert_eq!(
            out,
            vec![
                ("A".into(), "1".into()),
                ("B".into(), "9".into()),
                ("C".into(), "3".into()),
            ]
        );
    }
}
