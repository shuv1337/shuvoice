//! Spawn and supervise a single external worker process.
//!
//! # Child environment isolation
//!
//! [`WorkerProcess::spawn`] always starts from a cleared environment
//! ([`std::process::Command::env_clear`]), then restores a narrow
//! parent-environment allowlist ([`CHILD_ENV_ALLOWLIST`]) covering runtime
//! basics (PATH/HOME/locale/temp/XDG), TLS CA bundles, CUDA/GPU library
//! paths, OpenMP/BLAS thread knobs, and public HF/Torch cache **paths**.
//!
//! Credentials and agent state are **never** auto-forwarded: API keys,
//! `SSH_*`, `AWS_*`, `GH_TOKEN` / `GITHUB_TOKEN`, generic `*_TOKEN` /
//! `*_SECRET`, and proxy variables (which often embed credentials). Callers
//! that deliberately need an excluded key must pass it via
//! [`WorkerSpawnConfig::env`] / [`WorkerSpawnConfig::env_pair`], which are
//! applied **last** and win on conflicts.
//!
//! Spawns are always typed argv (program + args) — never a shell.

use std::collections::VecDeque;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::task::JoinHandle;
use tokio::time::timeout;

use crate::client::{NegotiatedSession, WorkerClient};
use crate::error::{ProtocolError, WorkerProcessError};
use crate::stderr_tail::{self, DEFAULT_STDERR_TAIL_BYTES};

/// How long to wait for the handshake by default.
pub const DEFAULT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// How long to wait for graceful `close` + exit.
pub const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);
/// How long to wait after `kill` before giving up.
pub const DEFAULT_KILL_TIMEOUT: Duration = Duration::from_secs(2);

/// Parent environment keys restored into worker children after
/// [`Command::env_clear`]. Explicit [`WorkerSpawnConfig::env`] overlays are
/// applied last and win on key conflicts.
///
/// Intentionally **excludes** credentials and agent state:
/// `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, `SSH_*`, `AWS_*`,
/// `GITHUB_TOKEN` / `GH_TOKEN`, generic `*_TOKEN` / `*_SECRET`, cloud SDKs,
/// and proxy variables (`HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` /
/// `NO_PROXY` and lowercase forms) whose values often embed credentials.
/// Opt in to any excluded key only via explicit spawn env overlays.
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
    // (http://user:pass@host). Opt in via explicit spawn env overlays.
    // CUDA / GPU runtime
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_ROOT",
    "CUDA_MODULE_LOADING",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    // CPU math thread knobs (not secrets)
    "OMP_NUM_THREADS",
    "OMP_SCHEDULE",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    // Model/cache locations (paths only — not tokens)
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "PYTORCH_CUDA_ALLOC_CONF",
    // Python runtime knobs (PYTHONPATH is set explicitly by spawn overlays)
    "PYTHONHOME",
    "PYTHONNOUSERSITE",
    "PYTHONSAFEPATH",
    "PYTHONHASHSEED",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONWARNINGS",
];

/// Build the isolated child environment from the current process environment:
/// allowlisted non-empty parent vars, then explicit overlays (which may
/// introduce non-allowlisted keys).
///
/// Overlay keys replace allowlisted values on conflict. This is the deliberate
/// opt-in escape hatch used by [`WorkerProcess::spawn`].
#[must_use]
pub fn build_isolated_child_env(overlays: &[(OsString, OsString)]) -> Vec<(OsString, OsString)> {
    build_isolated_child_env_from(std::env::vars_os(), overlays)
}

/// Filter an arbitrary parent environment through [`CHILD_ENV_ALLOWLIST`], then
/// apply `overlays` last (overlay keys win; may introduce non-allowlisted keys).
///
/// Prefer [`build_isolated_child_env`] at spawn sites. This entry point exists
/// so hosts/tests can reason about isolation without mutating process env.
#[must_use]
pub fn build_isolated_child_env_from<I, K, V>(
    parent: I,
    overlays: &[(OsString, OsString)],
) -> Vec<(OsString, OsString)>
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<OsString>,
    V: Into<OsString>,
{
    use std::collections::HashMap;

    let parent_map: HashMap<OsString, OsString> = parent
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect();

    let mut base = Vec::with_capacity(CHILD_ENV_ALLOWLIST.len() + overlays.len());
    for key in CHILD_ENV_ALLOWLIST {
        if let Some(val) = parent_map.get(OsStr::new(key))
            && !val.is_empty()
        {
            base.push((OsString::from(*key), val.clone()));
        }
    }
    merge_env_overlays(base, overlays)
}

fn merge_env_overlays(
    mut base: Vec<(OsString, OsString)>,
    overlays: &[(OsString, OsString)],
) -> Vec<(OsString, OsString)> {
    for (k, v) in overlays {
        if let Some(slot) = base.iter_mut().find(|(bk, _)| bk == k) {
            slot.1 = v.clone();
        } else {
            base.push((k.clone(), v.clone()));
        }
    }
    base
}

/// Explicit spawn specification — never passed through a shell.
#[derive(Debug, Clone)]
pub struct WorkerSpawnConfig {
    /// Absolute or PATH-resolved executable.
    pub program: PathBuf,
    /// Argument vector (not shell-joined).
    pub args: Vec<OsString>,
    /// Explicit environment overlays applied **after** the isolated allowlist
    /// (`env_clear` + [`CHILD_ENV_ALLOWLIST`] restored from the parent).
    ///
    /// This is the deliberate opt-in escape hatch for keys such as
    /// `PYTHONPATH`, device/venv hints, or a credential-bearing proxy —
    /// never auto-inherited from the parent. Overlay keys win on conflicts.
    pub env: Vec<(OsString, OsString)>,
    /// Optional working directory.
    pub current_dir: Option<PathBuf>,
    /// Handshake timeout.
    pub handshake_timeout: Duration,
    /// Graceful close + wait timeout.
    pub shutdown_timeout: Duration,
    /// Forced kill wait timeout.
    pub kill_timeout: Duration,
    /// Max stderr bytes retained (redacted).
    pub stderr_tail_bytes: usize,
    /// Client name sent in `hello`.
    pub client_name: String,
}

impl WorkerSpawnConfig {
    /// Build a config for `program` with no args.
    #[must_use]
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            env: Vec::new(),
            current_dir: None,
            handshake_timeout: DEFAULT_HANDSHAKE_TIMEOUT,
            shutdown_timeout: DEFAULT_SHUTDOWN_TIMEOUT,
            kill_timeout: DEFAULT_KILL_TIMEOUT,
            stderr_tail_bytes: DEFAULT_STDERR_TAIL_BYTES,
            client_name: "shuvoice".into(),
        }
    }

    #[must_use]
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    #[must_use]
    pub fn env_pair(mut self, key: impl Into<OsString>, val: impl Into<OsString>) -> Self {
        self.env.push((key.into(), val.into()));
        self
    }

    #[must_use]
    pub fn current_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.current_dir = Some(dir.into());
        self
    }

    #[must_use]
    pub fn client_name(mut self, name: impl Into<String>) -> Self {
        self.client_name = name.into();
        self
    }

    #[must_use]
    pub fn handshake_timeout(mut self, d: Duration) -> Self {
        self.handshake_timeout = d;
        self
    }

    #[must_use]
    pub fn shutdown_timeout(mut self, d: Duration) -> Self {
        self.shutdown_timeout = d;
        self
    }

    #[must_use]
    pub fn kill_timeout(mut self, d: Duration) -> Self {
        self.kill_timeout = d;
        self
    }

    #[must_use]
    pub fn stderr_tail_bytes(mut self, n: usize) -> Self {
        self.stderr_tail_bytes = n;
        self
    }

    #[must_use]
    pub fn program(&self) -> &Path {
        &self.program
    }
}

/// Exit information for a worker child.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerExitStatus {
    pub code: Option<i32>,
    pub success: bool,
    /// Redacted stderr tail captured from the child.
    pub stderr_tail: String,
}

/// A live worker process with an established protocol session.
pub struct WorkerProcess {
    child: Child,
    client: WorkerClient<ChildStdout, ChildStdin>,
    session: NegotiatedSession,
    stderr_task: JoinHandle<String>,
    config: WorkerSpawnConfig,
    /// Process id at spawn time (for orphan diagnostics).
    pid: Option<u32>,
}

impl WorkerProcess {
    /// Spawn `config.program` with explicit argv and complete the v1 handshake.
    ///
    /// The child environment is isolated: [`Command::env_clear`], then
    /// [`CHILD_ENV_ALLOWLIST`] restored from the parent, then
    /// [`WorkerSpawnConfig::env`] overlays last. Never a shell.
    pub async fn spawn(config: WorkerSpawnConfig) -> Result<Self, WorkerProcessError> {
        let mut command = Command::new(&config.program);
        command
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .env_clear();

        if let Some(dir) = &config.current_dir {
            command.current_dir(dir);
        }
        for (k, v) in build_isolated_child_env(&config.env) {
            command.env(k, v);
        }

        let mut child = command.spawn().map_err(WorkerProcessError::Spawn)?;
        let pid = child.id();

        let stdin = child.stdin.take().ok_or_else(|| {
            WorkerProcessError::Spawn(std::io::Error::other("missing stdin pipe"))
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            WorkerProcessError::Spawn(std::io::Error::other("missing stdout pipe"))
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            WorkerProcessError::Spawn(std::io::Error::other("missing stderr pipe"))
        })?;

        let max_stderr = config.stderr_tail_bytes;
        let stderr_task = tokio::spawn(async move { read_stderr_tail(stderr, max_stderr).await });

        let mut client = WorkerClient::new(stdout, stdin);
        let handshake = timeout(
            config.handshake_timeout,
            client.handshake(config.client_name.clone()),
        )
        .await;

        let session = match handshake {
            Err(_) => {
                // Timed out — kill child and surface typed error.
                let stderr_tail =
                    kill_and_collect(&mut child, stderr_task, config.kill_timeout).await;
                return Err(WorkerProcessError::HandshakeTimeout { stderr_tail });
            }
            Ok(Err(err)) => {
                let stderr_tail =
                    kill_and_collect(&mut child, stderr_task, config.kill_timeout).await;
                return Err(map_handshake_error(err, stderr_tail));
            }
            Ok(Ok(session)) => session.clone(),
        };

        Ok(Self {
            child,
            client,
            session,
            stderr_task,
            config,
            pid,
        })
    }

    #[must_use]
    pub fn session(&self) -> &NegotiatedSession {
        &self.session
    }

    #[must_use]
    pub fn client(&self) -> &WorkerClient<ChildStdout, ChildStdin> {
        &self.client
    }

    #[must_use]
    pub fn client_mut(&mut self) -> &mut WorkerClient<ChildStdout, ChildStdin> {
        &mut self.client
    }

    #[must_use]
    pub fn pid(&self) -> Option<u32> {
        self.pid
    }

    #[must_use]
    pub fn config(&self) -> &WorkerSpawnConfig {
        &self.config
    }

    /// Non-blocking poll: `Ok(None)` if still running.
    pub fn try_status(&mut self) -> Result<Option<std::process::ExitStatus>, WorkerProcessError> {
        self.child.try_wait().map_err(WorkerProcessError::Io)
    }

    /// Graceful close → wait → timed kill. Consumes the process.
    pub async fn shutdown(mut self) -> Result<WorkerExitStatus, WorkerProcessError> {
        let shutdown_timeout = self.config.shutdown_timeout;
        let kill_timeout = self.config.kill_timeout;

        // Best-effort protocol close; ignore protocol errors (peer may already be dead).
        let _ = timeout(shutdown_timeout, self.client.close()).await;

        // Drop pipes so the child sees EOF on stdin.
        // Client is dropped by moving fields out.
        let WorkerProcess {
            mut child,
            client,
            stderr_task,
            ..
        } = self;
        drop(client);

        match timeout(shutdown_timeout, child.wait()).await {
            Ok(Ok(status)) => {
                let stderr_tail = finish_stderr(stderr_task).await;
                Ok(WorkerExitStatus {
                    code: status.code(),
                    success: status.success(),
                    stderr_tail,
                })
            }
            Ok(Err(err)) => {
                let stderr_tail = finish_stderr(stderr_task).await;
                Err(WorkerProcessError::IoWithStderr {
                    source: err,
                    stderr_tail,
                })
            }
            Err(_elapsed) => {
                let _ = child.start_kill();
                match timeout(kill_timeout, child.wait()).await {
                    Ok(Ok(status)) => {
                        let stderr_tail = finish_stderr(stderr_task).await;
                        Ok(WorkerExitStatus {
                            code: status.code(),
                            success: status.success(),
                            stderr_tail,
                        })
                    }
                    Ok(Err(err)) => {
                        let stderr_tail = finish_stderr(stderr_task).await;
                        Err(WorkerProcessError::IoWithStderr {
                            source: err,
                            stderr_tail,
                        })
                    }
                    Err(_) => {
                        let stderr_tail = finish_stderr(stderr_task).await;
                        Err(WorkerProcessError::ShutdownTimeout { stderr_tail })
                    }
                }
            }
        }
    }

    /// Force-kill without protocol close (crash recovery path).
    pub async fn kill(self) -> WorkerExitStatus {
        let kill_timeout = self.config.kill_timeout;
        let WorkerProcess {
            mut child,
            client,
            stderr_task,
            ..
        } = self;
        drop(client);
        let _ = child.start_kill();
        let status = timeout(kill_timeout, child.wait()).await;
        let stderr_tail = finish_stderr(stderr_task).await;
        match status {
            Ok(Ok(status)) => WorkerExitStatus {
                code: status.code(),
                success: status.success(),
                stderr_tail,
            },
            _ => WorkerExitStatus {
                code: None,
                success: false,
                stderr_tail,
            },
        }
    }
}

// Child is kill_on_drop; dropping WorkerProcess without shutdown still kills.
// Explicit Drop is not required beyond field drops.

async fn read_stderr_tail(mut stderr: impl AsyncReadExt + Unpin, max_bytes: usize) -> String {
    let mut ring: VecDeque<u8> = VecDeque::new();
    let mut buf = [0u8; 512];
    loop {
        match stderr.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => stderr_tail::push_tail(&mut ring, max_bytes, &buf[..n]),
            Err(_) => break,
        }
    }
    let bytes: Vec<u8> = ring.into_iter().collect();
    stderr_tail::redact_stderr_tail(&bytes)
}

async fn finish_stderr(task: JoinHandle<String>) -> String {
    match timeout(Duration::from_secs(1), task).await {
        Ok(Ok(s)) => s,
        _ => String::new(),
    }
}

async fn kill_and_collect(
    child: &mut Child,
    stderr_task: JoinHandle<String>,
    kill_timeout: Duration,
) -> String {
    let _ = child.start_kill();
    let _ = timeout(kill_timeout, child.wait()).await;
    finish_stderr(stderr_task).await
}

fn map_handshake_error(err: ProtocolError, stderr_tail: String) -> WorkerProcessError {
    match err {
        ProtocolError::UnsupportedVersion { remote, local } => {
            WorkerProcessError::UnsupportedVersion {
                remote,
                local,
                stderr_tail,
            }
        }
        ProtocolError::Handshake(message) => {
            let lower = message.to_ascii_lowercase();
            if lower.contains("unsupported_version")
                || (lower.contains("unsupported") && lower.contains("version"))
            {
                // Prefer typed version errors; remote may be unknown if only a string.
                WorkerProcessError::UnsupportedVersion {
                    remote: 0,
                    local: crate::limits::PROTOCOL_VERSION,
                    stderr_tail,
                }
            } else if lower.contains("dependency_missing") || lower.starts_with("dependency") {
                WorkerProcessError::DependencyMissing {
                    message,
                    stderr_tail,
                }
            } else {
                WorkerProcessError::Handshake {
                    message,
                    stderr_tail,
                }
            }
        }
        ProtocolError::Worker { code, message, .. }
            if code == "dependency_missing" || code == "dependency" =>
        {
            WorkerProcessError::DependencyMissing {
                message,
                stderr_tail,
            }
        }
        other => WorkerProcessError::Handshake {
            message: other.to_string(),
            stderr_tail,
        },
    }
}

/// Helper for tests/docs: format program+args without shell quoting secrets.
#[must_use]
pub fn format_argv_for_log(program: &Path, args: &[impl AsRef<OsStr>]) -> String {
    let mut parts = vec![program.display().to_string()];
    for a in args {
        parts.push(a.as_ref().to_string_lossy().into_owned());
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(s: &str) -> OsString {
        OsString::from(s)
    }

    fn parent(pairs: &[(&str, &str)]) -> Vec<(OsString, OsString)> {
        pairs.iter().map(|(k, v)| (os(k), os(v))).collect()
    }

    #[test]
    fn child_env_allowlist_excludes_secrets_and_proxies() {
        assert!(CHILD_ENV_ALLOWLIST.contains(&"PATH"));
        assert!(CHILD_ENV_ALLOWLIST.contains(&"HOME"));
        assert!(CHILD_ENV_ALLOWLIST.contains(&"LANG"));
        assert!(CHILD_ENV_ALLOWLIST.contains(&"XDG_CACHE_HOME"));
        assert!(CHILD_ENV_ALLOWLIST.contains(&"SSL_CERT_FILE"));
        assert!(CHILD_ENV_ALLOWLIST.contains(&"CUDA_VISIBLE_DEVICES"));
        assert!(CHILD_ENV_ALLOWLIST.contains(&"OMP_NUM_THREADS"));
        assert!(CHILD_ENV_ALLOWLIST.contains(&"HF_HOME"));
        assert!(CHILD_ENV_ALLOWLIST.contains(&"TOKENIZERS_PARALLELISM"));
        assert!(!CHILD_ENV_ALLOWLIST.iter().any(|k| k.contains("API_KEY")));
        assert!(!CHILD_ENV_ALLOWLIST.iter().any(|k| {
            *k != "TOKENIZERS_PARALLELISM" && (k.contains("TOKEN") || k.ends_with("_TOKEN"))
        }));
        assert!(!CHILD_ENV_ALLOWLIST.iter().any(|k| k.contains("SECRET")));
        assert!(!CHILD_ENV_ALLOWLIST.contains(&"HTTP_PROXY"));
        assert!(!CHILD_ENV_ALLOWLIST.contains(&"HTTPS_PROXY"));
        assert!(!CHILD_ENV_ALLOWLIST.contains(&"http_proxy"));
        assert!(!CHILD_ENV_ALLOWLIST.contains(&"https_proxy"));
        assert!(!CHILD_ENV_ALLOWLIST.contains(&"ALL_PROXY"));
        assert!(!CHILD_ENV_ALLOWLIST.contains(&"NO_PROXY"));
        assert!(!CHILD_ENV_ALLOWLIST.contains(&"SSH_AUTH_SOCK"));
        assert!(!CHILD_ENV_ALLOWLIST.contains(&"OPENAI_API_KEY"));
        assert!(!CHILD_ENV_ALLOWLIST.contains(&"ELEVENLABS_API_KEY"));
        assert!(!CHILD_ENV_ALLOWLIST.contains(&"AWS_SECRET_ACCESS_KEY"));
        assert!(!CHILD_ENV_ALLOWLIST.contains(&"GH_TOKEN"));
        assert!(!CHILD_ENV_ALLOWLIST.contains(&"GITHUB_TOKEN"));
    }

    #[test]
    fn build_isolated_child_env_from_strips_sentinels_preserves_allowlist_overlay_wins() {
        let parent_env = parent(&[
            ("PATH", "/usr/bin:/bin"),
            ("HOME", "/home/test"),
            ("LANG", "C.UTF-8"),
            ("SSL_CERT_FILE", "/tmp/shuvoice-test-ca-bundle.pem"),
            ("XDG_CACHE_HOME", "/tmp/shuvoice-test-xdg-cache"),
            ("HF_HOME", "/tmp/shuvoice-test-hf-home"),
            ("CUDA_VISIBLE_DEVICES", "0"),
            ("OMP_NUM_THREADS", "4"),
            ("OPENAI_API_KEY", "sk-test-sentinel-openai"),
            ("ELEVENLABS_API_KEY", "el-test-sentinel-eleven"),
            ("SSH_AUTH_SOCK", "/tmp/ssh-agent.sock"),
            ("AWS_SECRET_ACCESS_KEY", "aws-secret-sentinel"),
            ("AWS_ACCESS_KEY_ID", "aws-key-sentinel"),
            ("GH_TOKEN", "gh-sentinel"),
            ("GITHUB_TOKEN", "ghs-sentinel"),
            ("SOME_SERVICE_TOKEN", "generic-token-sentinel"),
            ("MY_SECRET", "generic-secret-sentinel"),
            (
                "HTTP_PROXY",
                "http://proxy-user:proxy-pass-sentinel@127.0.0.1:8080",
            ),
            (
                "HTTPS_PROXY",
                "http://proxy-user:proxy-pass-sentinel@127.0.0.1:8443",
            ),
            (
                "http_proxy",
                "http://proxy-user:proxy-pass-sentinel@127.0.0.1:8080",
            ),
            (
                "https_proxy",
                "http://proxy-user:proxy-pass-sentinel@127.0.0.1:8443",
            ),
            (
                "ALL_PROXY",
                "socks5://proxy-user:proxy-pass-sentinel@127.0.0.1:1080",
            ),
            // Empty allowlisted values are dropped.
            ("TMPDIR", ""),
        ]);

        let env = build_isolated_child_env_from(
            parent_env.clone(),
            &[(os("PYTHONPATH"), os("/workers"))],
        );
        let keys: Vec<String> = env
            .iter()
            .map(|(k, _)| k.to_string_lossy().into_owned())
            .collect();

        assert!(keys.iter().any(|k| k == "PATH"));
        assert!(keys.iter().any(|k| k == "HOME"));
        assert!(keys.iter().any(|k| k == "LANG"));
        assert!(keys.iter().any(|k| k == "PYTHONPATH"));
        assert!(keys.iter().any(|k| k == "CUDA_VISIBLE_DEVICES"));
        assert!(keys.iter().any(|k| k == "OMP_NUM_THREADS"));
        assert!(
            !keys.iter().any(|k| k == "TMPDIR"),
            "empty values must drop"
        );
        assert_eq!(
            env.iter()
                .find(|(k, _)| k == "SSL_CERT_FILE")
                .map(|(_, v)| v.to_string_lossy().into_owned())
                .as_deref(),
            Some("/tmp/shuvoice-test-ca-bundle.pem")
        );
        assert_eq!(
            env.iter()
                .find(|(k, _)| k == "XDG_CACHE_HOME")
                .map(|(_, v)| v.to_string_lossy().into_owned())
                .as_deref(),
            Some("/tmp/shuvoice-test-xdg-cache")
        );
        assert_eq!(
            env.iter()
                .find(|(k, _)| k == "HF_HOME")
                .map(|(_, v)| v.to_string_lossy().into_owned())
                .as_deref(),
            Some("/tmp/shuvoice-test-hf-home")
        );

        for banned in [
            "OPENAI_API_KEY",
            "ELEVENLABS_API_KEY",
            "SSH_AUTH_SOCK",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_ACCESS_KEY_ID",
            "GH_TOKEN",
            "GITHUB_TOKEN",
            "SOME_SERVICE_TOKEN",
            "MY_SECRET",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "ALL_PROXY",
        ] {
            assert!(
                !keys.iter().any(|k| k == banned),
                "banned key {banned} leaked into isolated env: {keys:?}"
            );
        }
        assert!(
            env.iter()
                .all(|(_, v)| !v.to_string_lossy().contains("proxy-pass-sentinel")),
            "credential-bearing proxy value leaked: {env:?}"
        );
        assert!(
            env.iter()
                .all(|(_, v)| !v.to_string_lossy().contains("sk-test-sentinel")),
            "api key sentinel leaked: {env:?}"
        );

        // Explicit overlay is the deliberate opt-in path and wins on conflict.
        let env_opt_in = build_isolated_child_env_from(
            parent_env,
            &[
                (os("HTTPS_PROXY"), os("http://explicit-only@127.0.0.1:9")),
                (os("PATH"), os("/explicit/bin:/usr/bin")),
            ],
        );
        assert!(
            env_opt_in.iter().any(|(k, v)| {
                k == "HTTPS_PROXY" && v.to_string_lossy().contains("explicit-only")
            })
        );
        assert_eq!(
            env_opt_in
                .iter()
                .find(|(k, _)| k == "PATH")
                .map(|(_, v)| v.to_string_lossy().into_owned())
                .as_deref(),
            Some("/explicit/bin:/usr/bin")
        );
        assert_eq!(
            env.iter()
                .find(|(k, _)| k == "PYTHONPATH")
                .map(|(_, v)| v.to_string_lossy().into_owned())
                .as_deref(),
            Some("/workers")
        );
    }

    #[test]
    fn build_isolated_child_env_reads_process_env_allowlist_only() {
        // Smoke: live process env path returns only allowlisted keys when no overlays.
        let env = build_isolated_child_env(&[]);
        for (k, _) in &env {
            let key = k.to_string_lossy();
            assert!(
                CHILD_ENV_ALLOWLIST.iter().any(|a| *a == key),
                "live builder emitted non-allowlisted key {key}"
            );
        }
    }
}
