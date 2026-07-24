//! App-facing text injection and selection adapters over `shuvoice-io`.
//!
//! # Design
//!
//! - [`IoTextInjector`] wraps a mutex-guarded [`StreamingTyper`] (or any
//!   [`SyncInjectBackend`]) and runs blocking subprocess work on Tokio's
//!   `spawn_blocking` pool so the session actor never stalls.
//! - Final commit maps **any** [`CommitOutcome`] (including
//!   [`CommitOutcome::CommittedClipboardNotRestored`]) to `Ok(())` so the
//!   session latches and does **not** retry (retry would duplicate text).
//! - Clipboard-restore soft failures bump a counter and invoke an optional
//!   payload-free callback; they never surface as `Err`.
//! - [`InjectError`] is always treated as retryable and mapped to `Err(String)`
//!   using the error's static `Display` label only — never transcript bytes.
//! - [`IoSelection`] similarly offloads `wl-paste` work via `spawn_blocking`.
//!
//! # Integration notes
//!
//! Expected crate deps (declared by the integration owner):
//! - `shuvoice-app` (traits)
//! - `shuvoice-io` (StreamingTyper / SelectionCapture)
//! - `shuvoice-core` (Config → TyperConfig helper)
//! - `async-trait`, `tokio` (already in `shuvoice-cli`)
//!
//! No extra features required for this module.

#![allow(clippy::collapsible_if)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::let_and_return)]
#![allow(clippy::double_must_use)]
#![allow(clippy::result_unit_err)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use shuvoice_app::traits::{SelectionCapture as AppSelectionCapture, TextInjector};
use shuvoice_core::{Config, InjectionMode};
use shuvoice_io::{
    CommitOutcome, FinalInjectionMode, InjectError, SelectionCapture as IoSelectionCapture,
    SelectionError, StreamingTyper, TyperConfig,
};
use tracing::warn;

/// Payload-free hook fired when final text landed but clipboard restore failed.
pub type ClipboardWarningHook = Arc<dyn Fn() + Send + Sync + 'static>;

/// Synchronous inject backend. Production uses [`StreamingTyper`]; tests use fakes.
pub trait SyncInjectBackend: Send + 'static {
    fn update_partial(&mut self, text: &str) -> Result<(), InjectError>;
    fn commit_final(&mut self, text: &str) -> Result<CommitOutcome, InjectError>;
    fn reset(&mut self) -> Result<(), InjectError>;
}

impl SyncInjectBackend for StreamingTyper {
    fn update_partial(&mut self, text: &str) -> Result<(), InjectError> {
        StreamingTyper::update_partial(self, text)
    }

    fn commit_final(&mut self, text: &str) -> Result<CommitOutcome, InjectError> {
        StreamingTyper::commit_final(self, text)
    }

    fn reset(&mut self) -> Result<(), InjectError> {
        StreamingTyper::reset(self)
    }
}

/// Build a [`TyperConfig`] from the validated runtime [`Config`].
#[must_use]
pub fn typer_config_from_app_config(cfg: &Config) -> TyperConfig {
    TyperConfig {
        final_injection_mode: map_injection_mode(cfg.typing_final_injection_mode),
        preserve_clipboard: cfg.preserve_clipboard,
        clipboard_settle_delay: Duration::from_millis(u64::from(
            cfg.typing_clipboard_settle_delay_ms,
        )),
        retry_attempts: cfg.typing_retry_attempts,
        retry_delay: Duration::from_millis(u64::from(cfg.typing_retry_delay_ms)),
        subprocess_timeout: Duration::from_secs_f64(cfg.typing_subprocess_timeout.max(1.0)),
    }
}

#[must_use]
pub fn map_injection_mode(mode: InjectionMode) -> FinalInjectionMode {
    match mode {
        InjectionMode::Auto => FinalInjectionMode::Auto,
        InjectionMode::Clipboard => FinalInjectionMode::Clipboard,
        InjectionMode::Direct => FinalInjectionMode::Direct,
    }
}

/// `TextInjector` adapter: mutex + `spawn_blocking` over a sync inject backend.
pub struct IoTextInjector<B: SyncInjectBackend = StreamingTyper> {
    backend: Arc<Mutex<B>>,
    clipboard_warnings: Arc<AtomicU64>,
    on_clipboard_warning: Option<ClipboardWarningHook>,
}

impl IoTextInjector<StreamingTyper> {
    /// Construct with a live [`StreamingTyper`].
    #[must_use]
    pub fn new(typer: StreamingTyper) -> Self {
        Self::from_backend(typer)
    }

    /// Construct from validated app config (default process runner).
    #[must_use]
    pub fn from_config(cfg: &Config) -> Self {
        Self::new(StreamingTyper::new(
            typer_config_from_app_config(cfg),
            Arc::new(shuvoice_io::StdCommandRunner),
        ))
    }

    /// Default typer settings (tests / fallbacks).
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(StreamingTyper::with_defaults())
    }
}

impl<B: SyncInjectBackend> IoTextInjector<B> {
    /// Construct around any [`SyncInjectBackend`] (production or test fake).
    #[must_use]
    pub fn from_backend(backend: B) -> Self {
        Self {
            backend: Arc::new(Mutex::new(backend)),
            clipboard_warnings: Arc::new(AtomicU64::new(0)),
            on_clipboard_warning: None,
        }
    }

    /// Install a payload-free soft-warning callback for clipboard-restore failure.
    #[must_use]
    pub fn with_clipboard_warning_hook(mut self, hook: impl Fn() + Send + Sync + 'static) -> Self {
        self.on_clipboard_warning = Some(Arc::new(hook));
        self
    }

    /// Soft clipboard-restore warning count (committed text, restore failed).
    #[must_use]
    pub fn clipboard_warning_count(&self) -> u64 {
        self.clipboard_warnings.load(Ordering::Relaxed)
    }

    /// Shared counter handle (metrics / diagnostics).
    #[must_use]
    pub fn clipboard_warning_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.clipboard_warnings)
    }

    fn map_inject_err(err: InjectError) -> String {
        // InjectError Display is a static label (no payload fields).
        err.to_string()
    }
}

#[async_trait]
impl<B: SyncInjectBackend> TextInjector for IoTextInjector<B> {
    async fn update_partial(&self, text: &str) -> Result<(), String> {
        let backend = Arc::clone(&self.backend);
        // Own the text so the blocking task does not borrow across await.
        let owned = text.to_owned();
        let result = tokio::task::spawn_blocking(move || {
            let mut guard = backend
                .lock()
                .map_err(|_| "text injector lock poisoned".to_string())?;
            guard.update_partial(&owned).map_err(Self::map_inject_err)
        })
        .await
        .map_err(|_| "text injector task failed".to_string())?;
        result
    }

    async fn commit_final(&self, text: &str) -> Result<(), String> {
        let backend = Arc::clone(&self.backend);
        let warnings = Arc::clone(&self.clipboard_warnings);
        let hook = self.on_clipboard_warning.clone();
        let owned = text.to_owned();
        let result = tokio::task::spawn_blocking(move || {
            let mut guard = backend
                .lock()
                .map_err(|_| "text injector lock poisoned".to_string())?;
            match guard.commit_final(&owned) {
                Ok(outcome) => {
                    if outcome.needs_clipboard_warning() {
                        warnings.fetch_add(1, Ordering::Relaxed);
                        // Static label only — never include clipboard or transcript bytes.
                        warn!("final inject committed but clipboard restore failed");
                        if let Some(hook) = hook {
                            hook();
                        }
                    }
                    Ok(())
                }
                Err(err) => Err(Self::map_inject_err(err)),
            }
        })
        .await
        .map_err(|_| "text injector task failed".to_string())?;
        result
    }

    async fn reset(&self) -> Result<(), String> {
        let backend = Arc::clone(&self.backend);
        let result = tokio::task::spawn_blocking(move || {
            let mut guard = backend
                .lock()
                .map_err(|_| "text injector lock poisoned".to_string())?;
            guard.reset().map_err(Self::map_inject_err)
        })
        .await
        .map_err(|_| "text injector task failed".to_string())?;
        result
    }
}

/// `SelectionCapture` adapter over `shuvoice_io::SelectionCapture`.
#[derive(Clone)]
pub struct IoSelection {
    inner: Arc<IoSelectionCapture>,
}

impl IoSelection {
    #[must_use]
    pub fn new(inner: IoSelectionCapture) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(IoSelectionCapture::default())
    }

    #[must_use]
    pub fn from_runner(runner: Arc<dyn shuvoice_io::CommandRunner>) -> Self {
        Self::new(IoSelectionCapture::new(runner))
    }
}

fn map_selection_err(err: SelectionError) -> String {
    // SelectionError variants are static labels (no captured text payload).
    err.to_string()
}

#[async_trait]
impl AppSelectionCapture for IoSelection {
    async fn capture_selection(&self) -> Result<String, String> {
        let inner = Arc::clone(&self.inner);
        let result = tokio::task::spawn_blocking(move || {
            inner.capture_selection().map_err(map_selection_err)
        })
        .await
        .map_err(|_| "selection capture task failed".to_string())?;
        result
    }

    async fn capture_clipboard(&self) -> Result<String, String> {
        let inner = Arc::clone(&self.inner);
        let result = tokio::task::spawn_blocking(move || {
            inner.capture_clipboard().map_err(map_selection_err)
        })
        .await
        .map_err(|_| "clipboard capture task failed".to_string())?;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[derive(Debug, Default)]
    struct FakeBackend {
        partials: Vec<String>,
        finals: Vec<String>,
        resets: u32,
        next_commit: Option<Result<CommitOutcome, InjectError>>,
        next_partial: Option<Result<(), InjectError>>,
    }

    impl SyncInjectBackend for FakeBackend {
        fn update_partial(&mut self, text: &str) -> Result<(), InjectError> {
            if let Some(res) = self.next_partial.take() {
                if res.is_ok() {
                    self.partials.push(text.to_string());
                }
                return res;
            }
            self.partials.push(text.to_string());
            Ok(())
        }

        fn commit_final(&mut self, text: &str) -> Result<CommitOutcome, InjectError> {
            if let Some(res) = self.next_commit.take() {
                if res.is_ok() {
                    self.finals.push(text.to_string());
                }
                return res;
            }
            self.finals.push(text.to_string());
            Ok(CommitOutcome::Committed)
        }

        fn reset(&mut self) -> Result<(), InjectError> {
            self.resets += 1;
            self.partials.clear();
            Ok(())
        }
    }

    #[tokio::test]
    async fn commit_ok_latches_without_warning() {
        let inj = IoTextInjector::from_backend(FakeBackend::default());
        assert!(inj.commit_final("hello world").await.is_ok());
        assert_eq!(inj.clipboard_warning_count(), 0);
    }

    #[tokio::test]
    async fn clipboard_not_restored_is_ok_soft_warning() {
        let hooks = Arc::new(AtomicUsize::new(0));
        let hooks2 = Arc::clone(&hooks);
        let mut backend = FakeBackend::default();
        backend.next_commit = Some(Ok(CommitOutcome::CommittedClipboardNotRestored));
        let inj = IoTextInjector::from_backend(backend).with_clipboard_warning_hook(move || {
            hooks2.fetch_add(1, Ordering::SeqCst);
        });

        // Soft success: session must latch (Ok), no retry signal.
        let res = inj.commit_final("secret transcript").await;
        assert!(
            res.is_ok(),
            "clipboard-not-restored must not be Err: {res:?}"
        );
        assert_eq!(inj.clipboard_warning_count(), 1);
        assert_eq!(hooks.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn commit_err_is_retryable_string_without_payload() {
        let mut backend = FakeBackend::default();
        backend.next_commit = Some(Err(InjectError::ClipboardInjectFailed));
        let inj = IoTextInjector::from_backend(backend);
        let err = inj
            .commit_final("do-not-echo-this-transcript")
            .await
            .expect_err("expected retryable err");
        // Static label only.
        assert_eq!(err, InjectError::ClipboardInjectFailed.to_string());
        assert!(!err.contains("do-not-echo"));
        assert_eq!(inj.clipboard_warning_count(), 0);
    }

    #[tokio::test]
    async fn partial_and_reset_round_trip() {
        let backend = FakeBackend::default();
        let backend_handle = {
            // Access via injector only.
            let inj = IoTextInjector::from_backend(backend);
            inj.update_partial("hi").await.unwrap();
            inj.reset().await.unwrap();
            inj
        };
        // Ensure no panics; warning counter untouched.
        assert_eq!(backend_handle.clipboard_warning_count(), 0);
    }

    #[tokio::test]
    async fn selection_spawn_blocking_maps_empty() {
        use shuvoice_io::{RunOutput, ScriptedRunner};

        let runner = ScriptedRunner::new();
        runner.set_dynamic(|_| {
            Ok(RunOutput {
                status_code: Some(0),
                stdout: b"   ".to_vec(),
                stderr: Vec::new(),
                success: true,
            })
        });
        let sel = IoSelection::from_runner(Arc::new(runner));
        let err = sel.capture_selection().await.unwrap_err();
        assert!(
            err.to_lowercase().contains("no selected") || err.to_lowercase().contains("empty"),
            "unexpected err: {err}"
        );
    }

    #[tokio::test]
    async fn selection_returns_text_without_logging_requirement() {
        use shuvoice_io::{RunOutput, ScriptedRunner};

        let runner = ScriptedRunner::new();
        runner.set_dynamic(|_| {
            Ok(RunOutput {
                status_code: Some(0),
                stdout: b"selected phrase".to_vec(),
                stderr: Vec::new(),
                success: true,
            })
        });
        let sel = IoSelection::from_runner(Arc::new(runner));
        let text = sel.capture_selection().await.unwrap();
        assert_eq!(text, "selected phrase");
    }

    #[test]
    fn typer_config_maps_injection_mode() {
        let mut cfg = Config::default();
        cfg.typing_final_injection_mode = InjectionMode::Direct;
        cfg.preserve_clipboard = true;
        cfg.typing_clipboard_settle_delay_ms = 55;
        let tc = typer_config_from_app_config(&cfg);
        assert_eq!(tc.final_injection_mode, FinalInjectionMode::Direct);
        assert!(tc.preserve_clipboard);
        assert_eq!(tc.clipboard_settle_delay, Duration::from_millis(55));
    }
}
