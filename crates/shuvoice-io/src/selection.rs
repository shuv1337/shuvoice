//! Wayland selection / clipboard capture via `wl-paste`.

use std::sync::Arc;
use std::time::Duration;

use crate::error::{ProcessError, SelectionError};
use crate::process::{CommandRunner, RunOptions, StdCommandRunner, argv};

const SELECTION_TIMEOUT: Duration = Duration::from_secs(2);

/// Capture helpers backed by a [`CommandRunner`].
#[derive(Clone)]
pub struct SelectionCapture {
    runner: Arc<dyn CommandRunner>,
    timeout: Duration,
}

impl Default for SelectionCapture {
    fn default() -> Self {
        Self::new(Arc::new(StdCommandRunner))
    }
}

impl SelectionCapture {
    #[must_use]
    pub fn new(runner: Arc<dyn CommandRunner>) -> Self {
        Self {
            runner,
            timeout: SELECTION_TIMEOUT,
        }
    }

    /// Capture system clipboard only (`wl-paste --no-newline`).
    pub fn capture_clipboard(&self) -> Result<String, SelectionError> {
        match self.capture_wl_paste(&[]) {
            Some(text) => Ok(text),
            None => Err(SelectionError::EmptyClipboard),
        }
    }

    /// Primary selection first, then clipboard fallback.
    pub fn capture_selection(&self) -> Result<String, SelectionError> {
        if let Some(text) = self.capture_wl_paste(&["--primary"]) {
            return Ok(text);
        }
        match self.capture_clipboard() {
            Ok(text) => Ok(text),
            Err(SelectionError::EmptyClipboard) => Err(SelectionError::EmptySelection),
            Err(other) => Err(other),
        }
    }

    fn capture_wl_paste(&self, extra: &[&str]) -> Option<String> {
        let mut args = argv(["wl-paste", "--no-newline"]);
        for e in extra {
            args.push((*e).to_string());
        }
        let opts = RunOptions {
            timeout: self.timeout,
            check: true,
            ..RunOptions::default()
        };
        match self.runner.run(&args, &opts) {
            Ok(out) => {
                let text = out.stdout_lossy();
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            }
            Err(ProcessError::Timeout { .. }) => {
                tracing::warn!("Selection capture command timed out: wl-paste");
                None
            }
            Err(err) => {
                tracing::debug!("Selection capture command failed: {err}");
                None
            }
        }
    }
}

/// Convenience using the default runner.
pub fn capture_clipboard() -> Result<String, SelectionError> {
    SelectionCapture::default().capture_clipboard()
}

/// Convenience using the default runner.
pub fn capture_selection() -> Result<String, SelectionError> {
    SelectionCapture::default().capture_selection()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::{RunOutput, ScriptedRunner};

    #[test]
    fn prefers_primary() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|argv| {
            if argv.iter().any(|a| a == "--primary") {
                Ok(RunOutput {
                    status_code: Some(0),
                    stdout: b"primary text".to_vec(),
                    stderr: Vec::new(),
                    success: true,
                })
            } else {
                Ok(RunOutput {
                    status_code: Some(0),
                    stdout: b"clipboard text".to_vec(),
                    stderr: Vec::new(),
                    success: true,
                })
            }
        });
        let cap = SelectionCapture::new(Arc::new(r.clone()));
        assert_eq!(cap.capture_selection().unwrap(), "primary text");
        let calls = r.calls();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].iter().any(|a| a == "--primary"));
    }

    #[test]
    fn falls_back_to_clipboard() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|argv| {
            if argv.iter().any(|a| a == "--primary") {
                Err(ProcessError::ExitCode {
                    program: "wl-paste".into(),
                    code: 1,
                })
            } else {
                Ok(RunOutput {
                    status_code: Some(0),
                    stdout: b"clipboard text".to_vec(),
                    stderr: Vec::new(),
                    success: true,
                })
            }
        });
        let cap = SelectionCapture::new(Arc::new(r));
        assert_eq!(cap.capture_selection().unwrap(), "clipboard text");
    }

    #[test]
    fn clipboard_only_never_passes_primary() {
        let r = ScriptedRunner::new();
        r.push_ok(b"clipboard text");
        let cap = SelectionCapture::new(Arc::new(r.clone()));
        assert_eq!(cap.capture_clipboard().unwrap(), "clipboard text");
        assert!(
            r.calls()
                .iter()
                .all(|c| !c.iter().any(|a| a == "--primary"))
        );
    }

    #[test]
    fn empty_both_raises() {
        let r = ScriptedRunner::new();
        r.set_dynamic(|_| {
            Ok(RunOutput {
                status_code: Some(0),
                stdout: b"   ".to_vec(),
                stderr: Vec::new(),
                success: true,
            })
        });
        let cap = SelectionCapture::new(Arc::new(r));
        let err = cap.capture_selection().unwrap_err();
        assert!(matches!(err, SelectionError::EmptySelection));
    }
}
