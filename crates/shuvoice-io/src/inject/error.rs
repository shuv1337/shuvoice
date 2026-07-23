//! Typed injection outcomes.
//!
//! # Privacy
//!
//! [`InjectError`] and [`CommitOutcome`] never carry transcript or clipboard
//! payload bytes. Their `Display` / `Debug` implementations are static labels
//! only, so logs and UI toasts cannot accidentally leak dictated text.

use thiserror::Error;

/// Text-injection failure.
///
/// Variants describe *which step* failed so the app composition can decide
/// whether to retry, latch, or surface a toast. No variant holds user text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum InjectError {
    /// Could not erase previously injected partial text.
    #[error("partial backspace failed")]
    PartialBackspace,

    /// Erased (or had nothing to erase) but could not type the new partial suffix.
    #[error("partial type failed")]
    PartialType,

    /// Direct-mode final injection failed (backspace and/or type).
    #[error("direct type failed")]
    DirectType,

    /// Clipboard paste failed and the direct-typing fallback also failed.
    ///
    /// Safe to retry: no final text was confirmed inserted.
    #[error("clipboard paste and direct fallback both failed")]
    ClipboardInjectFailed,

    /// Payload exceeds the argv safety ceiling used by direct type backends.
    #[error("payload too large for argv injection")]
    PayloadTooLarge,

    /// Generic injection failure (reserved / catch-all).
    #[error("text injection failed")]
    Failed,
}

impl InjectError {
    /// `true` when retrying the same commit is safe (text was not confirmed inserted).
    #[must_use]
    pub const fn is_retryable(self) -> bool {
        // All current Err variants are retryable. Clipboard-restore issues are
        // returned as `Ok(CommitOutcome::CommittedClipboardNotRestored)` instead.
        true
    }
}

/// Successful final-commit outcome.
///
/// Composition **must latch** utterance commit on any `Ok` variant and **must
/// not retry** when the variant is [`CommittedClipboardNotRestored`] — the
/// focused window already contains the inserted text; retrying would duplicate it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitOutcome {
    /// Final text inserted; clipboard preservation (if any) restored cleanly.
    Committed,
    /// Final text inserted, but restoring the prior clipboard failed.
    ///
    /// Surface a non-fatal warning. Do **not** retry the commit.
    CommittedClipboardNotRestored,
}

impl CommitOutcome {
    /// Always `true` — both variants mean text landed in the target window.
    #[must_use]
    pub const fn text_inserted(self) -> bool {
        true
    }

    /// `true` only when preservation restore succeeded (or was not requested).
    #[must_use]
    pub const fn clipboard_restored(self) -> bool {
        matches!(self, Self::Committed)
    }

    /// `true` when composition should treat this as a soft warning, not a hard error.
    #[must_use]
    pub const fn needs_clipboard_warning(self) -> bool {
        matches!(self, Self::CommittedClipboardNotRestored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inject_error_display_has_no_payload_shaped_fields() {
        // Static labels only — guard against future variants accidentally
        // formatting user text via Debug/Display.
        let variants = [
            InjectError::PartialBackspace,
            InjectError::PartialType,
            InjectError::DirectType,
            InjectError::ClipboardInjectFailed,
            InjectError::PayloadTooLarge,
            InjectError::Failed,
        ];
        for v in variants {
            let d = format!("{v}");
            let dbg = format!("{v:?}");
            assert!(!d.is_empty());
            assert!(!dbg.is_empty());
            // No debug struct fields that could hold strings.
            assert!(
                !dbg.contains('{'),
                "InjectError Debug must stay field-less for privacy: {dbg}"
            );
            assert!(v.is_retryable());
        }
    }

    #[test]
    fn commit_outcome_latch_semantics() {
        assert!(CommitOutcome::Committed.text_inserted());
        assert!(CommitOutcome::Committed.clipboard_restored());
        assert!(!CommitOutcome::Committed.needs_clipboard_warning());

        assert!(CommitOutcome::CommittedClipboardNotRestored.text_inserted());
        assert!(!CommitOutcome::CommittedClipboardNotRestored.clipboard_restored());
        assert!(CommitOutcome::CommittedClipboardNotRestored.needs_clipboard_warning());
    }
}
