//! ASR circuit-breaker policy (pure state machine).

use std::time::{Duration, Instant};

/// Maximum consecutive ASR failures before opening the breaker.
pub const ASR_MAX_FAILURES: u32 = 10;
/// Cooldown before half-open recovery attempts.
pub const ASR_CIRCUIT_COOLDOWN: Duration = Duration::from_secs(30);
/// Transient overlay error toast duration.
pub const ERROR_TOAST_SECONDS: u64 = 5;

/// Outcome of recording a failure/success against the breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerAction {
    /// Failure counted; still closed.
    Counted { failures: u32 },
    /// Breaker opened after max failures.
    Opened,
    /// Failure ignored (already open, or recovered CUDA path).
    Ignored,
    /// Success cleared failure streak.
    ClosedClear,
}

/// Pure circuit breaker used by the session actor.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    pub max_failures: u32,
    pub cooldown: Duration,
    consecutive_failures: u32,
    disabled: bool,
    open_at: Option<Instant>,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitBreaker {
    pub fn new() -> Self {
        Self {
            max_failures: ASR_MAX_FAILURES,
            cooldown: ASR_CIRCUIT_COOLDOWN,
            consecutive_failures: 0,
            disabled: false,
            open_at: None,
        }
    }

    pub fn is_disabled(&self) -> bool {
        self.disabled
    }

    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }

    pub fn open_at(&self) -> Option<Instant> {
        self.open_at
    }

    /// Record a successful ASR call.
    pub fn on_success(&mut self) -> BreakerAction {
        self.consecutive_failures = 0;
        BreakerAction::ClosedClear
    }

    /// Record a non-recoverable failure. Returns whether the breaker opened.
    pub fn on_failure(&mut self, now: Instant) -> BreakerAction {
        if self.disabled {
            return BreakerAction::Ignored;
        }
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        if self.consecutive_failures >= self.max_failures {
            self.disabled = true;
            self.open_at = Some(now);
            BreakerAction::Opened
        } else {
            BreakerAction::Counted {
                failures: self.consecutive_failures,
            }
        }
    }

    /// CUDA-OOM recovery path: does not advance the failure counter.
    pub fn on_recovered_cuda_fallback(&mut self) -> BreakerAction {
        BreakerAction::Ignored
    }

    /// Force-open (e.g. repeated reset failures during start).
    pub fn force_open(&mut self, now: Instant) {
        self.disabled = true;
        self.open_at = Some(now);
    }

    /// Clear disabled state after a successful recovery reset.
    pub fn close_after_recovery(&mut self) {
        self.disabled = false;
        self.open_at = None;
        self.consecutive_failures = 0;
    }

    /// Whether cooldown has elapsed and a half-open recovery may be attempted.
    pub fn can_attempt_recovery(&self, now: Instant) -> bool {
        if !self.disabled {
            return false;
        }
        match self.open_at {
            Some(open_at) => now.duration_since(open_at) >= self.cooldown,
            None => false,
        }
    }

    /// On failed half-open recovery, refresh open timestamp.
    pub fn bump_open_timestamp(&mut self, now: Instant) {
        if self.disabled {
            self.open_at = Some(now);
        }
    }
}

/// Guard against spurious immediate re-starts during processing.
pub const PTT_REARM_GRACE: Duration = Duration::from_millis(350);

/// Return true when start should be ignored due to rearm grace.
pub fn should_ignore_start_during_rearm(processing: bool, since_stop: Duration) -> bool {
    processing && since_stop < PTT_REARM_GRACE
}

/// Minimum splash visibility after model load.
pub const MIN_SPLASH_VISIBLE: Duration = Duration::from_secs(2);

/// Remaining splash hold time in milliseconds.
pub fn remaining_splash_ms(shown_at: Option<Instant>, min_visible: Duration, now: Instant) -> u64 {
    let Some(shown_at) = shown_at else {
        return 0;
    };
    if min_visible.is_zero() {
        return 0;
    }
    let elapsed = now.saturating_duration_since(shown_at);
    min_visible.saturating_sub(elapsed).as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_after_max_failures() {
        let mut br = CircuitBreaker::new();
        let now = Instant::now();
        for i in 1..ASR_MAX_FAILURES {
            match br.on_failure(now) {
                BreakerAction::Counted { failures } => assert_eq!(failures, i),
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(br.on_failure(now), BreakerAction::Opened);
        assert!(br.is_disabled());
    }

    #[test]
    fn recovery_requires_cooldown() {
        let mut br = CircuitBreaker::new();
        let t0 = Instant::now();
        br.force_open(t0);
        assert!(!br.can_attempt_recovery(t0 + Duration::from_secs(10)));
        assert!(br.can_attempt_recovery(t0 + Duration::from_secs(30)));
        br.close_after_recovery();
        assert!(!br.is_disabled());
    }

    #[test]
    fn rearm_grace() {
        assert!(should_ignore_start_during_rearm(
            true,
            Duration::from_millis(100)
        ));
        assert!(!should_ignore_start_during_rearm(
            true,
            Duration::from_millis(400)
        ));
        assert!(!should_ignore_start_during_rearm(
            false,
            Duration::from_millis(0)
        ));
    }

    #[test]
    fn splash_remaining() {
        let t0 = Instant::now();
        assert_eq!(remaining_splash_ms(None, MIN_SPLASH_VISIBLE, t0), 0);
        assert_eq!(
            remaining_splash_ms(
                Some(t0),
                MIN_SPLASH_VISIBLE,
                t0 + Duration::from_millis(500)
            ),
            1500
        );
        assert_eq!(
            remaining_splash_ms(Some(t0), MIN_SPLASH_VISIBLE, t0 + Duration::from_secs(3)),
            0
        );
    }
}
