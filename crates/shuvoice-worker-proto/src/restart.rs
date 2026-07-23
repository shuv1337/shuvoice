//! Pure restart/backoff state machine (no I/O, no autonomous loops).

use std::time::Duration;

/// Configuration for bounded worker restart decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartPolicy {
    /// Maximum consecutive crash/restart attempts before giving up.
    pub max_attempts: u32,
    /// Backoff applied after the first failure.
    pub initial_backoff: Duration,
    /// Cap on exponential backoff.
    pub max_backoff: Duration,
    /// After the process has been healthy this long, consecutive failures reset.
    pub healthy_window: Duration,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 5,
            initial_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(30),
            healthy_window: Duration::from_secs(60),
        }
    }
}

impl RestartPolicy {
    /// Compute backoff delay after `consecutive_failures` failures (1-based).
    #[must_use]
    pub fn backoff_for_attempt(&self, consecutive_failures: u32) -> Duration {
        if consecutive_failures == 0 {
            return Duration::ZERO;
        }
        let shift = consecutive_failures.saturating_sub(1).min(16);
        let base = self.initial_backoff.as_millis();
        let scaled = base.saturating_mul(1u128 << shift);
        let capped = scaled.min(self.max_backoff.as_millis());
        Duration::from_millis(capped as u64)
    }
}

/// Decision returned by the restart state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartDecision {
    /// Caller may spawn immediately.
    RunNow,
    /// Caller should wait before the next spawn attempt.
    Wait(Duration),
    /// Stop restarting until the supervisor is reset manually.
    GiveUp { consecutive_failures: u32 },
}

/// Deterministic restart bookkeeping. Time is injected for testability.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RestartState {
    /// Total spawn attempts since construction / manual reset.
    pub total_spawns: u32,
    /// Consecutive failures since last healthy window reset.
    pub consecutive_failures: u32,
    /// Monotonic timestamp (ms) of last successful handshake, if any.
    last_healthy_ms: Option<u64>,
    /// Monotonic timestamp (ms) when the current process was started.
    last_start_ms: Option<u64>,
    /// When true, [`RestartDecision::GiveUp`] until [`RestartState::reset`].
    given_up: bool,
}

impl RestartState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn has_given_up(&self) -> bool {
        self.given_up
    }

    /// Clear give-up and failure counters (manual operator reset).
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Record that a spawn is about to begin at `now_ms`.
    pub fn on_spawn_start(&mut self, now_ms: u64) {
        self.total_spawns = self.total_spawns.saturating_add(1);
        self.last_start_ms = Some(now_ms);
    }

    /// Record a successful handshake / healthy process at `now_ms`.
    pub fn on_healthy(&mut self, now_ms: u64) {
        self.last_healthy_ms = Some(now_ms);
        // Do not zero consecutive_failures until healthy_window elapses —
        // that happens in `decide_after_exit` / `note_still_healthy`.
    }

    /// Note that the process is still alive and healthy; may clear failures.
    pub fn note_still_healthy(&mut self, now_ms: u64, policy: &RestartPolicy) {
        if let Some(healthy_at) = self.last_healthy_ms
            && now_ms.saturating_sub(healthy_at) >= duration_ms(policy.healthy_window)
        {
            self.consecutive_failures = 0;
            self.given_up = false;
        }
        self.last_healthy_ms = Some(now_ms);
    }

    /// Decide whether to restart after an unexpected exit.
    pub fn decide_after_failure(&mut self, now_ms: u64, policy: &RestartPolicy) -> RestartDecision {
        if self.given_up {
            return RestartDecision::GiveUp {
                consecutive_failures: self.consecutive_failures,
            };
        }

        // If we were healthy long enough before the crash, treat as fresh.
        if let Some(healthy_at) = self.last_healthy_ms
            && now_ms.saturating_sub(healthy_at) >= duration_ms(policy.healthy_window)
        {
            self.consecutive_failures = 0;
        }

        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        if self.consecutive_failures > policy.max_attempts {
            self.given_up = true;
            return RestartDecision::GiveUp {
                consecutive_failures: self.consecutive_failures,
            };
        }

        let wait = policy.backoff_for_attempt(self.consecutive_failures);
        if wait.is_zero() {
            RestartDecision::RunNow
        } else {
            RestartDecision::Wait(wait)
        }
    }

    /// Decide before an initial spawn or a spawn after the caller already
    /// honored a [`RestartDecision::Wait`] from [`Self::decide_after_failure`].
    ///
    /// Backoff delays are **only** produced by [`Self::decide_after_failure`];
    /// this method gates on give-up only so supervisors can `sleep(delay)` then
    /// call `ensure_running` without re-deferring forever.
    pub fn decide_before_start(&self, _policy: &RestartPolicy) -> RestartDecision {
        if self.given_up {
            RestartDecision::GiveUp {
                consecutive_failures: self.consecutive_failures,
            }
        } else {
            RestartDecision::RunNow
        }
    }
}

fn duration_ms(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> RestartPolicy {
        RestartPolicy {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(1),
            healthy_window: Duration::from_secs(10),
        }
    }

    #[test]
    fn backoff_grows_and_caps() {
        let p = policy();
        assert_eq!(p.backoff_for_attempt(1), Duration::from_millis(100));
        assert_eq!(p.backoff_for_attempt(2), Duration::from_millis(200));
        assert_eq!(p.backoff_for_attempt(3), Duration::from_millis(400));
        assert_eq!(p.backoff_for_attempt(10), Duration::from_secs(1));
    }

    #[test]
    fn gives_up_after_max_attempts() {
        let p = policy();
        let mut s = RestartState::new();
        assert_eq!(
            s.decide_after_failure(0, &p),
            RestartDecision::Wait(Duration::from_millis(100))
        );
        assert_eq!(
            s.decide_after_failure(1, &p),
            RestartDecision::Wait(Duration::from_millis(200))
        );
        assert_eq!(
            s.decide_after_failure(2, &p),
            RestartDecision::Wait(Duration::from_millis(400))
        );
        assert_eq!(
            s.decide_after_failure(3, &p),
            RestartDecision::GiveUp {
                consecutive_failures: 4
            }
        );
        assert!(s.has_given_up());
        assert_eq!(
            s.decide_after_failure(4, &p),
            RestartDecision::GiveUp {
                consecutive_failures: 4
            }
        );
    }

    #[test]
    fn healthy_window_resets_failures() {
        let p = policy();
        let mut s = RestartState::new();
        let _ = s.decide_after_failure(0, &p);
        s.on_healthy(1_000);
        s.note_still_healthy(1_000 + 10_000, &p);
        assert_eq!(s.consecutive_failures, 0);
        assert!(!s.has_given_up());
        // Next failure is attempt 1 again.
        assert_eq!(
            s.decide_after_failure(20_000, &p),
            RestartDecision::Wait(Duration::from_millis(100))
        );
    }

    #[test]
    fn manual_reset_clears_give_up() {
        let p = policy();
        let mut s = RestartState::new();
        for t in 0..5 {
            let _ = s.decide_after_failure(t, &p);
        }
        assert!(s.has_given_up());
        s.reset();
        assert!(!s.has_given_up());
        assert_eq!(s.decide_before_start(&p), RestartDecision::RunNow);
    }
}
