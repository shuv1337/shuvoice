//! Re-export of the pure ASR circuit breaker from `shuvoice-core`.
//!
//! Backend crates should not fork breaker policy. Orchestrators call
//! [`CircuitBreaker::on_recovered_cuda_fallback`] when GPU→CPU recovery
//! succeeds so the failure does not count.

pub use shuvoice_core::{
    ASR_CIRCUIT_COOLDOWN, ASR_MAX_FAILURES, BreakerAction, CircuitBreaker, ERROR_TOAST_SECONDS,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn opens_at_max_failures() {
        let mut b = CircuitBreaker::new();
        let t = Instant::now();
        for _ in 0..ASR_MAX_FAILURES - 1 {
            assert!(matches!(b.on_failure(t), BreakerAction::Counted { .. }));
        }
        assert_eq!(b.on_failure(t), BreakerAction::Opened);
        assert!(b.is_disabled());
    }

    #[test]
    fn cuda_recovery_does_not_count() {
        let mut b = CircuitBreaker::new();
        assert_eq!(b.on_recovered_cuda_fallback(), BreakerAction::Ignored);
        assert_eq!(b.consecutive_failures(), 0);
        assert!(!b.is_disabled());
    }

    #[test]
    fn recovery_after_cooldown() {
        let mut b = CircuitBreaker::new();
        let t0 = Instant::now();
        for _ in 0..ASR_MAX_FAILURES {
            let _ = b.on_failure(t0);
        }
        assert!(!b.can_attempt_recovery(t0));
        let t1 = t0 + ASR_CIRCUIT_COOLDOWN + Duration::from_millis(1);
        assert!(b.can_attempt_recovery(t1));
        b.close_after_recovery();
        assert!(!b.is_disabled());
    }
}
