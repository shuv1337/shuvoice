use std::time::{Duration, Instant};

use shuvoice_asr::{
    ASR_MAX_FAILURES, AsrBackend, AsrError, BreakerAction, CircuitBreaker, FallbackOutcome,
    MockAsrBackend, looks_like_cuda_oom_str,
};

#[test]
fn oom_markers_match_public_contract() {
    assert!(looks_like_cuda_oom_str("CUBLAS_STATUS_ALLOC_FAILED"));
    assert!(looks_like_cuda_oom_str(
        "bfc_arena.cc Failed to allocate memory for requested buffer"
    ));
    assert!(!looks_like_cuda_oom_str("MemcpyFromHost only"));
}

#[test]
fn circuit_opens_at_ten() {
    let mut b = CircuitBreaker::new();
    let t = Instant::now();
    for _ in 0..ASR_MAX_FAILURES - 1 {
        assert!(matches!(b.on_failure(t), BreakerAction::Counted { .. }));
    }
    assert_eq!(b.on_failure(t), BreakerAction::Opened);
    assert!(b.is_disabled());
}

#[tokio::test]
async fn mock_cpu_fallback_idempotent() {
    let mut m = MockAsrBackend::sherpa_streaming();
    let mut progress = |_f: Option<f32>, _m: &str| {};
    m.load(&mut progress).await.unwrap();
    let first = m.try_fallback_to_cpu().await.unwrap();
    assert!(matches!(first, FallbackOutcome::Applied { .. }));
    let second = m.try_fallback_to_cpu().await.unwrap();
    assert!(matches!(second, FallbackOutcome::NotApplicable { .. }));
}

#[tokio::test]
async fn recovered_cuda_oom_should_not_trip_breaker() {
    let mut breaker = CircuitBreaker::new();
    let t = Instant::now();
    let mut m = MockAsrBackend::sherpa_streaming();
    let mut progress = |_f: Option<f32>, _m: &str| {};
    m.load(&mut progress).await.unwrap();
    m.fail_next = Some(AsrError::cuda_oom("cublas_status_alloc_failed"));
    let err = m.process_chunk(&[0.1]).await.unwrap_err();
    assert!(matches!(err, AsrError::CudaOom(_)));
    let fb = m.try_fallback_to_cpu().await.unwrap();
    assert!(fb.applied());
    assert_eq!(breaker.on_recovered_cuda_fallback(), BreakerAction::Ignored);
    assert_eq!(breaker.consecutive_failures(), 0);
    assert!(!breaker.is_disabled());
    let _ = t;
    let _ = Duration::from_secs(1);
}
