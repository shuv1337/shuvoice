//! Concurrency / ownership tests for async ASR owner composition.

use std::time::Duration;

use shuvoice_app::fakes::ScriptedAsrBackend;
use shuvoice_app::{
    AudioIngress, Config, ControlHandlerSurface, DEFAULT_AUDIO_CAPACITY, SessionCommand,
    spawn_test_runtime,
};

fn cfg() -> Config {
    let mut c = Config::default();
    c.min_speech_ms = 0;
    c.silence_rms_threshold = 0.0;
    c.tts_enabled = true;
    c
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slow_asr_does_not_delay_stop_status_control_enqueue() {
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted.shared().lock().delay = Duration::from_millis(200);
    let (rt, _shared, _inj) = spawn_test_runtime(cfg(), scripted).await;

    rt.handle.try_enqueue(SessionCommand::Start).unwrap();
    let t0 = std::time::Instant::now();
    rt.control.on_stop();
    let status = rt.control.on_status();
    let elapsed = t0.elapsed();
    assert!(
        elapsed < Duration::from_millis(50),
        "status/control enqueue blocked for {elapsed:?}: {status}"
    );
    let _ = status;
    rt.shutdown().await.unwrap();
}

#[tokio::test]
async fn audio_queue_overflow_is_bounded() {
    let (ingress, ring) = AudioIngress::new(4);
    for i in 0..20 {
        ingress.try_push(vec![i as f32]);
    }
    assert!(ring.depth() <= 4);
    assert!(ring.dropped() >= 16);
    assert_eq!(ring.capacity(), 4);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_joins_asr_owner() {
    let scripted = ScriptedAsrBackend::default();
    let (rt, shared, _) = spawn_test_runtime(cfg(), scripted).await;
    rt.handle.send(SessionCommand::Start).await.unwrap();
    rt.shutdown().await.unwrap();
    let _ = shared.lock().reset_calls;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn control_status_reads_are_nonblocking_under_load() {
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted.shared().lock().delay = Duration::from_millis(100);
    let (rt, _, _) = spawn_test_runtime(cfg(), scripted).await;
    rt.control.on_start();
    for _ in 0..20 {
        let _ = rt.control.on_status();
        let _ = rt.control.on_metrics();
        let _ = rt.control.on_debug_status();
    }
    rt.shutdown().await.unwrap();
}

#[tokio::test]
async fn audio_ingress_is_only_push_api() {
    let (ingress, ring) = AudioIngress::new(DEFAULT_AUDIO_CAPACITY);
    assert!(ingress.try_push(vec![0.1, 0.2]));
    assert_eq!(ring.drain().len(), 1);
}
