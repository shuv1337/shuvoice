//! Worker client end-to-end against the in-process mock worker.

use shuvoice_asr::{
    AsrBackend, AsrBackendKind, WorkerAsrBackend, WorkerBackendKind, spawn_mock_worker, test_config,
};

#[tokio::test]
async fn nemo_worker_mock_roundtrip() {
    let (r, w) = spawn_mock_worker("nemo").await.unwrap();
    let mut cfg = test_config(AsrBackendKind::Nemo);
    cfg.core.right_context = 0;
    let _ = cfg.core.validate();
    let mut backend = WorkerAsrBackend::new(WorkerBackendKind::Nemo, cfg).with_duplex(r, w);
    let mut progress = |_f: Option<f32>, _m: &str| {};
    backend.load(&mut progress).await.unwrap();
    assert!(backend.capabilities().wants_raw_audio);
    // From handshake manifest + load Ack.result (mock returns 1280).
    assert_eq!(backend.native_chunk_samples(), 1280);
    assert_eq!(backend.capabilities().preferred_sample_rate, Some(16_000));
    backend.reset().await.unwrap();
    let t = backend.process_chunk(&[0.1, 0.2, 0.0]).await.unwrap();
    assert_eq!(t, "hello");
    let t2 = backend.process_chunk(&[0.3]).await.unwrap();
    assert_eq!(t2, "hello world");
    backend.shutdown().await.unwrap();
}

#[tokio::test]
async fn moonshine_worker_mock_load() {
    let (r, w) = spawn_mock_worker("moonshine").await.unwrap();
    let cfg = test_config(AsrBackendKind::Moonshine);
    let mut backend = WorkerAsrBackend::new(WorkerBackendKind::Moonshine, cfg).with_duplex(r, w);
    let mut progress = |_f: Option<f32>, _m: &str| {};
    backend.load(&mut progress).await.unwrap();
    assert!(backend.capabilities().wants_raw_audio);
    backend.shutdown().await.unwrap();
}

#[tokio::test]
async fn worker_without_command_errors_on_load() {
    let mut cfg = test_config(AsrBackendKind::Nemo);
    cfg.connect.worker_command = None;
    cfg.connect.worker_socket_path = None;
    cfg.connect.worker_spawn = None;
    let mut backend = WorkerAsrBackend::new(WorkerBackendKind::Nemo, cfg);
    let mut progress = |_f: Option<f32>, _m: &str| {};
    let err = backend.load(&mut progress).await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("external worker") || msg.contains("worker"),
        "{msg}"
    );
}
