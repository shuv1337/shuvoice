//! Cross-language e2e: Rust `WorkerClient` ↔ bundled Python fake workers over stdio.
//!
//! Spawns `workers/{nemo_asr,melotts,moonshine_asr}` with `--fake` so no ML
//! packages are required. Skips the entire module only when no Python
//! interpreter can be executed.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use shuvoice_worker_proto::{
    BackendKind, PROTOCOL_VERSION, PcmEncoding, ProtocolError, RequestId, WorkerClient,
};
use tokio::process::{Child, Command};
use uuid::Uuid;

// ── environment discovery ────────────────────────────────────────────────

fn python_executable() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("SHUVOICE_TEST_PYTHON") {
        let path = PathBuf::from(explicit);
        if path.as_os_str().is_empty() {
            return None;
        }
        return Some(path);
    }
    for candidate in ["python3", "python"] {
        if command_runs(candidate) {
            return Some(PathBuf::from(candidate));
        }
    }
    None
}

fn command_runs(bin: &str) -> bool {
    std::process::Command::new(bin)
        .arg("-c")
        .arg("import sys; sys.exit(0)")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn workers_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut candidates = vec![
        manifest_dir.join("../../workers"),
        manifest_dir.join("../../../workers"),
    ];
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("workers"));
        candidates.push(cwd.join("../workers"));
    }
    if let Ok(root) = std::env::var("CARGO_WORKSPACE_DIR") {
        candidates.push(PathBuf::from(root).join("workers"));
    }
    // Walk up from manifest looking for workers/nemo_asr.
    let mut cursor = manifest_dir.as_path();
    for _ in 0..6 {
        candidates.push(cursor.join("workers"));
        match cursor.parent() {
            Some(parent) => cursor = parent,
            None => break,
        }
    }

    for candidate in candidates {
        if is_workers_tree(&candidate) {
            return candidate.canonicalize().unwrap_or(candidate);
        }
    }
    panic!(
        "could not locate workers/ tree (expected nemo_asr + shuvoice_worker_proto). \
         CARGO_MANIFEST_DIR={}",
        env!("CARGO_MANIFEST_DIR")
    );
}

fn is_workers_tree(path: &Path) -> bool {
    path.join("nemo_asr").join("__main__.py").is_file()
        && path.join("melotts").join("__main__.py").is_file()
        && path.join("moonshine_asr").join("__main__.py").is_file()
        && path
            .join("shuvoice_worker_proto")
            .join("__init__.py")
            .is_file()
}

// ── child process helper ─────────────────────────────────────────────────

struct PythonWorker {
    child: Child,
    /// Kept so Drop can best-effort reap.
    module: &'static str,
}

impl PythonWorker {
    async fn spawn(
        module: &'static str,
    ) -> (
        Self,
        WorkerClient<impl tokio::io::AsyncRead + Unpin, impl tokio::io::AsyncWrite + Unpin>,
    ) {
        let python = python_executable().unwrap_or_else(|| {
            panic!("internal: spawn called without python (tests should skip first)")
        });
        let workers = workers_dir();

        let mut child = Command::new(&python)
            .args(["-m", module, "--fake"])
            .current_dir(&workers)
            .env("PYTHONPATH", &workers)
            .env("SHUVOICE_WORKER_FAKE", "1")
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap_or_else(|e| {
                panic!(
                    "failed to spawn `{python:?} -m {module} --fake` in {}: {e}",
                    workers.display()
                )
            });

        let stdin = child.stdin.take().expect("worker stdin piped");
        let stdout = child.stdout.take().expect("worker stdout piped");
        let client = WorkerClient::new(stdout, stdin);
        (Self { child, module }, client)
    }

    async fn expect_clean_exit(mut self) {
        // Give the worker a moment after close.
        let status = tokio::time::timeout(Duration::from_secs(5), self.child.wait())
            .await
            .unwrap_or_else(|_| panic!("{}: timed out waiting for exit", self.module))
            .unwrap_or_else(|e| panic!("{}: wait failed: {e}", self.module));

        if !status.success() {
            let stderr = self.read_stderr().await;
            panic!(
                "{}: non-zero exit {:?}\n--- stderr ---\n{stderr}",
                self.module,
                status.code()
            );
        }
    }

    async fn read_stderr(&mut self) -> String {
        use tokio::io::AsyncReadExt;
        let mut buf = String::new();
        if let Some(mut err) = self.child.stderr.take() {
            let mut bytes = Vec::new();
            let _ = err.read_to_end(&mut bytes).await;
            buf = String::from_utf8_lossy(&bytes).into_owned();
        }
        buf
    }
}

fn require_python() {
    if python_executable().is_none() {
        eprintln!("skipping python worker e2e: no python3/python executable available");
        // Use ignore-style early return via panic-free skip:
        // cargo doesn't have built-in skip from runtime without the `ignore` attribute,
        // so we use a soft skip by returning from each test after checking.
    }
}

fn python_or_skip() -> Option<PathBuf> {
    python_executable()
}

// ── tests ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn nemo_asr_fake_full_lifecycle() {
    let Some(_) = python_or_skip() else {
        eprintln!("SKIP nemo_asr_fake_full_lifecycle: python unavailable");
        return;
    };

    let (worker, mut client) = PythonWorker::spawn("nemo_asr").await;

    let session = client
        .handshake("rust-e2e-nemo")
        .await
        .expect("handshake")
        .clone();
    assert_eq!(session.protocol_version, PROTOCOL_VERSION);
    assert_eq!(session.manifest.backend_id, "nemo");
    assert_eq!(session.manifest.kind, BackendKind::Asr);
    let asr = session.manifest.asr.as_ref().expect("asr caps");
    assert!(asr.wants_raw_audio);
    assert!(asr.supports_streaming);
    assert_eq!(asr.native_sample_rate_hz, Some(16_000));

    client
        .load(serde_json::json!({
            "model_name": "fake-nemo",
            "right_context": 0,
            "device": "cpu",
        }))
        .await
        .expect("load");

    // right_context=0 → 1280 native samples (mirrors asr_nemo semantics).
    let samples = vec![0.01_f32; 1280];
    let partial = client
        .process_chunk(&samples, 16_000)
        .await
        .expect("process_chunk");
    assert_eq!(partial.text, "step-1");
    let chunk_id = partial.request_id;

    let finished = client.finish(Some(1_000)).await.expect("finish");
    assert_eq!(finished.text, "step-1");
    // finish uses its own request id — correlation must still be a valid UUID.
    assert_ne!(finished.request_id, Uuid::nil());
    assert_ne!(finished.request_id, chunk_id);

    client.reset().await.expect("reset");

    // Cancel a synthetic id (idle cancel) — must ack, not crash.
    let cancel_id = Uuid::nil();
    client.cancel(cancel_id).await.expect("cancel");

    let utterance = client
        .process_utterance(&samples, 16_000)
        .await
        .expect("process_utterance");
    assert!(
        utterance.text.starts_with("step-"),
        "unexpected transcript {}",
        utterance.text
    );

    client.close().await.expect("close");
    worker.expect_clean_exit().await;
}

#[tokio::test]
async fn melotts_fake_voices_and_pcm_frames() {
    let Some(_) = python_or_skip() else {
        eprintln!("SKIP melotts_fake_voices_and_pcm_frames: python unavailable");
        return;
    };

    let (worker, mut client) = PythonWorker::spawn("melotts").await;

    let session = client
        .handshake("rust-e2e-melo")
        .await
        .expect("handshake")
        .clone();
    assert_eq!(session.protocol_version, PROTOCOL_VERSION);
    assert_eq!(session.manifest.backend_id, "melotts");
    assert_eq!(session.manifest.kind, BackendKind::Tts);
    let tts = session.manifest.tts.as_ref().expect("tts caps");
    assert!(!tts.requires_api_key);
    assert_eq!(tts.default_sample_rate_hz, Some(44_100));

    let voices = client.list_voices().await.expect("list_voices");
    assert!(
        voices.iter().any(|v| v.id == "EN-US"),
        "missing EN-US in {voices:?}"
    );

    let result = client
        .synthesize("hello from rust", Some("EN-US".into()), Some(1.0))
        .await
        .expect("synthesize");
    assert_eq!(result.sample_rate_hz, Some(44_100));
    assert!(!result.pcm.is_empty(), "expected non-empty PCM body");
    // Client requests f32_le; worker should honor or still deliver binary audio.
    match result.encoding {
        PcmEncoding::F32Le => assert_eq!(result.pcm.len() % 4, 0, "f32 frame alignment"),
        PcmEncoding::I16Le => assert_eq!(result.pcm.len() % 2, 0, "i16 frame alignment"),
    }
    let synth_id = result.request_id;
    assert_ne!(synth_id, Uuid::nil());

    // Idle cancel still correlates.
    client.cancel(Uuid::nil()).await.expect("cancel");

    client.close().await.expect("close");
    worker.expect_clean_exit().await;
}

#[tokio::test]
async fn moonshine_asr_fake_utterance_lifecycle() {
    let Some(_) = python_or_skip() else {
        eprintln!("SKIP moonshine_asr_fake_utterance_lifecycle: python unavailable");
        return;
    };

    let (worker, mut client) = PythonWorker::spawn("moonshine_asr").await;

    let session = client
        .handshake("rust-e2e-moon")
        .await
        .expect("handshake")
        .clone();
    assert_eq!(session.protocol_version, PROTOCOL_VERSION);
    assert_eq!(session.manifest.backend_id, "moonshine");
    assert_eq!(session.manifest.kind, BackendKind::Asr);
    let asr = session.manifest.asr.as_ref().expect("asr caps");
    assert!(asr.wants_raw_audio);
    assert_eq!(asr.native_sample_rate_hz, Some(16_000));

    client
        .load(serde_json::json!({"model_name": "moonshine/tiny"}))
        .await
        .expect("load");

    let samples = vec![0.0_f32; 1600];
    let final_tr = client
        .process_utterance(&samples, 16_000)
        .await
        .expect("process_utterance");
    assert!(
        final_tr.text.starts_with("moon-samples-"),
        "got {}",
        final_tr.text
    );
    assert_ne!(final_tr.request_id, Uuid::nil());

    client.reset().await.expect("reset");
    client.close().await.expect("close");
    worker.expect_clean_exit().await;
}

#[tokio::test]
async fn intentional_error_does_not_leak_secret_or_transcript() {
    let Some(_) = python_or_skip() else {
        eprintln!("SKIP intentional_error_does_not_leak_secret_or_transcript: python unavailable");
        return;
    };

    let secret = "LEAKME_SECRET_TOKEN_9f3a2c1b";

    // 1) MeloTTS: oversize text containing the secret → text_too_long, no echo.
    {
        let (worker, mut client) = PythonWorker::spawn("melotts").await;
        client.handshake("rust-e2e-leak-melo").await.expect("hs");
        let long = secret.repeat(400); // >> 5000 chars
        assert!(long.len() > 5000);
        let err = client
            .synthesize(long.clone(), Some("EN-US".into()), Some(1.0))
            .await
            .expect_err("expected text_too_long");
        let rendered = render_error(&err);
        assert!(
            !rendered.contains(secret),
            "protocol error leaked secret: {rendered}"
        );
        assert!(
            !rendered.contains(&long[..64.min(long.len())]),
            "protocol error leaked transcript prefix: {rendered}"
        );
        match &err {
            ProtocolError::Worker { code, message, .. } => {
                assert_eq!(code, "text_too_long");
                assert!(!message.contains(secret));
            }
            other => panic!("expected Worker error, got {other:?}"),
        }
        client.close().await.ok();
        worker.expect_clean_exit().await;
    }

    // 2) NeMo: process_chunk before load → not_loaded; secret only in our local buffer.
    {
        let (worker, mut client) = PythonWorker::spawn("nemo_asr").await;
        client.handshake("rust-e2e-leak-nemo").await.expect("hs");
        // Plant secret as "audio" content — errors must not echo sample-derived text.
        let mut samples = vec![0.0_f32; 64];
        // Encode a recognizable pattern into samples (not a protocol field).
        for (i, ch) in secret.bytes().enumerate() {
            if i < samples.len() {
                samples[i] = f32::from(ch) / 255.0;
            }
        }
        let err = client
            .process_chunk(&samples, 16_000)
            .await
            .expect_err("expected not_loaded");
        let rendered = render_error(&err);
        assert!(
            !rendered.contains(secret),
            "not_loaded error leaked secret: {rendered}"
        );
        match &err {
            ProtocolError::Worker { code, .. } => assert_eq!(code, "not_loaded"),
            other => panic!("expected Worker error, got {other:?}"),
        }
        client.close().await.ok();
        worker.expect_clean_exit().await;
    }
}

#[tokio::test]
async fn request_ids_correlate_across_asr_roundtrip() {
    let Some(_) = python_or_skip() else {
        eprintln!("SKIP request_ids_correlate_across_asr_roundtrip: python unavailable");
        return;
    };

    let (worker, mut client) = PythonWorker::spawn("nemo_asr").await;
    client.handshake("rust-e2e-ids").await.expect("hs");
    client
        .load(serde_json::json!({"right_context": 1, "device": "cpu"}))
        .await
        .expect("load");

    let samples = vec![0.02_f32; 2560]; // right_context=1 → 2560
    let t1 = client
        .process_chunk(&samples, 16_000)
        .await
        .expect("chunk1");
    let t2 = client
        .process_chunk(&samples, 16_000)
        .await
        .expect("chunk2");
    assert_ne!(t1.request_id, t2.request_id);
    assert_eq!(t1.text, "step-1");
    assert_eq!(t2.text, "step-2");

    // Cancel uses the caller's id and must ack that same id (exercised inside client.cancel).
    let custom: RequestId = Uuid::from_u128(0x1111_2222_3333_4444);
    client.cancel(custom).await.expect("cancel custom");

    client.close().await.expect("close");
    worker.expect_clean_exit().await;
}

fn render_error(err: &ProtocolError) -> String {
    format!("{err:?} | {err}")
}

// Silence unused warning if someone gates tests differently.
#[allow(dead_code)]
fn _touch_require_python() {
    require_python();
}
