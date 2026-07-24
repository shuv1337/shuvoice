//! WorkerProcess / WorkerSupervisor integration tests.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use shuvoice_worker_proto::{
    BackendKind, PROTOCOL_VERSION, RestartDecision, RestartPolicy, RestartState, WorkerProcess,
    WorkerProcessError, WorkerSpawnConfig, WorkerSupervisor, redact_text,
};

// ── helpers ──────────────────────────────────────────────────────────────

fn python_executable() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("SHUVOICE_TEST_PYTHON") {
        let path = PathBuf::from(explicit);
        if !path.as_os_str().is_empty() {
            return Some(path);
        }
        return None;
    }
    for candidate in ["python3", "python"] {
        if std::process::Command::new(candidate)
            .arg("-c")
            .arg("import sys; sys.exit(0)")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return Some(PathBuf::from(candidate));
        }
    }
    None
}

fn workers_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut candidates = vec![
        manifest_dir.join("../../workers"),
        manifest_dir.join("../../../workers"),
    ];
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("workers"));
    }
    let mut cursor = manifest_dir.as_path();
    for _ in 0..6 {
        candidates.push(cursor.join("workers"));
        if let Some(parent) = cursor.parent() {
            cursor = parent;
        } else {
            break;
        }
    }
    for c in candidates {
        if c.join("nemo_asr/__main__.py").is_file() {
            return c.canonicalize().unwrap_or(c);
        }
    }
    panic!("workers/ not found");
}

fn fake_worker_config(module: &str) -> Option<WorkerSpawnConfig> {
    let python = python_executable()?;
    let workers = workers_dir();
    Some(
        WorkerSpawnConfig::new(python)
            .args(["-m", module, "--fake"])
            .current_dir(&workers)
            .env_pair("PYTHONPATH", workers.as_os_str())
            .env_pair("SHUVOICE_WORKER_FAKE", "1")
            .env_pair("PYTHONUNBUFFERED", "1")
            .client_name("supervisor-e2e")
            .handshake_timeout(Duration::from_secs(8))
            .shutdown_timeout(Duration::from_secs(3))
            .kill_timeout(Duration::from_secs(2)),
    )
}

// ── pure unit ────────────────────────────────────────────────────────────

#[test]
fn restart_state_machine_is_bounded() {
    let policy = RestartPolicy {
        max_attempts: 2,
        initial_backoff: Duration::from_millis(50),
        max_backoff: Duration::from_millis(200),
        healthy_window: Duration::from_secs(5),
    };
    let mut state = RestartState::new();
    assert_eq!(state.decide_before_start(&policy), RestartDecision::RunNow);
    assert!(matches!(
        state.decide_after_failure(0, &policy),
        RestartDecision::Wait(_)
    ));
    assert!(matches!(
        state.decide_after_failure(1, &policy),
        RestartDecision::Wait(_)
    ));
    assert_eq!(
        state.decide_after_failure(2, &policy),
        RestartDecision::GiveUp {
            consecutive_failures: 3
        }
    );
}

#[test]
fn redact_never_keeps_api_key_material() {
    let raw = "ERROR api_key=sk-live-ABCDEFG password=hunter2\n";
    let red = redact_text(raw);
    assert!(!red.contains("sk-live"));
    assert!(!red.contains("hunter2"));
    assert!(red.contains("REDACTED"));
}

// ── process spawn e2e ────────────────────────────────────────────────────

#[tokio::test]
async fn spawn_nemo_fake_handshake_and_clean_shutdown() {
    let Some(cfg) = fake_worker_config("nemo_asr") else {
        eprintln!("SKIP: python unavailable");
        return;
    };
    let mut proc = WorkerProcess::spawn(cfg).await.expect("spawn+handshake");
    assert_eq!(proc.session().protocol_version, PROTOCOL_VERSION);
    assert_eq!(proc.session().manifest.backend_id, "nemo");
    assert_eq!(proc.session().manifest.kind, BackendKind::Asr);
    assert!(proc.pid().is_some());

    proc.client_mut()
        .load(serde_json::json!({"right_context": 0, "device": "cpu"}))
        .await
        .expect("load");
    let samples = vec![0.0_f32; 1280];
    let tr = proc
        .client_mut()
        .process_chunk(&samples, 16_000)
        .await
        .expect("chunk");
    assert_eq!(tr.text, "step-1");

    let exit = proc.shutdown().await.expect("shutdown");
    assert!(exit.success, "exit={:?}", exit.code);
    // stderr tail must not contain transcript text
    assert!(!exit.stderr_tail.contains("step-1"));
}

#[tokio::test]
async fn spawn_melotts_fake_and_moonshine_fake() {
    let Some(melo_cfg) = fake_worker_config("melotts") else {
        eprintln!("SKIP: python unavailable");
        return;
    };
    let proc = WorkerProcess::spawn(melo_cfg).await.expect("melo");
    assert_eq!(proc.session().manifest.backend_id, "melotts");
    let exit = proc.shutdown().await.expect("melo shutdown");
    assert!(exit.success);

    let moon_cfg = fake_worker_config("moonshine_asr").expect("python");
    let proc = WorkerProcess::spawn(moon_cfg).await.expect("moon");
    assert_eq!(proc.session().manifest.backend_id, "moonshine");
    let exit = proc.shutdown().await.expect("moon shutdown");
    assert!(exit.success);
}

#[tokio::test]
async fn crashing_helper_surfaces_typed_error_or_clean_fail() {
    let Some(python) = python_executable() else {
        eprintln!("SKIP: python unavailable");
        return;
    };
    // Immediate non-zero exit before any handshake bytes.
    let cfg = WorkerSpawnConfig::new(&python)
        .args(["-c", "import sys; sys.exit(17)"])
        .handshake_timeout(Duration::from_secs(3))
        .kill_timeout(Duration::from_secs(1))
        .client_name("crash-helper");

    let err = match WorkerProcess::spawn(cfg).await {
        Ok(_) => panic!("must fail handshake"),
        Err(e) => e,
    };
    match err {
        WorkerProcessError::HandshakeTimeout { .. }
        | WorkerProcessError::Handshake { .. }
        | WorkerProcessError::Crashed { .. } => {}
        other => panic!("unexpected error variant: {other:?}"),
    }
    // Ensure no secret-looking material required in display.
    let shown = format!("{err}");
    assert!(!shown.contains("password"));
}

#[tokio::test]
async fn handshake_timeout_kills_hung_worker() {
    let Some(python) = python_executable() else {
        eprintln!("SKIP: python unavailable");
        return;
    };
    let cfg = WorkerSpawnConfig::new(&python)
        .args(["-c", "import time,sys; time.sleep(60)"])
        .handshake_timeout(Duration::from_millis(400))
        .kill_timeout(Duration::from_secs(2))
        .client_name("hang-helper");

    let started = std::time::Instant::now();
    let err = match WorkerProcess::spawn(cfg).await {
        Ok(_) => panic!("hang should timeout"),
        Err(e) => e,
    };
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "timeout path took too long: {:?}",
        started.elapsed()
    );
    assert!(
        matches!(err, WorkerProcessError::HandshakeTimeout { .. }),
        "got {err:?}"
    );
}

#[tokio::test]
async fn drop_prevents_orphans_via_kill_on_drop() {
    let Some(cfg) = fake_worker_config("nemo_asr") else {
        eprintln!("SKIP: python unavailable");
        return;
    };
    let proc = WorkerProcess::spawn(cfg).await.expect("spawn");
    let pid = proc.pid().expect("pid");
    // Drop without shutdown — Child::kill_on_drop must reap.
    drop(proc);
    // Give the runtime a moment to deliver SIGKILL.
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        !pid_exists(pid),
        "worker pid {pid} still alive after drop (orphan)"
    );
}

#[tokio::test]
async fn supervisor_ensure_running_and_restart_policy() {
    let Some(cfg) = fake_worker_config("nemo_asr") else {
        eprintln!("SKIP: python unavailable");
        return;
    };
    let policy = RestartPolicy {
        max_attempts: 2,
        initial_backoff: Duration::from_millis(10),
        max_backoff: Duration::from_millis(50),
        healthy_window: Duration::from_secs(30),
    };
    let mut sup = WorkerSupervisor::new(cfg, policy);

    {
        let proc = sup.ensure_running().await.expect("first start");
        assert_eq!(proc.session().manifest.backend_id, "nemo");
    }

    // Simulate failure path: kill current and ask supervisor to restart.
    let exit = {
        let proc = sup.take_process().expect("running");
        proc.kill().await
    };
    // First failure → Wait or RunNow depending on attempt count.
    let result = sup.restart_after_failure(Some(exit)).await;
    match result {
        Ok(_) => {}
        Err(WorkerProcessError::RestartDeferred { delay }) => {
            assert!(delay > Duration::ZERO);
            tokio::time::sleep(delay).await;
            sup.ensure_running()
                .await
                .expect("spawn after honoring backoff");
        }
        Err(e) => panic!("unexpected: {e:?}"),
    }

    let exit = sup
        .shutdown()
        .await
        .expect("shutdown")
        .expect("had process");
    assert!(exit.success || exit.code.is_some());
}

#[tokio::test]
async fn supervisor_gives_up_after_bounded_failures() {
    let Some(python) = python_executable() else {
        eprintln!("SKIP: python unavailable");
        return;
    };
    // Always-crash worker.
    let cfg = WorkerSpawnConfig::new(python)
        .args(["-c", "import sys; sys.exit(1)"])
        .handshake_timeout(Duration::from_millis(500))
        .kill_timeout(Duration::from_millis(500))
        .client_name("always-crash");
    let policy = RestartPolicy {
        max_attempts: 2,
        initial_backoff: Duration::ZERO,
        max_backoff: Duration::ZERO,
        healthy_window: Duration::from_secs(60),
    };
    let mut sup = WorkerSupervisor::new(cfg, policy);

    // ensure_running will fail handshake each time; drive restart_after_failure.
    for _ in 0..5 {
        match sup.ensure_running().await {
            Ok(_) => panic!("crash worker should not handshake"),
            Err(WorkerProcessError::RestartExhausted { .. }) => return,
            Err(WorkerProcessError::RestartDeferred { delay }) => {
                tokio::time::sleep(delay).await;
            }
            Err(_spawn_err) => {
                // Count as failure into the state machine.
                match sup.restart_after_failure(None).await {
                    Ok(_) => panic!("should not run"),
                    Err(WorkerProcessError::RestartExhausted {
                        consecutive_failures,
                    }) => {
                        assert!(consecutive_failures >= 2);
                        return;
                    }
                    Err(WorkerProcessError::RestartDeferred { delay }) => {
                        if !delay.is_zero() {
                            tokio::time::sleep(delay).await;
                        }
                    }
                    Err(_) => {}
                }
            }
        }
    }
    assert!(
        sup.restart_state().has_given_up(),
        "expected give up after bounded failures"
    );
}

fn pid_exists(pid: u32) -> bool {
    // Linux: /proc/<pid> exists while the process (or zombie briefly) remains.
    // After kill_on_drop + short wait, it should be gone.
    Path::new(&format!("/proc/{pid}")).exists()
}

#[tokio::test]
async fn ensure_running_honors_backoff_after_crash_without_double_count() {
    let Some(python) = python_executable() else {
        eprintln!("SKIP: python unavailable");
        return;
    };
    // Always-crash worker.
    let cfg = WorkerSpawnConfig::new(python)
        .args(["-c", "import sys; sys.exit(1)"])
        .handshake_timeout(Duration::from_millis(400))
        .kill_timeout(Duration::from_millis(400))
        .client_name("backoff-crash");
    let policy = RestartPolicy {
        max_attempts: 3,
        initial_backoff: Duration::from_millis(200),
        max_backoff: Duration::from_millis(200),
        healthy_window: Duration::from_secs(60),
    };
    let mut sup = WorkerSupervisor::new(cfg, policy);

    // First ensure_running → spawn fails → deferred or exhausted path via record_failure.
    let err1 = match sup.ensure_running().await {
        Ok(_) => panic!("crash worker should not handshake"),
        Err(e) => e,
    };
    match err1 {
        WorkerProcessError::RestartDeferred { delay } => {
            assert!(
                delay > Duration::ZERO,
                "expected positive backoff, got {delay:?}"
            );
            // Immediate retry must still be gated.
            let err2 = match sup.ensure_running().await {
                Ok(_) => panic!("backoff bypassed with live process"),
                Err(e) => e,
            };
            assert!(
                matches!(err2, WorkerProcessError::RestartDeferred { .. }),
                "backoff bypassed: {err2:?}"
            );
            // restart_after_failure must NOT double-count into instant give-up.
            let before = sup.restart_state().consecutive_failures;
            let err3 = match sup.restart_after_failure(None).await {
                Ok(_) => panic!("should still be in backoff window"),
                Err(e) => e,
            };
            assert!(
                matches!(err3, WorkerProcessError::RestartDeferred { .. }),
                "expected still deferred, got {err3:?}"
            );
            assert_eq!(
                sup.restart_state().consecutive_failures,
                before,
                "restart_after_failure double-counted failures"
            );
        }
        WorkerProcessError::RestartExhausted { .. } => {
            // With aggressive policies this can happen; still OK if given up without panic.
        }
        other => panic!("unexpected first error: {other:?}"),
    }
}
