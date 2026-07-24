use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream;
use parking_lot::Mutex;
use shuvoice_tts::{
    AudioEncoding, BackendId, Capabilities, EventInfo, FakeAudioOutputFactory, PlayerState,
    SynthesisRequest, SynthesisStream, TtsBackend, TtsError, TtsPlayer, VoiceInfo,
    chunk_to_samples,
};
use tokio_util::sync::CancellationToken;

struct FakeBackend {
    sample_rate: u32,
    chunks: Mutex<Vec<Bytes>>,
    delay: Duration,
    fail: bool,
    requests: Mutex<Vec<SynthesisRequest>>,
}

impl FakeBackend {
    fn with_pcm(samples: &[i16]) -> Self {
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        Self {
            sample_rate: 24_000,
            chunks: Mutex::new(vec![Bytes::from(bytes)]),
            delay: Duration::from_millis(5),
            fail: false,
            requests: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl TtsBackend for FakeBackend {
    fn id(&self) -> BackendId {
        BackendId::Kokoro
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_speed_control: true,
            ..Capabilities::default()
        }
    }
    fn sample_rate_hz(&self) -> u32 {
        self.sample_rate
    }
    fn dependency_errors(&self) -> Vec<String> {
        Vec::new()
    }
    async fn synthesize_stream(
        &self,
        request: SynthesisRequest,
        cancel: CancellationToken,
    ) -> Result<SynthesisStream, TtsError> {
        self.requests.lock().push(request);
        if self.fail {
            return Err(TtsError::backend("boom"));
        }
        let chunks = self.chunks.lock().clone();
        let delay = self.delay;
        let stream = stream::unfold(
            (chunks, 0usize, cancel, delay),
            |(chunks, idx, cancel, delay)| async move {
                if cancel.is_cancelled() || idx >= chunks.len() {
                    return None;
                }
                tokio::time::sleep(delay).await;
                let item = Ok(chunks[idx].clone());
                Some((item, (chunks, idx + 1, cancel, delay)))
            },
        );
        Ok(SynthesisStream {
            sample_rate_hz: self.sample_rate,
            encoding: AudioEncoding::PcmS16Le,
            chunks: Box::pin(stream),
        })
    }
    async fn list_voices(&self) -> Result<Vec<VoiceInfo>, TtsError> {
        Ok(vec![VoiceInfo::new("v", "V")])
    }
}

async fn wait_until(mut pred: impl FnMut() -> bool, timeout: Duration) {
    let start = std::time::Instant::now();
    while !pred() {
        if start.elapsed() > timeout {
            panic!("timeout waiting for condition");
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn player_basic_state_flow() {
    let backend = Arc::new(FakeBackend::with_pcm(&[1, 2, 3, 4]));
    let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
    let events = Arc::new(Mutex::new(Vec::<PlayerState>::new()));
    let events2 = Arc::clone(&events);
    let player = TtsPlayer::builder(backend, factory.clone())
        .on_event(move |ev| events2.lock().push(ev.state))
        .build();

    player.speak("hello", "v", "m").unwrap();
    wait_until(
        || player.state() == PlayerState::Idle,
        Duration::from_secs(2),
    )
    .await;

    let states = events.lock().clone();
    assert!(states.contains(&PlayerState::Synthesizing));
    assert!(states.contains(&PlayerState::Playing));
    assert!(states.last().copied() == Some(PlayerState::Idle));
    assert_eq!(factory.snapshot(), vec![1, 2, 3, 4]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn player_snapshots_speed_into_request() {
    let backend = Arc::new(FakeBackend::with_pcm(&[1]));
    let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
    let player = TtsPlayer::builder(backend.clone(), factory)
        .playback_speed(1.25)
        .build();
    player.speak("hi", "v", "m").unwrap();
    wait_until(
        || player.state() == PlayerState::Idle,
        Duration::from_secs(2),
    )
    .await;
    let reqs = backend.requests.lock().clone();
    assert_eq!(reqs[0].playback_speed, 1.25);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn player_restart_uses_latest_selected_speed() {
    let backend = Arc::new(FakeBackend::with_pcm(&[5, 6]));
    let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
    let player = TtsPlayer::builder(backend.clone(), factory)
        .playback_speed(1.0)
        .build();
    player.speak("once", "v", "m").unwrap();
    wait_until(
        || player.state() == PlayerState::Idle,
        Duration::from_secs(2),
    )
    .await;
    player.set_playback_speed(1.5);
    assert!(player.restart());
    wait_until(
        || player.state() == PlayerState::Idle,
        Duration::from_secs(2),
    )
    .await;
    let reqs = backend.requests.lock().clone();
    assert_eq!(reqs.last().unwrap().playback_speed, 1.5);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn player_pause_resume_and_stop() {
    let backend = Arc::new(FakeBackend {
        sample_rate: 24_000,
        chunks: Mutex::new(vec![
            Bytes::from(vec![1, 0, 2, 0]),
            Bytes::from(vec![3, 0, 4, 0]),
        ]),
        delay: Duration::from_millis(40),
        fail: false,
        requests: Mutex::new(Vec::new()),
    });
    let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
    let player = TtsPlayer::builder(backend, factory).build();
    player.speak("long", "v", "m").unwrap();
    wait_until(
        || {
            matches!(
                player.state(),
                PlayerState::Playing | PlayerState::Synthesizing
            )
        },
        Duration::from_secs(2),
    )
    .await;
    // Ensure playing
    wait_until(
        || player.state() == PlayerState::Playing,
        Duration::from_secs(2),
    )
    .await;
    assert!(player.pause());
    assert_eq!(player.state(), PlayerState::Paused);
    assert!(player.resume());
    assert!(player.stop());
    assert_eq!(player.state(), PlayerState::Idle);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn player_error_transition() {
    let backend = Arc::new(FakeBackend {
        sample_rate: 24_000,
        chunks: Mutex::new(Vec::new()),
        delay: Duration::ZERO,
        fail: true,
        requests: Mutex::new(Vec::new()),
    });
    let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
    let player = TtsPlayer::builder(backend, factory).build();
    player.speak("x", "v", "m").unwrap();
    wait_until(
        || player.state() == PlayerState::Error,
        Duration::from_secs(2),
    )
    .await;
    assert!(player.stop());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn player_recovers_from_transient_write_error() {
    let backend = Arc::new(FakeBackend::with_pcm(&[7, 8]));
    let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
    *factory.fail_times.lock() = 1;
    let player = TtsPlayer::builder(backend, factory.clone()).build();
    player.speak("ok", "v", "m").unwrap();
    wait_until(
        || player.state() == PlayerState::Idle,
        Duration::from_secs(2),
    )
    .await;
    assert_eq!(factory.snapshot(), vec![7, 8]);
    assert!(*factory.open_count.lock() >= 2);
}

#[test]
fn pcm_carry_and_status_payload_defaults() {
    let (samples, carry) = chunk_to_samples(&[0x10], &[]);
    assert!(samples.is_empty());
    assert_eq!(carry, vec![0x10]);
}

#[test]
fn piper_length_scale_inverse() {
    use shuvoice_tts::PiperTtsBackend;
    assert!((PiperTtsBackend::length_scale_for_speed(2.0).unwrap() - 0.5).abs() < 1e-9);
    assert!((PiperTtsBackend::length_scale_for_speed(0.5).unwrap() - 2.0).abs() < 1e-9);
}

#[test]
fn piper_sidecar_sample_rate() {
    use shuvoice_tts::piper_sample_rate_from_sidecar;
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("en.onnx");
    std::fs::write(&model, b"fake").unwrap();
    let sidecar = dir.path().join("en.onnx.json");
    std::fs::write(&sidecar, r#"{"audio":{"sample_rate":22050}}"#).unwrap();
    assert_eq!(piper_sample_rate_from_sidecar(&model), Some(22_050));
}

#[tokio::test]
async fn piper_discovers_voices() {
    use shuvoice_tts::{PiperConfig, PiperTtsBackend};
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.onnx"), b"x").unwrap();
    std::fs::write(dir.path().join("b.onnx"), b"x").unwrap();
    // Provide a fake piper binary script.
    let bin = dir.path().join("piper");
    std::fs::write(&bin, b"#!/bin/sh\ncat >/dev/null\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&bin).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&bin, perms).unwrap();
    }
    let cfg = PiperConfig {
        model_path: dir.path().to_path_buf(),
        default_voice_id: "default".into(),
        local_voice: None,
        max_chars: 5000,
        request_timeout: Duration::from_secs(2),
        piper_binary: Some(bin),
    };
    let backend = PiperTtsBackend::new(cfg).unwrap();
    let voices = backend.list_voices().await.unwrap();
    assert_eq!(voices.len(), 2);
}

#[tokio::test]
async fn melotts_frame_reader() {
    use shuvoice_tts::MeloTtsBackend;
    let pcm = vec![1u8, 0, 2, 0, 3, 0];
    let mut framed = Vec::new();
    framed.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    framed.extend_from_slice(&pcm);
    let mut cursor = &framed[..];
    let out = MeloTtsBackend::read_framed_pcm(&mut cursor).await.unwrap();
    assert_eq!(&out[..], &pcm[..]);
}

#[tokio::test]
async fn melotts_truncated_header() {
    use shuvoice_tts::MeloTtsBackend;
    let mut cursor = &[0u8, 1, 2][..];
    let err = MeloTtsBackend::read_framed_pcm(&mut cursor)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("Incomplete frame header"));
}

#[test]
fn melotts_request_json() {
    use shuvoice_tts::MeloTtsBackend;
    let s = MeloTtsBackend::build_request_json("hi", "EN-US", 1.25);
    let v: serde_json::Value = serde_json::from_str(&s).unwrap();
    assert_eq!(v["text"], "hi");
    assert_eq!(v["voice_id"], "EN-US");
    assert_eq!(v["speed"], 1.25);
}

#[test]
fn melotts_dependency_errors_missing_venv_legacy() {
    use shuvoice_tts::{MeloTtsBackend, MeloTtsConfig, MeloWireMode};
    let cfg = MeloTtsConfig {
        venv_path: PathBuf::from("/no/such/melotts-venv"),
        wire_mode: MeloWireMode::LegacyHelper,
        ..MeloTtsConfig::default()
    };
    let backend = MeloTtsBackend::new_for_test(cfg);
    let errors = backend.dependency_errors();
    assert!(!errors.is_empty());
    assert!(errors[0].contains("does not exist"));
}

#[test]
fn melotts_production_new_rejects_legacy_footgun() {
    use shuvoice_tts::{MeloTtsBackend, MeloTtsConfig, MeloWireMode};
    let cfg = MeloTtsConfig {
        wire_mode: MeloWireMode::LegacyHelper,
        helper_script: Some(PathBuf::from("/tmp/melo_helper.py")),
        ..MeloTtsConfig::default()
    };
    let backend = MeloTtsBackend::new(cfg);
    assert_eq!(backend.config().wire_mode, MeloWireMode::WorkerProto);
    assert!(backend.config().helper_script.is_none());
}

#[test]
fn melotts_child_env_allowlist_public_api() {
    use shuvoice_tts::{CHILD_ENV_ALLOWLIST, build_isolated_child_env};
    assert!(CHILD_ENV_ALLOWLIST.contains(&"PATH"));
    assert!(CHILD_ENV_ALLOWLIST.contains(&"CUDA_VISIBLE_DEVICES"));
    assert!(!CHILD_ENV_ALLOWLIST.iter().any(|k| k.contains("API_KEY")));
    assert!(!CHILD_ENV_ALLOWLIST.contains(&"HTTP_PROXY"));
    assert!(!CHILD_ENV_ALLOWLIST.contains(&"HTTPS_PROXY"));
    assert!(!CHILD_ENV_ALLOWLIST.contains(&"http_proxy"));
    assert!(!CHILD_ENV_ALLOWLIST.contains(&"https_proxy"));
    assert!(!CHILD_ENV_ALLOWLIST.contains(&"ALL_PROXY"));
    assert!(!CHILD_ENV_ALLOWLIST.contains(&"NO_PROXY"));
    // SAFETY: test mutates process env and restores it before returning.
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "sk-sentinel-public-api");
        std::env::set_var("ELEVENLABS_API_KEY", "el-sentinel-public-api");
        std::env::set_var(
            "HTTPS_PROXY",
            "http://proxy-user:proxy-pass-sentinel@127.0.0.1:8443",
        );
        std::env::set_var(
            "HTTP_PROXY",
            "http://proxy-user:proxy-pass-sentinel@127.0.0.1:8080",
        );
    }
    let env = build_isolated_child_env(&[]);
    assert!(!env.iter().any(|(k, _)| k == "OPENAI_API_KEY"));
    assert!(!env.iter().any(|(k, _)| k == "ELEVENLABS_API_KEY"));
    assert!(!env.iter().any(|(k, _)| k == "HTTPS_PROXY"));
    assert!(!env.iter().any(|(k, _)| k == "HTTP_PROXY"));
    assert!(env.iter().all(|(_, v)| !v.contains("proxy-pass-sentinel")));
    // Explicit overlay remains the deliberate opt-in path.
    let env_opt_in = build_isolated_child_env(&[(
        "HTTPS_PROXY".into(),
        "http://explicit-only@127.0.0.1:9".into(),
    )]);
    assert!(
        env_opt_in
            .iter()
            .any(|(k, v)| k == "HTTPS_PROXY" && v.contains("explicit-only"))
    );
    // SAFETY: restore process env mutated above.
    unsafe {
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("ELEVENLABS_API_KEY");
        std::env::remove_var("HTTPS_PROXY");
        std::env::remove_var("HTTP_PROXY");
    }
}

#[test]
#[cfg(feature = "worker-proto")]
fn melotts_worker_proto_resolve_spawn_shape() {
    use shuvoice_tts::{MeloTtsBackend, MeloTtsConfig, MeloWireMode};
    use std::fs;
    use std::path::Path;

    let root = tempfile::tempdir().unwrap();
    let melotts = root.path().join("melotts");
    fs::create_dir_all(&melotts).unwrap();
    fs::write(melotts.join("__main__.py"), b"# test\n").unwrap();
    let proto = root.path().join("shuvoice_worker_proto");
    fs::create_dir_all(&proto).unwrap();
    fs::write(proto.join("__init__.py"), b"").unwrap();

    let venv = tempfile::tempdir().unwrap();
    let bin = venv.path().join("bin");
    fs::create_dir_all(&bin).unwrap();
    let python = bin.join("python");
    fs::write(&python, b"#!/bin/sh\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&python, perms).unwrap();
    }

    let cfg = MeloTtsConfig {
        venv_path: venv.path().to_path_buf(),
        device: "cpu".into(),
        wire_mode: MeloWireMode::WorkerProto,
        worker_root: Some(root.path().to_path_buf()),
        helper_script: Some(PathBuf::from("/tmp/melo_helper.py")),
        ..MeloTtsConfig::default()
    };
    let spawn = MeloTtsBackend::new(cfg).resolve_spawn().unwrap();
    assert_eq!(spawn.program, python);
    assert_eq!(
        spawn.args,
        vec![
            "-m".to_string(),
            "melotts".to_string(),
            "--device".to_string(),
            "cpu".to_string()
        ]
    );
    assert_eq!(spawn.current_dir.as_deref(), Some(root.path()));
    assert!(
        spawn
            .env
            .iter()
            .any(|(k, v)| k == "PYTHONPATH" && Path::new(v) == root.path())
    );
    assert!(spawn.args.iter().all(|a| !a.contains("melo_helper")));
}

#[cfg(feature = "worker-proto")]
mod melotts_worker_proto_process {
    use std::path::{Path, PathBuf};
    use std::process::Stdio;
    use std::time::{Duration, Instant};

    use futures_util::StreamExt;
    use shuvoice_tts::{
        MeloTtsBackend, MeloTtsConfig, MeloWireMode, MeloWorkerSpawn, SynthesisRequest, TtsBackend,
        TtsError,
    };
    use tokio_util::sync::CancellationToken;

    fn python_executable() -> Option<PathBuf> {
        if let Some(explicit) = std::env::var_os("SHUVOICE_TEST_PYTHON") {
            let path = PathBuf::from(explicit);
            if path.as_os_str().is_empty() {
                return None;
            }
            return Some(path);
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

    fn workers_dir() -> Option<PathBuf> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut candidates = vec![
            manifest_dir.join("../../workers"),
            manifest_dir.join("../../../workers"),
        ];
        if let Ok(cwd) = std::env::current_dir() {
            candidates.push(cwd.join("workers"));
            candidates.push(cwd.join("../workers"));
        }
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
                return Some(candidate.canonicalize().unwrap_or(candidate));
            }
        }
        None
    }

    fn is_workers_tree(path: &Path) -> bool {
        path.join("melotts").join("__main__.py").is_file()
            && path
                .join("shuvoice_worker_proto")
                .join("__init__.py")
                .is_file()
    }

    fn hung_spawn(python: PathBuf) -> MeloWorkerSpawn {
        // Deliberately never speaks worker-proto; sleeps forever.
        MeloWorkerSpawn::new(python).args(["-c", "import time\ntime.sleep(3600)\n"])
    }

    #[tokio::test]
    async fn fake_bundled_worker_synthesizes_pcm_and_rate() {
        let Some(python) = python_executable() else {
            eprintln!("SKIP melotts fake worker: python unavailable");
            return;
        };
        let Some(workers) = workers_dir() else {
            eprintln!("SKIP melotts fake worker: workers/ tree not found");
            return;
        };

        let spawn = MeloWorkerSpawn::new(python)
            .args(["-m", "melotts", "--fake", "--device", "cpu"])
            .current_dir(&workers)
            .env_pair("PYTHONPATH", workers.to_string_lossy())
            .env_pair("SHUVOICE_WORKER_FAKE", "1")
            .env_pair("PYTHONUNBUFFERED", "1");

        let cfg = MeloTtsConfig {
            wire_mode: MeloWireMode::WorkerProto,
            worker_spawn: Some(spawn),
            request_timeout: Duration::from_secs(10),
            max_chars: 5000,
            default_voice_id: "EN-US".into(),
            ..MeloTtsConfig::default()
        };
        let backend = MeloTtsBackend::new(cfg);
        let mut stream = backend
            .synthesize_stream(
                SynthesisRequest::new("hello from tts crate", "EN-US", "melotts", 1.0),
                CancellationToken::new(),
            )
            .await
            .expect("synthesize_stream starts");
        assert_eq!(stream.sample_rate_hz, 44_100);

        let mut pcm = Vec::new();
        while let Some(item) = stream.chunks.next().await {
            let chunk = item.expect("pcm chunk");
            pcm.extend_from_slice(&chunk);
        }
        assert!(!pcm.is_empty(), "expected non-empty PCM from fake worker");
        assert_eq!(pcm.len() % 2, 0, "i16le alignment");
    }

    #[tokio::test]
    async fn hung_worker_times_out_within_bound() {
        let Some(python) = python_executable() else {
            eprintln!("SKIP hung timeout: python unavailable");
            return;
        };
        let cfg = MeloTtsConfig {
            worker_spawn: Some(hung_spawn(python)),
            request_timeout: Duration::from_millis(400),
            ..MeloTtsConfig::default()
        };
        let backend = MeloTtsBackend::new(cfg);
        let start = Instant::now();
        let result = backend
            .synthesize_stream(
                SynthesisRequest::new("timeout please", "EN-US", "melotts", 1.0),
                CancellationToken::new(),
            )
            .await;
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(3),
            "timeout wall-clock too large: {elapsed:?}"
        );
        match result {
            Err(err) => assert!(
                matches!(err, TtsError::TimedOut(_))
                    || err.to_string().to_ascii_lowercase().contains("timed out")
                    || err.to_string().to_ascii_lowercase().contains("timeout"),
                "unexpected err {err}"
            ),
            Ok(_) => panic!("expected timeout error"),
        }
    }

    #[tokio::test]
    async fn hung_worker_cancel_returns_promptly() {
        let Some(python) = python_executable() else {
            eprintln!("SKIP hung cancel: python unavailable");
            return;
        };
        let cfg = MeloTtsConfig {
            worker_spawn: Some(hung_spawn(python)),
            request_timeout: Duration::from_secs(30),
            ..MeloTtsConfig::default()
        };
        let backend = MeloTtsBackend::new(cfg);
        let cancel = CancellationToken::new();
        let cancel_task = cancel.clone();
        let join = tokio::spawn(async move {
            backend
                .synthesize_stream(
                    SynthesisRequest::new("cancel please", "EN-US", "melotts", 1.0),
                    cancel_task,
                )
                .await
        });
        tokio::time::sleep(Duration::from_millis(80)).await;
        let start = Instant::now();
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(3), join)
            .await
            .expect("cancel must not hang past 3s")
            .expect("join");
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(2),
            "cancel wall-clock too large: {elapsed:?}"
        );
        match result {
            Err(TtsError::Cancelled) => {}
            Err(err) => panic!("expected Cancelled, got error: {err}"),
            Ok(_) => panic!("expected Cancelled, got Ok stream"),
        }
    }

    #[tokio::test]
    async fn child_env_strips_api_key_sentinels_process() {
        let Some(python) = python_executable() else {
            eprintln!("SKIP env isolation process: python unavailable");
            return;
        };
        let marker = tempfile::NamedTempFile::new().unwrap();
        let marker_path = marker.path().to_path_buf();
        // SAFETY: test-only sentinels in process env; removed after the probe.
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "sk-process-sentinel-openai");
            std::env::set_var("ELEVENLABS_API_KEY", "el-process-sentinel-eleven");
        }
        let marker_str = marker_path.display().to_string();
        let code = format!(
            "import os, pathlib, time\np = pathlib.Path({marker_str:?})\np.write_text('openai=' + str('OPENAI_API_KEY' in os.environ) + ' eleven=' + str('ELEVENLABS_API_KEY' in os.environ) + ' path=' + str(bool(os.environ.get('PATH'))))\ntime.sleep(30)\n"
        );
        let spawn = MeloWorkerSpawn::new(python).args(["-c", &code]);
        let cfg = MeloTtsConfig {
            worker_spawn: Some(spawn),
            request_timeout: Duration::from_millis(500),
            ..MeloTtsConfig::default()
        };
        let backend = MeloTtsBackend::new(cfg);
        let _ = backend
            .synthesize_stream(
                SynthesisRequest::new("env probe", "EN-US", "melotts", 1.0),
                CancellationToken::new(),
            )
            .await;
        let mut body = String::new();
        for _ in 0..20 {
            body = std::fs::read_to_string(&marker_path).unwrap_or_default();
            if body.contains("openai=") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        // SAFETY: restore process env mutated for the child probe.
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("ELEVENLABS_API_KEY");
        }
        assert!(
            body.contains("openai=False"),
            "OPENAI_API_KEY leaked into child env; marker={body:?}"
        );
        assert!(
            body.contains("eleven=False"),
            "ELEVENLABS_API_KEY leaked into child env; marker={body:?}"
        );
        assert!(
            body.contains("path=True"),
            "PATH should be preserved; marker={body:?}"
        );
    }

    #[tokio::test]
    async fn pre_cancelled_token_returns_immediately() {
        let Some(python) = python_executable() else {
            eprintln!("SKIP pre-cancel: python unavailable");
            return;
        };
        let Some(workers) = workers_dir() else {
            eprintln!("SKIP pre-cancel: workers missing");
            return;
        };
        let spawn = MeloWorkerSpawn::new(python)
            .args(["-m", "melotts", "--fake", "--device", "cpu"])
            .current_dir(&workers)
            .env_pair("PYTHONPATH", workers.to_string_lossy())
            .env_pair("SHUVOICE_WORKER_FAKE", "1")
            .env_pair("PYTHONUNBUFFERED", "1");
        let cfg = MeloTtsConfig {
            worker_spawn: Some(spawn),
            request_timeout: Duration::from_secs(10),
            ..MeloTtsConfig::default()
        };
        let backend = MeloTtsBackend::new(cfg);
        let cancel = CancellationToken::new();
        cancel.cancel();
        let start = Instant::now();
        let result = backend
            .synthesize_stream(
                SynthesisRequest::new("already cancelled", "EN-US", "melotts", 1.0),
                cancel,
            )
            .await;
        assert!(start.elapsed() < Duration::from_secs(2));
        match result {
            Err(TtsError::Cancelled) => {}
            Err(err) => panic!("expected Cancelled, got {err}"),
            Ok(_) => panic!("expected Cancelled"),
        }
    }
}

// Silence unused import in some rustc versions
#[allow(dead_code)]
fn _event_info() -> EventInfo {
    EventInfo::default()
}
