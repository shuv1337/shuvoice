//! Setup / model / preflight tests with fake runners + scripted HTTP (no network/sudo).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serial_test::serial;
use shuvoice_cli::commands::setup;
use shuvoice_cli::commands::{model, preflight};
use shuvoice_cli::error::{EXIT_DEPENDENCY, EXIT_FAILURE, EXIT_SUCCESS};
use shuvoice_cli::setup::SetupContext;
use shuvoice_cli::setup::http::{
    HttpDownloader, ReqwestDownloader, ScriptedDownloader, host_is_allowed, publish_paired_files,
    validate_download_url,
};
use shuvoice_cli::setup::install::{
    HostProfile, NEMO_WORKER_VENV_NAME, asr_feature_guidance, asr_install_pipelines,
    asr_install_plans, run_install_pipelines,
};
use shuvoice_cli::setup::melotts::{
    melotts_install_commands, melotts_missing_dependencies, melotts_venv_valid, run_melotts_install,
};
use shuvoice_cli::setup::piper::{
    CURATED_PIPER_VOICES, ensure_local_piper_ready, get_curated_piper_voice, piper_install_hints,
    recommended_piper_voice, validate_piper_voice_artifacts,
};
use shuvoice_cli::setup::sherpa_model::{
    format_sherpa_provider_line, is_complete_sherpa_dir, model_status_line,
    sherpa_cuda_provider_errors, sherpa_model_dir, sherpa_runtime_errors,
};
use shuvoice_core::{AsrBackendKind, ComputeProvider, Config, TtsBackendKind};
use shuvoice_io::process::{RunOutput, ScriptedRunner};
use tempfile::TempDir;

/// RAII env var restore for serial tests.
struct EnvGuard {
    key: &'static str,
    prev: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        let prev = std::env::var_os(key);
        // SAFETY: callers are `#[serial]` and Drop restores the previous value.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: restores the process-global env value captured in `set` under serial tests.
        unsafe {
            match &self.prev {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

fn cfg_sherpa() -> Config {
    let mut c = Config::default();
    c.asr_backend = AsrBackendKind::Sherpa;
    let _ = c.validate();
    c
}

fn with_xdg(tmp: &TempDir) -> (EnvGuard, EnvGuard, EnvGuard) {
    let config = tmp.path().join("config");
    let data = tmp.path().join("data");
    let runtime = tmp.path().join("runtime");
    std::fs::create_dir_all(config.join("shuvoice")).unwrap();
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&runtime).unwrap();
    std::fs::write(config.join("shuvoice/config.toml"), "config_version = 1\n").unwrap();
    (
        EnvGuard::set("XDG_CONFIG_HOME", &config),
        EnvGuard::set("XDG_DATA_HOME", &data),
        EnvGuard::set("XDG_RUNTIME_DIR", &runtime),
    )
}

// ─── install plans ──────────────────────────────────────────────────────────

#[test]
fn sherpa_native_install_plan_is_empty_no_python_wheels() {
    let host = HostProfile {
        is_arch: true,
        has_yay: true,
        has_uv: true,
        in_venv: true,
        has_nvidia_smi: true,
        ..HostProfile::default()
    };
    let plans = asr_install_plans(AsrBackendKind::Sherpa, &host);
    assert!(
        plans.is_empty(),
        "native Sherpa must not pip-install wheels: {plans:?}"
    );
    let guide = asr_feature_guidance(AsrBackendKind::Sherpa);
    assert!(guide.iter().any(|g| g.contains("asr-sherpa")));
}

#[test]
fn nemo_moonshine_get_isolated_worker_venv_plans() {
    let host = HostProfile {
        has_uv: true,
        ..HostProfile::default()
    };
    let pipes = asr_install_pipelines(AsrBackendKind::Nemo, &host);
    assert!(pipes.len() >= 2, "uv + python fallback: {pipes:?}");
    for p in &pipes {
        assert_eq!(p.len(), 2, "sequential venv then pip: {p:?}");
    }
    assert!(
        pipes
            .iter()
            .flatten()
            .any(|c| c.iter().any(|a| a.contains(NEMO_WORKER_VENV_NAME)))
    );
    // No shell metacharacters.
    for cmd in asr_install_plans(AsrBackendKind::Nemo, &host) {
        for arg in &cmd {
            assert!(!arg.contains('|') && !arg.contains(';') && !arg.contains("&&"));
        }
    }
    let moon = asr_install_pipelines(AsrBackendKind::Moonshine, &host);
    assert!(
        moon.iter()
            .flatten()
            .any(|c| c.join(" ").contains("moonshine"))
    );
    assert!(
        moon.iter()
            .flatten()
            .any(|c| c.iter().any(|a| a.contains("workers-moonshine-venv")))
    );
}

#[test]
fn sequential_pipeline_runs_all_venv_and_pip_steps() {
    let runner = ScriptedRunner::new();
    let calls = Arc::new(std::sync::Mutex::new(Vec::<Vec<String>>::new()));
    let calls2 = calls.clone();
    runner.set_dynamic(move |argv| {
        calls2.lock().unwrap().push(argv.to_vec());
        Ok(RunOutput {
            status_code: Some(0),
            stdout: Vec::new(),
            stderr: Vec::new(),
            success: true,
        })
    });
    let pipeline = vec![
        vec!["true".into(), "venv".into(), "/tmp/x".into()],
        vec![
            "true".into(),
            "-m".into(),
            "pip".into(),
            "install".into(),
            "pkg".into(),
        ],
    ];
    let n = run_install_pipelines(std::slice::from_ref(&pipeline), &runner).unwrap();
    assert_eq!(n, 2);
    let recorded = calls.lock().unwrap();
    assert_eq!(recorded.len(), 2);
    assert!(recorded[0].iter().any(|a| a == "venv"));
    assert!(recorded[1].iter().any(|a| a == "pip"));
}

#[test]
fn sequential_pipeline_does_not_stop_after_first_success() {
    let runner = ScriptedRunner::new();
    let calls = Arc::new(std::sync::Mutex::new(0usize));
    let c2 = calls.clone();
    runner.set_dynamic(move |argv| {
        *c2.lock().unwrap() += 1;
        let ok = argv.iter().any(|a| a == "venv"); // pip fails
        Ok(RunOutput {
            status_code: Some(if ok { 0 } else { 1 }),
            stdout: Vec::new(),
            stderr: Vec::new(),
            success: ok,
        })
    });
    let pipeline = vec![
        vec!["true".into(), "venv".into()],
        vec!["true".into(), "pip".into(), "install".into()],
    ];
    let err = run_install_pipelines(std::slice::from_ref(&pipeline), &runner).unwrap_err();
    assert!(err.contains("failed") || err.contains("exit"), "{err}");
    assert_eq!(*calls.lock().unwrap(), 2, "must attempt pip after venv");
}

// ─── sherpa model completeness ──────────────────────────────────────────────

#[test]
fn sherpa_dir_complete_requires_tokens_and_onnx() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("model");
    std::fs::create_dir_all(&dir).unwrap();
    assert!(!is_complete_sherpa_dir(&dir));
    std::fs::write(dir.join("tokens.txt"), "a").unwrap();
    std::fs::write(dir.join("encoder.onnx"), "x").unwrap();
    std::fs::write(dir.join("decoder.onnx"), "x").unwrap();
    std::fs::write(dir.join("joiner.onnx"), "x").unwrap();
    assert!(is_complete_sherpa_dir(&dir));
}

#[test]
fn model_status_reports_missing_and_present() {
    let tmp = TempDir::new().unwrap();
    let mut c = cfg_sherpa();
    c.sherpa_model_dir = Some(tmp.path().join("m").display().to_string());
    let s = model_status_line(&c);
    assert!(s.contains("missing"), "{s}");
    let dir = tmp.path().join("m");
    std::fs::create_dir_all(&dir).unwrap();
    for name in ["tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"] {
        std::fs::write(dir.join(name), "x").unwrap();
    }
    let s = model_status_line(&c);
    assert!(s.contains("present"), "{s}");
}

#[test]
#[serial]
fn sherpa_model_dir_expands_tilde() {
    let tmp = TempDir::new().unwrap();
    let _home = EnvGuard::set("HOME", tmp.path());
    let mut c = cfg_sherpa();
    c.sherpa_model_dir = Some("~/models/sherpa-x".into());
    let dir = sherpa_model_dir(&c);
    assert_eq!(dir, tmp.path().join("models/sherpa-x"));
}

// ─── model download (already present / feature gate) ────────────────────────

#[tokio::test]
async fn model_download_skips_when_present() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("sherpa-model");
    std::fs::create_dir_all(&dir).unwrap();
    for name in ["tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"] {
        std::fs::write(dir.join(name), "x").unwrap();
    }
    let mut c = cfg_sherpa();
    c.sherpa_model_dir = Some(dir.display().to_string());
    let msg = model::download_model_impl(&c, None).await.unwrap();
    assert!(msg.contains("already present"), "{msg}");
}

#[tokio::test]
async fn model_download_openai_skipped() {
    let mut c = Config::default();
    c.asr_backend = AsrBackendKind::OpenaiRealtime;
    let _ = c.validate();
    let msg = model::download_model_impl(&c, None).await.unwrap();
    assert!(msg.contains("skipped"));
}

#[tokio::test]
async fn model_download_nemo_and_moonshine_are_consistent_skips() {
    for backend in [AsrBackendKind::Nemo, AsrBackendKind::Moonshine] {
        let mut c = Config::default();
        c.asr_backend = backend;
        let _ = c.validate();
        let msg = model::download_model_impl(&c, None).await.unwrap();
        assert!(msg.contains("skipped"), "{backend:?}: {msg}");
        assert!(msg.contains("worker") || msg.contains("lazily"), "{msg}");
        let status = model::download_model(&c).await;
        assert_eq!(status.code, EXIT_SUCCESS, "{backend:?} must not exit 78");
    }
}

// ─── HTTP policy ────────────────────────────────────────────────────────────

#[test]
fn https_and_curated_host_policy() {
    assert!(
        validate_download_url("https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/x")
            .is_ok()
    );
    assert!(host_is_allowed("cdn.huggingface.co"));
    assert!(validate_download_url("http://huggingface.co/x").is_err());
    assert!(validate_download_url("https://evil.example/x").is_err());
    assert!(validate_download_url("file:///tmp/x").is_err());
}

#[tokio::test]
async fn production_downloader_rejects_file_urls() {
    let dl = ReqwestDownloader::default();
    let tmp = TempDir::new().unwrap();
    let dest = tmp.path().join("out.bin");
    let mut prog = |_: Option<f32>, _: &str| {};
    let err = dl
        .download_to_file("file:///etc/hosts", &dest, 1024, &mut prog)
        .await
        .unwrap_err();
    assert!(err.contains("file://") || err.contains("disabled"), "{err}");
}

#[tokio::test]
async fn test_seam_allows_file_urls() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src.bin");
    std::fs::write(&src, b"hello-file-url").unwrap();
    let dest = tmp.path().join("dest.bin");
    let dl = ReqwestDownloader::with_file_urls();
    let mut prog = |_: Option<f32>, _: &str| {};
    dl.download_to_file(&format!("file://{}", src.display()), &dest, 1024, &mut prog)
        .await
        .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), b"hello-file-url");
}

#[test]
fn publish_paired_is_transactional() {
    let tmp = TempDir::new().unwrap();
    let stage_m = tmp.path().join("s.onnx");
    let stage_s = tmp.path().join("s.onnx.json");
    let final_m = tmp.path().join("f.onnx");
    let final_s = tmp.path().join("f.onnx.json");
    std::fs::write(&final_m, b"old-model").unwrap();
    std::fs::write(&final_s, b"old-side").unwrap();
    std::fs::write(&stage_m, b"new-model-bytes").unwrap();
    // Missing stage sidecar → fail and leave finals intact.
    let err = publish_paired_files(&stage_m, &stage_s, &final_m, &final_s).unwrap_err();
    assert!(!err.is_empty());
    assert_eq!(std::fs::read(&final_m).unwrap(), b"old-model");
    assert_eq!(std::fs::read(&final_s).unwrap(), b"old-side");

    std::fs::write(&stage_s, br#"{"sample_rate":22050}"#).unwrap();
    // stage_m may have been consumed? rewrite
    std::fs::write(&stage_m, b"new-model-bytes").unwrap();
    publish_paired_files(&stage_m, &stage_s, &final_m, &final_s).unwrap();
    assert_eq!(std::fs::read(&final_m).unwrap(), b"new-model-bytes");
}

// ─── Piper curated download ─────────────────────────────────────────────────

#[tokio::test]
#[serial]
async fn piper_download_atomic_and_validates() {
    let tmp = TempDir::new().unwrap();
    let model_dir = tmp.path().join("piper");
    let voice = recommended_piper_voice();
    let dl = ScriptedDownloader::default();
    dl.insert(voice.model_url, vec![0u8; 2048]);
    dl.insert(
        voice.sidecar_url,
        br#"{"audio":{"sample_rate":22050}}"#.to_vec(),
    );

    let runner = ScriptedRunner::new();
    let bin_dir = tmp.path().join("bin");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let stub = bin_dir.join("piper");
    std::fs::write(&stub, b"#!/bin/sh\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    let mut new_path = std::env::split_paths(&path).collect::<Vec<_>>();
    new_path.insert(0, bin_dir);
    let joined = std::env::join_paths(new_path).unwrap();
    let _path_guard = EnvGuard::set("PATH", &joined);

    let mut prog = |_: Option<f32>, _: &str| {};
    let result = ensure_local_piper_ready(voice, &model_dir, false, &dl, &runner, &mut prog)
        .await
        .unwrap();
    assert_eq!(result.status, "ok", "{}", result.message);
    assert!(model_dir.join(format!("{}.onnx", voice.stem)).is_file());
    assert!(
        model_dir
            .join(format!("{}.onnx.json", voice.stem))
            .is_file()
    );
    let (_, rate) = validate_piper_voice_artifacts(&model_dir, Some(voice.stem)).unwrap();
    assert_eq!(rate, Some(22050));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let model = model_dir.join(format!("{}.onnx", voice.stem));
        let side = model_dir.join(format!("{}.onnx.json", voice.stem));
        let m = std::fs::metadata(&model).unwrap().permissions().mode() & 0o777;
        let s = std::fs::metadata(&side).unwrap().permissions().mode() & 0o777;
        assert_eq!(m, 0o600, "piper model mode {m:#o}");
        assert_eq!(s, 0o600, "piper sidecar mode {s:#o}");
    }
}

#[test]
fn curated_voice_lookup() {
    assert_eq!(
        get_curated_piper_voice("en_US-amy-medium").unwrap().stem,
        "en_US-amy-medium"
    );
    assert!(get_curated_piper_voice("nope").is_err());
    assert_eq!(CURATED_PIPER_VOICES.len(), 6);
}

#[test]
fn piper_hints_cover_non_arch() {
    let hints = piper_install_hints();
    assert!(!hints.is_empty());
    let joined = hints.join(" | ");
    assert!(
        joined.to_ascii_lowercase().contains("non-arch")
            || joined.to_ascii_lowercase().contains("manual")
            || joined.contains("PATH"),
        "{joined}"
    );
}

#[test]
fn validate_piper_requires_sidecar() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("v.onnx"), b"xxxx").unwrap();
    let err = validate_piper_voice_artifacts(tmp.path(), Some("v")).unwrap_err();
    assert!(err.to_ascii_lowercase().contains("sidecar"), "{err}");
}

#[test]
#[serial]
fn validate_piper_expands_tilde_paths() {
    let tmp = TempDir::new().unwrap();
    let _home = EnvGuard::set("HOME", tmp.path());
    let dir = tmp.path().join("voices");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("v.onnx"), b"xxxx").unwrap();
    std::fs::write(dir.join("v.onnx.json"), br#"{"sample_rate":16000}"#).unwrap();
    let (_, rate) = validate_piper_voice_artifacts(Path::new("~/voices"), Some("v")).unwrap();
    assert_eq!(rate, Some(16000));
}

// ─── MeloTTS install plan / idempotent ──────────────────────────────────────

#[test]
fn melotts_install_commands_are_argv_lists() {
    let venv = PathBuf::from("/tmp/melotts-test-venv");
    let cmds = melotts_install_commands(&venv);
    assert_eq!(cmds.len(), 4);
    assert_eq!(cmds[0][0], "uv");
    assert!(cmds[1].contains(&"venv".into()));
    for cmd in &cmds {
        for a in cmd {
            assert!(!a.contains("&&"));
        }
    }
}

#[test]
#[serial]
fn melotts_install_skips_venv_when_valid() {
    let tmp = TempDir::new().unwrap();
    let venv = tmp.path().join("venv");
    std::fs::create_dir_all(venv.join("bin")).unwrap();
    let py = venv.join("bin/python");
    std::fs::write(&py, b"#!/bin/sh\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    assert!(melotts_venv_valid(&venv));

    let runner = ScriptedRunner::new();
    runner.set_dynamic(|argv| {
        assert!(!argv.is_empty());
        Ok(RunOutput {
            status_code: Some(0),
            stdout: Vec::new(),
            stderr: Vec::new(),
            success: true,
        })
    });
    let bin = tmp.path().join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let p = bin.join("uv");
    std::fs::write(&p, b"#!/bin/sh\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let old = std::env::var_os("PATH");
    let _path = EnvGuard::set(
        "PATH",
        format!(
            "{}:{}",
            bin.display(),
            old.as_ref()
                .map(|s| s.to_string_lossy())
                .unwrap_or_default()
        ),
    );
    run_melotts_install(&venv, &runner, true).unwrap();
    let calls = runner.calls();
    assert!(
        !calls
            .iter()
            .any(|c| c.get(1).map(|s| s.as_str()) == Some("venv")),
        "venv create should be skipped when already valid: {calls:?}"
    );
}

#[test]
fn melotts_missing_when_absent() {
    let missing = melotts_missing_dependencies(std::path::Path::new("/no/such/melotts-venv"));
    assert!(!missing.is_empty());
}

// ─── Sherpa CUDA fail-closed ────────────────────────────────────────────────

#[test]
fn sherpa_cuda_provider_fail_closed_with_cpu_guidance() {
    let mut c = cfg_sherpa();
    c.sherpa_provider = ComputeProvider::Cuda;
    let _ = c.validate();
    let errs = sherpa_cuda_provider_errors(&c);
    assert!(!errs.is_empty());
    let joined = errs.join(" ");
    assert!(
        joined.contains("unsupported") || joined.contains("does not support CUDA"),
        "{joined}"
    );
    assert!(joined.contains("cpu") || joined.contains("CPU"), "{joined}");
    let line = format_sherpa_provider_line(&c);
    assert!(line.contains("requested=cuda"), "{line}");
    assert!(line.contains("unsupported"), "{line}");
    assert!(
        !line.contains("effective=cuda"),
        "must not lie effective=cuda: {line}"
    );
}

#[test]
fn sherpa_cpu_provider_reports_effective_cpu_only() {
    let mut c = cfg_sherpa();
    c.sherpa_provider = ComputeProvider::Cpu;
    let _ = c.validate();
    assert!(sherpa_cuda_provider_errors(&c).is_empty());
    let line = format_sherpa_provider_line(&c);
    assert_eq!(line, "[INFO] Sherpa provider: requested=cpu effective=cpu");
}

#[tokio::test]
#[serial]
async fn setup_sherpa_cuda_exits_78_fail_closed() {
    let tmp = TempDir::new().unwrap();
    let _xdg = with_xdg(&tmp);
    let mut c = Config::default();
    c.asr_backend = AsrBackendKind::Sherpa;
    c.sherpa_provider = ComputeProvider::Cuda;
    // Avoid other gates when feature is on.
    c.sherpa_decode_mode = shuvoice_core::SherpaDecodeMode::OfflineInstant;
    let _ = c.validate();
    let status = setup::run_setup(
        &c,
        setup::SetupOptions {
            install_missing: false,
            skip_model_download: true,
            skip_preflight: true,
            tts_local_voice: None,
            tts_local_model_dir: None,
            non_interactive: true,
        },
    )
    .await;
    assert_eq!(
        status.code, EXIT_DEPENDENCY,
        "cuda must fail closed with 78, got {}",
        status.code
    );
}

#[tokio::test]
#[serial]
async fn preflight_sherpa_cuda_not_ready() {
    let tmp = TempDir::new().unwrap();
    let _xdg = with_xdg(&tmp);
    let mut c = Config::default();
    c.asr_backend = AsrBackendKind::Sherpa;
    c.sherpa_provider = ComputeProvider::Cuda;
    c.tts_enabled = false;
    // Provide complete model so only CUDA gate fails among ASR checks.
    let dir = tmp.path().join("sherpa-model");
    std::fs::create_dir_all(&dir).unwrap();
    for name in ["tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"] {
        std::fs::write(dir.join(name), "x").unwrap();
    }
    c.sherpa_model_dir = Some(dir.display().to_string());
    let _ = c.validate();
    assert!(!sherpa_runtime_errors(&c).is_empty());
    let status = preflight::run_preflight(&c).await;
    assert_eq!(status.code, EXIT_FAILURE);
}

// ─── setup dependency exit 78 ───────────────────────────────────────────────

/// Explicit no-default-features / missing asr-sherpa gate: setup must exit 78.
#[cfg(not(feature = "asr-sherpa"))]
#[tokio::test]
#[serial]
async fn setup_without_asr_sherpa_feature_exits_78() {
    let tmp = TempDir::new().unwrap();
    let _xdg = with_xdg(&tmp);
    let mut c = Config::default();
    c.asr_backend = AsrBackendKind::Sherpa;
    c.sherpa_provider = ComputeProvider::Cpu;
    let _ = c.validate();

    let status = setup::run_setup(
        &c,
        setup::SetupOptions {
            install_missing: false,
            skip_model_download: true,
            skip_preflight: true,
            tts_local_voice: None,
            tts_local_model_dir: None,
            non_interactive: true,
        },
    )
    .await;
    assert_eq!(
        status.code, EXIT_DEPENDENCY,
        "missing asr-sherpa feature must exit 78, got {}",
        status.code
    );
}

#[cfg(feature = "asr-sherpa")]
#[tokio::test]
#[serial]
async fn setup_missing_sherpa_feature_exits_78_without_feature() {
    let tmp = TempDir::new().unwrap();
    let _xdg = with_xdg(&tmp);
    let mut c = Config::default();
    c.asr_backend = AsrBackendKind::Sherpa;
    c.sherpa_model_name = shuvoice_core::PARAKEET_TDT_V3_INT8_MODEL_NAME.into();
    c.sherpa_decode_mode = shuvoice_core::SherpaDecodeMode::Streaming;
    c.sherpa_enable_parakeet_streaming = false;
    let _ = c.validate();

    let status = setup::run_setup(
        &c,
        setup::SetupOptions {
            install_missing: false,
            skip_model_download: true,
            skip_preflight: true,
            tts_local_voice: None,
            tts_local_model_dir: None,
            non_interactive: true,
        },
    )
    .await;
    assert_eq!(
        status.code, EXIT_DEPENDENCY,
        "expected 78 got {}",
        status.code
    );
}

#[cfg(feature = "asr-sherpa")]
#[tokio::test]
#[serial]
async fn setup_parakeet_offline_instant_passes_gate() {
    let tmp = TempDir::new().unwrap();
    let _xdg = with_xdg(&tmp);
    let mut c = Config::default();
    c.asr_backend = AsrBackendKind::Sherpa;
    c.sherpa_model_name = shuvoice_core::PARAKEET_TDT_V3_INT8_MODEL_NAME.into();
    c.sherpa_decode_mode = shuvoice_core::SherpaDecodeMode::OfflineInstant;
    let _ = c.validate();
    let data = std::env::var("XDG_DATA_HOME").unwrap();
    let dir = PathBuf::from(data)
        .join("shuvoice/models/sherpa")
        .join(&c.sherpa_model_name);
    std::fs::create_dir_all(&dir).unwrap();
    for name in ["tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"] {
        std::fs::write(dir.join(name), "x").unwrap();
    }
    c.sherpa_model_dir = Some(dir.display().to_string());

    let status = setup::run_setup(
        &c,
        setup::SetupOptions {
            install_missing: false,
            skip_model_download: true,
            skip_preflight: true,
            tts_local_voice: None,
            tts_local_model_dir: None,
            non_interactive: true,
        },
    )
    .await;
    assert_eq!(status.code, EXIT_SUCCESS, "got {}", status.code);
}

#[tokio::test]
#[serial]
async fn setup_nemo_without_install_does_not_pass_then_exit_78() {
    let tmp = TempDir::new().unwrap();
    let _xdg = with_xdg(&tmp);
    let mut c = Config::default();
    c.asr_backend = AsrBackendKind::Nemo;
    let _ = c.validate();
    let status = setup::run_setup(
        &c,
        setup::SetupOptions {
            install_missing: false,
            skip_model_download: false,
            skip_preflight: true,
            tts_local_voice: None,
            tts_local_model_dir: None,
            non_interactive: true,
        },
    )
    .await;
    // Guidance-only path must not crash with 78 after a fake PASS.
    assert_eq!(status.code, EXIT_SUCCESS, "got {}", status.code);
}

#[tokio::test]
#[serial]
async fn setup_nemo_install_missing_failure_exits_78_without_pass() {
    let tmp = TempDir::new().unwrap();
    let _xdg = with_xdg(&tmp);
    let mut c = Config::default();
    c.asr_backend = AsrBackendKind::Nemo;
    let _ = c.validate();
    let runner = ScriptedRunner::new();
    runner.set_dynamic(|_argv| {
        Ok(RunOutput {
            status_code: Some(1),
            stdout: Vec::new(),
            stderr: b"fail".to_vec(),
            success: false,
        })
    });
    let ctx = SetupContext {
        runner: Arc::new(runner),
        downloader: Arc::new(ScriptedDownloader::default()),
        sherpa_archive_url_override: None,
    };
    let status = setup::run_setup_with_ctx(
        &c,
        setup::SetupOptions {
            install_missing: true,
            skip_model_download: true,
            skip_preflight: true,
            tts_local_voice: None,
            tts_local_model_dir: None,
            non_interactive: true,
        },
        &ctx,
    )
    .await;
    assert_eq!(status.code, EXIT_DEPENDENCY, "got {}", status.code);
}

#[cfg(feature = "asr-sherpa")]
#[tokio::test]
#[serial]
async fn setup_persists_local_voice_only_after_success() {
    let tmp = TempDir::new().unwrap();
    let _xdg = with_xdg(&tmp);

    let bin = tmp.path().join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let stub = bin.join("piper");
    std::fs::write(&stub, b"#!/bin/sh\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let old_path = std::env::var_os("PATH");
    let _path = EnvGuard::set(
        "PATH",
        format!(
            "{}:{}",
            bin.display(),
            old_path
                .as_ref()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        ),
    );

    let voice = get_curated_piper_voice("en_US-amy-medium").unwrap();
    let dl = ScriptedDownloader::default();
    dl.insert(voice.model_url, vec![1u8; 4096]);
    dl.insert(voice.sidecar_url, br#"{"sample_rate":22050}"#.to_vec());

    let mut c = Config::default();
    c.asr_backend = AsrBackendKind::Sherpa;
    c.tts_backend = TtsBackendKind::Local;
    let sdir = PathBuf::from(std::env::var("XDG_DATA_HOME").unwrap())
        .join("shuvoice/models/sherpa")
        .join(&c.sherpa_model_name);
    std::fs::create_dir_all(&sdir).unwrap();
    for name in ["tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"] {
        std::fs::write(sdir.join(name), "x").unwrap();
    }
    c.sherpa_model_dir = Some(sdir.display().to_string());
    let _ = c.validate();

    let piper_dir = tmp.path().join("piper-models");
    let ctx = SetupContext {
        runner: Arc::new(ScriptedRunner::new()),
        downloader: Arc::new(dl),
        sherpa_archive_url_override: None,
    };

    let status = setup::run_setup_with_ctx(
        &c,
        setup::SetupOptions {
            install_missing: false,
            skip_model_download: false,
            skip_preflight: true,
            tts_local_voice: Some("en_US-amy-medium".into()),
            tts_local_model_dir: Some(piper_dir.clone()),
            non_interactive: true,
        },
        &ctx,
    )
    .await;
    assert_eq!(status.code, EXIT_SUCCESS, "got {}", status.code);

    let reloaded = Config::load().unwrap();
    assert_eq!(reloaded.tts_backend, TtsBackendKind::Local);
    assert_eq!(
        reloaded.tts_local_voice.as_deref(),
        Some("en_US-amy-medium")
    );
    assert!(reloaded.tts_local_model_path.is_some());
}

#[cfg(feature = "asr-sherpa")]
#[tokio::test]
#[serial]
async fn failed_piper_download_leaves_config_unchanged() {
    let tmp = TempDir::new().unwrap();
    let _xdg = with_xdg(&tmp);

    let bin = tmp.path().join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let stub = bin.join("piper");
    std::fs::write(&stub, b"#!/bin/sh\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let old_path = std::env::var_os("PATH");
    let _path = EnvGuard::set(
        "PATH",
        format!(
            "{}:{}",
            bin.display(),
            old_path
                .as_ref()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        ),
    );

    let voice = get_curated_piper_voice("en_US-amy-medium").unwrap();
    let dl = ScriptedDownloader::default();
    // Fail model download.
    dl.fail(voice.model_url);

    let mut c = Config::default();
    c.asr_backend = AsrBackendKind::Sherpa;
    c.tts_backend = TtsBackendKind::Elevenlabs; // not local yet
    c.tts_enabled = false;
    let sdir = PathBuf::from(std::env::var("XDG_DATA_HOME").unwrap())
        .join("shuvoice/models/sherpa")
        .join(&c.sherpa_model_name);
    std::fs::create_dir_all(&sdir).unwrap();
    for name in ["tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"] {
        std::fs::write(sdir.join(name), "x").unwrap();
    }
    c.sherpa_model_dir = Some(sdir.display().to_string());
    // Force local TTS path.
    c.tts_backend = TtsBackendKind::Local;
    let _ = c.validate();
    c.save_to_path(Config::config_path()).unwrap();
    let before = Config::load().unwrap();

    let ctx = SetupContext {
        runner: Arc::new(ScriptedRunner::new()),
        downloader: Arc::new(dl),
        sherpa_archive_url_override: None,
    };
    let status = setup::run_setup_with_ctx(
        &c,
        setup::SetupOptions {
            install_missing: false,
            skip_model_download: false,
            skip_preflight: true,
            tts_local_voice: Some("en_US-amy-medium".into()),
            tts_local_model_dir: Some(tmp.path().join("piper-fail")),
            non_interactive: true,
        },
        &ctx,
    )
    .await;
    assert_ne!(status.code, EXIT_SUCCESS);
    let after = Config::load().unwrap();
    assert_eq!(before.tts_local_voice, after.tts_local_voice);
    assert_eq!(before.tts_local_model_path, after.tts_local_model_path);
    assert_eq!(before.tts_default_voice_id, after.tts_default_voice_id);
}

// ─── preflight ──────────────────────────────────────────────────────────────

#[tokio::test]
#[serial]
async fn preflight_fails_missing_required_sherpa_model() {
    let tmp = TempDir::new().unwrap();
    let _xdg = with_xdg(&tmp);
    let mut c = Config::default();
    c.asr_backend = AsrBackendKind::Sherpa;
    c.sherpa_model_dir = Some(tmp.path().join("empty-sherpa").display().to_string());
    // Disable TTS to avoid unrelated fails; binaries may still fail.
    c.tts_enabled = false;
    let _ = c.validate();
    let status = preflight::run_preflight(&c).await;
    // Must be NOT READY because model missing (and possibly other deps).
    assert_eq!(status.code, EXIT_FAILURE);
}

#[tokio::test]
#[serial]
async fn preflight_api_key_presence_without_value() {
    let tmp = TempDir::new().unwrap();
    let _xdg = with_xdg(&tmp);
    let _k1 = EnvGuard::set("ELEVENLABS_API_KEY", "sk_test_secret_value_do_not_print");
    let _k2 = EnvGuard::set("OPENAI_API_KEY", "sk-openai-secret");
    let mut c = Config::default();
    c.tts_backend = TtsBackendKind::Elevenlabs;
    c.tts_enabled = true;
    c.tts_api_key_env = "ELEVENLABS_API_KEY".into();
    c.asr_backend = AsrBackendKind::OpenaiRealtime;
    c.openai_realtime_api_key_env = "OPENAI_API_KEY".into();
    let _ = c.validate();

    let s = shuvoice_cli::setup::tts_api_key_env_status(&c).unwrap();
    assert!(s.contains("is set"));
    assert!(!s.contains("sk_test"));
    let s = shuvoice_cli::setup::openai_asr_key_status(&c).unwrap();
    assert!(s.contains("is set"));
    assert!(!s.contains("sk-openai"));
}

#[test]
fn rejects_raw_key_as_env_name() {
    let mut c = Config::default();
    c.tts_enabled = true;
    c.tts_backend = TtsBackendKind::Elevenlabs;
    c.tts_api_key_env = "sk_live_abc".into();
    let err = shuvoice_cli::setup::tts_api_key_env_status(&c).unwrap_err();
    assert!(err.contains("raw API key"));
}

// ─── Kokoro preflight via httpmock ──────────────────────────────────────────

#[tokio::test]
async fn kokoro_voice_endpoint_preflight() {
    let server = httpmock::MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/v1/audio/voices");
        then.status(200)
            .json_body(serde_json::json!({"voices": ["af_heart", "af_sarah"]}));
    });
    let n = shuvoice_cli::setup::check_kokoro_voices(&format!("{}/v1", server.base_url()))
        .await
        .unwrap();
    assert_eq!(n, 2);
    m.assert();
}

#[tokio::test]
async fn kokoro_voice_endpoint_failure() {
    let server = httpmock::MockServer::start();
    let _m = server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/v1/audio/voices");
        then.status(503);
    });
    let err = shuvoice_cli::setup::check_kokoro_voices(&format!("{}/v1", server.base_url()))
        .await
        .unwrap_err();
    assert!(err.contains("503") || err.contains("HTTP"), "{err}");
}

#[tokio::test]
async fn kokoro_requires_usable_voices() {
    let server = httpmock::MockServer::start();
    let _m = server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/v1/audio/voices");
        then.status(200)
            .json_body(serde_json::json!({"voices": ["", "  "]}));
    });
    let err = shuvoice_cli::setup::check_kokoro_voices(&format!("{}/v1", server.base_url()))
        .await
        .unwrap_err();
    assert!(err.contains("usable"), "{err}");
}

#[tokio::test]
async fn kokoro_rejects_oversized_body() {
    let server = httpmock::MockServer::start();
    let huge = "x".repeat(2_000_000);
    let _m = server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/v1/audio/voices");
        then.status(200)
            .header("content-type", "application/json")
            .body(format!(r#"{{"voices":["{huge}"]}}"#));
    });
    let err = shuvoice_cli::setup::check_kokoro_voices(&format!("{}/v1", server.base_url()))
        .await
        .unwrap_err();
    assert!(
        err.contains("cap") || err.contains("exceeds") || err.contains("body"),
        "{err}"
    );
}
