//! Finite install command plans (no shell strings). Injectable CommandRunner.
//!
//! Plans distinguish:
//! - **Sequential pipeline**: every argv step must succeed (e.g. `venv` then `pip install`).
//! - **Fallback alternatives**: ordered pipelines; first fully-successful pipeline wins
//!   (e.g. `uv` path, else `python -m venv` + pip; or `yay` else `paru`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shuvoice_core::{AsrBackendKind, data_dir};
use shuvoice_io::process::{CommandRunner, RunOptions, StdCommandRunner};

/// Conventional isolated worker venv directory names under the ShuVoice data dir.
/// Kept in lockstep with `workers/README.md`.
pub const NEMO_WORKER_VENV_NAME: &str = "workers-nemo-venv";
pub const MOONSHINE_WORKER_VENV_NAME: &str = "workers-moonshine-venv";
pub const MELOTTS_VENV_NAME: &str = "melotts-venv";

/// OS / package-manager detection for install planning.
#[derive(Debug, Clone, Default)]
pub struct HostProfile {
    pub is_arch: bool,
    pub has_yay: bool,
    pub has_paru: bool,
    pub has_uv: bool,
    pub in_venv: bool,
    pub has_nvidia_smi: bool,
}

impl HostProfile {
    pub fn detect() -> Self {
        let is_arch = Path::new("/etc/arch-release").exists()
            || std::fs::read_to_string("/etc/os-release")
                .map(|t| t.contains("ID=arch") || t.contains("ID_LIKE=arch"))
                .unwrap_or(false);
        Self {
            is_arch,
            has_yay: which::which("yay").is_ok(),
            has_paru: which::which("paru").is_ok(),
            has_uv: which::which("uv").is_ok(),
            in_venv: std::env::var_os("VIRTUAL_ENV").is_some(),
            has_nvidia_smi: which::which("nvidia-smi").is_ok(),
        }
    }
}

/// A single argv plan step.
pub type Argv = Vec<String>;

/// Ordered sequential steps that must all succeed.
pub type Pipeline = Vec<Argv>;

/// Build ordered **alternative** install pipelines for an ASR backend.
///
/// Each outer element is a fallback alternative. Within a pipeline every step
/// runs sequentially and must succeed before the pipeline is considered done.
///
/// Native Rust defaults:
/// - Sherpa: **no** Python wheel / CUDA RUNPATH repair (empty).
/// - NeMo / Moonshine: isolated optional worker venv pipelines only.
/// - OpenAI Realtime: feature-flag rebuild guidance (empty).
pub fn asr_install_pipelines(backend: AsrBackendKind, host: &HostProfile) -> Vec<Pipeline> {
    match backend {
        AsrBackendKind::Sherpa | AsrBackendKind::OpenaiRealtime => Vec::new(),
        AsrBackendKind::Nemo => {
            worker_venv_pipelines(host, NEMO_WORKER_VENV_NAME, &["nemo-toolkit[asr]", "torch"])
        }
        AsrBackendKind::Moonshine => {
            worker_venv_pipelines(host, MOONSHINE_WORKER_VENV_NAME, &["useful-moonshine-onnx"])
        }
    }
}

/// Flat view of all argv steps across alternative pipelines (for inspection/tests).
pub fn asr_install_plans(backend: AsrBackendKind, host: &HostProfile) -> Vec<Argv> {
    asr_install_pipelines(backend, host)
        .into_iter()
        .flatten()
        .collect()
}

fn worker_venv_pipelines(host: &HostProfile, venv_name: &str, packages: &[&str]) -> Vec<Pipeline> {
    let venv = worker_venv_path(venv_name);
    let python = venv.join("bin").join("python");
    let mut alts = Vec::new();

    // Preferred: uv venv + uv pip (sequential).
    if host.has_uv || which::which("uv").is_ok() {
        let mut pip = vec![
            "uv".into(),
            "pip".into(),
            "install".into(),
            "--python".into(),
            python.display().to_string(),
        ];
        pip.extend(packages.iter().map(|s| (*s).to_string()));
        alts.push(vec![
            vec![
                "uv".into(),
                "venv".into(),
                "--python".into(),
                "3.12".into(),
                venv.display().to_string(),
            ],
            pip,
        ]);
    }

    // Fallback alternative: python3 -m venv + pip (sequential).
    let mut pip = vec![
        python.display().to_string(),
        "-m".into(),
        "pip".into(),
        "install".into(),
    ];
    pip.extend(packages.iter().map(|s| (*s).to_string()));
    alts.push(vec![
        vec![
            "python3".into(),
            "-m".into(),
            "venv".into(),
            venv.display().to_string(),
        ],
        pip,
    ]);

    alts
}

pub fn worker_venv_path(venv_name: &str) -> PathBuf {
    data_dir().join(venv_name)
}

pub fn worker_venv_dir_for_backend(backend: AsrBackendKind) -> Option<PathBuf> {
    match backend {
        AsrBackendKind::Nemo => Some(worker_venv_path(NEMO_WORKER_VENV_NAME)),
        AsrBackendKind::Moonshine => Some(worker_venv_path(MOONSHINE_WORKER_VENV_NAME)),
        _ => None,
    }
}

pub fn worker_venv_python_ready(venv_dir: &Path) -> bool {
    let python = venv_dir.join("bin").join("python");
    if !python.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(&python)
            .map(|m| m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Human guidance when install plans are empty (native feature rebuild).
pub fn asr_feature_guidance(backend: AsrBackendKind) -> Vec<String> {
    match backend {
        AsrBackendKind::Sherpa => vec![
            "Native Sherpa via shuvoice-asr: rebuild with --features asr-sherpa (static sherpa-onnx; no Python wheel/CUDA repair).".into(),
            "CUDA is unsupported on the native static build: keep sherpa_provider = \"cpu\".".into(),
            "Optional system audio/UI deps: gtk4 gtk4-layer-shell wtype wl-clipboard pipewire.".into(),
        ],
        AsrBackendKind::OpenaiRealtime => vec![
            "OpenAI Realtime ASR: rebuild with --features asr-openai.".into(),
            "Set OPENAI_API_KEY (or configured env name) in ~/.config/shuvoice/local.dev.".into(),
        ],
        AsrBackendKind::Nemo => vec![
            "NeMo is an optional external worker (no native Rust runtime).".into(),
            format!(
                "Use setup --install-missing to create isolated venv at {} , then configure worker_command.",
                worker_venv_path(NEMO_WORKER_VENV_NAME).display()
            ),
        ],
        AsrBackendKind::Moonshine => vec![
            "Moonshine is an optional external worker (no native Rust runtime).".into(),
            format!(
                "Use setup --install-missing to create isolated venv at {} , then configure worker_command.",
                worker_venv_path(MOONSHINE_WORKER_VENV_NAME).display()
            ),
        ],
    }
}

/// Run ordered alternative pipelines. Within each pipeline every step must succeed.
///
/// Returns the number of argv steps executed in the winning pipeline.
pub fn run_install_pipelines(
    pipelines: &[Pipeline],
    runner: &dyn CommandRunner,
) -> Result<usize, String> {
    if pipelines.is_empty() {
        return Ok(0);
    }
    let mut last_err =
        String::from("Automatic install failed or no supported installer available.");
    for (idx, pipeline) in pipelines.iter().enumerate() {
        if pipeline.is_empty() {
            continue;
        }
        println!(
            "Trying install pipeline {}/{} ({} step(s))…",
            idx + 1,
            pipelines.len(),
            pipeline.len()
        );
        match run_sequential_pipeline(pipeline, runner) {
            Ok(n) => return Ok(n),
            Err(e) => {
                println!("  Pipeline {}/{} failed: {e}", idx + 1, pipelines.len());
                last_err = e;
            }
        }
    }
    Err(last_err)
}

/// Back-compat wrapper: treat a flat argv list as **alternatives** (one step each).
/// Prefer [`run_install_pipelines`] for sequential venv+pip plans.
pub fn run_install_plans(plans: &[Argv], runner: &dyn CommandRunner) -> Result<usize, String> {
    let pipelines: Vec<Pipeline> = plans.iter().cloned().map(|step| vec![step]).collect();
    run_install_pipelines(&pipelines, runner)
}

fn run_sequential_pipeline(steps: &[Argv], runner: &dyn CommandRunner) -> Result<usize, String> {
    let opts = RunOptions {
        check: false,
        timeout: std::time::Duration::from_secs(1800),
        ..RunOptions::default()
    };
    let mut ran = 0usize;
    for cmd in steps {
        if cmd.is_empty() {
            continue;
        }
        if which::which(&cmd[0]).is_err() && !Path::new(&cmd[0]).is_file() {
            return Err(format!("executable not found: {}", cmd[0]));
        }
        println!("  Running: {}", cmd.join(" "));
        let out = runner
            .run(cmd, &opts)
            .map_err(|e| format!("install command failed: {e}"))?;
        ran += 1;
        if !out.success {
            return Err(format!(
                "command failed (exit {:?}): {}",
                out.status_code,
                cmd.join(" ")
            ));
        }
    }
    if ran == 0 {
        return Err("empty install pipeline".into());
    }
    Ok(ran)
}

pub fn default_runner() -> Arc<dyn CommandRunner> {
    Arc::new(StdCommandRunner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shuvoice_io::process::{RunOutput, ScriptedRunner};

    #[test]
    fn worker_names_match_workers_readme_convention() {
        assert_eq!(NEMO_WORKER_VENV_NAME, "workers-nemo-venv");
        assert_eq!(MOONSHINE_WORKER_VENV_NAME, "workers-moonshine-venv");
        assert_eq!(MELOTTS_VENV_NAME, "melotts-venv");
    }

    #[test]
    fn uv_and_python_are_fallback_alternatives_each_sequential() {
        let host = HostProfile {
            has_uv: true,
            ..HostProfile::default()
        };
        let pipes = asr_install_pipelines(AsrBackendKind::Nemo, &host);
        assert!(
            pipes.len() >= 2,
            "uv path + python fallback expected: {pipes:?}"
        );
        // Each pipeline is sequential venv then pip.
        for p in &pipes {
            assert_eq!(p.len(), 2, "venv + pip: {p:?}");
            assert!(
                p[0].iter().any(|a| a == "venv"),
                "first step creates venv: {:?}",
                p[0]
            );
            assert!(
                p[1].iter().any(|a| a == "install" || a == "pip"),
                "second step installs: {:?}",
                p[1]
            );
        }
        assert!(pipes[0][0][0] == "uv");
        assert!(pipes[1][0][0] == "python3");
        assert!(
            pipes
                .iter()
                .flatten()
                .any(|c| c.iter().any(|a| a.contains(NEMO_WORKER_VENV_NAME)))
        );
    }

    #[test]
    fn run_pipelines_requires_all_sequential_steps() {
        let runner = ScriptedRunner::new();
        let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Vec<String>>::new()));
        let calls2 = calls.clone();
        runner.set_dynamic(move |argv| {
            calls2.lock().unwrap().push(argv.to_vec());
            // First step (venv) succeeds; second (pip) fails.
            let ok = argv.iter().any(|a| a == "venv");
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
        assert!(
            err.contains("failed") || err.contains("pip") || err.contains("exit"),
            "{err}"
        );
        let recorded = calls.lock().unwrap();
        assert_eq!(recorded.len(), 2, "expected both steps, got {recorded:?}");
    }

    #[test]
    fn run_pipelines_tries_next_alternative_after_failure() {
        let runner = ScriptedRunner::new();
        let n = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let n2 = n.clone();
        runner.set_dynamic(move |argv| {
            let i = n2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // First pipeline (2 steps) both fail at step 0; second pipeline succeeds both.
            let ok = i >= 1;
            let _ = argv;
            Ok(RunOutput {
                status_code: Some(if ok { 0 } else { 1 }),
                stdout: Vec::new(),
                stderr: Vec::new(),
                success: ok,
            })
        });
        let pipelines = vec![
            vec![vec!["true".into(), "a".into()]],
            vec![
                vec!["true".into(), "b".into()],
                vec!["true".into(), "c".into()],
            ],
        ];
        let ran = run_install_pipelines(&pipelines, &runner).unwrap();
        assert_eq!(ran, 2);
        assert_eq!(n.load(std::sync::atomic::Ordering::SeqCst), 3);
    }
}
