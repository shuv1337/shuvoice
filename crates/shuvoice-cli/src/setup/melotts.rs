//! MeloTTS isolated worker-venv installation plans.

use std::path::{Path, PathBuf};

use shuvoice_core::{Config, data_dir, expand_user_path};
use shuvoice_io::process::{CommandRunner, RunOptions};

use super::install::MELOTTS_VENV_NAME;

pub fn default_melotts_venv_dir() -> PathBuf {
    data_dir().join(MELOTTS_VENV_NAME)
}

pub fn melotts_venv_dir(config: &Config) -> PathBuf {
    config
        .tts_melotts_venv_path
        .as_ref()
        .map(expand_user_path)
        .unwrap_or_else(default_melotts_venv_dir)
}

pub fn melotts_venv_valid(venv_dir: &Path) -> bool {
    let python = venv_dir.join("bin").join("python");
    python.is_file()
        && std::fs::metadata(&python)
            .map(|m| {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    m.permissions().mode() & 0o111 != 0
                }
                #[cfg(not(unix))]
                {
                    let _ = m;
                    true
                }
            })
            .unwrap_or(false)
}

/// Sequential MeloTTS install pipeline (all steps must succeed, in order).
pub fn melotts_install_commands(venv_dir: &Path) -> Vec<Vec<String>> {
    let python_bin = venv_dir.join("bin").join("python");
    vec![
        vec![
            "uv".into(),
            "python".into(),
            "install".into(),
            "3.12".into(),
        ],
        vec![
            "uv".into(),
            "venv".into(),
            "--python".into(),
            "3.12".into(),
            venv_dir.display().to_string(),
        ],
        vec![
            python_bin.display().to_string(),
            "-m".into(),
            "pip".into(),
            "install".into(),
            "melotts".into(),
        ],
        vec![
            python_bin.display().to_string(),
            "-m".into(),
            "unidic".into(),
            "download".into(),
        ],
    ]
}

pub fn melotts_missing_dependencies(venv_dir: &Path) -> Vec<String> {
    let mut missing = Vec::new();
    if !venv_dir.is_dir() {
        missing.push(format!("venv missing ({})", venv_dir.display()));
        return missing;
    }
    if !melotts_venv_valid(venv_dir) {
        missing.push(format!(
            "venv python not executable ({})",
            venv_dir.display()
        ));
    }
    missing
}

/// Run MeloTTS **sequential** install plan. Skips venv creation when already valid.
pub fn run_melotts_install(
    venv_dir: &Path,
    runner: &dyn CommandRunner,
    already_valid: bool,
) -> Result<(), String> {
    let opts = RunOptions {
        check: false,
        timeout: std::time::Duration::from_secs(1800),
        ..RunOptions::default()
    };
    for cmd in melotts_install_commands(venv_dir) {
        // Skip `uv venv` when already valid.
        if already_valid && cmd.get(1).map(|s| s.as_str()) == Some("venv") {
            continue;
        }
        let exe = &cmd[0];
        let found = if exe.contains('/') {
            Path::new(exe).is_file()
        } else {
            which::which(exe).is_ok()
        };
        if !found {
            // uv python install is optional if python already present.
            if cmd.get(1).map(|s| s.as_str()) == Some("python") {
                continue;
            }
            return Err(format!("executable not found: {exe}"));
        }
        let out = runner
            .run(&cmd, &opts)
            .map_err(|e| format!("command failed: {e}"))?;
        if !out.success {
            return Err(format!(
                "command failed (exit {:?}): {}",
                out.status_code,
                cmd.join(" ")
            ));
        }
    }
    if !melotts_venv_valid(venv_dir) {
        return Err(format!(
            "MeloTTS venv still not valid after install: {}",
            venv_dir.display()
        ));
    }
    Ok(())
}

pub fn format_melotts_report(venv_dir: &Path, missing: &[String]) -> String {
    let mut lines = vec!["MeloTTS:".into()];
    let present = venv_dir.is_dir();
    lines.push(format!(
        "  Venv: {} ({})",
        if present { "present" } else { "missing" },
        venv_dir.display()
    ));
    if present {
        lines.push(format!(
            "  Python: {}",
            if melotts_venv_valid(venv_dir) {
                "executable"
            } else {
                "not executable"
            }
        ));
    }
    if missing.is_empty() {
        lines.push(format!("  Status: ready ({})", venv_dir.display()));
    } else {
        for m in missing {
            lines.push(format!("  Missing: {m}"));
        }
    }
    lines.join("\n")
}
