//! Process-level child environment isolation probe for [`WorkerProcess::spawn`].

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use shuvoice_worker_proto::{
    CHILD_ENV_ALLOWLIST, WorkerProcess, WorkerSpawnConfig, build_isolated_child_env,
};

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

fn which_absolute(program: &Path) -> Option<PathBuf> {
    if program.is_absolute() {
        return Some(program.to_path_buf());
    }
    let name = program.as_os_str();
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn marker_val<'a>(body: &'a str, key: &str) -> &'a str {
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix(key)
            && let Some(rest) = rest.strip_prefix('=')
        {
            return rest;
        }
    }
    ""
}

/// Env-guard helpers. `set_var`/`remove_var` are unsafe on current rustc; this
/// integration crate does not `forbid(unsafe_code)`.
struct EnvGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn set(key: &'static str, val: &str) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: test-only process env mutation; restored on drop.
        unsafe {
            std::env::set_var(key, val);
        }
        Self { key, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: restore prior value captured at set time.
        unsafe {
            match &self.previous {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

#[test]
fn public_allowlist_and_builder_surface() {
    assert!(CHILD_ENV_ALLOWLIST.contains(&"PATH"));
    assert!(CHILD_ENV_ALLOWLIST.contains(&"SSL_CERT_FILE"));
    assert!(!CHILD_ENV_ALLOWLIST.contains(&"OPENAI_API_KEY"));
    assert!(!CHILD_ENV_ALLOWLIST.contains(&"HTTPS_PROXY"));
    let _guards = [
        EnvGuard::set("OPENAI_API_KEY", "sk-surface-sentinel"),
        EnvGuard::set(
            "HTTPS_PROXY",
            "http://proxy-user:proxy-pass-surface@127.0.0.1:8443",
        ),
    ];
    let env = build_isolated_child_env(&[]);
    assert!(!env.iter().any(|(k, _)| k == "OPENAI_API_KEY"));
    assert!(!env.iter().any(|(k, _)| k == "HTTPS_PROXY"));
    assert!(
        env.iter()
            .all(|(_, v)| !v.to_string_lossy().contains("proxy-pass-surface"))
    );
}

/// Process-level probe through [`WorkerProcess::spawn`]'s isolation path.
///
/// Uses a non-protocol Python child that writes an env marker, then sleeps.
/// Handshake fails (timeout or EOF after kill), but the marker proves which
/// keys were present in the real child environment built by spawn.
#[tokio::test]
async fn spawn_child_env_isolation_process_probe() {
    let Some(python) = python_executable() else {
        eprintln!("SKIP env isolation process probe: python unavailable");
        return;
    };
    let python = which_absolute(&python).unwrap_or(python);

    let marker_dir = std::env::temp_dir().join(format!(
        "shuvoice-worker-env-probe-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&marker_dir).expect("marker dir");
    let marker_path = marker_dir.join("env-marker.txt");
    let script_path = marker_dir.join("probe.py");
    let marker_str = marker_path.to_string_lossy().into_owned();

    let script = format!(
        r#"import os, pathlib, time, sys
p = pathlib.Path({marker_str:?})
keys = [
    "OPENAI_API_KEY",
    "ELEVENLABS_API_KEY",
    "SSH_AUTH_SOCK",
    "AWS_SECRET_ACCESS_KEY",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "PATH",
    "SSL_CERT_FILE",
    "XDG_CACHE_HOME",
    "PROBE_OVERLAY",
    "PYTHONPATH",
]
lines = [f"{{k}}={{os.environ.get(k)!r}}" for k in keys]
p.write_text("\n".join(lines))
sys.stderr.write("probe-marker-written\n")
sys.stderr.flush()
time.sleep(30)
"#
    );
    std::fs::write(&script_path, script).expect("write probe script");

    let _guards = [
        EnvGuard::set("OPENAI_API_KEY", "sk-process-sentinel-openai"),
        EnvGuard::set("ELEVENLABS_API_KEY", "el-process-sentinel-eleven"),
        EnvGuard::set("SSH_AUTH_SOCK", "/tmp/ssh-agent-process.sock"),
        EnvGuard::set("AWS_SECRET_ACCESS_KEY", "aws-process-secret-sentinel"),
        EnvGuard::set("GH_TOKEN", "gh-process-sentinel"),
        EnvGuard::set("GITHUB_TOKEN", "ghs-process-sentinel"),
        EnvGuard::set(
            "HTTPS_PROXY",
            "http://proxy-user:proxy-pass-process@127.0.0.1:8443",
        ),
        EnvGuard::set(
            "HTTP_PROXY",
            "http://proxy-user:proxy-pass-process@127.0.0.1:8080",
        ),
        EnvGuard::set("SSL_CERT_FILE", "/tmp/shuvoice-process-ca.pem"),
        EnvGuard::set("XDG_CACHE_HOME", "/tmp/shuvoice-process-xdg-cache"),
    ];
    let _path_guard = if std::env::var_os("PATH").is_none() {
        Some(EnvGuard::set("PATH", "/usr/bin:/bin"))
    } else {
        None
    };

    let cfg = WorkerSpawnConfig::new(&python)
        .args([script_path.as_os_str()])
        .env_pair("PROBE_OVERLAY", "overlay-wins-sentinel")
        .env_pair("PYTHONPATH", "/explicit/workers-overlay")
        // Give the probe time to flush the marker before handshake gives up.
        .handshake_timeout(Duration::from_millis(1500))
        .kill_timeout(Duration::from_secs(2))
        .client_name("env-isolation-probe");

    let spawn_result = WorkerProcess::spawn(cfg).await;
    let spawn_err = match spawn_result {
        Ok(_) => panic!("probe child is not a protocol worker; expected handshake failure"),
        Err(e) => e,
    };
    eprintln!("spawn_err={spawn_err}");

    let mut body = String::new();
    for _ in 0..80 {
        body = std::fs::read_to_string(&marker_path).unwrap_or_default();
        if body.contains("OPENAI_API_KEY=") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    // Keep dir on failure for diagnosis; remove on success path below.
    assert!(
        !body.is_empty(),
        "child never wrote env marker at {}; spawn_err={spawn_err}; script={}; python={}; dirents={:?}",
        marker_path.display(),
        script_path.display(),
        python.display(),
        std::fs::read_dir(&marker_dir)
            .map(|rd| rd
                .filter_map(|e| e.ok().map(|e| e.file_name()))
                .collect::<Vec<_>>())
            .unwrap_or_default()
    );
    let _ = std::fs::remove_dir_all(&marker_dir);

    assert_eq!(
        marker_val(&body, "OPENAI_API_KEY"),
        "None",
        "OPENAI_API_KEY leaked; marker={body:?}"
    );
    assert_eq!(
        marker_val(&body, "ELEVENLABS_API_KEY"),
        "None",
        "ELEVENLABS_API_KEY leaked; marker={body:?}"
    );
    assert_eq!(
        marker_val(&body, "SSH_AUTH_SOCK"),
        "None",
        "SSH_AUTH_SOCK leaked; marker={body:?}"
    );
    assert_eq!(
        marker_val(&body, "AWS_SECRET_ACCESS_KEY"),
        "None",
        "AWS_SECRET_ACCESS_KEY leaked; marker={body:?}"
    );
    assert_eq!(
        marker_val(&body, "GH_TOKEN"),
        "None",
        "GH_TOKEN leaked; marker={body:?}"
    );
    assert_eq!(
        marker_val(&body, "GITHUB_TOKEN"),
        "None",
        "GITHUB_TOKEN leaked; marker={body:?}"
    );
    assert_eq!(
        marker_val(&body, "HTTPS_PROXY"),
        "None",
        "HTTPS_PROXY leaked; marker={body:?}"
    );
    assert_eq!(
        marker_val(&body, "HTTP_PROXY"),
        "None",
        "HTTP_PROXY leaked; marker={body:?}"
    );
    assert!(
        !body.contains("proxy-pass-process"),
        "proxy credential leaked; marker={body:?}"
    );
    assert!(
        !body.contains("sk-process-sentinel"),
        "api key sentinel leaked; marker={body:?}"
    );

    let path_val = marker_val(&body, "PATH");
    assert!(
        path_val != "None" && path_val.len() > 2,
        "PATH should be preserved from parent; marker={body:?}"
    );
    assert_eq!(
        marker_val(&body, "SSL_CERT_FILE"),
        "'/tmp/shuvoice-process-ca.pem'",
        "SSL_CERT_FILE should be preserved; marker={body:?}"
    );
    assert_eq!(
        marker_val(&body, "XDG_CACHE_HOME"),
        "'/tmp/shuvoice-process-xdg-cache'",
        "XDG_CACHE_HOME should be preserved; marker={body:?}"
    );
    assert_eq!(
        marker_val(&body, "PROBE_OVERLAY"),
        "'overlay-wins-sentinel'",
        "explicit overlay should win; marker={body:?}"
    );
    assert_eq!(
        marker_val(&body, "PYTHONPATH"),
        "'/explicit/workers-overlay'",
        "explicit PYTHONPATH overlay should be present; marker={body:?}"
    );
}
