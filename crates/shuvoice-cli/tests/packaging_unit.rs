//! Packaging contract checks owned by the CLI crate.
//!
//! These assertions lock the Rust desktop cutover for AUR/CI:
//! locked release build, both bins, systemd exit-78 unit, bundled workers
//! tree, and no Python application install path.

use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

/// Body of a top-level `name=(...)` assignment (not `make`/`opt` prefixes).
fn bash_array_body(content: &str, name: &str) -> String {
    let mut out = String::new();
    let mut in_array = false;
    let header = format!("{name}=(");
    for line in content.lines() {
        let trimmed = line.trim();
        if !in_array {
            if trimmed.starts_with(&header) {
                in_array = true;
                if let Some(rest) = trimmed.strip_prefix(&header) {
                    if rest.trim() == ")" {
                        break;
                    }
                    if let Some(rest) = rest.strip_suffix(')') {
                        out.push_str(rest);
                        break;
                    }
                    out.push_str(rest);
                    out.push('\n');
                }
            }
            continue;
        }
        if trimmed == ")" || trimmed.starts_with(')') {
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    assert!(
        in_array || !out.is_empty(),
        "missing {name}=() array in PKGBUILD"
    );
    out
}

#[test]
fn systemd_unit_preserves_exec_exit_78_and_rust_log() {
    let content = read("packaging/systemd/user/shuvoice.service");
    assert!(
        content.contains("ExecStart=/usr/bin/shuvoice"),
        "ExecStart path must remain /usr/bin/shuvoice"
    );
    assert!(
        content.contains("RestartPreventExitStatus=78"),
        "dependency exit 78 must remain RestartPreventExitStatus"
    );
    assert!(
        content.contains("Environment=RUST_LOG=info")
            || content.contains("Environment=\"RUST_LOG=info\""),
        "user unit must set RUST_LOG for journald defaults"
    );
    assert!(
        content.contains("WantedBy=graphical-session.target"),
        "user graphical session install target required"
    );
}

#[test]
fn pkgbuild_builds_locked_desktop_release_bins() {
    let content = read("packaging/PKGBUILD");

    // Exact locked desktop release build (not a loose cargo reference).
    assert!(
        content.contains("cargo build --release --locked -p shuvoice-cli --features desktop"),
        "PKGBUILD must build the locked desktop release of shuvoice-cli"
    );
    assert!(
        content.contains("--bin shuvoice") && content.contains("--bin shuvoice-waybar"),
        "PKGBUILD must request both shuvoice and shuvoice-waybar bins"
    );

    // Installed binary destinations.
    assert!(
        content.contains("target/release/shuvoice")
            && content.contains("\"$pkgdir/usr/bin/shuvoice\""),
        "PKGBUILD must install target/release/shuvoice to /usr/bin/shuvoice"
    );
    assert!(
        content.contains("target/release/shuvoice-waybar")
            && content.contains("\"$pkgdir/usr/bin/shuvoice-waybar\""),
        "PKGBUILD must install target/release/shuvoice-waybar to /usr/bin/shuvoice-waybar"
    );

    // User service unit.
    assert!(
        content.contains("shuvoice.service")
            && content.contains("\"$pkgdir/usr/lib/systemd/user/shuvoice.service\""),
        "PKGBUILD must install the user service unit"
    );

    // Bundled optional workers under the discovery path.
    assert!(
        content.contains("/usr/lib/shuvoice/workers"),
        "PKGBUILD must install workers under /usr/lib/shuvoice/workers"
    );
    for pkg in [
        "shuvoice_worker_proto",
        "nemo_asr",
        "moonshine_asr",
        "melotts",
    ] {
        assert!(
            content.contains(pkg),
            "PKGBUILD must package worker module '{pkg}'"
        );
    }
    // Source entrypoints only (*.py via find); no tests tree install.
    assert!(
        content.contains("-name '*.py'") || content.contains("-name \"*.py\""),
        "PKGBUILD must install worker *.py sources"
    );
    assert!(
        !content.contains("workers/tests"),
        "PKGBUILD must not ship worker tests/"
    );
    assert!(
        !content.contains("__pycache__"),
        "PKGBUILD must not reference __pycache__"
    );

    // Must not install the legacy Python application / pyproject package.
    let lowered = content.to_ascii_lowercase();
    for banned in [
        "pyproject.toml",
        "python -m build",
        "python -m installer",
        "python-build",
        "python-installer",
        "python-hatchling",
        "pip install",
        "site-packages",
        "python -m shuvoice",
    ] {
        assert!(
            !lowered.contains(banned),
            "PKGBUILD must not install the Python application via {banned}"
        );
    }
}

#[test]
fn pkgbuild_has_no_python_app_hard_depends() {
    let content = read("packaging/PKGBUILD");
    let depends = bash_array_body(&content, "depends");
    for banned in [
        "python",
        "python-numpy",
        "python-sounddevice",
        "python-sherpa-onnx",
        "python-gobject",
        "python-pytorch",
        "python-pytorch-cuda",
    ] {
        let single = format!("'{banned}'");
        let double = format!("\"{banned}\"");
        assert!(
            !depends.contains(&single) && !depends.contains(&double),
            "depends must not hard-require Python app/ML package {banned}; got:\n{depends}"
        );
    }

    // Isolated worker venvs: system pytorch is not useful; uv is the setup preference.
    let optdepends = bash_array_body(&content, "optdepends");
    assert!(
        !optdepends.contains("python-pytorch"),
        "optdepends must not list system python-pytorch (workers use isolated venvs)"
    );
    assert!(
        optdepends.contains("'uv:") || optdepends.contains("\"uv:"),
        "optdepends should list uv for setup-managed worker venvs"
    );
}
