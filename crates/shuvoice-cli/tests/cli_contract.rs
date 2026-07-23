//! assert_cmd contract tests for the shuvoice CLI surface.

use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;
use serial_test::serial;
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

fn with_xdg<F: FnOnce(PathBuf)>(f: F) {
    let dir = tempdir().unwrap();
    let config = dir.path().join("config");
    let data = dir.path().join("data");
    let runtime = dir.path().join("runtime");
    fs::create_dir_all(&config).unwrap();
    fs::create_dir_all(&data).unwrap();
    fs::create_dir_all(&runtime).unwrap();

    // SAFETY: serial tests mutate process env under #[serial].
    unsafe {
        std::env::set_var("XDG_CONFIG_HOME", &config);
        std::env::set_var("XDG_DATA_HOME", &data);
        std::env::set_var("XDG_RUNTIME_DIR", &runtime);
    }
    f(config);
    // SAFETY: restore process env after serial test mutation above.
    unsafe {
        std::env::remove_var("XDG_CONFIG_HOME");
        std::env::remove_var("XDG_DATA_HOME");
        std::env::remove_var("XDG_RUNTIME_DIR");
    }
}

#[test]
fn help_lists_required_subcommands() {
    let mut cmd = cargo_bin_cmd!("shuvoice");
    cmd.arg("--help");
    cmd.assert().success().stdout(
        predicate::str::contains("run")
            .and(predicate::str::contains("control"))
            .and(predicate::str::contains("preflight"))
            .and(predicate::str::contains("setup"))
            .and(predicate::str::contains("wizard"))
            .and(predicate::str::contains("config"))
            .and(predicate::str::contains("model"))
            .and(predicate::str::contains("audio"))
            .and(predicate::str::contains("diagnostics")),
    );
}

#[test]
fn control_help_lists_tts_speak_clipboard() {
    let mut cmd = cargo_bin_cmd!("shuvoice");
    cmd.args(["control", "--help"]);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("tts_speak_clipboard"));
}

#[test]
#[serial]
fn config_path_and_validate_defaults() {
    with_xdg(|config_home| {
        let mut cmd = cargo_bin_cmd!("shuvoice");
        cmd.arg("config").arg("path");
        cmd.assert().success().stdout(predicate::str::contains(
            config_home
                .join("shuvoice")
                .join("config.toml")
                .to_string_lossy()
                .as_ref(),
        ));

        let mut cmd = cargo_bin_cmd!("shuvoice");
        cmd.arg("config").arg("validate");
        cmd.assert()
            .success()
            .stdout(predicate::str::contains("OK (schema="));
    });
}

#[test]
#[serial]
fn config_set_injection_mode_and_effective() {
    with_xdg(|config_home| {
        let cfg = config_home.join("shuvoice").join("config.toml");
        fs::create_dir_all(cfg.parent().unwrap()).unwrap();
        fs::write(&cfg, "config_version = 1\n").unwrap();

        let mut cmd = cargo_bin_cmd!("shuvoice");
        cmd.args(["config", "set", "typing_final_injection_mode", "direct"]);
        cmd.assert().success().stdout(predicate::str::contains(
            "OK set typing_final_injection_mode=direct",
        ));

        let text = fs::read_to_string(&cfg).unwrap();
        assert!(text.contains("typing_final_injection_mode"));
        assert!(text.contains("direct"));

        let mut cmd = cargo_bin_cmd!("shuvoice");
        cmd.args(["config", "set", "overlay_debug_mode", "true"]);
        cmd.assert().success();

        let mut cmd = cargo_bin_cmd!("shuvoice");
        cmd.arg("config").arg("effective");
        cmd.assert()
            .success()
            .stdout(predicate::str::contains("overlay_debug_mode"));
    });
}

#[test]
fn legacy_flags_mutually_exclusive_exit_2() {
    let mut cmd = cargo_bin_cmd!("shuvoice");
    cmd.args(["--preflight", "--wizard"]);
    cmd.assert().failure().code(predicate::eq(2));
}

#[test]
#[serial]
fn control_missing_socket_exits_1() {
    with_xdg(|_| {
        let mut cmd = cargo_bin_cmd!("shuvoice");
        cmd.args(["control", "ping", "--control-wait-sec", "0"]);
        cmd.assert()
            .failure()
            .code(predicate::eq(1))
            .stderr(predicate::str::contains("ERROR:"));
    });
}

#[test]
#[serial]
fn run_dependency_exit_is_78() {
    with_xdg(|config_home| {
        let cfg = config_home.join("shuvoice").join("config.toml");
        fs::create_dir_all(cfg.parent().unwrap()).unwrap();
        fs::write(
            &cfg,
            "config_version = 1\n\n[asr]\nasr_backend = \"nemo\"\n",
        )
        .unwrap();
        let data = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap();
        fs::create_dir_all(data.join("shuvoice")).unwrap();
        fs::write(data.join("shuvoice").join(".wizard-done"), b"1").unwrap();

        let mut cmd = cargo_bin_cmd!("shuvoice");
        cmd.arg("run");
        cmd.assert().failure().code(predicate::eq(78));
    });
}

#[test]
fn waybar_status_prints_json() {
    let mut cmd = cargo_bin_cmd!("shuvoice-waybar");
    cmd.arg("status");
    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("\"text\""),
        "expected waybar json, got: {stdout}"
    );
    assert!(stdout.contains("\"class\""));
}
