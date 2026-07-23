//! Compatibility checks for the CLI's frozen public vocabulary.

use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;

/// Tokens retained from the pre-rewrite public CLI contract.
const REQUIRED_TOP_LEVEL: &[&str] = &[
    "run",
    "control",
    "preflight",
    "setup",
    "wizard",
    "config",
    "model",
    "audio",
    "diagnostics",
];

const REQUIRED_CONTROL_CMDS: &[&str] = &[
    "start",
    "stop",
    "toggle",
    "status",
    "ping",
    "metrics",
    "debug_status",
    "tts_speak",
    "tts_speak_clipboard",
    "tts_pause",
    "tts_resume",
    "tts_toggle_pause",
    "tts_restart",
    "tts_stop",
    "tts_status",
];

const REQUIRED_LEGACY_FLAGS: &[&str] = &[
    "--download-model",
    "--preflight",
    "--list-audio-devices",
    "--wizard",
    "--control",
    "--control-wait-sec",
];

const REQUIRED_RUNTIME_OVERRIDES: &[&str] = &[
    "--asr-backend",
    "--device",
    "--right-context",
    "--sherpa-model-dir",
    "--sherpa-model-name",
    "--sherpa-provider",
    "--sherpa-num-threads",
    "--sherpa-chunk-ms",
    "--moonshine-model-name",
    "--moonshine-model-dir",
    "--moonshine-model-precision",
    "--moonshine-chunk-ms",
    "--moonshine-max-window-sec",
    "--moonshine-max-tokens",
    "--moonshine-provider",
    "--moonshine-onnx-threads",
    "--audio-device",
    "--input-gain",
    "--output-mode",
    "--control-socket",
];

const REQUIRED_CONFIG_SUBS: &[&str] = &["effective", "path", "validate", "set"];
const REQUIRED_CONFIG_SET_KEYS: &[&str] = &[
    "typing_final_injection_mode",
    "typing_text_case",
    "overlay_debug_mode",
];

fn help_stdout(args: &[&str]) -> String {
    let mut cmd = cargo_bin_cmd!("shuvoice");
    cmd.args(args);
    let out = cmd.output().expect("run shuvoice");
    assert!(
        out.status.success(),
        "help failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn top_level_help_covers_compatibility_vocabulary() {
    let help = help_stdout(&["--help"]);
    for token in REQUIRED_TOP_LEVEL
        .iter()
        .chain(REQUIRED_LEGACY_FLAGS.iter())
        .chain(REQUIRED_RUNTIME_OVERRIDES.iter())
    {
        assert!(
            help.contains(token),
            "top-level help missing compatibility token: {token}\n{help}"
        );
    }
}

#[test]
fn control_help_covers_compatibility_commands() {
    let help = help_stdout(&["control", "--help"]);
    for token in REQUIRED_CONTROL_CMDS {
        assert!(
            help.contains(token),
            "control help missing command: {token}\n{help}"
        );
    }
    assert!(help.contains("--control-wait-sec"));
    assert!(help.contains("--control-socket"));
}

#[test]
fn config_help_covers_compatibility_subcommands_and_set_keys() {
    let help = help_stdout(&["config", "--help"]);
    for token in REQUIRED_CONFIG_SUBS {
        assert!(
            help.contains(token),
            "config help missing subcommand: {token}\n{help}"
        );
    }
    let set_help = help_stdout(&["config", "set", "--help"]);
    for token in REQUIRED_CONFIG_SET_KEYS {
        assert!(
            set_help.contains(token),
            "config set help missing key: {token}\n{set_help}"
        );
    }
}

#[test]
fn setup_help_covers_compatibility_flags() {
    let help = help_stdout(&["setup", "--help"]);
    for token in [
        "--install-missing",
        "--skip-model-download",
        "--skip-preflight",
        "--tts-local-voice",
        "--tts-local-model-dir",
        "--non-interactive",
    ] {
        assert!(
            help.contains(token),
            "setup help missing flag: {token}\n{help}"
        );
    }
}

#[test]
fn model_and_audio_help_cover_compatibility_subcommands() {
    let model = help_stdout(&["model", "--help"]);
    assert!(model.contains("download"), "{model}");
    let audio = help_stdout(&["audio", "--help"]);
    assert!(audio.contains("list-devices"), "{audio}");
}

#[test]
fn diagnostics_help_has_json_flag() {
    let help = help_stdout(&["diagnostics", "--help"]);
    assert!(help.contains("--json"), "{help}");
}

#[test]
fn control_command_allowlist_matches_core_and_control_crates() {
    use shuvoice_cli::control::ControlCmd;
    use shuvoice_control::CONTROL_COMMANDS as CTRL;
    use shuvoice_core::CONTROL_COMMANDS as CORE;

    assert_eq!(CORE, CTRL);
    assert_eq!(CORE.len(), 15);
    // Every clap variant maps to a wire token present in the shared allowlist.
    for cmd in [
        ControlCmd::Start,
        ControlCmd::Stop,
        ControlCmd::Toggle,
        ControlCmd::Status,
        ControlCmd::Ping,
        ControlCmd::Metrics,
        ControlCmd::DebugStatus,
        ControlCmd::TtsSpeak,
        ControlCmd::TtsSpeakClipboard,
        ControlCmd::TtsPause,
        ControlCmd::TtsResume,
        ControlCmd::TtsTogglePause,
        ControlCmd::TtsRestart,
        ControlCmd::TtsStop,
        ControlCmd::TtsStatus,
    ] {
        assert!(
            CORE.contains(&cmd.as_str()),
            "clap ControlCmd {} missing from shared allowlist",
            cmd.as_str()
        );
    }
}

#[test]
fn waybar_help_lists_compatibility_actions() {
    let mut cmd = cargo_bin_cmd!("shuvoice-waybar");
    cmd.arg("--help");
    cmd.assert().success().stdout(
        predicate::str::contains("status")
            .and(predicate::str::contains("toggle-record"))
            .and(predicate::str::contains("launch-wizard"))
            .and(predicate::str::contains("service-toggle")),
    );
}
