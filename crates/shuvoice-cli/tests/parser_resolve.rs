use clap::Parser;
use shuvoice_cli::control::ControlCmd;
use shuvoice_cli::parser::{Cli, ResolvedCommand, resolve_command};

fn parse(args: &[&str]) -> Cli {
    Cli::try_parse_from(std::iter::once("shuvoice").chain(args.iter().copied())).unwrap()
}

#[test]
fn default_route_is_run() {
    let cli = parse(&[]);
    let (route, warnings) = resolve_command(&cli).unwrap();
    assert!(matches!(route, ResolvedCommand::Run { .. }));
    assert!(warnings.is_empty());
}

#[test]
fn legacy_preflight_maps() {
    let cli = parse(&["--preflight"]);
    let (route, warnings) = resolve_command(&cli).unwrap();
    assert!(matches!(route, ResolvedCommand::Preflight { .. }));
    assert_eq!(warnings.len(), 1);
}

#[test]
fn subcommand_control_tts_clipboard() {
    let cli = parse(&["control", "tts_speak_clipboard", "--control-wait-sec", "0"]);
    let (route, warnings) = resolve_command(&cli).unwrap();
    match route {
        ResolvedCommand::Control {
            command, wait_sec, ..
        } => {
            assert_eq!(command, ControlCmd::TtsSpeakClipboard);
            assert_eq!(wait_sec, 0.0);
        }
        other => panic!("unexpected {other:?}"),
    }
    assert!(warnings.is_empty());
}

#[test]
fn config_set_route() {
    let cli = parse(&["config", "set", "typing_text_case", "lowercase"]);
    let (route, _) = resolve_command(&cli).unwrap();
    assert!(matches!(route, ResolvedCommand::ConfigSet { .. }));
}

#[test]
fn setup_flags_route() {
    let cli = parse(&[
        "setup",
        "--install-missing",
        "--skip-model-download",
        "--skip-preflight",
        "--non-interactive",
    ]);
    let (route, _) = resolve_command(&cli).unwrap();
    match route {
        ResolvedCommand::Setup {
            install_missing,
            skip_model_download,
            skip_preflight,
            non_interactive,
            ..
        } => {
            assert!(install_missing);
            assert!(skip_model_download);
            assert!(skip_preflight);
            assert!(non_interactive);
        }
        other => panic!("unexpected {other:?}"),
    }
}
