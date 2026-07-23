//! Application composition and CLI dispatch for ShuVoice.

pub mod application;
pub mod commands;
pub mod compose;
pub mod config;
pub mod control;
pub mod env_loader;
pub mod error;
pub mod logging;
pub mod parser;
pub mod setup;
pub mod waybar;

// Sync bootstrap (call before Tokio). Re-exported for embedders/docs.
pub use crate::env_loader::{bootstrap_local_dev_env, local_dev_bootstrapped};

use clap::Parser;
use tracing::warn;

use crate::commands::{
    audio, control as control_cmd, diagnostics, model, preflight, run, setup as setup_cmd, wizard,
};
use crate::config::{cmd_effective, cmd_path, cmd_set, cmd_validate, load_effective_config};
use crate::env_loader::local_dev_env_path;
use crate::error::{EXIT_FAILURE, EXIT_SUCCESS, EXIT_USAGE, ExitStatus};
use crate::logging::configure_logging;
use crate::parser::{Cli, ResolvedCommand, resolve_command};
use crate::waybar::{WaybarCli, run_waybar};

/// Synchronous process bootstrap for the `shuvoice` binary.
///
/// Loads `local.dev` **before** any Tokio runtime exists, then builds a
/// multi-thread runtime and `block_on`s async dispatch.
///
/// # Bootstrap safety
///
/// `local.dev` applies `env::set_var` (unsafe). That must not race other
/// threads. This helper is the supported binary entry; do not call
/// [`bootstrap_local_dev_env`] after `Runtime::new` / `#[tokio::main]`.
pub fn run_blocking() -> ExitStatus {
    let loaded = bootstrap_local_dev_env();
    // Logging after env load so RUST_LOG / similar from local.dev apply.
    // Verbose is parsed inside the async path after runtime start — default
    // filter uses RUST_LOG if set by bootstrap; `--verbose` still upgrades
    // when parse runs (subscriber already init: try_init is best-effort).
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(err) => {
            eprintln!("ERROR: failed to start async runtime: {err}");
            return ExitStatus::code(EXIT_FAILURE).with_message(format!("async runtime: {err}"));
        }
    };
    if loaded > 0 {
        // Subscriber may not exist yet; message emitted after configure_logging
        // inside run_with_args when possible. Keep a debug breadcrumb via stderr
        // only at trace-less bootstrap — actual debug log is in run_with_args
        // when bootstrapped count is visible via local_dev_bootstrapped().
        let _ = loaded;
    }
    rt.block_on(run_with_args(std::env::args_os()))
}

/// Library entry used by tests / embedders that already own a runtime.
///
/// **Does not** load `local.dev`. Call [`bootstrap_local_dev_env`] first from
/// a single-threaded context if process env should include it.
pub async fn run() -> ExitStatus {
    run_with_args(std::env::args_os()).await
}

/// Parse argv and dispatch.
///
/// **Does not** mutate the process environment. Env bootstrap is
/// [`bootstrap_local_dev_env`] / [`run_blocking`] only.
pub async fn run_with_args<I, T>(args: I) -> ExitStatus
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(err) => {
            use clap::error::ErrorKind;
            let code = match err.kind() {
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => EXIT_SUCCESS,
                _ => EXIT_USAGE,
            };
            let _ = err.print();
            return ExitStatus::code(code);
        }
    };

    configure_logging(cli.verbose);
    if local_dev_bootstrapped() {
        tracing::debug!(
            "local.dev bootstrap already applied (path {})",
            local_dev_env_path().display()
        );
    }

    let (resolved, warnings) = match resolve_command(&cli) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitStatus::code(EXIT_USAGE);
        }
    };
    for message in warnings {
        warn!("{message}");
    }

    dispatch(resolved).await
}

async fn dispatch(resolved: ResolvedCommand) -> ExitStatus {
    match resolved {
        ResolvedCommand::Run { overrides } => run::run_app(&overrides).await,
        ResolvedCommand::Wizard => wizard::run_wizard_command(),
        ResolvedCommand::AudioListDevices => audio::list_devices(),
        ResolvedCommand::ConfigPath => cmd_path(),
        ResolvedCommand::ConfigValidate => cmd_validate(),
        ResolvedCommand::ConfigEffective => cmd_effective(),
        ResolvedCommand::ConfigSet { key, value } => cmd_set(key, value),
        ResolvedCommand::Preflight { overrides } => match load_effective_config(&overrides) {
            Ok(cfg) => preflight::run_preflight(&cfg).await,
            Err(err) => {
                eprintln!("ERROR: {err}");
                ExitStatus::code(EXIT_FAILURE)
            }
        },
        ResolvedCommand::Setup {
            overrides,
            install_missing,
            skip_model_download,
            skip_preflight,
            tts_local_voice,
            tts_local_model_dir,
            non_interactive,
        } => match load_effective_config(&overrides) {
            Ok(cfg) => {
                setup_cmd::run_setup(
                    &cfg,
                    setup_cmd::SetupOptions {
                        install_missing,
                        skip_model_download,
                        skip_preflight,
                        tts_local_voice,
                        tts_local_model_dir,
                        non_interactive,
                    },
                )
                .await
            }
            Err(err) => {
                eprintln!("ERROR: {err}");
                ExitStatus::code(EXIT_FAILURE)
            }
        },
        ResolvedCommand::Control {
            command,
            wait_sec,
            socket,
            overrides,
        } => {
            let socket = match socket {
                Some(s) => Some(s),
                None => match load_effective_config(&overrides) {
                    Ok(cfg) => cfg.control_socket,
                    Err(_) => overrides.control_socket,
                },
            };
            control_cmd::execute(command, socket.as_deref(), wait_sec).await
        }
        ResolvedCommand::ModelDownload { overrides } => match load_effective_config(&overrides) {
            Ok(cfg) => model::download_model(&cfg).await,
            Err(err) => {
                eprintln!("ERROR: {err}");
                ExitStatus::code(EXIT_FAILURE)
            }
        },
        ResolvedCommand::Diagnostics { overrides, json } => {
            match load_effective_config(&overrides) {
                Ok(cfg) => diagnostics::execute(&cfg, json).await,
                Err(err) => {
                    eprintln!("ERROR: {err}");
                    ExitStatus::code(EXIT_FAILURE)
                }
            }
        }
    }
}

/// Synchronous process bootstrap for the `shuvoice-waybar` binary.
///
/// Loads `local.dev` before creating a Tokio runtime (see [`run_blocking`]).
pub fn run_waybar_blocking() -> ExitStatus {
    let _loaded = bootstrap_local_dev_env();
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(err) => {
            eprintln!("ERROR: failed to start async runtime: {err}");
            return ExitStatus::code(EXIT_FAILURE).with_message(format!("async runtime: {err}"));
        }
    };
    rt.block_on(run_waybar_main())
}

/// Library entry for waybar when a runtime already exists.
///
/// **Does not** load `local.dev` (see [`bootstrap_local_dev_env`]).
pub async fn run_waybar_main() -> ExitStatus {
    let cli = match WaybarCli::try_parse() {
        Ok(cli) => cli,
        Err(err) => {
            use clap::error::ErrorKind;
            let code = match err.kind() {
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => EXIT_SUCCESS,
                _ => EXIT_USAGE,
            };
            let _ = err.print();
            return ExitStatus::code(code);
        }
    };
    configure_logging(false);
    run_waybar(cli).await
}

#[cfg(test)]
mod bootstrap_tests {
    use super::*;

    /// Async dispatch must never call the env loader (structural + behavioral).
    #[test]
    fn run_with_args_source_does_not_load_local_dev() {
        let src = include_str!("lib.rs");
        // Extract the run_with_args function body roughly.
        let start = src
            .find("pub async fn run_with_args")
            .expect("run_with_args present");
        let rest = &src[start..];
        let end = rest
            .find(
                "
async fn dispatch",
            )
            .unwrap_or(rest.len());
        let body = &rest[..end];
        assert!(
            !body.contains("load_local_dev_env"),
            "run_with_args must not load local.dev (bootstrap owns env mutation)"
        );
        assert!(
            !body.contains("bootstrap_local_dev_env"),
            "run_with_args must not bootstrap env (would race if called under runtime)"
        );
        assert!(
            !body.contains("set_var"),
            "run_with_args must not mutate process env"
        );
    }

    #[test]
    fn run_waybar_main_source_does_not_load_local_dev() {
        let src = include_str!("lib.rs");
        let start = src
            .find("pub async fn run_waybar_main")
            .expect("run_waybar_main present");
        let rest = &src[start..];
        // stop at bootstrap_tests or end of function next pub/mod
        let end = rest
            .find(
                "
#[cfg(test)]",
            )
            .unwrap_or(rest.len());
        let body = &rest[..end];
        assert!(
            !body.contains("load_local_dev_env"),
            "run_waybar_main must not load local.dev"
        );
        assert!(!body.contains("bootstrap_local_dev_env"), "{body}");
    }

    #[test]
    fn binary_bootstrap_helpers_exist_and_are_sync() {
        let src = include_str!("lib.rs");
        assert!(src.contains("pub fn run_blocking()"));
        assert!(src.contains("pub fn run_waybar_blocking()"));
        // Sync helpers must call bootstrap before Runtime build.
        for name in ["run_blocking", "run_waybar_blocking"] {
            let start = src
                .find(&format!("pub fn {name}"))
                .unwrap_or_else(|| panic!("{name}"));
            let chunk = &src[start..start + 800.min(src.len() - start)];
            let boot_at = chunk
                .find("bootstrap_local_dev_env")
                .expect("bootstrap call");
            let rt_at = chunk
                .find("Builder::new_multi_thread")
                .expect("runtime build");
            assert!(
                boot_at < rt_at,
                "{name}: bootstrap_local_dev_env must appear before Tokio Runtime build"
            );
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn async_run_with_args_does_not_mutate_env() {
        // Behavioral: under an existing runtime, dispatch must not set env vars
        // from local.dev. We assert a sentinel key is unchanged.
        const KEY: &str = "SHUVOICE_BOOTSTRAP_SENTINEL_ASYNC_PATH";
        let prev = std::env::var_os(KEY);
        // SAFETY: test-only env mutation on the current thread; no concurrent
        // env writers in this test; restored before return below.
        unsafe {
            std::env::remove_var(KEY);
        }
        let status = run_with_args(["shuvoice", "--help"]).await;
        assert_eq!(status.code, EXIT_SUCCESS);
        let after = std::env::var_os(KEY);
        assert!(after.is_none(), "async path must not invent env keys");
        // SAFETY: restore process env to the value captured before this test.
        unsafe {
            match prev {
                Some(v) => std::env::set_var(KEY, v),
                None => std::env::remove_var(KEY),
            }
        }
    }
}
