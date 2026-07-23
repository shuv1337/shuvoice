//! Journald-aware logging setup for the CLI.

use tracing_subscriber::EnvFilter;

pub fn configure_logging(verbose: bool) {
    let journald = std::env::var_os("JOURNAL_STREAM").is_some();
    let default_level = if verbose { "debug" } else { "info" };
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));

    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_writer(std::io::stderr);

    if journald {
        let _ = builder.without_time().try_init();
    } else {
        let _ = builder.try_init();
    }
}
