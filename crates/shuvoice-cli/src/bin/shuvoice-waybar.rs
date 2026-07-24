//! `shuvoice-waybar` process entry.
//!
//! Bootstrap order matches `shuvoice`: load `local.dev` before any Tokio runtime.

fn main() -> std::process::ExitCode {
    let status = shuvoice_cli::run_waybar_blocking();
    let code = if (0..=255).contains(&status.code) {
        status.code as u8
    } else {
        1
    };
    std::process::ExitCode::from(code)
}
