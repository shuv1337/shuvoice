//! `shuvoice` process entry.
//!
//! Bootstrap order (security):
//! 1. Load `local.dev` on this thread (no other threads yet)
//! 2. Build Tokio multi-thread runtime
//! 3. `block_on` async CLI dispatch
//!
//! Never use `#[tokio::main]` here — that would spawn workers before env load.

fn main() -> std::process::ExitCode {
    let status = shuvoice_cli::run_blocking();
    if let Some(message) = status.message.as_ref() {
        // Messages for failures are usually already printed by commands.
        if status.code != 0 && !message.is_empty() && !message.starts_with("ERROR") {
            let _ = message;
        }
    }
    // Prefer returning ExitCode over process::exit so Tokio/runtime destructors flush.
    exit_code_from_i32(status.code)
}

fn exit_code_from_i32(code: i32) -> std::process::ExitCode {
    let code = if (0..=255).contains(&code) {
        code as u8
    } else {
        1
    };
    std::process::ExitCode::from(code)
}
