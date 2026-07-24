//! Process bootstrap for `local.dev` via `shuvoice-io`.
//!
//! # Bootstrap safety
//!
//! `load_local_dev_env` mutates the process environment via `unsafe` `set_var`.
//! That is only sound **before** any other threads exist (including Tokio
//! worker threads created by `#[tokio::main]` / `Runtime::new`).
//!
//! Call [`bootstrap_local_dev_env`] from a plain synchronous `fn main` **before**
//! building a Tokio runtime. Async library entrypoints (`run_with_args`,
//! `run_waybar_main`) must **not** load env.

use std::sync::atomic::{AtomicBool, Ordering};

use shuvoice_io::env_loader::load_local_dev_env;
pub use shuvoice_io::env_loader::local_dev_env_path;

/// Exactly-once guard: local.dev is applied at most once per process.
static LOCAL_DEV_BOOTSTRAPPED: AtomicBool = AtomicBool::new(false);

/// Load `local.dev` into the process environment (exactly once).
///
/// # Bootstrap contract
///
/// Must be invoked from a **single-threaded** context before any Tokio runtime
/// (or other threads) are created. Subsequent calls are no-ops and return `0`.
///
/// IO errors are treated as zero loads (historical CLI tolerance).
pub fn bootstrap_local_dev_env() -> usize {
    // Exactly-once: first caller wins; later callers skip mutation entirely.
    if LOCAL_DEV_BOOTSTRAPPED.swap(true, Ordering::SeqCst) {
        return 0;
    }
    load_local_dev_env_best_effort(false)
}

/// Whether [`bootstrap_local_dev_env`] has already run in this process.
#[must_use]
pub fn local_dev_bootstrapped() -> bool {
    LOCAL_DEV_BOOTSTRAPPED.load(Ordering::SeqCst)
}

/// Load local.dev into the process environment, returning applied count.
///
/// Prefer [`bootstrap_local_dev_env`] at process start. This helper does **not**
/// enforce the exactly-once / pre-runtime contract — it is the raw best-effort
/// loader used by bootstrap and tests.
///
/// IO errors are treated as zero loads (matching historical CLI tolerance).
fn load_local_dev_env_best_effort(override_existing: bool) -> usize {
    match load_local_dev_env(None, override_existing) {
        Ok(n) => n,
        Err(err) => {
            // Tracing may not be configured yet during early bootstrap; prefer
            // eprintln only if a subscriber is absent would lose the message.
            // warn! is fine after logging init; during bootstrap it is a no-op
            // until the subscriber exists (events still discarded safely).
            tracing::warn!("Could not load local.dev: {err}");
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_is_exactly_once() {
        // May already be true if another test bootstrapped; still must not panic.
        let _first = bootstrap_local_dev_env();
        assert!(local_dev_bootstrapped());
        let second = bootstrap_local_dev_env();
        assert_eq!(second, 0, "second bootstrap must be a no-op");
    }
}
