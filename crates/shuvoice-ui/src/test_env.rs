//! Process-global env lock for tests that mutate `std::env`.
//!
//! Multiple tests in this crate touch `XDG_*` / `HOME` / branding overrides.
//! Without serialization they race and flake (e.g. one test clears XDG while
//! another still expects it).

#![cfg(test)]

use std::ffi::{OsStr, OsString};
use std::sync::{Mutex, MutexGuard, OnceLock};

static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn lock() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// RAII guard: holds the crate env mutex and restores prior values on drop
/// (including panic unwind).
pub struct EnvGuard {
    _guard: MutexGuard<'static, ()>,
    saved: Vec<(String, Option<OsString>)>,
}

impl EnvGuard {
    /// Acquire the global lock and snapshot the listed variables.
    pub fn acquire(keys: &[&str]) -> Self {
        let _guard = lock();
        let mut saved = Vec::with_capacity(keys.len());
        for key in keys {
            saved.push(((*key).to_string(), std::env::var_os(key)));
        }
        Self { _guard, saved }
    }

    /// Set a variable for the duration of this guard.
    ///
    /// # Safety
    ///
    /// Caller must hold this guard (process-global env mutation is serialized).
    pub fn set(&self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) {
        let key = key.as_ref();
        // SAFETY: EnvGuard holds ENV_LOCK; tests are the only mutators.
        unsafe {
            std::env::set_var(key, value);
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, prior) in self.saved.drain(..) {
            // SAFETY: still holding ENV_LOCK via `_guard`; restore snapshot.
            unsafe {
                match prior {
                    Some(value) => std::env::set_var(&key, value),
                    None => std::env::remove_var(&key),
                }
            }
        }
    }
}

/// Run `body` with temporary XDG_CONFIG_HOME / XDG_DATA_HOME / HOME under a lock.
pub fn with_isolated_xdg(body: impl FnOnce(&std::path::Path)) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = dir.path().join("config");
    let data = dir.path().join("data");
    std::fs::create_dir_all(&config).expect("config dir");
    std::fs::create_dir_all(&data).expect("data dir");

    let guard = EnvGuard::acquire(&["XDG_CONFIG_HOME", "XDG_DATA_HOME", "HOME"]);
    guard.set("XDG_CONFIG_HOME", &config);
    guard.set("XDG_DATA_HOME", &data);
    guard.set("HOME", dir.path());

    body(dir.path());
    // guard drops here → full restore even on panic
    drop(guard);
    // keep tempdir alive until after restore
    drop(dir);
}
