//! gtk4-layer-shell presence probe (multiarch + dynamic loader).

use std::ffi::CString;
use std::path::PathBuf;

const SONAMES: &[&str] = &["libgtk4-layer-shell.so.0", "libgtk4-layer-shell.so"];

/// Well-known multiarch / FHS library directories.
fn candidate_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/usr/local/lib64"),
        PathBuf::from("/lib"),
        PathBuf::from("/lib64"),
    ];
    // Debian/Ubuntu multiarch
    for arch in [
        "x86_64-linux-gnu",
        "aarch64-linux-gnu",
        "arm-linux-gnueabihf",
        "i386-linux-gnu",
        "riscv64-linux-gnu",
    ] {
        dirs.push(PathBuf::from(format!("/usr/lib/{arch}")));
        dirs.push(PathBuf::from(format!("/lib/{arch}")));
        dirs.push(PathBuf::from(format!("/usr/local/lib/{arch}")));
    }
    // Fedora-style
    for arch in ["x86_64", "aarch64", "i686"] {
        dirs.push(PathBuf::from(format!("/usr/lib64/{arch}")));
    }
    if let Ok(ld) = std::env::var("LD_LIBRARY_PATH") {
        for part in std::env::split_paths(&ld) {
            if !part.as_os_str().is_empty() {
                dirs.push(part);
            }
        }
    }
    dirs
}

/// Return the first filesystem hit for the layer-shell library, if any.
pub fn layer_shell_library_path() -> Option<PathBuf> {
    for dir in candidate_dirs() {
        for name in SONAMES {
            let path = dir.join(name);
            if path.is_file() || path.exists() {
                return Some(path);
            }
        }
    }
    None
}

/// True when the library is loadable via path probe or `dlopen` soname resolution.
pub fn layer_shell_present() -> bool {
    if layer_shell_library_path().is_some() {
        return true;
    }
    #[cfg(unix)]
    {
        for name in SONAMES {
            if dlopen_probe(name) {
                return true;
            }
        }
    }
    false
}

/// Human-readable detail for preflight.
pub fn layer_shell_detail() -> Result<String, String> {
    if let Some(path) = layer_shell_library_path() {
        return Ok(format!("{} present", path.display()));
    }
    #[cfg(unix)]
    {
        for name in SONAMES {
            if dlopen_probe(name) {
                return Ok(format!("{name} loadable via dynamic linker"));
            }
        }
    }
    Err(
        "libgtk4-layer-shell.so not found (checked multiarch lib dirs + dlopen). \
         Install: pacman -S gtk4-layer-shell  |  apt install libgtk-4-layer-shell0  |  \
         dnf install gtk4-layer-shell"
            .into(),
    )
}

#[cfg(unix)]
fn dlopen_probe(name: &str) -> bool {
    let Ok(c) = CString::new(name) else {
        return false;
    };
    // SAFETY: probe-only dlopen/dlclose of a well-formed soname; handle closed immediately.
    unsafe {
        let h = libc::dlopen(c.as_ptr(), libc::RTLD_LAZY);
        if h.is_null() {
            false
        } else {
            libc::dlclose(h);
            true
        }
    }
}

#[cfg(not(unix))]
fn dlopen_probe(_name: &str) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn candidate_dirs_include_multiarch() {
        let dirs = candidate_dirs();
        assert!(dirs.iter().any(|d| d.ends_with("x86_64-linux-gnu")));
        assert!(dirs.iter().any(|d| d == Path::new("/usr/lib")));
    }
}
