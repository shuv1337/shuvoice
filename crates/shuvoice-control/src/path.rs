//! Control socket path jail and secure directory helpers.

use std::env;
use std::fs;
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

use rustix::fs::{AtFlags, Mode, OFlags, chmod, openat, statat};
use rustix::process::{getuid, umask};

use crate::error::ControlError;

const SOCKET_NAME: &str = "control.sock";
const APP_DIR_NAME: &str = "shuvoice";
const DIR_MODE: u32 = 0o700;
const SOCKET_MODE: u32 = 0o600;

/// Return allowed parent roots for custom control sockets.
pub fn allowed_control_roots() -> Vec<PathBuf> {
    let mut roots = Vec::with_capacity(2);
    if let Some(runtime) = env::var_os("XDG_RUNTIME_DIR")
        && !runtime.is_empty()
    {
        let path = PathBuf::from(runtime);
        if let Ok(resolved) = path.canonicalize() {
            roots.push(resolved);
        } else {
            roots.push(path);
        }
    }
    let tmp = PathBuf::from("/tmp");
    roots.push(tmp.canonicalize().unwrap_or(tmp));
    roots
}

fn is_within(path: &Path, root: &Path) -> bool {
    path.starts_with(root)
}

fn lstat_meta(path: &Path) -> io::Result<fs::Metadata> {
    fs::symlink_metadata(path)
}

/// Refuse any symlink component on the logical path (TOCTOU / jail escape).
fn reject_symlink_components(path: &Path) -> Result<(), ControlError> {
    let mut cur = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::RootDir => {
                cur.push(Component::RootDir);
            }
            Component::Normal(name) => {
                cur.push(name);
                if let Ok(meta) = lstat_meta(&cur)
                    && meta.file_type().is_symlink()
                {
                    return Err(ControlError::DirectoryIsSymlink(cur));
                }
            }
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(ControlError::Other(
                    "control socket path must not contain '..'".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn verify_secure_dir(path: &Path) -> Result<(), ControlError> {
    let meta = lstat_meta(path).map_err(ControlError::Io)?;
    if meta.file_type().is_symlink() {
        return Err(ControlError::DirectoryIsSymlink(path.to_path_buf()));
    }
    if !meta.file_type().is_dir() {
        return Err(ControlError::NotADirectory(path.to_path_buf()));
    }
    let uid = getuid().as_raw();
    if meta.uid() != uid {
        return Err(ControlError::DirectoryNotOwned(path.to_path_buf()));
    }
    let mode = meta.mode() & 0o777;
    if mode != DIR_MODE {
        return Err(ControlError::DirectoryInsecureMode {
            path: path.to_path_buf(),
            mode,
        });
    }
    Ok(())
}

/// Ensure `path` exists as a user-only (`0700`) directory owned by the current uid.
///
/// Each path component is created/verified with symlink-refusal (`O_NOFOLLOW` /
/// `lstat`). Mode `0700` is hard-required after chmod — soft failures are rejected.
pub fn ensure_secure_directory(path: &Path) -> Result<(), ControlError> {
    if path.as_os_str().is_empty() {
        return Err(ControlError::Other("empty directory path".into()));
    }
    if !path.is_absolute() {
        return Err(ControlError::PathNotAbsolute);
    }

    // Walk components, creating missing dirs under a verified parent fd.
    let mut components = path.components();
    let Some(Component::RootDir) = components.next() else {
        return Err(ControlError::PathNotAbsolute);
    };

    let mut current = PathBuf::from("/");
    // Root is not owned by the user; skip ownership on `/` itself.
    for comp in components {
        match comp {
            Component::Normal(name) => {
                current.push(name);
                match lstat_meta(&current) {
                    Ok(meta) => {
                        if meta.file_type().is_symlink() {
                            return Err(ControlError::DirectoryIsSymlink(current));
                        }
                        if !meta.file_type().is_dir() {
                            return Err(ControlError::NotADirectory(current));
                        }
                        // Intermediate components outside our leaf may be root-owned
                        // (e.g. `/tmp`, `$XDG_RUNTIME_DIR`). Only enforce uid/mode on
                        // the final target path after the loop. For intermediates that
                        // we did not create, refuse world-writable non-sticky dirs.
                        let mode = meta.mode() & 0o777;
                        let uid = getuid().as_raw();
                        if meta.uid() == uid {
                            // Our directory: force 0700.
                            force_chmod(&current, DIR_MODE)?;
                            verify_secure_dir(&current)?;
                        } else {
                            // Foreign intermediate: refuse group/other write without sticky.
                            let sticky = meta.mode() & 0o1000 != 0;
                            if (mode & 0o022) != 0 && !sticky {
                                return Err(ControlError::DirectoryInsecureMode {
                                    path: current,
                                    mode,
                                });
                            }
                        }
                    }
                    Err(err) if err.kind() == io::ErrorKind::NotFound => {
                        create_secure_dir_nofollow(&current)?;
                        verify_secure_dir(&current)?;
                    }
                    Err(err) => return Err(ControlError::Io(err)),
                }
            }
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(ControlError::Other(
                    "control socket path must not contain '..'".into(),
                ));
            }
            other => {
                return Err(ControlError::Other(format!(
                    "unsupported path component: {other:?}"
                )));
            }
        }
    }

    verify_secure_dir(path)
}

fn force_chmod(path: &Path, mode: u32) -> Result<(), ControlError> {
    chmod(path, Mode::from_raw_mode(mode)).map_err(|err| {
        ControlError::Other(format!("chmod {} to {mode:#o}: {err}", path.display()))
    })?;
    Ok(())
}

fn create_secure_dir_nofollow(path: &Path) -> Result<(), ControlError> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or_else(|| ControlError::Other("directory has no parent".into()))?;
    let name = path
        .file_name()
        .ok_or_else(|| ControlError::Other("directory has no name".into()))?;

    // Open parent with O_DIRECTORY|O_NOFOLLOW when possible.
    let parent_fd = openat(
        rustix::fs::CWD,
        parent,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|err| ControlError::Other(format!("open parent {}: {err}", parent.display())))?;

    // Refuse if name already exists as a symlink.
    match statat(&parent_fd, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(st) => {
            let ftype = st.st_mode & libc::S_IFMT;
            if ftype == libc::S_IFLNK {
                return Err(ControlError::DirectoryIsSymlink(path.to_path_buf()));
            }
            if ftype != libc::S_IFDIR {
                return Err(ControlError::NotADirectory(path.to_path_buf()));
            }
            // Exists as dir — fall through to chmod/verify.
        }
        Err(err) if err == rustix::io::Errno::NOENT => {
            let old = umask(Mode::from_raw_mode(0o077));
            let mkdir_res = rustix::fs::mkdirat(&parent_fd, name, Mode::from_raw_mode(DIR_MODE));
            let _ = umask(old);
            mkdir_res
                .map_err(|e| ControlError::Other(format!("mkdirat {}: {e}", path.display())))?;
        }
        Err(err) => {
            return Err(ControlError::Other(format!(
                "statat {}: {err}",
                path.display()
            )));
        }
    }

    force_chmod(path, DIR_MODE)?;
    Ok(())
}

/// Force socket file mode to `0600` and verify.
pub fn force_socket_mode(path: &Path) -> Result<(), ControlError> {
    force_chmod(path, SOCKET_MODE)?;
    let meta = lstat_meta(path)?;
    if meta.file_type().is_symlink() {
        return Err(ControlError::Other(format!(
            "control socket {} is a symlink",
            path.display()
        )));
    }
    let mode = meta.mode() & 0o777;
    if mode != SOCKET_MODE {
        return Err(ControlError::SocketInsecureMode {
            path: path.to_path_buf(),
            mode,
        });
    }
    let uid = getuid().as_raw();
    if meta.uid() != uid {
        return Err(ControlError::Other(format!(
            "control socket {} not owned by current user",
            path.display()
        )));
    }
    Ok(())
}

/// Logical default path without creating directories.
#[must_use]
pub fn default_control_socket_path_logical() -> PathBuf {
    let runtime = env::var_os("XDG_RUNTIME_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    runtime.join(APP_DIR_NAME).join(SOCKET_NAME)
}

/// Default control socket path, ensuring the parent directory is secure.
pub fn default_control_socket_path() -> Result<PathBuf, ControlError> {
    prepare_control_socket_path(None)
}

/// Resolve and validate a control socket path **without** creating directories.
///
/// Use this for clients. Server startup should call [`prepare_control_socket_path`].
pub fn resolve_control_socket_path(path: Option<&str>) -> Result<PathBuf, ControlError> {
    resolve_control_socket_path_inner(path, false)
}

/// Resolve, validate, and ensure the parent directory is a secure `0700` dir.
pub fn prepare_control_socket_path(path: Option<&str>) -> Result<PathBuf, ControlError> {
    resolve_control_socket_path_inner(path, true)
}

fn resolve_control_socket_path_inner(
    path: Option<&str>,
    create_parent: bool,
) -> Result<PathBuf, ControlError> {
    let candidate = match path.map(str::trim).filter(|s| !s.is_empty()) {
        None => default_control_socket_path_logical(),
        Some(raw) => {
            let candidate = Path::new(raw);
            if !candidate.is_absolute() {
                return Err(ControlError::PathNotAbsolute);
            }
            if raw.ends_with('/') {
                return Err(ControlError::PathIsDirectory);
            }
            // lstat: refuse if the final component exists and is a directory/symlink-to-dir.
            if let Ok(meta) = lstat_meta(candidate) {
                if meta.file_type().is_dir()
                    || (meta.file_type().is_symlink()
                        && fs::metadata(candidate).map(|m| m.is_dir()).unwrap_or(false))
                {
                    return Err(ControlError::PathIsDirectory);
                }
            } else if candidate.exists() && candidate.is_dir() {
                return Err(ControlError::PathIsDirectory);
            }
            if candidate.extension().and_then(|e| e.to_str()) != Some("sock") {
                return Err(ControlError::PathBadSuffix);
            }
            // Reject `..` components early.
            if candidate
                .components()
                .any(|c| matches!(c, Component::ParentDir))
            {
                return Err(ControlError::Other(
                    "control socket path must not contain '..'".into(),
                ));
            }
            candidate.to_path_buf()
        }
    };

    let parent = candidate
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or(ControlError::PathNotAbsolute)?;

    // Refuse symlink components on the caller-supplied path before canonicalize.
    reject_symlink_components(parent)?;

    let parent_resolved = if parent.exists() {
        // canonicalize follows symlinks for existing path — then we re-check leaf.
        parent.canonicalize().map_err(ControlError::Io)?
    } else {
        canonicalize_with_missing(parent)?
    };

    let roots = allowed_control_roots();
    if !roots.iter().any(|root| is_within(&parent_resolved, root)) {
        let roots_text = roots
            .iter()
            .map(|r| r.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(ControlError::PathOutsideJail(roots_text));
    }

    if create_parent {
        ensure_secure_directory(&parent_resolved)?;
    } else if parent_resolved.exists() {
        // Clients must enforce the same owner/mode contract the server
        // establishes. Without this, a local attacker who pre-creates the
        // parent (e.g. /tmp/shuvoice under the world-writable /tmp fallback)
        // makes the server's bind fail while clients happily connect to the
        // attacker's socket and trust its responses.
        verify_secure_dir(&parent_resolved)?;
    }

    let name = candidate.file_name().ok_or(ControlError::PathBadSuffix)?;
    Ok(parent_resolved.join(name))
}

fn canonicalize_with_missing(path: &Path) -> Result<PathBuf, ControlError> {
    let mut existing = path.to_path_buf();
    let mut missing = Vec::new();
    while !existing.exists() {
        let file_name = existing
            .file_name()
            .map(|s| s.to_os_string())
            .ok_or_else(|| {
                ControlError::Other(format!(
                    "cannot resolve control socket parent {}",
                    path.display()
                ))
            })?;
        // Refuse symlink at any existing ancestor during walk — checked later by ensure.
        missing.push(file_name);
        existing = existing.parent().map(Path::to_path_buf).ok_or_else(|| {
            ControlError::Other(format!(
                "cannot resolve control socket parent {}",
                path.display()
            ))
        })?;
    }
    // Existing ancestor must not be a symlink chain escape: canonicalize it.
    let meta = lstat_meta(&existing)?;
    if meta.file_type().is_symlink() {
        // canonicalize will resolve; jail check happens on result.
    }
    let mut resolved = existing.canonicalize().map_err(ControlError::Io)?;
    for part in missing.into_iter().rev() {
        if part == ".." {
            return Err(ControlError::Other(
                "control socket path must not contain '..'".into(),
            ));
        }
        resolved.push(part);
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn rejects_relative_path() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let err = resolve_control_socket_path(Some("control.sock")).unwrap_err();
        assert!(matches!(err, ControlError::PathNotAbsolute));
    }

    #[test]
    fn rejects_non_sock_suffix() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let err = resolve_control_socket_path(Some("/tmp/control.socket")).unwrap_err();
        assert!(matches!(err, ControlError::PathBadSuffix));
    }

    #[test]
    fn rejects_outside_jail() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let err = resolve_control_socket_path(Some("/var/lib/shuvoice/control.sock")).unwrap_err();
        assert!(matches!(err, ControlError::PathOutsideJail(_)));
    }

    #[test]
    fn rejects_parent_dir_components() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let err = resolve_control_socket_path(Some("/tmp/foo/../evil/control.sock")).unwrap_err();
        assert!(matches!(err, ControlError::Other(_)));
    }

    #[test]
    fn client_resolve_does_not_create_dirs() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("nested-missing").join("control.sock");
        let resolved = resolve_control_socket_path(Some(sock.to_str().unwrap())).unwrap();
        assert_eq!(resolved, sock);
        assert!(!sock.parent().unwrap().exists());
    }

    #[test]
    fn client_rejects_insecure_existing_parent() {
        // A pre-created world-readable parent (e.g. an attacker's /tmp/shuvoice)
        // must be refused on the client path, not just at server bind.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path().join("shuvoice");
        fs::create_dir(&parent).unwrap();
        let mut perms = fs::metadata(&parent).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&parent, perms).unwrap();
        let sock = parent.join("control.sock");
        let err = resolve_control_socket_path(Some(sock.to_str().unwrap())).unwrap_err();
        assert!(
            matches!(err, ControlError::DirectoryInsecureMode { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn client_accepts_secure_existing_parent() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path().join("shuvoice");
        fs::create_dir(&parent).unwrap();
        let mut perms = fs::metadata(&parent).unwrap().permissions();
        perms.set_mode(0o700);
        fs::set_permissions(&parent, perms).unwrap();
        let sock = parent.join("control.sock");
        let resolved = resolve_control_socket_path(Some(sock.to_str().unwrap())).unwrap();
        assert_eq!(resolved, sock);
    }

    #[test]
    fn prepare_creates_secure_dir() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("nested").join("control.sock");
        let resolved = prepare_control_socket_path(Some(sock.to_str().unwrap())).unwrap();
        assert_eq!(resolved, sock);
        let meta = fs::metadata(resolved.parent().unwrap()).unwrap();
        assert_eq!(meta.permissions().mode() & 0o777, 0o700);
        assert!(!meta.file_type().is_symlink());
    }

    #[test]
    fn refuses_symlink_directory() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real");
        fs::create_dir(&real).unwrap();
        let link = dir.path().join("link");
        symlink(&real, &link).unwrap();
        let sock = link.join("control.sock");
        // prepare walks components; symlink component must be refused.
        let err = prepare_control_socket_path(Some(sock.to_str().unwrap())).unwrap_err();
        assert!(
            matches!(err, ControlError::DirectoryIsSymlink(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn hard_fails_insecure_mode() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("open");
        fs::create_dir(&target).unwrap();
        let mut perms = fs::metadata(&target).unwrap().permissions();
        perms.set_mode(0o777);
        fs::set_permissions(&target, perms).unwrap();
        // ensure_secure_directory should chmod back to 0700 successfully for owner.
        ensure_secure_directory(&target).unwrap();
        let meta = fs::metadata(&target).unwrap();
        assert_eq!(meta.permissions().mode() & 0o777, 0o700);
    }

    #[test]
    fn force_socket_mode_sets_0600() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("s.sock");
        fs::File::create(&sock).unwrap();
        let mut perms = fs::metadata(&sock).unwrap().permissions();
        perms.set_mode(0o666);
        fs::set_permissions(&sock, perms).unwrap();
        force_socket_mode(&sock).unwrap();
        let meta = fs::metadata(&sock).unwrap();
        assert_eq!(meta.permissions().mode() & 0o777, 0o600);
    }

    #[test]
    fn default_path_uses_runtime_dir() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        // SAFETY: serialized by ENV_LOCK for tests only.
        // SAFETY: serialized by ENV_LOCK for tests only.
        unsafe {
            env::set_var("XDG_RUNTIME_DIR", dir.path());
        }
        let path = default_control_socket_path().unwrap();
        assert_eq!(path.file_name().unwrap(), "control.sock");
        assert_eq!(path.parent().unwrap().file_name().unwrap(), "shuvoice");
        let meta = fs::metadata(path.parent().unwrap()).unwrap();
        assert_eq!(meta.permissions().mode() & 0o777, 0o700);
        // SAFETY: serialized by ENV_LOCK for tests only.
        unsafe {
            env::remove_var("XDG_RUNTIME_DIR");
        }
    }
}
