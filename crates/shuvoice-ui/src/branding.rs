//! Branding asset location helpers.

use std::env;
use std::path::{Path, PathBuf};

/// Preferred logo filenames (badge first, then lockups).
pub const LOGO_FILENAMES: &[&str] = &[
    "shuvoice-variant-dark-badge.png",
    "shuvoice_variant_dark_lockup_alt.png",
    "shuvoice-variant-dark-lockup.png",
    "shuvoice-variant-light-lockup.png",
];

/// Directories searched for branding assets.
///
/// `repo_root` should be the repository root when running from a checkout.
/// Packaged doc paths are always included.
pub fn branding_directories(repo_root: Option<&Path>) -> Vec<PathBuf> {
    let mut directories: Vec<PathBuf> = Vec::new();

    if let Ok(custom) = env::var("SHUVOICE_BRANDING_DIR")
        && !custom.trim().is_empty()
    {
        directories.push(PathBuf::from(custom).expand_user());
    }

    if let Some(root) = repo_root {
        directories.push(root.join("docs/assets/branding"));
    }

    directories.push(PathBuf::from(
        "/usr/share/doc/shuvoice/docs/assets/branding",
    ));
    directories.push(PathBuf::from(
        "/usr/share/doc/shuvoice-git/docs/assets/branding",
    ));

    // De-duplicate while preserving order (by expanded display path).
    let mut seen = Vec::new();
    let mut unique = Vec::new();
    for directory in directories {
        let key = directory.to_string_lossy().to_string();
        if seen.iter().any(|s: &String| s == &key) {
            continue;
        }
        seen.push(key);
        unique.push(directory);
    }
    unique
}

/// All candidate logo paths in search order.
pub fn logo_candidates(repo_root: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for directory in branding_directories(repo_root) {
        for filename in LOGO_FILENAMES {
            candidates.push(directory.join(filename));
        }
    }
    candidates
}

/// First existing logo path, if any.
pub fn find_logo(repo_root: Option<&Path>) -> Option<PathBuf> {
    logo_candidates(repo_root).into_iter().find(|p| p.is_file())
}

trait ExpandUser {
    fn expand_user(self) -> PathBuf;
}

impl ExpandUser for PathBuf {
    fn expand_user(self) -> PathBuf {
        let s = self.to_string_lossy();
        if let Some(rest) = s.strip_prefix("~/") {
            if let Some(home) = env::var_os("HOME") {
                return PathBuf::from(home).join(rest);
            }
        } else if s == "~"
            && let Some(home) = env::var_os("HOME")
        {
            return PathBuf::from(home);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn logo_candidates_include_packaged_doc_paths() {
        let c = logo_candidates(None);
        assert!(c.iter().any(|p| p.starts_with("/usr/share/doc/shuvoice/")));
        assert!(
            c.iter()
                .any(|p| p.starts_with("/usr/share/doc/shuvoice-git/"))
        );
        assert!(
            c.iter()
                .any(|p| p.ends_with("shuvoice-variant-dark-badge.png"))
        );
    }

    #[test]
    fn find_logo_returns_first_existing_candidate() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let first = dir.join(LOGO_FILENAMES[0]);
        let second = dir.join(LOGO_FILENAMES[1]);
        fs::write(&second, b"logo").unwrap();
        // Only second exists — serialize against other env-mutating tests.
        let guard = crate::test_env::EnvGuard::acquire(&["SHUVOICE_BRANDING_DIR"]);
        guard.set("SHUVOICE_BRANDING_DIR", dir);
        let found = find_logo(None);
        drop(guard);
        assert_eq!(found.as_deref(), Some(second.as_path()));
        let _ = first; // silence
    }
}
