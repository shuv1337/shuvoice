//! Injectable HTTPS download for setup (Piper/Kokoro/tests).
//!
//! Production policy:
//! - HTTPS only
//! - curated host allow-list (huggingface.co / github.com and subdomains)
//! - finite connect / overall / body timeouts
//! - safe redirect policy (HTTPS + allow-list, limited hop count)
//! - `file://` is **disabled** on the production downloader; tests use
//!   [`ScriptedDownloader`] or [`ReqwestDownloader::with_file_urls`] explicitly.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;

/// Progress callback: (fraction 0..=1 or None, message).
pub type ProgressFn<'a> = dyn FnMut(Option<f32>, &str) + Send + 'a;

pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const DEFAULT_OVERALL_TIMEOUT: Duration = Duration::from_secs(120);
pub const DEFAULT_BODY_TIMEOUT: Duration = Duration::from_secs(120);
pub const MAX_REDIRECTS: usize = 10;

/// Hosts permitted for curated Piper / setup downloads.
pub const ALLOWED_DOWNLOAD_HOSTS: &[&str] = &["huggingface.co", "github.com"];

/// Download seam used by Piper curated voices and tests.
#[async_trait]
pub trait HttpDownloader: Send + Sync {
    async fn download_to_file(
        &self,
        url: &str,
        dest: &Path,
        max_bytes: u64,
        progress: &mut ProgressFn<'_>,
    ) -> Result<(), String>;
}

/// reqwest-backed downloader (production).
#[derive(Debug, Clone)]
pub struct ReqwestDownloader {
    pub client: Option<reqwest::Client>,
    /// When true, `file://` URLs are accepted (explicit test seam only).
    pub allow_file_urls: bool,
    pub connect_timeout: Duration,
    pub overall_timeout: Duration,
    pub body_timeout: Duration,
}

impl Default for ReqwestDownloader {
    fn default() -> Self {
        Self {
            client: None,
            allow_file_urls: false,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            overall_timeout: DEFAULT_OVERALL_TIMEOUT,
            body_timeout: DEFAULT_BODY_TIMEOUT,
        }
    }
}

impl ReqwestDownloader {
    /// Explicit test seam that permits `file://` copies. Production must not use this.
    pub fn with_file_urls() -> Self {
        Self {
            allow_file_urls: true,
            ..Self::default()
        }
    }
}

pub fn host_is_allowed(host: &str) -> bool {
    let host = host.trim().trim_end_matches('.').to_ascii_lowercase();
    if host.is_empty() {
        return false;
    }
    ALLOWED_DOWNLOAD_HOSTS
        .iter()
        .any(|domain| host == *domain || host.ends_with(&format!(".{domain}")))
}

pub fn validate_download_url(url: &str) -> Result<(), String> {
    let parsed = reqwest::Url::parse(url).map_err(|e| format!("invalid URL: {e}"))?;
    if parsed.scheme() != "https" {
        return Err(format!("Download URL must use HTTPS scheme: {url}"));
    }
    let host = parsed.host_str().unwrap_or("");
    if !host_is_allowed(host) {
        return Err(format!(
            "Download URL domain {host:?} is not in the allowed list: {ALLOWED_DOWNLOAD_HOSTS:?}"
        ));
    }
    Ok(())
}

fn redirect_policy() -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(|attempt| {
        if attempt.previous().len() >= MAX_REDIRECTS {
            return attempt.error("too many redirects");
        }
        let scheme = attempt.url().scheme().to_string();
        let host = attempt.url().host_str().unwrap_or("").to_string();
        if scheme != "https" {
            return attempt.error(format!("redirect to non-HTTPS blocked ({scheme})"));
        }
        if !host_is_allowed(&host) {
            return attempt.error(format!("redirect host not allowed: {host}"));
        }
        attempt.follow()
    })
}

#[async_trait]
impl HttpDownloader for ReqwestDownloader {
    async fn download_to_file(
        &self,
        url: &str,
        dest: &Path,
        max_bytes: u64,
        progress: &mut ProgressFn<'_>,
    ) -> Result<(), String> {
        if let Some(path) = url.strip_prefix("file://") {
            if !self.allow_file_urls {
                return Err(
                    "file:// URLs are disabled on the production downloader (test seam only)"
                        .into(),
                );
            }
            let meta = fs::metadata(path).map_err(|e| e.to_string())?;
            if meta.len() > max_bytes {
                return Err(format!(
                    "file size {} exceeds max_bytes {max_bytes}",
                    meta.len()
                ));
            }
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            fs::copy(path, dest).map_err(|e| e.to_string())?;
            progress(Some(1.0), "Local file copied");
            return Ok(());
        }

        validate_download_url(url)?;

        let client = match &self.client {
            Some(c) => c.clone(),
            None => reqwest::Client::builder()
                .connect_timeout(self.connect_timeout)
                .timeout(self.overall_timeout)
                .redirect(redirect_policy())
                .build()
                .map_err(|e| format!("http client: {e}"))?,
        };

        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("download failed: {e}"))?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!("download failed: HTTP {status}"));
        }
        // Final URL after redirects must still satisfy policy (belt-and-suspenders).
        let final_url = response.url().clone();
        if final_url.scheme() != "https" || !host_is_allowed(final_url.host_str().unwrap_or("")) {
            return Err(format!("download final URL not permitted: {final_url}"));
        }
        if let Some(len) = response.content_length()
            && len > max_bytes
        {
            return Err(format!(
                "Content-Length {len} exceeds max_bytes {max_bytes}"
            ));
        }
        let total = response.content_length();
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let tmp = unique_tmp_path(dest);
        let mut file = File::create(&tmp).map_err(|e| e.to_string())?;
        let mut stream = response.bytes_stream();
        use futures_util::StreamExt;
        let mut downloaded = 0u64;
        let mut last = -1.0f32;
        let body_deadline = Instant::now() + self.body_timeout;
        while let Some(chunk) = stream.next().await {
            if Instant::now() > body_deadline {
                let _ = fs::remove_file(&tmp);
                return Err(format!(
                    "download body timed out after {}s",
                    self.body_timeout.as_secs()
                ));
            }
            let chunk = chunk.map_err(|e| format!("stream error: {e}"))?;
            downloaded = downloaded.saturating_add(chunk.len() as u64);
            if downloaded > max_bytes {
                let _ = fs::remove_file(&tmp);
                return Err(format!(
                    "download exceeded max_bytes {max_bytes} (got at least {downloaded})"
                ));
            }
            file.write_all(&chunk).map_err(|e| e.to_string())?;
            if let Some(total) = total.filter(|t| *t > 0) {
                let frac = downloaded as f32 / total as f32;
                if frac - last >= 0.05 || frac >= 1.0 {
                    last = frac;
                    progress(
                        Some(frac.min(1.0)),
                        &format!("Downloading… {}%", (frac * 100.0) as u32),
                    );
                }
            }
        }
        file.flush().map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        drop(file);
        fs::rename(&tmp, dest).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            e.to_string()
        })?;
        if let Some(parent) = dest.parent() {
            fsync_dir(parent);
        }
        progress(Some(1.0), "Download complete");
        Ok(())
    }
}

fn unique_tmp_path(dest: &Path) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let name = format!(
        ".{}.{}-{}.tmp-download",
        dest.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("download"),
        std::process::id(),
        nanos
    );
    dest.parent().unwrap_or_else(|| Path::new(".")).join(name)
}

pub(crate) fn fsync_dir(dir: &Path) {
    if let Ok(file) = fs::File::open(dir) {
        let _ = file.sync_all();
    }
}

pub(crate) fn fsync_file(path: &Path) -> Result<(), String> {
    let file = fs::OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())
}

/// Force Unix file mode `0600` on model artifacts (owner read/write only).
///
/// No-op on non-Unix. Best-effort independent of process umask.
pub fn force_private_file_mode(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("failed to set mode 0600 on {}: {e}", path.display()))?;
    }
    let _ = path;
    Ok(())
}

/// Scripted downloader for tests (URL → bytes map). Never hits the network.
#[derive(Default)]
pub struct ScriptedDownloader {
    pub files: std::sync::Mutex<std::collections::BTreeMap<String, Vec<u8>>>,
    pub fail_urls: std::sync::Mutex<std::collections::BTreeSet<String>>,
    pub status_for: std::sync::Mutex<std::collections::BTreeMap<String, u16>>,
}

impl ScriptedDownloader {
    pub fn insert(&self, url: &str, bytes: impl Into<Vec<u8>>) {
        self.files
            .lock()
            .unwrap()
            .insert(url.to_string(), bytes.into());
    }

    pub fn fail(&self, url: &str) {
        self.fail_urls.lock().unwrap().insert(url.to_string());
    }
}

#[async_trait]
impl HttpDownloader for ScriptedDownloader {
    async fn download_to_file(
        &self,
        url: &str,
        dest: &Path,
        max_bytes: u64,
        progress: &mut ProgressFn<'_>,
    ) -> Result<(), String> {
        if self.fail_urls.lock().unwrap().contains(url) {
            return Err("download failed: HTTP 404".into());
        }
        if let Some(code) = self.status_for.lock().unwrap().get(url).copied()
            && !(200..300).contains(&code)
        {
            return Err(format!("download failed: HTTP {code}"));
        }
        let bytes = self
            .files
            .lock()
            .unwrap()
            .get(url)
            .cloned()
            .ok_or_else(|| format!("download failed: no scripted body for {url}"))?;
        if bytes.len() as u64 > max_bytes {
            return Err(format!(
                "download exceeded max_bytes {max_bytes} (got {})",
                bytes.len()
            ));
        }
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let tmp = unique_tmp_path(dest);
        fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
        fs::rename(&tmp, dest).map_err(|e| e.to_string())?;
        progress(Some(1.0), "Download complete");
        Ok(())
    }
}

/// Atomic write helper used after successful multi-file downloads.
pub fn replace_file(src: &Path, dest: &Path) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::rename(src, dest).or_else(|_| {
        fs::copy(src, dest).map(|_| ()).map_err(|e| e.to_string())?;
        let _ = fs::remove_file(src);
        Ok(())
    })
}

/// Publish a staged model+sidecar pair transactionally with rollback + fsync.
///
/// Modes:
/// - neither final exists → rename both; on second failure remove first
/// - finals exist → move aside to `.bak-publish`, rename new, drop backups; rollback on failure
pub fn publish_paired_files(
    stage_model: &Path,
    stage_sidecar: &Path,
    final_model: &Path,
    final_sidecar: &Path,
) -> Result<(), String> {
    // Staged artifacts: lock down before fsync/publish.
    force_private_file_mode(stage_model)?;
    force_private_file_mode(stage_sidecar)?;
    fsync_file(stage_model)?;
    fsync_file(stage_sidecar)?;

    let parent = final_model
        .parent()
        .or_else(|| final_sidecar.parent())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;

    let bak_model = unique_backup_path(final_model);
    let bak_side = unique_backup_path(final_sidecar);
    let had_model = final_model.exists();
    let had_side = final_sidecar.exists();

    if had_model {
        fs::rename(final_model, &bak_model).map_err(|e| e.to_string())?;
    }
    if had_side {
        fs::rename(final_sidecar, &bak_side).map_err(|e| {
            // restore model backup
            if had_model {
                let _ = fs::rename(&bak_model, final_model);
            }
            e.to_string()
        })?;
    }

    let rollback = |err: String| -> String {
        let _ = fs::remove_file(final_model);
        let _ = fs::remove_file(final_sidecar);
        if had_model {
            let _ = fs::rename(&bak_model, final_model);
        }
        if had_side {
            let _ = fs::rename(&bak_side, final_sidecar);
        }
        err
    };

    if let Err(e) = fs::rename(stage_model, final_model) {
        return Err(rollback(e.to_string()));
    }
    if let Err(e) = fs::rename(stage_sidecar, final_sidecar) {
        return Err(rollback(e.to_string()));
    }

    // Final artifacts: re-assert 0600 after rename (belt-and-suspenders vs FS quirks).
    if let Err(e) = force_private_file_mode(final_model) {
        return Err(rollback(e));
    }
    if let Err(e) = force_private_file_mode(final_sidecar) {
        return Err(rollback(e));
    }
    if let Err(e) = fsync_file(final_model) {
        return Err(rollback(e));
    }
    if let Err(e) = fsync_file(final_sidecar) {
        return Err(rollback(e));
    }

    let _ = fs::remove_file(&bak_model);
    let _ = fs::remove_file(&bak_side);
    fsync_dir(parent);
    Ok(())
}

fn unique_backup_path(path: &Path) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    path.with_extension(format!("bak-publish-{}-{}", std::process::id(), nanos))
}

pub type SharedDownloader = Arc<dyn HttpDownloader>;

pub fn default_downloader() -> SharedDownloader {
    Arc::new(ReqwestDownloader::default())
}

/// Unique staging directory next to target (pid + nanos; never reuses a fixed name).
pub fn stage_dir(parent: &Path, prefix: &str) -> Result<PathBuf, String> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = parent.join(format!(".{prefix}-{}-{}", std::process::id(), nanos));
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn https_host_policy() {
        assert!(
            validate_download_url("https://huggingface.co/rhasspy/piper-voices/resolve/x").is_ok()
        );
        assert!(validate_download_url("https://cdn.huggingface.co/x").is_ok());
        assert!(validate_download_url("https://github.com/x").is_ok());
        assert!(validate_download_url("http://huggingface.co/x").is_err());
        assert!(validate_download_url("https://evil.example/x").is_err());
        assert!(validate_download_url("file:///tmp/x").is_err());
    }

    #[test]
    fn production_downloader_rejects_file_urls() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let dl = ReqwestDownloader::default();
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("out.bin");
        let mut prog = |_: Option<f32>, _: &str| {};
        let err = rt
            .block_on(dl.download_to_file("file:///etc/hosts", &dest, 1024, &mut prog))
            .unwrap_err();
        assert!(err.contains("file://") || err.contains("disabled"), "{err}");
    }

    #[test]
    fn publish_paired_rolls_back_on_missing_stage_side() {
        let tmp = tempfile::tempdir().unwrap();
        let stage_m = tmp.path().join("s.onnx");
        let stage_s = tmp.path().join("s.onnx.json");
        let final_m = tmp.path().join("f.onnx");
        let final_s = tmp.path().join("f.onnx.json");
        fs::write(&stage_m, b"model-bytes").unwrap();
        // sidecar missing → fsync_file fails
        let err = publish_paired_files(&stage_m, &stage_s, &final_m, &final_s).unwrap_err();
        assert!(!final_m.exists(), "model must not publish alone: {err}");
        assert!(!final_s.exists());
    }

    #[test]
    fn publish_paired_succeeds() {
        let tmp = tempfile::tempdir().unwrap();
        let stage_m = tmp.path().join("s.onnx");
        let stage_s = tmp.path().join("s.onnx.json");
        let final_m = tmp.path().join("f.onnx");
        let final_s = tmp.path().join("f.onnx.json");
        fs::write(&stage_m, b"model-bytes-long").unwrap();
        fs::write(&stage_s, br#"{"sample_rate":22050}"#).unwrap();
        // Deliberately loose modes before publish.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&stage_m, fs::Permissions::from_mode(0o644)).unwrap();
            fs::set_permissions(&stage_s, fs::Permissions::from_mode(0o644)).unwrap();
        }
        publish_paired_files(&stage_m, &stage_s, &final_m, &final_s).unwrap();
        assert!(final_m.is_file());
        assert!(final_s.is_file());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let m = fs::metadata(&final_m).unwrap().permissions().mode() & 0o777;
            let s = fs::metadata(&final_s).unwrap().permissions().mode() & 0o777;
            assert_eq!(m, 0o600, "final model mode {m:#o}");
            assert_eq!(s, 0o600, "final sidecar mode {s:#o}");
        }
    }
}
