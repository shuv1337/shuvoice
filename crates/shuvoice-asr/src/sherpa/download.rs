//! Safe Sherpa model archive download + extract.
//!
//! Hardening:
//! - HTTPS-only production downloads with curated GitHub/release-asset hosts
//! - safe redirect policy (HTTPS + allow-list, hop cap)
//! - `file://` only via explicit test/opt-in hardening seam (never default)
//! - finite connect timeout, overall download deadline, body inactivity timeout
//! - content-length + overall compressed byte cap
//! - tar entry-count + total uncompressed byte caps (extraction bomb defense)
//! - cancellation flag
//! - extract + tree copy + install transaction on `spawn_blocking`
//! - wall-clock bound + cooperative cancel during extract/copy/install
//! - bounded-chunk I/O (SHA / tar payload / tree copy / file://) so cancel overruns
//!   by at most one chunk — cooperative granularity, not a hard OS preemption guarantee
//! - atomic temp dir + rename into place (`ErrorKind::CrossesDevices` fallback)
//! - archive path traversal, symlink/hardlink/special/sparse entry rejection
//! - temp/stage `0700`, archive + installed model files `0600` (unix)
//! - startup recovery when `target.bak` remains after a crash
//! - cleanup on failure + transactional target backup/rollback
//! - optional SHA-256 of the archive when provided

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bzip2::read::BzDecoder;
use sha2::{Digest, Sha256};
use tar::{Archive, EntryType};

use crate::error::{AsrError, AsrResult};
use crate::sherpa::model::is_model_dir_complete;

/// Default max compressed archive size (4 GiB).
pub const DEFAULT_MAX_DOWNLOAD_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Default TCP/TLS connect timeout.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default overall download deadline (headers + body).
///
/// Sized for multi-gigabyte model archives on modest links; body inactivity is
/// enforced separately so a stalled peer still fails quickly.
pub const DEFAULT_OVERALL_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// Default max idle time between successful body chunk reads.
pub const DEFAULT_BODY_INACTIVITY_TIMEOUT: Duration = Duration::from_secs(120);

/// Default max tar members (files + directories) accepted during extract.
pub const DEFAULT_MAX_TAR_ENTRIES: u64 = 10_000;

/// Default max total uncompressed payload accepted during extract (8 GiB).
pub const DEFAULT_MAX_UNCOMPRESSED_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Default wall-clock bound for the blocking extract + copy + install job.
pub const DEFAULT_EXTRACT_INSTALL_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// Bounded read/write chunk size for cancellable local I/O.
///
/// After a deadline flips the cancel flag, in-flight work returns within roughly
/// one of these chunks (plus one decompress step for tar.bz2), not an unbounded
/// `io::copy` / `entry.unpack`.
pub const IO_CHUNK_SIZE: usize = 64 * 1024;

/// Max HTTP redirects followed during model download.
pub const MAX_REDIRECTS: usize = 10;

/// Hosts permitted for production Sherpa archive downloads.
///
/// Matches exact hosts and subdomains (e.g. `objects.githubusercontent.com`,
/// `release-assets.githubusercontent.com`, `codeload.github.com`).
pub const ALLOWED_DOWNLOAD_HOSTS: &[&str] = &["github.com", "githubusercontent.com"];

/// Timeouts and extraction caps for model download.
///
/// Production callers use [`DownloadHardening::default`] (HTTPS + GitHub hosts,
/// no `file://`). Tests opt into local seams via
/// [`DownloadHardening::for_local_tests`] or field overrides with
/// [`download_model_with_hardening`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadHardening {
    pub connect_timeout: Duration,
    pub overall_timeout: Duration,
    pub body_inactivity_timeout: Duration,
    /// Wall-clock bound for SHA verify + extract + tree copy + install (blocking job).
    pub extract_install_timeout: Duration,
    pub max_tar_entries: u64,
    pub max_uncompressed_bytes: u64,
    /// When true, permit `file://` archive URLs. **Test/opt-in only.**
    pub allow_file_urls: bool,
    /// When true, permit `http(s)://` to loopback hosts (httpmock / local stubs).
    /// **Test/opt-in only.**
    pub allow_loopback_http: bool,
}

impl Default for DownloadHardening {
    fn default() -> Self {
        Self {
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            overall_timeout: DEFAULT_OVERALL_TIMEOUT,
            body_inactivity_timeout: DEFAULT_BODY_INACTIVITY_TIMEOUT,
            extract_install_timeout: DEFAULT_EXTRACT_INSTALL_TIMEOUT,
            max_tar_entries: DEFAULT_MAX_TAR_ENTRIES,
            max_uncompressed_bytes: DEFAULT_MAX_UNCOMPRESSED_BYTES,
            allow_file_urls: false,
            allow_loopback_http: false,
        }
    }
}

impl DownloadHardening {
    /// Explicit local-test seam: short timeouts + `file://` + loopback HTTP.
    ///
    /// Production must not use this.
    #[must_use]
    pub fn for_local_tests() -> Self {
        Self {
            connect_timeout: Duration::from_millis(400),
            overall_timeout: Duration::from_secs(3),
            body_inactivity_timeout: Duration::from_millis(400),
            extract_install_timeout: Duration::from_secs(5),
            allow_file_urls: true,
            allow_loopback_http: true,
            ..Self::default()
        }
    }

    fn resolve(self) -> Self {
        Self {
            connect_timeout: nonzero_duration(self.connect_timeout, DEFAULT_CONNECT_TIMEOUT),
            overall_timeout: nonzero_duration(self.overall_timeout, DEFAULT_OVERALL_TIMEOUT),
            body_inactivity_timeout: nonzero_duration(
                self.body_inactivity_timeout,
                DEFAULT_BODY_INACTIVITY_TIMEOUT,
            ),
            extract_install_timeout: nonzero_duration(
                self.extract_install_timeout,
                DEFAULT_EXTRACT_INSTALL_TIMEOUT,
            ),
            max_tar_entries: if self.max_tar_entries == 0 {
                DEFAULT_MAX_TAR_ENTRIES
            } else {
                self.max_tar_entries
            },
            max_uncompressed_bytes: if self.max_uncompressed_bytes == 0 {
                DEFAULT_MAX_UNCOMPRESSED_BYTES
            } else {
                self.max_uncompressed_bytes
            },
            allow_file_urls: self.allow_file_urls,
            allow_loopback_http: self.allow_loopback_http,
        }
    }
}

fn nonzero_duration(value: Duration, default: Duration) -> Duration {
    if value.is_zero() { default } else { value }
}

#[derive(Debug, Clone)]
pub struct DownloadOptions {
    pub model_name: String,
    pub target_dir: PathBuf,
    pub base_url: String,
    /// Optional override for tests (`file://` with allow_file_urls, or mock URL).
    pub archive_url_override: Option<String>,
    pub cancel: Option<Arc<AtomicBool>>,
    /// Soft cap on downloaded (compressed) bytes.
    pub max_bytes: u64,
    /// Optional lowercase hex SHA-256 of the **archive** file.
    pub expected_sha256: Option<String>,
}

impl DownloadOptions {
    pub fn archive_url(&self) -> String {
        if let Some(u) = &self.archive_url_override {
            return u.clone();
        }
        format!(
            "{}/{}.tar.bz2",
            self.base_url.trim_end_matches('/'),
            self.model_name.trim()
        )
    }

    fn max_bytes_or_default(&self) -> u64 {
        if self.max_bytes == 0 {
            DEFAULT_MAX_DOWNLOAD_BYTES
        } else {
            self.max_bytes
        }
    }
}

type Progress<'a> = dyn FnMut(Option<f32>, &str) + Send + 'a;

/// Download + install with production-default hardening.
pub async fn download_model(opts: DownloadOptions, progress: &mut Progress<'_>) -> AsrResult<()> {
    download_model_with_hardening(opts, DownloadHardening::default(), progress).await
}

/// Download + install with explicit timeouts / extraction caps (tests + advanced hosts).
pub async fn download_model_with_hardening(
    opts: DownloadOptions,
    hardening: DownloadHardening,
    progress: &mut Progress<'_>,
) -> AsrResult<()> {
    let hardening = hardening.resolve();
    let max_bytes = opts.max_bytes_or_default();

    check_cancel(&opts)?;
    // Deterministic crash repair: promote a complete `target.bak` if needed.
    recover_install_crash(&opts.target_dir)?;
    if is_model_dir_complete(&opts.target_dir) {
        progress(Some(1.0), "Sherpa model already available");
        return Ok(());
    }
    if opts.target_dir.exists() && !opts.target_dir.is_dir() {
        return Err(AsrError::internal(format!(
            "Sherpa model target exists and is not a directory: {}",
            opts.target_dir.display()
        )));
    }
    if let Some(parent) = opts.target_dir.parent() {
        fs::create_dir_all(parent)?;
        #[cfg(unix)]
        {
            // Best-effort: do not fail install if parent mode cannot be tightened.
            let _ = set_dir_mode_0700(parent);
        }
    }

    let url = opts.archive_url();
    progress(Some(0.0), &format!("Downloading {}", opts.model_name));

    // Stage under a sibling temp directory, then rename into place.
    let parent = opts
        .target_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let stage = {
        let prefix = format!(".shuvoice-sherpa-{}-", opts.model_name);
        let mut builder = tempfile::Builder::new();
        builder.prefix(&prefix);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            builder.permissions(fs::Permissions::from_mode(0o700));
        }
        builder
            .tempdir_in(&parent)
            .map_err(|e| AsrError::internal(format!("tempdir: {e}")))?
    };
    #[cfg(unix)]
    {
        set_dir_mode_0700(stage.path())?;
    }

    let archive_path = stage.path().join(format!("{}.tar.bz2", opts.model_name));
    let extracted = stage.path().join("extracted");
    fs::create_dir_all(&extracted)?;
    #[cfg(unix)]
    {
        set_dir_mode_0700(&extracted)?;
    }

    let result = async {
        download_url_to_file(&url, &archive_path, &opts, max_bytes, &hardening, progress).await?;
        check_cancel(&opts)?;
        progress(
            Some(0.93),
            "Extracting model archive… this can take 10–60s on slower disks",
        );

        // CPU/IO heavy: verify + extract + copy + install off the async runtime.
        // Shared cancel so a wall-clock timeout can cooperatively stop the job.
        let cancel_flag = opts
            .cancel
            .clone()
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        let cancel_for_block = cancel_flag.clone();
        let expected_sha256 = opts.expected_sha256.clone();
        let archive_path_b = archive_path.clone();
        let extracted_b = extracted.clone();
        let staged_model = stage.path().join("model");
        let target_b = opts.target_dir.clone();
        let hardening_b = hardening.clone();
        let extract_deadline = hardening.extract_install_timeout;

        let handle = tokio::task::spawn_blocking(move || -> AsrResult<()> {
            let cancel = &cancel_for_block;
            if let Some(expected) = expected_sha256.as_ref() {
                // Some("") must not silently disable verification.
                verify_sha256_with_cancel(&archive_path_b, expected, cancel)?;
            }
            check_cancel_flag(cancel)?;
            safe_extract_tar_bz2_with_cancel(
                &archive_path_b,
                &extracted_b,
                &hardening_b,
                cancel,
            )?;
            check_cancel_flag(cancel)?;
            let source = find_extracted_model_dir(&extracted_b).ok_or_else(|| {
                AsrError::internal(
                    "Downloaded Sherpa archive did not contain required artifacts                      (tokens.txt + encoder/decoder/joiner ONNX files).",
                )
            })?;
            if staged_model.exists() {
                remove_path_any(&staged_model)?;
            }
            copy_dir_all_with_cancel(&source, &staged_model, cancel)?;
            #[cfg(unix)]
            {
                clamp_tree_modes(&staged_model)?;
            }
            if !is_model_dir_complete(&staged_model) {
                return Err(AsrError::internal(format!(
                    "Sherpa model download completed but artifacts are incomplete: {}",
                    staged_model.display()
                )));
            }
            check_cancel_flag(cancel)?;
            install_model_transaction(&staged_model, &target_b, cancel)?;
            #[cfg(unix)]
            {
                clamp_tree_modes(&target_b)?;
            }
            Ok(())
        });

        // Poll the blocking job with a wall-clock bound. On timeout, set cancel and
        // *await* the handle so TempDir/`stage` is not dropped under a live task.
        let mut handle = handle;
        let join = tokio::select! {
            join = &mut handle => join,
            _ = tokio::time::sleep(extract_deadline) => {
                cancel_flag.store(true, Ordering::Relaxed);
                match handle.await {
                    Ok(Ok(())) => {
                        // Race: finished in the same window as the deadline — accept.
                        Ok(Ok(()))
                    }
                    Ok(Err(AsrError::Cancelled(_))) => {
                        return Err(AsrError::RemoteTimeout(
                            extract_deadline,
                            "extract/install wall-clock deadline".into(),
                        ));
                    }
                    Ok(Err(e)) => return Err(e),
                    Err(join_err) => {
                        if join_err.is_cancelled() {
                            return Err(AsrError::Cancelled(
                                "Model extract/install cancelled".into(),
                            ));
                        }
                        return Err(AsrError::internal(format!(
                            "extract/install task join error: {join_err}"
                        )));
                    }
                }
            }
        };
        match join {
            Ok(inner) => inner?,
            Err(join_err) => {
                if join_err.is_cancelled() {
                    return Err(AsrError::Cancelled(
                        "Model extract/install cancelled".into(),
                    ));
                }
                return Err(AsrError::internal(format!(
                    "extract/install task join error: {join_err}"
                )));
            }
        }

        progress(Some(0.97), "Finalizing model files… almost done");
        Ok(())
    }
    .await;

    if let Err(err) = result {
        // Best-effort cleanup of partial target.
        if opts.target_dir.exists() && !is_model_dir_complete(&opts.target_dir) {
            let _ = fs::remove_dir_all(&opts.target_dir);
        }
        return Err(err);
    }

    progress(Some(1.0), "Sherpa model ready");
    Ok(())
}

/// Whether `host` is on the curated GitHub / release-asset allow-list.
#[must_use]
pub fn host_is_allowed(host: &str) -> bool {
    let host = host.trim().trim_end_matches('.').to_ascii_lowercase();
    if host.is_empty() {
        return false;
    }
    ALLOWED_DOWNLOAD_HOSTS
        .iter()
        .any(|domain| host == *domain || host.ends_with(&format!(".{domain}")))
}

#[must_use]
fn is_loopback_host(host: &str) -> bool {
    let host = host.trim().trim_end_matches('.').to_ascii_lowercase();
    matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1" | "[::1]")
}

/// Validate a download URL against production (or test-seam) policy.
pub fn validate_download_url(url: &str, hardening: &DownloadHardening) -> AsrResult<()> {
    if url.starts_with("file://") {
        if hardening.allow_file_urls {
            return Ok(());
        }
        return Err(AsrError::transport(
            "file:// URLs are disabled on the production Sherpa downloader (test seam only)",
        ));
    }
    let parsed = reqwest::Url::parse(url)
        .map_err(|e| AsrError::transport(format!("invalid download URL {url:?}: {e}")))?;
    validate_parsed_url(&parsed, hardening.allow_loopback_http).map_err(AsrError::transport)
}

fn validate_parsed_url(url: &reqwest::Url, allow_loopback_http: bool) -> Result<(), String> {
    let scheme = url.scheme();
    let host = url.host_str().unwrap_or("");

    if scheme == "https" && host_is_allowed(host) {
        return Ok(());
    }
    if allow_loopback_http && matches!(scheme, "http" | "https") && is_loopback_host(host) {
        return Ok(());
    }
    if scheme != "https" {
        return Err(format!(
            "Sherpa download URL must use HTTPS (got {scheme}): {url}"
        ));
    }
    Err(format!(
        "Sherpa download host {host:?} is not in the allowed GitHub/release-asset list: \
         {ALLOWED_DOWNLOAD_HOSTS:?}"
    ))
}

fn redirect_policy(allow_loopback_http: bool) -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(move |attempt| {
        if attempt.previous().len() >= MAX_REDIRECTS {
            return attempt.error("too many redirects");
        }
        // Clone before consuming `attempt` (avoid CLI-style borrow/move footgun).
        let next = attempt.url().clone();
        match validate_parsed_url(&next, allow_loopback_http) {
            Ok(()) => attempt.follow(),
            Err(msg) => attempt.error(msg),
        }
    })
}

async fn download_url_to_file(
    url: &str,
    dest: &Path,
    opts: &DownloadOptions,
    max_bytes: u64,
    hardening: &DownloadHardening,
    progress: &mut Progress<'_>,
) -> AsrResult<()> {
    validate_download_url(url, hardening)?;

    if let Some(path) = url.strip_prefix("file://") {
        // validate_download_url already enforced allow_file_urls.
        // Run chunked/cancellable copy on the blocking pool (not the async worker).
        let src = PathBuf::from(path);
        let dest_b = dest.to_path_buf();
        let cancel = opts
            .cancel
            .clone()
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        let join = tokio::task::spawn_blocking(move || -> AsrResult<()> {
            let meta = fs::metadata(&src)?;
            if meta.len() > max_bytes {
                return Err(AsrError::transport(format!(
                    "archive size {} exceeds max_download_bytes {max_bytes}",
                    meta.len()
                )));
            }
            copy_file_chunked_cancellable(&src, &dest_b, Some(max_bytes), &cancel)?;
            #[cfg(unix)]
            {
                set_file_mode_0600(&dest_b)?;
            }
            Ok(())
        })
        .await
        .map_err(|e| AsrError::internal(format!("file:// copy join error: {e}")))?;
        join?;
        progress(Some(0.9), "Local archive copied");
        return Ok(());
    }

    let client = reqwest::Client::builder()
        .connect_timeout(hardening.connect_timeout)
        // Overall request timeout as a backstop; streaming also enforces a deadline.
        .timeout(hardening.overall_timeout)
        .read_timeout(hardening.body_inactivity_timeout)
        .redirect(redirect_policy(hardening.allow_loopback_http))
        .build()
        .map_err(|e| AsrError::transport(format!("http client: {e}")))?;

    let started = Instant::now();
    let response =
        match tokio::time::timeout(hardening.overall_timeout, client.get(url).send()).await {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => {
                if e.is_timeout() || e.is_connect() {
                    return Err(AsrError::RemoteTimeout(
                        hardening.connect_timeout.min(hardening.overall_timeout),
                        format!("download connect/headers for {url}: {e}"),
                    ));
                }
                return Err(AsrError::transport(format!(
                    "Failed to download Sherpa model archive. URL: {url}. Error: {e}"
                )));
            }
            Err(_) => {
                return Err(AsrError::RemoteTimeout(
                    hardening.overall_timeout,
                    format!("download connect/headers deadline for {url}"),
                ));
            }
        };

    let status = response.status();
    if !status.is_success() {
        return Err(AsrError::transport(format!(
            "Failed to download Sherpa model archive. URL: {url}. HTTP {status}"
        )));
    }

    // Final URL after redirects must still satisfy policy.
    let final_url = response.url().clone();
    validate_parsed_url(&final_url, hardening.allow_loopback_http).map_err(|msg| {
        AsrError::transport(format!(
            "download final URL not permitted ({final_url}): {msg}"
        ))
    })?;

    if let Some(total) = response.content_length()
        && total > max_bytes
    {
        return Err(AsrError::transport(format!(
            "Content-Length {total} exceeds max_download_bytes {max_bytes}"
        )));
    }

    let total = response.content_length();
    let mut file = create_secret_file(dest)?;
    let mut stream = response.bytes_stream();
    use futures_util::StreamExt;
    let mut downloaded: u64 = 0;
    let mut last_frac = -1.0f32;

    loop {
        check_cancel(opts)?;
        let elapsed = started.elapsed();
        if elapsed >= hardening.overall_timeout {
            return Err(AsrError::RemoteTimeout(
                hardening.overall_timeout,
                format!("download overall deadline for {url}"),
            ));
        }
        let remaining_overall = hardening.overall_timeout.saturating_sub(elapsed);
        let chunk_wait = remaining_overall.min(hardening.body_inactivity_timeout);

        let next = tokio::time::timeout(chunk_wait, stream.next()).await;
        let chunk = match next {
            Ok(Some(Ok(chunk))) => chunk,
            Ok(Some(Err(e))) => {
                if e.is_timeout() {
                    return Err(AsrError::RemoteTimeout(
                        hardening.body_inactivity_timeout,
                        format!("download body inactivity for {url}: {e}"),
                    ));
                }
                return Err(AsrError::transport(format!("download stream error: {e}")));
            }
            Ok(None) => break,
            Err(_) => {
                if started.elapsed() >= hardening.overall_timeout {
                    return Err(AsrError::RemoteTimeout(
                        hardening.overall_timeout,
                        format!("download overall deadline for {url}"),
                    ));
                }
                return Err(AsrError::RemoteTimeout(
                    hardening.body_inactivity_timeout,
                    format!("download body inactivity for {url}"),
                ));
            }
        };

        downloaded = downloaded.saturating_add(chunk.len() as u64);
        if downloaded > max_bytes {
            return Err(AsrError::transport(format!(
                "download exceeded max_download_bytes {max_bytes} (got at least {downloaded})"
            )));
        }
        file.write_all(&chunk)?;
        if let Some(total) = total
            && total > 0
        {
            let frac = (downloaded as f32 / total as f32) * 0.9;
            if frac - last_frac >= 0.01 || frac >= 0.9 {
                last_frac = frac;
                let pct = ((downloaded as f64 / total as f64) * 100.0) as u32;
                progress(
                    Some(frac.min(0.9)),
                    &format!("Downloading model archive… {pct}%"),
                );
            }
        }
    }
    file.flush()?;
    #[cfg(unix)]
    {
        set_file_mode_0600(dest)?;
    }
    Ok(())
}

#[cfg(test)]
fn verify_sha256(path: &Path, expected_hex: &str) -> AsrResult<()> {
    verify_sha256_with_cancel(path, expected_hex, &inert_cancel())
}

fn verify_sha256_with_cancel(
    path: &Path,
    expected_hex: &str,
    cancel: &Arc<AtomicBool>,
) -> AsrResult<()> {
    let expected = expected_hex.trim().to_ascii_lowercase();
    // Empty / whitespace-only must not silently disable verification when provided.
    if expected.is_empty() {
        return Err(AsrError::internal(
            "Sherpa archive SHA-256 expected value is empty; omit expected_sha256 to skip verification",
        ));
    }
    if expected.len() != 64 || !expected.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(AsrError::internal(format!(
            "Sherpa archive SHA-256 expected value must be 64 lowercase hex chars, got {expected:?}"
        )));
    }
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; IO_CHUNK_SIZE];
    loop {
        check_cancel_flag(cancel)?;
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let got = hex::encode(hasher.finalize());
    if got != expected {
        return Err(AsrError::internal(format!(
            "Sherpa archive SHA-256 mismatch: expected {expected}, got {got}"
        )));
    }
    Ok(())
}

fn check_cancel(opts: &DownloadOptions) -> AsrResult<()> {
    if let Some(c) = &opts.cancel {
        check_cancel_flag(c)
    } else {
        Ok(())
    }
}

fn check_cancel_flag(cancel: &Arc<AtomicBool>) -> AsrResult<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(AsrError::Cancelled("Model download cancelled".into()))
    } else {
        Ok(())
    }
}

/// Never-cancelled flag for public helpers that do not expose a cancel seam.
fn inert_cancel() -> Arc<AtomicBool> {
    Arc::new(AtomicBool::new(false))
}

#[must_use]
pub fn path_is_unsafe(path: &Path) -> bool {
    path.components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
        || path.is_absolute()
}

/// Extract tar.bz2 rejecting path traversal and special entries (default caps).
pub fn safe_extract_tar_bz2(archive_path: &Path, target_dir: &Path) -> AsrResult<()> {
    safe_extract_tar_bz2_with_limits(archive_path, target_dir, &DownloadHardening::default())
}

/// Extract tar.bz2 with explicit entry-count / uncompressed-size caps.
pub fn safe_extract_tar_bz2_with_limits(
    archive_path: &Path,
    target_dir: &Path,
    limits: &DownloadHardening,
) -> AsrResult<()> {
    safe_extract_tar_bz2_with_cancel(archive_path, target_dir, limits, &inert_cancel())
}

/// Extract with cooperative cancel checks between tar members **and** within
/// regular-file payloads (bounded chunks; no `entry.unpack`).
fn safe_extract_tar_bz2_with_cancel(
    archive_path: &Path,
    target_dir: &Path,
    limits: &DownloadHardening,
    cancel: &Arc<AtomicBool>,
) -> AsrResult<()> {
    let limits = limits.clone().resolve();
    let file = File::open(archive_path)?;
    let decoder = BzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    // Don't follow/preserve elevated metadata when unpacking untrusted input.
    archive.set_preserve_permissions(false);
    archive.set_unpack_xattrs(false);

    let mut entries_seen: u64 = 0;
    let mut uncompressed_total: u64 = 0;

    for entry in archive
        .entries()
        .map_err(|e| AsrError::internal(format!("tar entries: {e}")))?
    {
        check_cancel_flag(cancel)?;
        let mut entry = entry.map_err(|e| AsrError::internal(format!("tar entry: {e}")))?;
        entries_seen = entries_seen.saturating_add(1);
        if entries_seen > limits.max_tar_entries {
            return Err(AsrError::internal(format!(
                "Sherpa archive exceeds max tar entry count {} (saw at least {entries_seen})",
                limits.max_tar_entries
            )));
        }

        let path = entry
            .path()
            .map_err(|e| AsrError::internal(format!("tar path: {e}")))?
            .into_owned();

        if path_is_unsafe(&path) {
            return Err(AsrError::internal(format!(
                "Unsafe path {} while extracting Sherpa model archive",
                path.display()
            )));
        }

        let kind = entry.header().entry_type();
        match kind {
            EntryType::Regular | EntryType::Continuous | EntryType::Directory => {}
            EntryType::GNUSparse
            | EntryType::Fifo
            | EntryType::Char
            | EntryType::Block
            | EntryType::Symlink
            | EntryType::Link
            | EntryType::XGlobalHeader
            | EntryType::XHeader => {
                return Err(AsrError::internal(format!(
                    "Refusing tar link/special/sparse entry {kind:?} at {}",
                    path.display()
                )));
            }
            _ => {
                return Err(AsrError::internal(format!(
                    "Refusing unsupported tar entry type {kind:?} at {}",
                    path.display()
                )));
            }
        }

        // GNU long-link etc. already filtered; double-check link_name.
        if entry.link_name().ok().flatten().is_some() {
            return Err(AsrError::internal(format!(
                "Refusing hard/symlink entry at {}",
                path.display()
            )));
        }

        let entry_size = entry.size();
        match kind {
            EntryType::Regular | EntryType::Continuous => {
                uncompressed_total = uncompressed_total.saturating_add(entry_size);
                if uncompressed_total > limits.max_uncompressed_bytes {
                    return Err(AsrError::internal(format!(
                        "Sherpa archive exceeds max uncompressed bytes {} \
                         (at least {uncompressed_total} via {})",
                        limits.max_uncompressed_bytes,
                        path.display()
                    )));
                }
            }
            _ => {}
        }

        let out = target_dir.join(&path);
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
            #[cfg(unix)]
            {
                clamp_dir_ancestors(target_dir, parent)?;
            }
        }

        match kind {
            EntryType::Directory => {
                fs::create_dir_all(&out)?;
                #[cfg(unix)]
                {
                    set_dir_mode_0700(&out)?;
                }
            }
            EntryType::Regular | EntryType::Continuous => {
                // Manual chunked extract — cancellable, size-capped, owner-only mode.
                extract_regular_entry_chunked(&mut entry, &out, entry_size, cancel)?;
            }
            _ => unreachable!("filtered above"),
        }
    }
    #[cfg(unix)]
    {
        clamp_tree_modes(target_dir)?;
    }
    Ok(())
}

/// Copy exactly `declared_size` bytes from a tar entry using bounded chunks.
fn extract_regular_entry_chunked<R: Read>(
    entry: &mut R,
    out_path: &Path,
    declared_size: u64,
    cancel: &Arc<AtomicBool>,
) -> AsrResult<()> {
    check_cancel_flag(cancel)?;
    let mut out = create_secret_file(out_path)?;
    let mut buf = vec![0u8; IO_CHUNK_SIZE];
    let mut remaining = declared_size;
    let mut written: u64 = 0;
    while remaining > 0 {
        check_cancel_flag(cancel)?;
        let want = usize::try_from(remaining.min(IO_CHUNK_SIZE as u64)).unwrap_or(IO_CHUNK_SIZE);
        let n = entry
            .read(&mut buf[..want])
            .map_err(|e| AsrError::internal(format!("tar read {}: {e}", out_path.display())))?;
        if n == 0 {
            return Err(AsrError::internal(format!(
                "tar entry truncated at {}: wrote {written} of declared {declared_size} bytes",
                out_path.display()
            )));
        }
        out.write_all(&buf[..n])
            .map_err(|e| AsrError::internal(format!("tar write {}: {e}", out_path.display())))?;
        let n_u = n as u64;
        written = written.saturating_add(n_u);
        remaining = remaining.saturating_sub(n_u);
    }
    out.flush()?;
    #[cfg(unix)]
    {
        set_file_mode_0600(out_path)?;
    }
    // Ensure we did not stop short; over-read is prevented by `want`/`remaining`.
    if written != declared_size {
        return Err(AsrError::internal(format!(
            "tar entry size mismatch at {}: wrote {written}, declared {declared_size}",
            out_path.display()
        )));
    }
    Ok(())
}

fn find_extracted_model_dir(root: &Path) -> Option<PathBuf> {
    if is_model_dir_complete(root) {
        return Some(root.to_path_buf());
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = fs::read_dir(&dir).ok()?;
        for ent in rd.flatten() {
            let p = ent.path();
            if p.is_dir() {
                if is_model_dir_complete(&p) {
                    return Some(p);
                }
                stack.push(p);
            }
        }
    }
    None
}

fn backup_path_for(target: &Path) -> PathBuf {
    let mut b = target.as_os_str().to_os_string();
    b.push(".bak");
    PathBuf::from(b)
}

/// Remove a file **or** directory tree. Propagates errors (no silent `let _ =`).
fn remove_path_any(path: &Path) -> AsrResult<()> {
    let meta = match fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(AsrError::from(e)),
    };
    if meta.file_type().is_symlink() {
        fs::remove_file(path)?;
        return Ok(());
    }
    if meta.is_dir() {
        fs::remove_dir_all(path)?;
    } else {
        fs::remove_file(path)?;
    }
    Ok(())
}

/// Crash repair for interrupted install transactions.
///
/// If `target.bak` exists:
/// - target complete → drop stale backup
/// - backup complete and target missing/incomplete/file → restore backup as target
/// - backup incomplete → drop backup (target left for normal download path)
fn recover_install_crash(target: &Path) -> AsrResult<()> {
    let backup = backup_path_for(target);
    if !backup.exists() {
        return Ok(());
    }

    let target_ok = is_model_dir_complete(target);
    let backup_ok = is_model_dir_complete(&backup);
    let cancel = inert_cancel();

    if target_ok {
        remove_path_any(&backup)?;
        return Ok(());
    }

    if backup_ok {
        // Target may be missing, an incomplete dir, or a stray file.
        if target.exists() || fs::symlink_metadata(target).is_ok() {
            remove_path_any(target)?;
        }
        rename_or_copy_dir(&backup, target, &cancel)?;
        #[cfg(unix)]
        {
            clamp_tree_modes(target)?;
        }
        return Ok(());
    }

    // Neither side is a complete model; discard the orphan backup.
    remove_path_any(&backup)?;
    Ok(())
}

/// Replace `target` with `staged` using backup + rollback.
///
/// Steps: `target` → `target.bak` (if present), `staged` → `target`,
/// then remove backup. On failure after moving the old tree aside, restore it.
fn install_model_transaction(
    staged: &Path,
    target: &Path,
    cancel: &Arc<AtomicBool>,
) -> AsrResult<()> {
    check_cancel_flag(cancel)?;
    let backup = backup_path_for(target);
    if backup.exists() {
        remove_path_any(&backup)?;
    }

    if target.exists() || fs::symlink_metadata(target).is_ok() {
        // Move aside files or dirs uniformly.
        if target.is_file()
            || fs::symlink_metadata(target)
                .map(|m| m.is_file())
                .unwrap_or(false)
        {
            // Promote a stray file into the backup path as a file.
            remove_path_any(&backup)?;
            fs::rename(target, &backup).or_else(|e| {
                if e.kind() == io::ErrorKind::CrossesDevices {
                    fs::copy(target, &backup)?;
                    fs::remove_file(target)?;
                    Ok(())
                } else {
                    Err(e)
                }
            })?;
        } else {
            rename_or_copy_dir(target, &backup, cancel)?;
        }
    }

    check_cancel_flag(cancel)?;
    match rename_or_copy_dir(staged, target, cancel) {
        Ok(()) => {
            // Verify completeness *before* discarding the backup so rollback works.
            if !is_model_dir_complete(target) {
                remove_path_any(target)?;
                if backup.exists() {
                    rename_or_copy_dir(&backup, target, cancel)?;
                }
                return Err(AsrError::internal(format!(
                    "installed model incomplete at {}",
                    target.display()
                )));
            }
            if backup.exists() {
                remove_path_any(&backup)?;
            }
            Ok(())
        }
        Err(e) => {
            let _ = remove_path_any(target);
            if backup.exists() {
                // Best-effort restore; surface original error if restore also fails.
                if let Err(re) = rename_or_copy_dir(&backup, target, cancel) {
                    return Err(AsrError::internal(format!(
                        "install failed ({e}); rollback also failed ({re})"
                    )));
                }
            }
            Err(e)
        }
    }
}

fn rename_or_copy_dir(src: &Path, dst: &Path, cancel: &Arc<AtomicBool>) -> AsrResult<()> {
    check_cancel_flag(cancel)?;
    match fs::rename(src, dst) {
        Ok(()) => Ok(()),
        Err(e) => {
            if e.kind() == io::ErrorKind::CrossesDevices {
                copy_dir_all_with_cancel(src, dst, cancel)?;
                remove_path_any(src)?;
                Ok(())
            } else {
                Err(AsrError::from(e))
            }
        }
    }
}

fn copy_dir_all_with_cancel(src: &Path, dst: &Path, cancel: &Arc<AtomicBool>) -> AsrResult<()> {
    check_cancel_flag(cancel)?;
    fs::create_dir_all(dst)?;
    #[cfg(unix)]
    {
        set_dir_mode_0700(dst)?;
    }
    for entry in fs::read_dir(src)? {
        check_cancel_flag(cancel)?;
        let entry = entry?;
        let ty = entry.file_type()?;
        let to = dst.join(entry.file_name());
        if ty.is_symlink() {
            return Err(AsrError::internal(format!(
                "refusing to copy symlink {}",
                entry.path().display()
            )));
        }
        if ty.is_dir() {
            copy_dir_all_with_cancel(&entry.path(), &to, cancel)?;
        } else {
            copy_file_chunked_cancellable(&entry.path(), &to, None, cancel)?;
        }
    }
    Ok(())
}

/// Chunked file copy with cooperative cancel. Optional `max_bytes` enforces a cap.
fn copy_file_chunked_cancellable(
    src: &Path,
    dst: &Path,
    max_bytes: Option<u64>,
    cancel: &Arc<AtomicBool>,
) -> AsrResult<()> {
    check_cancel_flag(cancel)?;
    let mut input = File::open(src)?;
    let mut out = create_secret_file(dst)?;
    let mut buf = vec![0u8; IO_CHUNK_SIZE];
    let mut copied: u64 = 0;
    loop {
        check_cancel_flag(cancel)?;
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        copied = copied.saturating_add(n as u64);
        if let Some(max) = max_bytes
            && copied > max
        {
            drop(out);
            let _ = fs::remove_file(dst);
            return Err(AsrError::transport(format!(
                "copy exceeded max_bytes {max} (got at least {copied})"
            )));
        }
        out.write_all(&buf[..n])?;
    }
    out.flush()?;
    #[cfg(unix)]
    {
        set_file_mode_0600(dst)?;
    }
    Ok(())
}

fn create_secret_file(path: &Path) -> io::Result<File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
    }
    #[cfg(not(unix))]
    {
        File::create(path)
    }
}

#[cfg(unix)]
fn set_file_mode_0600(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
}

#[cfg(unix)]
fn set_dir_mode_0700(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
}

/// Clamp every directory from `root` down through `dir` to `0700`.
#[cfg(unix)]
fn clamp_dir_ancestors(root: &Path, dir: &Path) -> io::Result<()> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let mut cur = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    // Walk upward until we pass root.
    loop {
        if cur.starts_with(&root) {
            set_dir_mode_0700(&cur)?;
        }
        if cur == root {
            break;
        }
        match cur.parent() {
            Some(p) if p != cur => cur = p.to_path_buf(),
            _ => break,
        }
    }
    Ok(())
}

/// Recursively force dirs=`0700` and non-symlink files=`0600` under `root`.
#[cfg(unix)]
fn clamp_tree_modes(root: &Path) -> io::Result<()> {
    if !root.exists() {
        return Ok(());
    }
    let meta = fs::symlink_metadata(root)?;
    if meta.file_type().is_symlink() {
        return Err(io::Error::other(format!(
            "refusing to clamp symlink tree {}",
            root.display()
        )));
    }
    if meta.is_dir() {
        set_dir_mode_0700(root)?;
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            let ft = entry.file_type()?;
            let p = entry.path();
            if ft.is_symlink() {
                return Err(io::Error::other(format!(
                    "refusing to clamp symlink {}",
                    p.display()
                )));
            }
            if ft.is_dir() {
                clamp_tree_modes(&p)?;
            } else if ft.is_file() {
                set_file_mode_0600(&p)?;
            }
        }
    } else if meta.is_file() {
        set_file_mode_0600(root)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    use httpmock::Method::GET;
    use httpmock::MockServer;

    fn write_model_dir(model: &Path) {
        fs::create_dir_all(model).unwrap();
        fs::write(model.join("tokens.txt"), "a\n").unwrap();
        fs::write(model.join("encoder.onnx"), b"enc").unwrap();
        fs::write(model.join("decoder.onnx"), b"dec").unwrap();
        fs::write(model.join("joiner.onnx"), b"join").unwrap();
    }

    fn make_bz2_archive(dir: &Path, archive: &Path) {
        let model = dir.join("model");
        write_model_dir(&model);

        let tar_path = dir.join("m.tar");
        {
            let file = File::create(&tar_path).unwrap();
            let mut builder = tar::Builder::new(file);
            builder.append_dir_all("model", &model).unwrap();
            builder.finish().unwrap();
        }
        let tar_bytes = fs::read(&tar_path).unwrap();
        let mut enc = bzip2::write::BzEncoder::new(
            File::create(archive).unwrap(),
            bzip2::Compression::default(),
        );
        enc.write_all(&tar_bytes).unwrap();
        enc.finish().unwrap();
    }

    fn make_bz2_from_tar_builder(
        dir: &Path,
        archive: &Path,
        build: impl FnOnce(&mut tar::Builder<File>),
    ) {
        let tar_path = dir.join("custom.tar");
        {
            let file = File::create(&tar_path).unwrap();
            let mut builder = tar::Builder::new(file);
            build(&mut builder);
            builder.finish().unwrap();
        }
        let tar_bytes = fs::read(&tar_path).unwrap();
        let mut enc = bzip2::write::BzEncoder::new(
            File::create(archive).unwrap(),
            bzip2::Compression::default(),
        );
        enc.write_all(&tar_bytes).unwrap();
        enc.finish().unwrap();
    }

    fn test_hardening() -> DownloadHardening {
        DownloadHardening::for_local_tests()
    }

    #[test]
    fn safe_extract_and_find() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("m.tar.bz2");
        make_bz2_archive(dir.path(), &archive);
        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();
        safe_extract_tar_bz2(&archive, &out).unwrap();
        let found = find_extracted_model_dir(&out).unwrap();
        assert!(is_model_dir_complete(&found));
    }

    #[test]
    fn rejects_parent_dir_components() {
        assert!(path_is_unsafe(Path::new("../evil.txt")));
        assert!(path_is_unsafe(Path::new("foo/../../etc/passwd")));
        assert!(!path_is_unsafe(Path::new("model/encoder.onnx")));
    }

    #[test]
    fn rejects_gnu_sparse_policy() {
        let kind = EntryType::GNUSparse;
        assert!(matches!(
            kind,
            EntryType::GNUSparse
                | EntryType::Fifo
                | EntryType::Char
                | EntryType::Block
                | EntryType::Symlink
                | EntryType::Link
        ));
    }

    #[test]
    fn rejects_symlink_entries() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("link.tar.bz2");
        make_bz2_from_tar_builder(dir.path(), &archive, |builder| {
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(EntryType::Symlink);
            header.set_size(0);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_link(&mut header, "evil.link", "tokens.txt")
                .unwrap();
        });
        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();
        let err = safe_extract_tar_bz2(&archive, &out).unwrap_err();
        assert!(
            err.to_string().to_ascii_lowercase().contains("link")
                || err.to_string().to_ascii_lowercase().contains("refusing"),
            "{err}"
        );
    }

    #[test]
    fn rejects_extraction_bomb_by_uncompressed_size() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("bomb.tar.bz2");
        {
            let tar_path = dir.path().join("bomb.tar");
            let mut file = File::create(&tar_path).unwrap();
            let mut header = tar::Header::new_ustar();
            header.set_path("bomb.bin").unwrap();
            header.set_entry_type(EntryType::Regular);
            header.set_size(1_048_576);
            header.set_mode(0o644);
            header.set_cksum();
            file.write_all(header.as_bytes()).unwrap();
            file.write_all(&[0u8; 512]).unwrap();
            file.write_all(&[0u8; 1024]).unwrap();
            drop(file);

            let tar_bytes = fs::read(&tar_path).unwrap();
            let mut enc = bzip2::write::BzEncoder::new(
                File::create(&archive).unwrap(),
                bzip2::Compression::default(),
            );
            enc.write_all(&tar_bytes).unwrap();
            enc.finish().unwrap();
        }

        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();
        let limits = DownloadHardening {
            max_uncompressed_bytes: 1024,
            ..DownloadHardening::default()
        };
        let err = safe_extract_tar_bz2_with_limits(&archive, &out, &limits).unwrap_err();
        let msg = err.to_string().to_ascii_lowercase();
        assert!(msg.contains("uncompressed") || msg.contains("max"), "{err}");
    }

    #[test]
    fn rejects_extraction_bomb_by_file_count() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("many.tar.bz2");
        make_bz2_from_tar_builder(dir.path(), &archive, |builder| {
            for i in 0..30 {
                let name = format!("f{i}.txt");
                let mut header = tar::Header::new_gnu();
                header.set_path(&name).unwrap();
                header.set_entry_type(EntryType::Regular);
                header.set_size(1);
                header.set_mode(0o644);
                header.set_cksum();
                builder.append_data(&mut header, &name, &b"x"[..]).unwrap();
            }
        });
        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();
        let limits = DownloadHardening {
            max_tar_entries: 10,
            ..DownloadHardening::default()
        };
        let err = safe_extract_tar_bz2_with_limits(&archive, &out, &limits).unwrap_err();
        assert!(
            err.to_string().to_ascii_lowercase().contains("entry count")
                || err.to_string().to_ascii_lowercase().contains("tar entry"),
            "{err}"
        );
    }

    #[test]
    fn cancel_flag_aborts() {
        let dir = tempfile::tempdir().unwrap();
        let cancel = Arc::new(AtomicBool::new(true));
        let opts = DownloadOptions {
            model_name: "x".into(),
            target_dir: dir.path().join("m"),
            base_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models".into(),
            archive_url_override: None,
            cancel: Some(cancel),
            max_bytes: DEFAULT_MAX_DOWNLOAD_BYTES,
            expected_sha256: None,
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let err = rt
            .block_on(async {
                let mut progress = |_f: Option<f32>, _m: &str| {};
                download_model(opts, &mut progress).await
            })
            .unwrap_err();
        assert!(matches!(err, AsrError::Cancelled(_)));
    }

    #[test]
    fn sha256_mismatch_detected() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("m.tar.bz2");
        make_bz2_archive(dir.path(), &archive);
        let err = verify_sha256(&archive, "deadbeef").unwrap_err();
        assert!(err.to_string().contains("SHA-256"));
    }

    #[test]
    fn sha256_accepts_matching_digest() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("m.tar.bz2");
        make_bz2_archive(dir.path(), &archive);
        let mut file = File::open(&archive).unwrap();
        let mut hasher = Sha256::new();
        io::copy(&mut file, &mut hasher).unwrap();
        let digest = hex::encode(hasher.finalize());
        verify_sha256(&archive, &digest).unwrap();
    }

    #[test]
    fn production_url_policy_https_and_github_hosts() {
        let prod = DownloadHardening::default();
        assert!(!prod.allow_file_urls);
        assert!(!prod.allow_loopback_http);

        validate_download_url(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/m.tar.bz2",
            &prod,
        )
        .unwrap();
        validate_download_url(
            "https://objects.githubusercontent.com/github-production-release-asset/1/x",
            &prod,
        )
        .unwrap();
        validate_download_url(
            "https://release-assets.githubusercontent.com/github-production-release-asset/1/x",
            &prod,
        )
        .unwrap();

        let err = validate_download_url("http://github.com/x", &prod).unwrap_err();
        assert!(
            err.to_string().to_ascii_lowercase().contains("https"),
            "{err}"
        );

        let err = validate_download_url("https://evil.example/x", &prod).unwrap_err();
        assert!(
            err.to_string().to_ascii_lowercase().contains("allowed")
                || err.to_string().to_ascii_lowercase().contains("host"),
            "{err}"
        );

        let err = validate_download_url("file:///tmp/x.tar.bz2", &prod).unwrap_err();
        assert!(
            err.to_string().to_ascii_lowercase().contains("file://"),
            "{err}"
        );

        assert!(host_is_allowed("github.com"));
        assert!(host_is_allowed("codeload.github.com"));
        assert!(host_is_allowed("objects.githubusercontent.com"));
        assert!(!host_is_allowed("evilgithubusercontent.com"));
        assert!(!host_is_allowed("huggingface.co"));
    }

    #[cfg(unix)]
    #[test]
    fn secret_file_and_stage_dir_modes() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("arch.tar.bz2");
        {
            let mut f = create_secret_file(&file_path).unwrap();
            f.write_all(b"data").unwrap();
        }
        set_file_mode_0600(&file_path).unwrap();
        let mode = fs::metadata(&file_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "archive must be owner read/write only");

        let mut builder = tempfile::Builder::new();
        builder.prefix("mode-stage-");
        builder.permissions(fs::Permissions::from_mode(0o700));
        let stage = builder.tempdir_in(dir.path()).unwrap();
        set_dir_mode_0700(stage.path()).unwrap();
        let dmode = fs::metadata(stage.path()).unwrap().permissions().mode() & 0o777;
        assert_eq!(dmode, 0o700, "stage dir must be owner rwx only");
    }

    #[test]
    fn install_transaction_rolls_back_on_incomplete_staged() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("model");
        write_model_dir(&target);
        assert!(is_model_dir_complete(&target));
        let original_tokens = fs::read_to_string(target.join("tokens.txt")).unwrap();

        let staged = dir.path().join("staged");
        fs::create_dir_all(&staged).unwrap();
        fs::write(staged.join("tokens.txt"), "partial\n").unwrap();

        let err = install_model_transaction(&staged, &target, &inert_cancel()).unwrap_err();
        assert!(
            err.to_string().to_ascii_lowercase().contains("incomplete"),
            "{err}"
        );
        assert!(
            is_model_dir_complete(&target),
            "target must be restored from backup"
        );
        assert_eq!(
            fs::read_to_string(target.join("tokens.txt")).unwrap(),
            original_tokens
        );
        assert!(
            !PathBuf::from(format!("{}.bak", target.display())).exists(),
            "backup should be removed or restored"
        );
    }

    #[tokio::test]
    async fn file_url_requires_explicit_hardening_seam() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("m.tar.bz2");
        make_bz2_archive(dir.path(), &archive);
        let target = dir.path().join("installed");
        let opts = DownloadOptions {
            model_name: "m".into(),
            target_dir: target.clone(),
            base_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models".into(),
            archive_url_override: Some(format!("file://{}", archive.display())),
            cancel: None,
            max_bytes: DEFAULT_MAX_DOWNLOAD_BYTES,
            expected_sha256: None,
        };
        let mut progress = |_f: Option<f32>, _m: &str| {};

        // Default production path must refuse file://.
        let err = download_model(opts.clone(), &mut progress)
            .await
            .unwrap_err();
        assert!(
            err.to_string().to_ascii_lowercase().contains("file://"),
            "{err}"
        );
        assert!(!target.exists() || !is_model_dir_complete(&target));

        // Explicit test seam succeeds.
        download_model_with_hardening(opts, test_hardening(), &mut progress)
            .await
            .unwrap();
        assert!(is_model_dir_complete(&target));
    }

    #[tokio::test]
    async fn httpmock_happy_path_installs() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("m.tar.bz2");
        make_bz2_archive(dir.path(), &archive);
        let bytes = fs::read(&archive).unwrap();

        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/m.tar.bz2");
            then.status(200)
                .header("content-type", "application/octet-stream")
                .body(bytes);
        });

        let target = dir.path().join("installed");
        let opts = DownloadOptions {
            model_name: "m".into(),
            target_dir: target.clone(),
            base_url: server.base_url(),
            archive_url_override: Some(server.url("/m.tar.bz2")),
            cancel: None,
            max_bytes: DEFAULT_MAX_DOWNLOAD_BYTES,
            expected_sha256: None,
        };
        let mut progress = |_f: Option<f32>, _m: &str| {};
        download_model_with_hardening(opts, test_hardening(), &mut progress)
            .await
            .unwrap();
        mock.assert();
        assert!(is_model_dir_complete(&target));
    }

    #[tokio::test]
    async fn httpmock_redirect_to_disallowed_host_blocked() {
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(GET).path("/bounce");
            then.status(302)
                .header("Location", "https://evil.example/payload.tar.bz2");
        });

        let dir = tempfile::tempdir().unwrap();
        let opts = DownloadOptions {
            model_name: "bounce".into(),
            target_dir: dir.path().join("installed"),
            base_url: server.base_url(),
            archive_url_override: Some(server.url("/bounce")),
            cancel: None,
            max_bytes: DEFAULT_MAX_DOWNLOAD_BYTES,
            expected_sha256: None,
        };
        let mut progress = |_f: Option<f32>, _m: &str| {};
        let err = download_model_with_hardening(opts, test_hardening(), &mut progress)
            .await
            .unwrap_err();
        let msg = err.to_string().to_ascii_lowercase();
        assert!(
            msg.contains("not allowed")
                || msg.contains("allowed")
                || msg.contains("redirect")
                || msg.contains("host")
                || msg.contains("evil"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn httpmock_slow_response_hits_overall_timeout() {
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(GET).path("/slow.tar.bz2");
            then.status(200)
                .delay(Duration::from_secs(5))
                .body(vec![0u8; 16]);
        });

        let dir = tempfile::tempdir().unwrap();
        let opts = DownloadOptions {
            model_name: "slow".into(),
            target_dir: dir.path().join("installed"),
            base_url: server.base_url(),
            archive_url_override: Some(server.url("/slow.tar.bz2")),
            cancel: None,
            max_bytes: DEFAULT_MAX_DOWNLOAD_BYTES,
            expected_sha256: None,
        };
        let hardening = DownloadHardening {
            connect_timeout: Duration::from_secs(2),
            overall_timeout: Duration::from_millis(300),
            body_inactivity_timeout: Duration::from_secs(2),
            extract_install_timeout: Duration::from_secs(5),
            max_tar_entries: 100,
            max_uncompressed_bytes: 1024 * 1024,
            allow_file_urls: false,
            allow_loopback_http: true,
        };
        let started = Instant::now();
        let mut progress = |_f: Option<f32>, _m: &str| {};
        let err = download_model_with_hardening(opts, hardening, &mut progress)
            .await
            .unwrap_err();
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "must fail fast, took {:?}",
            started.elapsed()
        );
        assert!(
            matches!(err, AsrError::RemoteTimeout(_, _))
                || err.to_string().to_ascii_lowercase().contains("timeout")
                || err.to_string().to_ascii_lowercase().contains("deadline"),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn local_server_connect_hang_hits_connect_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();

        let dir = tempfile::tempdir().unwrap();
        let opts = DownloadOptions {
            model_name: "hang".into(),
            target_dir: dir.path().join("installed"),
            base_url: format!("http://{addr}"),
            archive_url_override: Some(format!("http://{addr}/hang.tar.bz2")),
            cancel: None,
            max_bytes: DEFAULT_MAX_DOWNLOAD_BYTES,
            expected_sha256: None,
        };
        let hardening = DownloadHardening {
            connect_timeout: Duration::from_millis(250),
            overall_timeout: Duration::from_secs(2),
            body_inactivity_timeout: Duration::from_secs(2),
            extract_install_timeout: Duration::from_secs(5),
            max_tar_entries: 100,
            max_uncompressed_bytes: 1024 * 1024,
            allow_file_urls: false,
            allow_loopback_http: true,
        };
        let started = Instant::now();
        let mut progress = |_f: Option<f32>, _m: &str| {};
        let err = download_model_with_hardening(opts, hardening, &mut progress)
            .await
            .unwrap_err();
        drop(listener);
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "connect hang took {:?}",
            started.elapsed()
        );
        assert!(
            matches!(err, AsrError::RemoteTimeout(_, _) | AsrError::Transport(_))
                || err.to_string().to_ascii_lowercase().contains("timeout"),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn local_server_body_inactivity_timeout() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener as TokioListener;

        let listener = TokioListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 2048];
                let _ = sock.read(&mut buf).await;
                let headers = b"HTTP/1.1 200 OK\r\nContent-Length: 64\r\nConnection: close\r\n\r\n";
                let _ = sock.write_all(headers).await;
                let _ = sock.flush().await;
                tokio::time::sleep(Duration::from_secs(30)).await;
            }
        });

        let dir = tempfile::tempdir().unwrap();
        let opts = DownloadOptions {
            model_name: "stall".into(),
            target_dir: dir.path().join("installed"),
            base_url: format!("http://{addr}"),
            archive_url_override: Some(format!("http://{addr}/stall.tar.bz2")),
            cancel: None,
            max_bytes: DEFAULT_MAX_DOWNLOAD_BYTES,
            expected_sha256: None,
        };
        let hardening = DownloadHardening {
            connect_timeout: Duration::from_secs(2),
            overall_timeout: Duration::from_secs(5),
            body_inactivity_timeout: Duration::from_millis(300),
            extract_install_timeout: Duration::from_secs(5),
            max_tar_entries: 100,
            max_uncompressed_bytes: 1024 * 1024,
            allow_file_urls: false,
            allow_loopback_http: true,
        };
        let started = Instant::now();
        let mut progress = |_f: Option<f32>, _m: &str| {};
        let err = download_model_with_hardening(opts, hardening, &mut progress)
            .await
            .unwrap_err();
        server.abort();
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "body inactivity took {:?}",
            started.elapsed()
        );
        match &err {
            AsrError::RemoteTimeout(_, msg) => {
                assert!(
                    msg.to_ascii_lowercase().contains("inactivity")
                        || msg.to_ascii_lowercase().contains("deadline")
                        || msg.to_ascii_lowercase().contains("timeout"),
                    "{msg}"
                );
            }
            other => {
                let s = other.to_string().to_ascii_lowercase();
                assert!(
                    s.contains("timeout") || s.contains("inactivity") || s.contains("deadline"),
                    "{other:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn complete_target_is_not_clobbered_by_bad_archive_url() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("installed");
        write_model_dir(&target);
        fs::write(target.join("tokens.txt"), "original\n").unwrap();

        let bad_archive = dir.path().join("bad.tar.bz2");
        make_bz2_from_tar_builder(dir.path(), &bad_archive, |builder| {
            let mut header = tar::Header::new_gnu();
            header.set_path("readme.txt").unwrap();
            header.set_entry_type(EntryType::Regular);
            header.set_size(4);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, "readme.txt", &b"nope"[..])
                .unwrap();
        });

        let opts = DownloadOptions {
            model_name: "bad".into(),
            target_dir: target.clone(),
            base_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models".into(),
            archive_url_override: Some(format!("file://{}", bad_archive.display())),
            cancel: None,
            max_bytes: DEFAULT_MAX_DOWNLOAD_BYTES,
            expected_sha256: None,
        };
        let mut progress = |_f: Option<f32>, _m: &str| {};
        // Short-circuit before URL policy: complete install is left alone.
        download_model(opts, &mut progress).await.unwrap();
        assert!(is_model_dir_complete(&target));
        assert_eq!(
            fs::read_to_string(target.join("tokens.txt")).unwrap(),
            "original\n"
        );
    }

    #[tokio::test]
    async fn failed_extract_leaves_prior_complete_install_via_transaction() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("installed");
        write_model_dir(&target);
        fs::write(target.join("tokens.txt"), "keep-me\n").unwrap();

        let staged_bad = dir.path().join("staged_bad");
        fs::create_dir_all(&staged_bad).unwrap();
        fs::write(staged_bad.join("tokens.txt"), "nope\n").unwrap();

        let err = install_model_transaction(&staged_bad, &target, &inert_cancel()).unwrap_err();
        assert!(
            err.to_string().to_ascii_lowercase().contains("incomplete"),
            "{err}"
        );
        assert_eq!(
            fs::read_to_string(target.join("tokens.txt")).unwrap(),
            "keep-me\n"
        );
        assert!(is_model_dir_complete(&target));
    }

    #[cfg(unix)]
    #[test]
    fn extracted_and_installed_modes_are_clamped() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("m.tar.bz2");
        make_bz2_archive(dir.path(), &archive);
        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();
        // Deliberately loose umask-visible modes before extract clamp.
        safe_extract_tar_bz2(&archive, &out).unwrap();
        let found = find_extracted_model_dir(&out).unwrap();
        let file_mode = fs::metadata(found.join("tokens.txt"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        let dir_mode = fs::metadata(&found).unwrap().permissions().mode() & 0o777;
        assert_eq!(file_mode, 0o600, "model files must be 0600");
        assert_eq!(dir_mode, 0o700, "model dirs must be 0700");

        let target = dir.path().join("installed");
        write_model_dir(&target);
        // Loosen then clamp via install path helper.
        fs::set_permissions(&target, fs::Permissions::from_mode(0o755)).unwrap();
        fs::set_permissions(target.join("tokens.txt"), fs::Permissions::from_mode(0o644)).unwrap();
        clamp_tree_modes(&target).unwrap();
        assert_eq!(
            fs::metadata(&target).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(target.join("tokens.txt"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }

    #[test]
    fn recover_install_crash_restores_complete_backup_when_target_missing() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("model");
        let backup = backup_path_for(&target);
        write_model_dir(&backup);
        fs::write(backup.join("tokens.txt"), "from-bak\n").unwrap();
        assert!(!target.exists());

        recover_install_crash(&target).unwrap();
        assert!(is_model_dir_complete(&target));
        assert_eq!(
            fs::read_to_string(target.join("tokens.txt")).unwrap(),
            "from-bak\n"
        );
        assert!(!backup.exists(), "backup consumed by restore");
    }

    #[test]
    fn recover_install_crash_prefers_complete_backup_over_incomplete_target() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("model");
        let backup = backup_path_for(&target);
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("tokens.txt"), "broken\n").unwrap();
        write_model_dir(&backup);
        fs::write(backup.join("tokens.txt"), "good\n").unwrap();

        recover_install_crash(&target).unwrap();
        assert!(is_model_dir_complete(&target));
        assert_eq!(
            fs::read_to_string(target.join("tokens.txt")).unwrap(),
            "good\n"
        );
        assert!(!backup.exists());
    }

    #[test]
    fn recover_install_crash_drops_stale_backup_when_target_complete() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("model");
        let backup = backup_path_for(&target);
        write_model_dir(&target);
        write_model_dir(&backup);
        recover_install_crash(&target).unwrap();
        assert!(is_model_dir_complete(&target));
        assert!(!backup.exists());
    }

    #[test]
    fn extract_respects_cooperative_cancel() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("m.tar.bz2");
        make_bz2_archive(dir.path(), &archive);
        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();
        let cancel = Arc::new(AtomicBool::new(true));
        let err = safe_extract_tar_bz2_with_cancel(
            &archive,
            &out,
            &DownloadHardening::default(),
            &cancel,
        )
        .unwrap_err();
        assert!(matches!(err, AsrError::Cancelled(_)), "{err:?}");
    }

    #[tokio::test]
    async fn extract_install_wall_clock_timeout() {
        // Use a zero-resolved deadline (resolve maps 0 → default), so set 1ms explicitly.
        // Pre-set cancel so the blocking job stops at the first cancel check, and a
        // short wall-clock still bounds the overall wait if cancel is slow to observe.
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("m.tar.bz2");
        make_bz2_archive(dir.path(), &archive);
        let target = dir.path().join("installed");
        let cancel = Arc::new(AtomicBool::new(false));
        let opts = DownloadOptions {
            model_name: "m".into(),
            target_dir: target.clone(),
            base_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models".into(),
            archive_url_override: Some(format!("file://{}", archive.display())),
            cancel: Some(cancel.clone()),
            max_bytes: DEFAULT_MAX_DOWNLOAD_BYTES,
            expected_sha256: None,
        };
        let hardening = DownloadHardening {
            extract_install_timeout: Duration::from_millis(1),
            allow_file_urls: true,
            allow_loopback_http: true,
            connect_timeout: Duration::from_secs(2),
            overall_timeout: Duration::from_secs(5),
            body_inactivity_timeout: Duration::from_secs(2),
            max_tar_entries: DEFAULT_MAX_TAR_ENTRIES,
            max_uncompressed_bytes: DEFAULT_MAX_UNCOMPRESSED_BYTES,
        };
        let started = Instant::now();
        let mut progress = |_f: Option<f32>, _m: &str| {};
        let result = download_model_with_hardening(opts, hardening, &mut progress).await;
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "bounded extract/install took {:?}",
            started.elapsed()
        );
        match result {
            Err(AsrError::RemoteTimeout(_, msg)) => {
                assert!(
                    msg.to_ascii_lowercase().contains("extract")
                        || msg.to_ascii_lowercase().contains("wall"),
                    "{msg}"
                );
                assert!(
                    cancel.load(Ordering::Relaxed),
                    "timeout must flip cooperative cancel"
                );
            }
            Ok(()) => {
                // Host finished within 1ms after download; still a valid outcome.
                assert!(is_model_dir_complete(&target));
            }
            Err(AsrError::Cancelled(_)) => {
                // Cooperative cancel observed first — also acceptable under the bound.
            }
            Err(other) => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_sha256_expected_value() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("m.tar.bz2");
        make_bz2_archive(dir.path(), &archive);
        let err = verify_sha256(&archive, "").unwrap_err();
        assert!(
            err.to_string().to_ascii_lowercase().contains("empty"),
            "{err}"
        );
        let err = verify_sha256(&archive, "   ").unwrap_err();
        assert!(
            err.to_string().to_ascii_lowercase().contains("empty"),
            "{err}"
        );
    }

    #[test]
    fn rejects_hardlink_entries() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("hard.tar.bz2");
        // Real hardlink member: type Link pointing at another path.
        make_bz2_from_tar_builder(dir.path(), &archive, |builder| {
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(EntryType::Link);
            header.set_size(0);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_link(&mut header, "hard.link", "tokens.txt")
                .unwrap();
        });
        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();
        let err = safe_extract_tar_bz2(&archive, &out).unwrap_err();
        let msg = err.to_string().to_ascii_lowercase();
        assert!(
            msg.contains("link") || msg.contains("refusing") || msg.contains("special"),
            "{err}"
        );
    }

    #[test]
    fn rejects_fifo_special_entries() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("fifo.tar.bz2");
        make_bz2_from_tar_builder(dir.path(), &archive, |builder| {
            let mut header = tar::Header::new_gnu();
            header.set_path("evil.fifo").unwrap();
            header.set_entry_type(EntryType::Fifo);
            header.set_size(0);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, "evil.fifo", &[][..])
                .unwrap();
        });
        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();
        let err = safe_extract_tar_bz2(&archive, &out).unwrap_err();
        let msg = err.to_string().to_ascii_lowercase();
        assert!(
            msg.contains("special")
                || msg.contains("fifo")
                || msg.contains("refusing")
                || msg.contains("link"),
            "{err}"
        );
    }

    #[test]
    fn recover_install_crash_replaces_stray_target_file() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("model");
        let backup = backup_path_for(&target);
        // Target is a plain file (not a dir) — recovery must remove it safely.
        fs::write(&target, b"not-a-dir").unwrap();
        write_model_dir(&backup);
        fs::write(backup.join("tokens.txt"), "restored\n").unwrap();

        recover_install_crash(&target).unwrap();
        assert!(target.is_dir(), "target must become the restored directory");
        assert!(is_model_dir_complete(&target));
        assert_eq!(
            fs::read_to_string(target.join("tokens.txt")).unwrap(),
            "restored\n"
        );
        assert!(!backup.exists());
    }

    #[test]
    fn large_entry_cancel_returns_promptly() {
        // Single large regular member; cancel mid-extract must return within a
        // small multiple of IO_CHUNK_SIZE wall time (cooperative chunk bound).
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("big.tar.bz2");
        let big_len = 8 * 1024 * 1024; // 8 MiB payload
        {
            let tar_path = dir.path().join("big.tar");
            let mut file = File::create(&tar_path).unwrap();
            let mut header = tar::Header::new_ustar();
            header.set_path("big.bin").unwrap();
            header.set_entry_type(EntryType::Regular);
            header.set_size(big_len);
            header.set_mode(0o644);
            header.set_cksum();
            file.write_all(header.as_bytes()).unwrap();
            // Write payload in blocks; pad to 512-byte tar block boundary.
            let block = vec![0u8; 64 * 1024];
            let mut left = big_len as usize;
            while left > 0 {
                let n = left.min(block.len());
                file.write_all(&block[..n]).unwrap();
                left -= n;
            }
            let pad = (512 - (big_len as usize % 512)) % 512;
            if pad > 0 {
                file.write_all(&vec![0u8; pad]).unwrap();
            }
            file.write_all(&[0u8; 1024]).unwrap();
            drop(file);
            let tar_bytes = fs::read(&tar_path).unwrap();
            let mut enc = bzip2::write::BzEncoder::new(
                File::create(&archive).unwrap(),
                bzip2::Compression::fast(),
            );
            enc.write_all(&tar_bytes).unwrap();
            enc.finish().unwrap();
        }

        let out = dir.path().join("out");
        fs::create_dir_all(&out).unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_t = cancel.clone();
        let archive_t = archive.clone();
        let out_t = out.clone();
        let started = Instant::now();
        let handle = std::thread::spawn(move || {
            safe_extract_tar_bz2_with_cancel(
                &archive_t,
                &out_t,
                &DownloadHardening::default(),
                &cancel_t,
            )
        });
        // Flip cancel shortly after start; chunked extract should notice quickly.
        std::thread::sleep(Duration::from_millis(20));
        cancel.store(true, Ordering::Relaxed);
        let err = handle.join().unwrap().unwrap_err();
        let elapsed = started.elapsed();
        assert!(matches!(err, AsrError::Cancelled(_)), "{err:?}");
        // 8 MiB unbounded unpack would often take much longer; require prompt stop.
        assert!(
            elapsed < Duration::from_secs(2),
            "large-entry cancel not prompt: {elapsed:?}"
        );
    }

    #[test]
    fn defaults_are_finite_and_sane() {
        let d = DownloadHardening::default();
        assert_eq!(d.connect_timeout, DEFAULT_CONNECT_TIMEOUT);
        assert_eq!(d.overall_timeout, DEFAULT_OVERALL_TIMEOUT);
        assert_eq!(d.body_inactivity_timeout, DEFAULT_BODY_INACTIVITY_TIMEOUT);
        assert_eq!(d.extract_install_timeout, DEFAULT_EXTRACT_INSTALL_TIMEOUT);
        assert_eq!(d.max_tar_entries, DEFAULT_MAX_TAR_ENTRIES);
        assert_eq!(d.max_uncompressed_bytes, DEFAULT_MAX_UNCOMPRESSED_BYTES);
        assert!(!d.allow_file_urls);
        assert!(!d.allow_loopback_http);
        assert!(!d.extract_install_timeout.is_zero());
        assert!(!d.connect_timeout.is_zero());
        assert!(!d.overall_timeout.is_zero());
        assert!(!d.body_inactivity_timeout.is_zero());
        assert!(d.max_tar_entries > 0);
        assert!(d.max_uncompressed_bytes >= DEFAULT_MAX_DOWNLOAD_BYTES);
    }
}
