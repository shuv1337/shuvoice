//! Private append-only JSON-lines log with single-generation size rotation.
//!
//! Writes happen on a dedicated thread so callers on async/actor paths never
//! touch the filesystem. The queue is bounded; when it is full new lines are
//! dropped (and counted) instead of blocking the caller.

use std::fs::{DirBuilder, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::thread::JoinHandle;

use tracing::warn;

const QUEUE_CAPACITY: usize = 256;

pub struct JsonlAppender {
    tx: Option<SyncSender<String>>,
    join: Option<JoinHandle<()>>,
    dropped: Arc<AtomicU64>,
}

impl JsonlAppender {
    /// Start the writer thread. The file and parent directory are created
    /// lazily on first write (`0600` file, `0700` directories).
    pub fn spawn(path: PathBuf, max_bytes: u64) -> std::io::Result<Self> {
        let (tx, rx) = sync_channel::<String>(QUEUE_CAPACITY);
        let join = std::thread::Builder::new()
            .name("shuvoice-jsonl-log".into())
            .spawn(move || writer_loop(&path, max_bytes, rx))?;
        Ok(Self {
            tx: Some(tx),
            join: Some(join),
            dropped: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Queue one line (without trailing newline). Returns `false` when dropped.
    pub fn append(&self, line: String) -> bool {
        let Some(tx) = self.tx.as_ref() else {
            return false;
        };
        match tx.try_send(line) {
            Ok(()) => true,
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl Drop for JsonlAppender {
    /// Flushes queued lines before returning.
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn writer_loop(path: &Path, max_bytes: u64, rx: Receiver<String>) {
    let mut file: Option<File> = None;
    let mut warned = false;
    for line in rx {
        match write_line(path, max_bytes, &mut file, &line) {
            Ok(()) => warned = false,
            Err(err) => {
                file = None;
                if !warned {
                    warn!(error = %err, "jsonl log write failed; dropping lines until it recovers");
                    warned = true;
                }
            }
        }
    }
}

fn write_line(
    path: &Path,
    max_bytes: u64,
    file: &mut Option<File>,
    line: &str,
) -> std::io::Result<()> {
    if let Some(f) = file.as_ref()
        && f.metadata()?.len() >= max_bytes
    {
        *file = None;
        std::fs::rename(path, rotated_path(path))?;
    }
    let f = match file {
        Some(f) => f,
        None => {
            if let Some(parent) = path.parent() {
                DirBuilder::new()
                    .recursive(true)
                    .mode(0o700)
                    .create(parent)?;
            }
            let opened = OpenOptions::new()
                .create(true)
                .append(true)
                .mode(0o600)
                .open(path)?;
            if opened.metadata()?.len() >= max_bytes {
                drop(opened);
                std::fs::rename(path, rotated_path(path))?;
                return write_line(path, max_bytes, file, line);
            }
            file.insert(opened)
        }
    };
    let mut buf = Vec::with_capacity(line.len() + 1);
    buf.extend_from_slice(line.as_bytes());
    buf.push(b'\n');
    f.write_all(&buf)?;
    f.flush()
}

/// `<path>.1`
pub fn rotated_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".1");
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn appends_lines_with_private_permissions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/log.jsonl");
        let log = JsonlAppender::spawn(path.clone(), 1024 * 1024).unwrap();
        assert!(log.append(r#"{"a":1}"#.into()));
        assert!(log.append(r#"{"a":2}"#.into()));
        drop(log);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{\"a\":1}\n{\"a\":2}\n"
        );
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        let dir_mode = std::fs::metadata(path.parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(dir_mode, 0o700);
    }

    #[test]
    fn rotates_to_single_generation_when_full() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.jsonl");
        std::fs::write(&path, "old-old-old\n").unwrap();
        let log = JsonlAppender::spawn(path.clone(), 10).unwrap();
        log.append("first-line".into());
        log.append("second".into());
        drop(log);
        assert_eq!(
            std::fs::read_to_string(rotated_path(&path)).unwrap(),
            "first-line\n"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second\n");
    }
}
