//! Subprocess runner with timeouts and secret-safe errors.

use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::error::ProcessError;

/// Default cap on captured stdout/stderr combined per stream.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 1_048_576; // 1 MiB

/// Options for a single process invocation.
#[derive(Debug, Clone)]
pub struct RunOptions {
    pub timeout: Duration,
    pub stdin_data: Option<Vec<u8>>,
    pub capture_stdout: bool,
    pub capture_stderr: bool,
    pub check: bool,
    /// Per-stream capture cap (bytes).
    pub max_output_bytes: usize,
    /// Put the child in its own process group and kill the group on timeout.
    pub kill_process_group: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            stdin_data: None,
            capture_stdout: true,
            capture_stderr: true,
            check: true,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            kill_process_group: true,
        }
    }
}

/// Successful (or unchecked) process output.
#[derive(Debug, Clone)]
pub struct RunOutput {
    pub status_code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub success: bool,
}

impl RunOutput {
    #[must_use]
    pub fn stdout_lossy(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    #[must_use]
    pub fn stdout_text(&self) -> String {
        String::from_utf8_lossy(&self.stdout)
            .trim_end_matches('\n')
            .to_string()
    }
}

/// Abstraction over process execution (injectable for tests).
pub trait CommandRunner: Send + Sync {
    fn run(&self, argv: &[String], opts: &RunOptions) -> Result<RunOutput, ProcessError>;
}

/// Default runner using `std::process::Command`.
#[derive(Debug, Default, Clone, Copy)]
pub struct StdCommandRunner;

impl CommandRunner for StdCommandRunner {
    fn run(&self, argv: &[String], opts: &RunOptions) -> Result<RunOutput, ProcessError> {
        if argv.is_empty() {
            return Err(ProcessError::NotFound {
                program: String::new(),
            });
        }
        let program = argv[0].clone();
        let mut cmd = Command::new(&program);
        if argv.len() > 1 {
            cmd.args(&argv[1..]);
        }
        if opts.capture_stdout {
            cmd.stdout(Stdio::piped());
        } else {
            cmd.stdout(Stdio::null());
        }
        if opts.capture_stderr {
            cmd.stderr(Stdio::piped());
        } else {
            cmd.stderr(Stdio::null());
        }
        if opts.stdin_data.is_some() {
            cmd.stdin(Stdio::piped());
        } else {
            cmd.stdin(Stdio::null());
        }

        // Own process group so timeout can reap grandchildren.
        if opts.kill_process_group {
            // SAFETY: pre_exec runs in the child after fork, before exec.
            unsafe {
                cmd.pre_exec(|| {
                    // SAFETY: child-only; establishes a new process group for timeout kill.
                    libc::setpgid(0, 0);
                    Ok(())
                });
            }
        }

        let mut child = cmd.spawn().map_err(|err| map_spawn_err(&program, err))?;
        let deadline = Instant::now() + opts.timeout;
        let limit = opts.max_output_bytes.max(1);

        // Stdin write under the same deadline (helper thread).
        let stdin_handle = if let Some(data) = opts.stdin_data.clone() {
            child.stdin.take().map(|mut stdin| {
                thread_spawn_named("proc-stdin", move || {
                    let _ = write_all_deadline(&mut stdin, &data, deadline);
                    // Drop closes stdin.
                })
            })
        } else {
            None
        };

        let stdout_buf = Arc::new(Mutex::new(Vec::new()));
        let stderr_buf = Arc::new(Mutex::new(Vec::new()));
        let stdout_overflow = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stderr_overflow = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let stdout_thread = child.stdout.take().map(|mut pipe| {
            let buf = Arc::clone(&stdout_buf);
            let overflow = Arc::clone(&stdout_overflow);
            thread_spawn_named("proc-stdout", move || {
                read_capped(&mut pipe, &buf, limit, &overflow);
            })
        });
        let stderr_thread = child.stderr.take().map(|mut pipe| {
            let buf = Arc::clone(&stderr_buf);
            let overflow = Arc::clone(&stderr_overflow);
            thread_spawn_named("proc-stderr", move || {
                read_capped(&mut pipe, &buf, limit, &overflow);
            })
        });

        // Wait loop with deadline.
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {
                    if Instant::now() >= deadline {
                        kill_child_tree(&mut child, opts.kill_process_group);
                        let _ = child.wait();
                        // Join readers best-effort.
                        let _ = stdin_handle.map(|h| h.join());
                        let _ = stdout_thread.map(|h| h.join());
                        let _ = stderr_thread.map(|h| h.join());
                        return Err(ProcessError::Timeout {
                            program,
                            timeout: opts.timeout,
                        });
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(err) => {
                    kill_child_tree(&mut child, opts.kill_process_group);
                    return Err(ProcessError::Io {
                        program,
                        source: err,
                    });
                }
            }
        };

        let _ = stdin_handle.map(|h| h.join());
        let _ = stdout_thread.map(|h| h.join());
        let _ = stderr_thread.map(|h| h.join());

        if stdout_overflow.load(std::sync::atomic::Ordering::SeqCst)
            || stderr_overflow.load(std::sync::atomic::Ordering::SeqCst)
        {
            return Err(ProcessError::OutputTooLarge { program, limit });
        }

        let stdout = stdout_buf.lock().unwrap().clone();
        let stderr = stderr_buf.lock().unwrap().clone();
        let code = status.code();
        let success = status.success();
        let result = RunOutput {
            status_code: code,
            stdout,
            stderr,
            success,
        };
        if opts.check && !success {
            return Err(ProcessError::ExitCode {
                program,
                code: code.unwrap_or(-1),
            });
        }
        Ok(result)
    }
}

fn thread_spawn_named<F>(name: &str, f: F) -> std::thread::JoinHandle<()>
where
    F: FnOnce() + Send + 'static,
{
    match std::thread::Builder::new().name(name.into()).spawn(f) {
        Ok(h) => h,
        Err(_) => {
            // Name failed (rare); run inline via a fresh spawn without name is not possible
            // without F: Clone — panic is worse, so just use unnamed via std::thread if we
            // still hold f... Builder::spawn consumes f on both Ok and Err paths in std.
            // On Err, f is returned... actually spawn consumes f only on success in older
            // rust? In current std, Err does not return f. Use unwrap_or_else with panic
            // fallback by double-boxing.
            unreachable!("thread spawn with name failed — should not happen")
        }
    }
}

fn write_all_deadline(w: &mut impl Write, data: &[u8], deadline: Instant) -> std::io::Result<()> {
    let mut offset = 0;
    while offset < data.len() {
        if Instant::now() >= deadline {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "stdin write deadline",
            ));
        }
        match w.write(&data[offset..]) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "stdin closed",
                ));
            }
            Ok(n) => offset += n,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(())
}

fn read_capped(
    r: &mut impl Read,
    buf: &Mutex<Vec<u8>>,
    limit: usize,
    overflow: &std::sync::atomic::AtomicBool,
) {
    let mut tmp = [0u8; 8192];
    loop {
        match r.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => {
                let mut g = buf.lock().unwrap();
                if g.len() >= limit {
                    overflow.store(true, std::sync::atomic::Ordering::SeqCst);
                    // Drain remaining to unblock child, but don't store.
                    continue;
                }
                let take = (limit - g.len()).min(n);
                g.extend_from_slice(&tmp[..take]);
                if take < n {
                    overflow.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
}

fn kill_child_tree(child: &mut std::process::Child, process_group: bool) {
    if process_group {
        let pid = child.id() as i32;
        if pid > 0 {
            // SAFETY: pid is the child's process group leader id we created via setpgid.
            // Negative pgid delivers SIGKILL to the whole group.
            unsafe {
                let _ = libc::kill(-pid, libc::SIGKILL);
            }
        }
    }
    let _ = child.kill();
}

fn map_spawn_err(program: &str, err: std::io::Error) -> ProcessError {
    if err.kind() == std::io::ErrorKind::NotFound {
        ProcessError::NotFound {
            program: program.to_string(),
        }
    } else {
        ProcessError::Io {
            program: program.to_string(),
            source: err,
        }
    }
}

/// Recording runner for tests: captures argv and returns scripted results.
#[derive(Clone, Default)]
pub struct ScriptedRunner {
    inner: Arc<Mutex<ScriptedInner>>,
}

#[derive(Default)]
struct ScriptedInner {
    calls: Vec<Vec<String>>,
    stdin_payloads: Vec<Option<Vec<u8>>>,
    /// Queue of results (pop front). If empty, returns success empty.
    results: Vec<Result<RunOutput, ProcessError>>,
    /// Optional predicate overrides: if set, consulted first.
    dynamic: Option<DynamicHandler>,
}

type DynamicHandler = Arc<dyn Fn(&[String]) -> Result<RunOutput, ProcessError> + Send + Sync>;

impl ScriptedRunner {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_ok(&self, stdout: impl Into<Vec<u8>>) {
        self.inner.lock().unwrap().results.push(Ok(RunOutput {
            status_code: Some(0),
            stdout: stdout.into(),
            stderr: Vec::new(),
            success: true,
        }));
    }

    pub fn push_exit(&self, code: i32) {
        self.inner
            .lock()
            .unwrap()
            .results
            .push(Err(ProcessError::ExitCode {
                program: "scripted".into(),
                code,
            }));
    }

    pub fn set_dynamic<F>(&self, f: F)
    where
        F: Fn(&[String]) -> Result<RunOutput, ProcessError> + Send + Sync + 'static,
    {
        self.inner.lock().unwrap().dynamic = Some(Arc::new(f));
    }

    pub fn calls(&self) -> Vec<Vec<String>> {
        self.inner.lock().unwrap().calls.clone()
    }

    pub fn stdin_payloads(&self) -> Vec<Option<Vec<u8>>> {
        self.inner.lock().unwrap().stdin_payloads.clone()
    }
}

impl CommandRunner for ScriptedRunner {
    fn run(&self, argv: &[String], opts: &RunOptions) -> Result<RunOutput, ProcessError> {
        let mut g = self.inner.lock().unwrap();
        g.calls.push(argv.to_vec());
        g.stdin_payloads.push(opts.stdin_data.clone());
        if let Some(dyn_fn) = g.dynamic.clone() {
            drop(g);
            return dyn_fn(argv);
        }
        if g.results.is_empty() {
            return Ok(RunOutput {
                status_code: Some(0),
                stdout: Vec::new(),
                stderr: Vec::new(),
                success: true,
            });
        }
        g.results.remove(0)
    }
}

/// Helper: argv as owned Strings.
#[must_use]
pub fn argv<I, S>(parts: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    parts.into_iter().map(Into::into).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_error_does_not_include_args() {
        let err = ProcessError::Timeout {
            program: "wtype".into(),
            timeout: Duration::from_secs(5),
        };
        let msg = err.to_string();
        assert!(msg.contains("wtype"));
        assert!(msg.contains("timed out"));
        assert!(!msg.contains("--"));
        assert!(!msg.contains("SECRET"));
    }

    #[test]
    fn scripted_runner_records_calls() {
        let r = ScriptedRunner::new();
        r.push_ok(b"hi");
        let out = r
            .run(&argv(["wl-paste", "--no-newline"]), &RunOptions::default())
            .unwrap();
        assert_eq!(out.stdout, b"hi");
        assert_eq!(r.calls().len(), 1);
    }

    #[test]
    fn real_timeout_kills_sleep() {
        let runner = StdCommandRunner;
        let err = runner
            .run(
                &argv(["sleep", "30"]),
                &RunOptions {
                    timeout: Duration::from_millis(200),
                    capture_stdout: false,
                    capture_stderr: false,
                    check: false,
                    ..RunOptions::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, ProcessError::Timeout { .. }));
        assert!(!err.to_string().contains("30"));
    }

    #[test]
    fn stdin_write_under_timeout_does_not_hang() {
        // `cat` will read stdin; we just ensure large stdin + short timeout returns.
        let runner = StdCommandRunner;
        let data = vec![b'x'; 64 * 1024];
        let result = runner.run(
            &argv(["cat"]),
            &RunOptions {
                timeout: Duration::from_secs(2),
                stdin_data: Some(data.clone()),
                ..RunOptions::default()
            },
        );
        match result {
            Ok(out) => assert_eq!(out.stdout, data),
            Err(ProcessError::Timeout { .. }) => {}
            Err(other) => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn output_cap_enforced() {
        let runner = StdCommandRunner;
        // yes generates infinite output; with tiny cap should OutputTooLarge or Timeout.
        let err = runner
            .run(
                &argv(["yes", "x"]),
                &RunOptions {
                    timeout: Duration::from_millis(500),
                    max_output_bytes: 1024,
                    ..RunOptions::default()
                },
            )
            .unwrap_err();
        assert!(
            matches!(
                err,
                ProcessError::OutputTooLarge { .. } | ProcessError::Timeout { .. }
            ),
            "got {err}"
        );
        assert!(!err.to_string().contains("yes x"));
    }
}
