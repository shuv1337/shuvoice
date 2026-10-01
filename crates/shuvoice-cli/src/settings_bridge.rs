//! `shuvoice settings-bridge`: versioned JSON-lines protocol on stdio for the
//! settings app.
//!
//! One request object per line; exactly one response line per request:
//!
//! ```text
//! -> {"v":1,"id":7,"op":"save","params":{"revision":"…","changes":{"overlay.font_size":24}}}
//! <- {"v":1,"id":7,"ok":true,"result":{"revision":"…","backup":"…","changed":[…]}}
//! <- {"v":1,"id":7,"ok":false,"error":{"kind":"conflict","message":"…",…}}
//! ```
//!
//! `apply` runs on a worker thread and streams progress events before its
//! response, so `status` and `cancel` stay responsive meanwhile:
//!
//! ```text
//! <- {"v":1,"id":8,"event":"progress","phase":"waiting_idle","busy":["recording"]}
//! <- {"v":1,"id":8,"event":"progress","phase":"saving"}
//! <- {"v":1,"id":8,"event":"progress","phase":"restarting"}
//! <- {"v":1,"id":8,"ok":true,"result":{"saved":{…},"restart":{"outcome":"ready",…}}}
//! ```
//!
//! Only the operations in [`handle`] and [`serve`] exist. Stdout carries
//! protocol only; logs go to stderr. Secret values never appear in responses.

use std::collections::BTreeMap;
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};
use shuvoice_control::{ControlCommand, send_control_command};
use shuvoice_core::Config;
use shuvoice_core::settings::{self, ApplyError, FIELDS, Section};

use crate::error::{EXIT_SUCCESS, ExitStatus};

pub const PROTOCOL_VERSION: u64 = 1;
/// Requests larger than this are rejected (and the rest of the line discarded).
pub const MAX_REQUEST_BYTES: usize = 256 * 1024;
const SERVICE: &str = "shuvoice.service";

/// Section order for the sidebar.
const SECTIONS: [Section; 5] = [
    Section::Speech,
    Section::Typing,
    Section::TextToSpeech,
    Section::Audio,
    Section::Appearance,
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    v: u64,
    id: u64,
    op: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChangeParams {
    #[serde(default)]
    revision: Option<String>,
    #[serde(default)]
    changes: BTreeMap<String, Value>,
}

/// Side-effect seams so the protocol is testable without the live service.
pub struct Context {
    pub config_path: PathBuf,
    pub env_present: Box<dyn Fn(&str) -> bool + Send + Sync>,
    pub status: Box<dyn Fn() -> Value + Send + Sync>,
    pub devices: Box<dyn Fn() -> Result<Value, String> + Send + Sync>,
    /// Start/restart the service and wait for readiness; returns the outcome.
    pub restart: Box<dyn Fn() -> Value + Send + Sync>,
    pub idle_poll: Duration,
    pub idle_timeout: Duration,
}

impl Context {
    pub fn live() -> Self {
        Self {
            config_path: Config::config_path(),
            env_present: Box::new(|name| std::env::var_os(name).is_some_and(|v| !v.is_empty())),
            status: Box::new(live_status),
            devices: Box::new(|| {
                crate::commands::audio::input_devices().map(|devices| json!({ "devices": devices }))
            }),
            restart: Box::new(|| {
                // Test/dev knob: save without touching the user service.
                if std::env::var_os("SHUVOICE_SETTINGS_NO_RESTART").is_some_and(|v| v == "1") {
                    return json!(crate::commands::wizard::RestartOutcome::NotActive {
                        state: "restart disabled".into()
                    });
                }
                json!(crate::commands::wizard::restart_live(SERVICE))
            }),
            idle_poll: Duration::from_millis(500),
            idle_timeout: Duration::from_secs(120),
        }
    }
}

/// Activities that block a restart: recording/processing dictation, or
/// read-aloud that is synthesizing, playing or paused.
fn busy_activities(status: &Value) -> Vec<String> {
    if status.get("active_state").and_then(Value::as_str) != Some("active") {
        return Vec::new();
    }
    let mut busy = Vec::new();
    if let Some(stt) = status.get("stt").and_then(Value::as_str)
        && matches!(stt, "recording" | "processing")
    {
        busy.push(stt.to_string());
    }
    if let Some(tts) = status.get("tts").and_then(Value::as_str)
        && matches!(tts, "synthesizing" | "playing" | "paused")
    {
        busy.push(format!("tts_{tts}"));
    }
    busy
}

fn live_status() -> Value {
    let socket = crate::config::load_config()
        .ok()
        .and_then(|config| config.control_socket);
    let probe = |command| {
        send_control_command(command, socket.as_deref(), Some(Duration::from_millis(500)))
            .ok()
            .and_then(|line| line.strip_prefix("OK ").map(str::to_string))
    };
    let ui_ready = probe(ControlCommand::DebugStatus)
        .and_then(|body| serde_json::from_str::<Value>(&body).ok())
        .and_then(|debug| debug.get("ui_ready").and_then(Value::as_bool));
    json!({
        "service": SERVICE,
        "active_state": shuvoice_io::waybar::service_active_state(SERVICE, None),
        "ui_ready": ui_ready,
        "stt": probe(ControlCommand::Status),
        "tts": probe(ControlCommand::TtsStatus),
    })
}

fn ok(id: u64, result: Value) -> Value {
    json!({ "v": PROTOCOL_VERSION, "id": id, "ok": true, "result": result })
}

fn err(id: Option<u64>, kind: &str, message: impl Into<String>, data: Value) -> Value {
    let mut error = json!({ "kind": kind, "message": message.into() });
    if let (Value::Object(target), Value::Object(extra)) = (&mut error, data) {
        target.extend(extra);
    }
    json!({ "v": PROTOCOL_VERSION, "id": id, "ok": false, "error": error })
}

fn params<T: for<'de> Deserialize<'de> + Default>(value: Value) -> Result<T, String> {
    if value.is_null() {
        return Ok(T::default());
    }
    serde_json::from_value(value).map_err(|e| format!("invalid params: {e}"))
}

/// Handle one request line and return the response object.
pub fn handle(line: &str, ctx: &Context) -> Value {
    let request: Request = match serde_json::from_str(line) {
        Ok(request) => request,
        Err(e) => {
            let id = serde_json::from_str::<Value>(line)
                .ok()
                .and_then(|v| v.get("id").and_then(Value::as_u64));
            return err(
                id,
                "protocol",
                format!("malformed request: {e}"),
                Value::Null,
            );
        }
    };
    let id = request.id;
    if request.v != PROTOCOL_VERSION {
        return err(
            Some(id),
            "protocol",
            format!(
                "unsupported protocol version {} (bridge speaks {PROTOCOL_VERSION})",
                request.v
            ),
            Value::Null,
        );
    }

    match request.op.as_str() {
        "hello" => ok(
            id,
            json!({
                "protocol": PROTOCOL_VERSION,
                "version": env!("CARGO_PKG_VERSION"),
                "config_path": ctx.config_path.display().to_string(),
            }),
        ),
        "schema" => ok(id, json!({ "sections": SECTIONS, "fields": FIELDS })),
        "snapshot" => match settings::snapshot(&ctx.config_path, &ctx.env_present) {
            Ok(snapshot) => ok(id, json!(snapshot)),
            Err(message) => err(Some(id), "io", message, Value::Null),
        },
        "validate" => {
            let p: ChangeParams = match params(request.params) {
                Ok(p) => p,
                Err(message) => return err(Some(id), "protocol", message, Value::Null),
            };
            match validate(ctx, &p.changes) {
                Ok(errors) => ok(id, json!({ "errors": errors })),
                Err(message) => err(Some(id), "io", message, Value::Null),
            }
        }
        "save" => {
            let p: ChangeParams = match params(request.params) {
                Ok(p) => p,
                Err(message) => return err(Some(id), "protocol", message, Value::Null),
            };
            let Some(revision) = p.revision else {
                return err(
                    Some(id),
                    "protocol",
                    "save requires params.revision",
                    Value::Null,
                );
            };
            if p.changes.is_empty() {
                return err(
                    Some(id),
                    "protocol",
                    "save requires at least one change",
                    Value::Null,
                );
            }
            match settings::apply(&ctx.config_path, &revision, &p.changes) {
                Ok(applied) => ok(id, json!(applied)),
                Err(ApplyError::Conflict { current_revision }) => err(
                    Some(id),
                    "conflict",
                    "the config file changed since it was loaded",
                    json!({ "current_revision": current_revision }),
                ),
                Err(ApplyError::Invalid { errors }) => err(
                    Some(id),
                    "invalid",
                    "some settings are invalid",
                    json!({ "errors": errors }),
                ),
                Err(ApplyError::Io { message }) => err(Some(id), "io", message, Value::Null),
            }
        }
        "status" => ok(id, (ctx.status)()),
        "devices" => match (ctx.devices)() {
            Ok(result) => ok(id, result),
            Err(message) => err(Some(id), "unavailable", message, Value::Null),
        },
        other => err(
            Some(id),
            "unknown_op",
            format!("unknown op: {other}"),
            Value::Null,
        ),
    }
}

fn validate(
    ctx: &Context,
    changes: &BTreeMap<String, Value>,
) -> Result<Vec<settings::FieldError>, String> {
    let raw = shuvoice_core::config::load_raw(&ctx.config_path).map_err(|e| e.to_string())?;
    let (migrated, _) =
        shuvoice_core::config::migrate_to_latest(&raw).map_err(|e| e.to_string())?;
    let current = settings::snapshot(&ctx.config_path, |_| false)?.values;
    Ok(
        match settings::validate_changes(&migrated, &current, changes) {
            Ok(_) => Vec::new(),
            Err(errors) => errors,
        },
    )
}

/// Shared, line-atomic writer for responses and events from several threads.
pub type Output = Arc<Mutex<dyn Write + Send>>;

fn emit(out: &Output, message: &Value) {
    let mut out = out.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    // A closed stdout means the app is gone; nothing useful to do on error.
    let _ = serde_json::to_writer(&mut *out, message);
    let _ = out.write_all(b"\n");
    let _ = out.flush();
}

fn progress(id: u64, phase: &str, extra: Value) -> Value {
    let mut event = json!({ "v": PROTOCOL_VERSION, "id": id, "event": "progress", "phase": phase });
    if let (Value::Object(target), Value::Object(extra)) = (&mut event, extra) {
        target.extend(extra);
    }
    event
}

/// `apply`: validate → wait for idle → save → restart, with progress events.
fn apply_op(
    id: u64,
    params_value: Value,
    ctx: &Context,
    out: &Output,
    cancel: &AtomicBool,
) -> Value {
    let p: ChangeParams = match params(params_value) {
        Ok(p) => p,
        Err(message) => return err(Some(id), "protocol", message, Value::Null),
    };
    let Some(revision) = p.revision else {
        return err(
            Some(id),
            "protocol",
            "apply requires params.revision",
            Value::Null,
        );
    };
    if p.changes.is_empty() {
        return err(
            Some(id),
            "protocol",
            "apply requires at least one change",
            Value::Null,
        );
    }
    match validate(ctx, &p.changes) {
        Ok(errors) if !errors.is_empty() => {
            return err(
                Some(id),
                "invalid",
                "some settings are invalid",
                json!({ "errors": errors }),
            );
        }
        Ok(_) => {}
        Err(message) => return err(Some(id), "io", message, Value::Null),
    }

    // Idle gate (a check, not a lock): never restart under an active
    // recording or read-aloud. Nothing has been saved yet.
    let started = Instant::now();
    let mut announced: Vec<String> = Vec::new();
    loop {
        let busy = busy_activities(&(ctx.status)());
        if busy.is_empty() {
            break;
        }
        if busy != announced {
            emit(out, &progress(id, "waiting_idle", json!({ "busy": busy })));
            announced = busy;
        }
        if cancel.load(Ordering::Acquire) {
            return err(
                Some(id),
                "cancelled",
                "apply cancelled; nothing was saved",
                Value::Null,
            );
        }
        if started.elapsed() >= ctx.idle_timeout {
            return err(
                Some(id),
                "busy",
                "ShuVoice stayed busy; nothing was saved",
                json!({ "busy": announced }),
            );
        }
        std::thread::sleep(ctx.idle_poll);
    }

    emit(out, &progress(id, "saving", Value::Null));
    let saved = match settings::apply(&ctx.config_path, &revision, &p.changes) {
        Ok(saved) => saved,
        Err(ApplyError::Conflict { current_revision }) => {
            return err(
                Some(id),
                "conflict",
                "the config file changed since it was loaded",
                json!({ "current_revision": current_revision }),
            );
        }
        Err(ApplyError::Invalid { errors }) => {
            return err(
                Some(id),
                "invalid",
                "some settings are invalid",
                json!({ "errors": errors }),
            );
        }
        Err(ApplyError::Io { message }) => return err(Some(id), "io", message, Value::Null),
    };

    emit(out, &progress(id, "restarting", Value::Null));
    ok(id, json!({ "saved": saved, "restart": (ctx.restart)() }))
}

/// Serve requests from `input` until EOF. `apply` runs on a worker thread
/// (one at a time); `cancel` stops an apply that is still waiting for idle.
pub fn serve(mut input: impl BufRead, out: Output, ctx: Arc<Context>) -> std::io::Result<()> {
    let cancel = Arc::new(AtomicBool::new(false));
    let mut worker: Option<std::thread::JoinHandle<()>> = None;
    let mut line = Vec::new();
    loop {
        line.clear();
        let read = (&mut input)
            .take(MAX_REQUEST_BYTES as u64 + 1)
            .read_until(b'\n', &mut line)?;
        if read == 0 {
            break;
        }
        if line.len() > MAX_REQUEST_BYTES && line.last() != Some(&b'\n') {
            // Discard the rest of the oversized line.
            let mut sink = Vec::new();
            input.read_until(b'\n', &mut sink)?;
            let message = format!("request exceeds {MAX_REQUEST_BYTES} bytes");
            emit(&out, &err(None, "protocol", message, Value::Null));
            continue;
        }
        let text = String::from_utf8_lossy(&line);
        let text = text.trim();
        if text.is_empty() {
            continue;
        }

        let request = serde_json::from_str::<Request>(text).ok();
        let running = worker.as_ref().is_some_and(|w| !w.is_finished());
        match request.as_ref().map(|r| (r.v, r.op.as_str())) {
            Some((PROTOCOL_VERSION, "apply")) if running => {
                let id = request.as_ref().map(|r| r.id);
                emit(
                    &out,
                    &err(id, "busy", "another apply is in progress", Value::Null),
                );
            }
            Some((PROTOCOL_VERSION, "apply")) => {
                let Request { id, params, .. } = request.expect("matched Some");
                cancel.store(false, Ordering::Release);
                let (ctx, out, cancel) = (Arc::clone(&ctx), Arc::clone(&out), Arc::clone(&cancel));
                worker = Some(std::thread::spawn(move || {
                    let response = apply_op(id, params, &ctx, &out, &cancel);
                    emit(&out, &response);
                }));
            }
            Some((PROTOCOL_VERSION, "cancel")) => {
                cancel.store(true, Ordering::Release);
                let id = request.expect("matched Some").id;
                emit(&out, &ok(id, json!({ "cancelling": running })));
            }
            _ => emit(&out, &handle(text, &ctx)),
        }
    }
    // The app closed: cancel a pending wait, let a started restart finish.
    cancel.store(true, Ordering::Release);
    if let Some(worker) = worker {
        let _ = worker.join();
    }
    Ok(())
}

/// Entry for `shuvoice settings-bridge`.
pub fn run_stdio() -> ExitStatus {
    let ctx = Arc::new(Context::live());
    let out: Output = Arc::new(Mutex::new(std::io::stdout()));
    match serve(std::io::stdin().lock(), out, ctx) {
        Ok(()) => ExitStatus::code(EXIT_SUCCESS),
        Err(e) => {
            eprintln!("ERROR: settings bridge I/O: {e}");
            ExitStatus::code(crate::error::EXIT_FAILURE)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(dir: &tempfile::TempDir) -> Context {
        Context {
            config_path: dir.path().join("config.toml"),
            env_present: Box::new(|name| name == "OPENAI_API_KEY"),
            status: Box::new(
                || json!({ "active_state": "active", "ui_ready": true, "stt": "idle" }),
            ),
            devices: Box::new(|| Ok(json!({ "devices": [{ "index": 0, "name": "Mic" }] }))),
            restart: Box::new(|| json!({ "outcome": "ready", "action": "restart" })),
            idle_poll: Duration::from_millis(5),
            idle_timeout: Duration::from_millis(200),
        }
    }

    /// Run `serve` over `lines` and return every emitted message.
    fn run(ctx: Context, lines: &[Value]) -> Vec<Value> {
        let input: String = lines.iter().map(|l| format!("{l}\n")).collect();
        let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let out: Output = buffer.clone();
        serve(input.as_bytes(), out, Arc::new(ctx)).unwrap();
        let bytes = buffer.lock().unwrap().clone();
        String::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn call(ctx: &Context, request: Value) -> Value {
        handle(&request.to_string(), ctx)
    }

    #[test]
    fn hello_schema_and_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx(&dir);
        let hello = call(&ctx, json!({"v":1,"id":1,"op":"hello"}));
        assert_eq!(hello["ok"], true);
        assert_eq!(hello["result"]["protocol"], 1);

        let schema = call(&ctx, json!({"v":1,"id":2,"op":"schema"}));
        assert_eq!(schema["result"]["sections"][0], "speech");
        assert_eq!(
            schema["result"]["fields"].as_array().unwrap().len(),
            FIELDS.len()
        );

        let snap = call(&ctx, json!({"v":1,"id":3,"op":"snapshot"}));
        assert_eq!(snap["id"], 3);
        assert_eq!(snap["result"]["revision"], settings::ABSENT_REVISION);
    }

    #[test]
    fn validate_save_and_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx(&dir);
        let bad = call(
            &ctx,
            json!({"v":1,"id":1,"op":"validate","params":{"changes":{"overlay.font_size":0}}}),
        );
        assert_eq!(bad["result"]["errors"][0]["field"], "overlay.font_size");

        let saved = call(
            &ctx,
            json!({"v":1,"id":2,"op":"save","params":{"revision":"absent","changes":{"overlay.font_size":30}}}),
        );
        assert_eq!(saved["ok"], true, "{saved}");
        let stale = call(
            &ctx,
            json!({"v":1,"id":3,"op":"save","params":{"revision":"absent","changes":{"overlay.font_size":31}}}),
        );
        assert_eq!(stale["error"]["kind"], "conflict");
        assert_eq!(
            stale["error"]["current_revision"],
            saved["result"]["revision"]
        );
    }

    #[test]
    fn rejects_malformed_unknown_and_wrong_version() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx(&dir);
        assert_eq!(handle("{not json", &ctx)["error"]["kind"], "protocol");
        assert_eq!(
            call(&ctx, json!({"v":2,"id":1,"op":"hello"}))["error"]["kind"],
            "protocol"
        );
        assert_eq!(
            call(&ctx, json!({"v":1,"id":1,"op":"rm_rf"}))["error"]["kind"],
            "unknown_op"
        );
        assert_eq!(
            call(&ctx, json!({"v":1,"id":1,"op":"hello","extra":true}))["error"]["kind"],
            "protocol"
        );
        assert_eq!(
            call(
                &ctx,
                json!({"v":1,"id":4,"op":"save","params":{"changes":{"overlay.font_size":20}}})
            )["error"]["kind"],
            "protocol"
        );
    }

    #[test]
    fn serve_answers_each_line_and_bounds_request_size() {
        let dir = tempfile::tempdir().unwrap();
        let huge = json!({"v":1,"id":9,"op":"hello","pad":"x".repeat(MAX_REQUEST_BYTES)});
        let lines = run(
            ctx(&dir),
            &[
                json!({"v":1,"id":1,"op":"status"}),
                huge,
                json!({"v":1,"id":2,"op":"devices"}),
            ],
        );
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert_eq!(lines[0]["result"]["ui_ready"], true);
        assert_eq!(lines[1]["error"]["kind"], "protocol");
        assert_eq!(lines[2]["result"]["devices"][0]["name"], "Mic");
    }

    fn apply_request(id: u64, revision: &str, font_size: i64) -> Value {
        json!({"v":1,"id":id,"op":"apply","params":{"revision":revision,"changes":{"overlay.font_size":font_size}}})
    }

    #[test]
    fn apply_saves_then_restarts_with_progress_events() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let lines = run(ctx(&dir), &[apply_request(5, "absent", 30)]);
        let phases: Vec<_> = lines
            .iter()
            .filter(|l| l["event"] == "progress")
            .map(|l| l["phase"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(phases, ["saving", "restarting"]);
        let done = lines.last().unwrap();
        assert_eq!(done["id"], 5);
        assert_eq!(done["result"]["restart"]["outcome"], "ready");
        assert!(
            std::fs::read_to_string(path)
                .unwrap()
                .contains("font_size = 30")
        );
    }

    /// Call `apply_op` directly (no EOF-triggered cancel) and collect output.
    fn apply_direct(context: Context, request: Value) -> (Value, Vec<Value>) {
        let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let out: Output = buffer.clone();
        let response = apply_op(
            request["id"].as_u64().unwrap(),
            request["params"].clone(),
            &context,
            &out,
            &AtomicBool::new(false),
        );
        let bytes = buffer.lock().unwrap().clone();
        let events = String::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        (response, events)
    }

    #[test]
    fn apply_waits_while_busy_and_times_out_without_saving() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut context = ctx(&dir);
        context.status =
            Box::new(|| json!({ "active_state": "active", "stt": "recording", "tts": "paused" }));
        context.restart = Box::new(|| panic!("must not restart while busy"));
        let (response, events) = apply_direct(context, apply_request(6, "absent", 30));
        assert_eq!(events[0]["phase"], "waiting_idle");
        assert_eq!(events[0]["busy"], json!(["recording", "tts_paused"]));
        assert_eq!(events.len(), 1, "busy set announced once: {events:?}");
        assert_eq!(response["error"]["kind"], "busy");
        assert!(!path.exists(), "nothing is saved while waiting");
    }

    #[test]
    fn apply_proceeds_once_idle() {
        let dir = tempfile::tempdir().unwrap();
        let polls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut context = ctx(&dir);
        let counter = Arc::clone(&polls);
        context.idle_timeout = Duration::from_secs(5);
        context.status = Box::new(move || {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            let stt = if n < 10 { "processing" } else { "idle" };
            json!({ "active_state": "active", "stt": stt })
        });
        let (response, events) = apply_direct(context, apply_request(7, "absent", 30));
        let phases: Vec<_> = events.iter().map(|e| e["phase"].clone()).collect();
        assert_eq!(
            phases,
            [json!("waiting_idle"), json!("saving"), json!("restarting")]
        );
        assert_eq!(response["ok"], true, "{response}");
    }

    #[test]
    fn a_second_apply_is_rejected_while_one_runs() {
        let dir = tempfile::tempdir().unwrap();
        let mut context = ctx(&dir);
        context.idle_timeout = Duration::from_secs(5);
        context.status = Box::new(|| json!({ "active_state": "active", "stt": "recording" }));
        let lines = run(
            context,
            &[
                apply_request(7, "absent", 30),
                apply_request(8, "absent", 31),
            ],
        );
        let second = lines.iter().find(|l| l["id"] == 8).unwrap();
        assert_eq!(second["error"]["kind"], "busy");
        assert_eq!(second["error"]["message"], "another apply is in progress");
    }

    #[test]
    fn closing_the_app_cancels_a_pending_wait() {
        let dir = tempfile::tempdir().unwrap();
        let mut context = ctx(&dir);
        context.idle_timeout = Duration::from_secs(30);
        context.status = Box::new(|| json!({ "active_state": "active", "stt": "recording" }));
        let started = Instant::now();
        let lines = run(context, &[apply_request(9, "absent", 30)]);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(lines.last().unwrap()["error"]["kind"], "cancelled");
    }

    #[test]
    fn stopped_service_is_not_busy() {
        assert!(busy_activities(&json!({"active_state":"inactive","stt":"recording"})).is_empty());
        assert!(
            busy_activities(
                &json!({"active_state":"active","stt":"error:asr_disabled","tts":"error"})
            )
            .is_empty()
        );
    }
}
