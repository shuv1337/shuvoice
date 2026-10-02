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
use shuvoice_core::settings::{self, ApplyError, Section};

use crate::error::{EXIT_SUCCESS, ExitStatus};

pub const PROTOCOL_VERSION: u64 = 1;
/// Requests larger than this are rejected (and the rest of the line discarded).
pub const MAX_REQUEST_BYTES: usize = 256 * 1024;
const SERVICE: &str = "shuvoice.service";

fn supported_features() -> Vec<&'static str> {
    let features = vec![
        "capabilities",
        "corrections_preview",
        "onboarding_defaults",
        "shortcut_get",
        "shortcut_set",
        "models",
        "model_download",
    ];
    #[cfg(feature = "audio")]
    {
        let mut features = features;
        features.push("output_devices");
        features
    }
    #[cfg(not(feature = "audio"))]
    {
        features
    }
}

/// Section order for the sidebar.
const SECTIONS: [Section; 7] = [
    Section::Speech,
    Section::Vocabulary,
    Section::Typing,
    Section::TextToSpeech,
    Section::Audio,
    Section::Appearance,
    Section::Advanced,
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
    onboarding: bool,
    #[serde(default)]
    revision: Option<String>,
    #[serde(default)]
    changes: BTreeMap<String, Value>,
}

/// Side-effect seams so the protocol is testable without the live service.
pub struct Context {
    /// Only validated inventories can authorize download destinations.
    pub model_catalog: Mutex<BTreeMap<String, Config>>,
    pub config_path: PathBuf,
    pub env_present: Box<dyn Fn(&str) -> bool + Send + Sync>,
    pub status: Box<dyn Fn() -> Value + Send + Sync>,
    pub devices: Box<dyn Fn() -> Result<Value, String> + Send + Sync>,
    /// Start/restart the service and wait for readiness; returns the outcome.
    pub restart: Box<dyn Fn() -> Value + Send + Sync>,
    pub reserve: Box<dyn Fn() -> Result<u64, String> + Send + Sync>,
    pub release: Box<dyn Fn(u64) + Send + Sync>,
    pub marker: Box<dyn Fn() -> Result<(), String> + Send + Sync>,
    pub idle_poll: Duration,
    pub idle_timeout: Duration,
}

impl Context {
    pub fn live() -> Self {
        let leases = Arc::new(Mutex::new(BTreeMap::<u64, Option<String>>::new()));
        let release_leases = leases.clone();
        Self {
            model_catalog: Mutex::new(BTreeMap::new()),
            config_path: Config::config_path(),
            env_present: Box::new(|name| std::env::var_os(name).is_some_and(|v| !v.is_empty())),
            status: Box::new(live_status),
            devices: Box::new(|| {
                let devices = crate::commands::audio::input_devices()?;
                let outputs = crate::commands::audio::output_devices()?;
                Ok(json!({ "devices": devices, "outputs": outputs }))
            }),
            reserve: Box::new(move || {
                let socket = saved_control_socket();
                let response = send_control_command(
                    ControlCommand::MaintenanceReserve,
                    socket.as_deref(),
                    Some(Duration::from_millis(600)),
                )
                .map_err(|e| e.to_string())?;
                if response.starts_with("OK reserved") {
                    let token = response
                        .split_whitespace()
                        .find_map(|part| {
                            part.strip_prefix("token=")
                                .and_then(|token| token.parse::<u64>().ok())
                        })
                        .filter(|token| *token > 0)
                        .ok_or("service returned an invalid maintenance token")?;
                    leases
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .insert(token, socket);
                    Ok(token)
                } else {
                    Err(response)
                }
            }),
            release: Box::new(move |token| {
                if let Some(socket) = release_leases
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .remove(&token)
                {
                    let _ = send_control_command(
                        ControlCommand::MaintenanceRelease(token),
                        socket.as_deref(),
                        Some(Duration::from_millis(600)),
                    );
                }
            }),
            marker: Box::new(|| {
                shuvoice_ui::wizard_controller::write_wizard_marker()
                    .map(drop)
                    .map_err(|e| e.to_string())
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

fn saved_control_socket() -> Option<String> {
    settings::draft(Config::config_path(), &BTreeMap::new())
        .ok()
        .and_then(|config| config.control_socket)
}

fn context_snapshot(ctx: &Context) -> Result<settings::Snapshot, String> {
    let mut snapshot = settings::snapshot(&ctx.config_path, &ctx.env_present)?;
    for secret in &mut snapshot.secrets {
        if secret.source == Some("env")
            && let Some(source) = shuvoice_io::env_loader::loaded_env_source(&secret.env)
        {
            secret.source = Some(source);
        }
    }
    Ok(snapshot)
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
    let socket = saved_control_socket();
    let debug = send_control_command(
        ControlCommand::DebugStatus,
        socket.as_deref(),
        Some(Duration::from_millis(500)),
    )
    .ok()
    .and_then(|line| {
        line.strip_prefix("OK ")
            .and_then(|body| serde_json::from_str::<Value>(body).ok())
    })
    .unwrap_or(Value::Null);
    let app = &debug["app"];
    let stt = if app["recording"] == true || app["start_pending"] == true {
        "recording"
    } else if app["processing"] == true || app["finalizing"] == true {
        "processing"
    } else if app.is_object() {
        "idle"
    } else {
        "unknown"
    };
    json!({
        "service": SERVICE,
        "active_state": shuvoice_io::waybar::service_active_state(SERVICE, None),
        "ui_ready": debug["ui_ready"],
        "stt": stt,
        "tts": app["tts_player_state"],
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
                "features": supported_features(),
            }),
        ),
        "schema" => ok(
            id,
            json!({ "sections": SECTIONS, "fields": settings::schema_fields(), "excluded": settings::EXCLUDED.iter().map(|(id, reason)| json!({"id":id,"reason":reason})).collect::<Vec<_>>() }),
        ),
        "onboarding_defaults" => match context_snapshot(ctx) {
            Ok(snapshot) => ok(
                id,
                json!({"values": settings::onboarding_defaults(&snapshot)}),
            ),
            Err(message) => err(Some(id), "io", message, Value::Null),
        },
        "capabilities" | "models" => {
            let p: ChangeParams = match params(request.params) {
                Ok(p) => p,
                Err(message) => return err(Some(id), "protocol", message, Value::Null),
            };
            match settings::draft(&ctx.config_path, &p.changes) {
                Ok(config) if request.op == "capabilities" => ok(
                    id,
                    json!({"vocabulary_hints": settings::vocabulary_capability(&config)}),
                ),
                Ok(config) => {
                    let inventory = model_inventory(&config);
                    if let Some(required) = inventory["required"].as_array() {
                        let mut catalog =
                            ctx.model_catalog.lock().unwrap_or_else(|p| p.into_inner());
                        catalog.clear();
                        for item in required {
                            if let Some(id) = item["id"].as_str() {
                                catalog.insert(id.into(), config.clone());
                            }
                        }
                    }
                    ok(id, inventory)
                }
                Err(error) => apply_error(id, error),
            }
        }
        "corrections_preview" => {
            #[derive(Default, Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Preview {
                text: String,
                #[serde(default)]
                changes: BTreeMap<String, Value>,
            }
            let p: Preview = match params(request.params) {
                Ok(p) => p,
                Err(message) => return err(Some(id), "protocol", message, Value::Null),
            };
            match settings::draft(&ctx.config_path, &p.changes) {
                Ok(config) => {
                    let options = shuvoice_core::postprocess::RenderOptions {
                        text_case: config.typing_text_case,
                        auto_capitalize: config.auto_capitalize,
                        replacements: config.compiled_text_replacements,
                    };
                    ok(
                        id,
                        json!({"output": shuvoice_core::postprocess::render_transcript_text(&p.text, &options), "builtins": *shuvoice_core::config::DEFAULT_TEXT_REPLACEMENTS}),
                    )
                }
                Err(error) => apply_error(id, error),
            }
        }
        "shortcut_get" => ok(
            id,
            shuvoice_ui::wizard_controller::settings_shortcut_get(
                &shuvoice_ui::wizard_controller::hyprland_config_candidates(),
            ),
        ),
        "shortcut_set" => {
            #[derive(Default, Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Shortcut {
                id: String,
                dry_run: bool,
            }
            let p: Shortcut = match params(request.params) {
                Ok(p) => p,
                Err(message) => return err(Some(id), "protocol", message, Value::Null),
            };
            ok(
                id,
                shuvoice_ui::wizard_controller::settings_shortcut_set(
                    &shuvoice_ui::wizard_controller::hyprland_config_candidates(),
                    &p.id,
                    p.dry_run,
                    &shuvoice_ui::wizard_controller::resolve_shuvoice_command(),
                ),
            )
        }
        "snapshot" => match context_snapshot(ctx) {
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
    match settings::draft(&ctx.config_path, changes) {
        Ok(_) => Ok(Vec::new()),
        Err(ApplyError::Invalid { errors }) => Ok(errors),
        Err(error) => Err(format!("{error:?}")),
    }
}

fn apply_error(id: u64, error: ApplyError) -> Value {
    match error {
        ApplyError::Conflict { current_revision } => err(
            Some(id),
            "conflict",
            "the config file changed since it was loaded",
            json!({"current_revision":current_revision}),
        ),
        ApplyError::Invalid { errors } => err(
            Some(id),
            "invalid",
            "some settings are invalid",
            json!({"errors":errors}),
        ),
        ApplyError::Io { message } => err(Some(id), "io", message, Value::Null),
    }
}

struct Reservation<'a>(&'a Context, u64, Instant);
impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        (self.0.release)(self.1);
    }
}

fn model_inventory(config: &Config) -> Value {
    use shuvoice_core::{AsrBackendKind, TtsBackendKind};
    let mut required = Vec::new();
    match config.asr_backend {
        AsrBackendKind::Sherpa => required.push(json!({"id":format!("sherpa:{}",config.sherpa_model_name),"label":config.sherpa_model_name,"installed":crate::setup::sherpa_model::is_complete_sherpa_dir(&crate::setup::sherpa_model::sherpa_model_dir(config)),"size_hint":null})),
        AsrBackendKind::Nemo | AsrBackendKind::Moonshine => required.push(json!({"id":format!("worker:{}",config.asr_backend.as_str()),"label":"Worker-managed model (downloaded on first worker load)","installed":false,"size_hint":null})),
        AsrBackendKind::OpenaiRealtime => {},
    }
    if config.tts_enabled && config.tts_backend == TtsBackendKind::Local {
        let dir = config
            .tts_local_model_path
            .as_ref()
            .map(shuvoice_core::expand_user_path)
            .unwrap_or_else(crate::setup::piper::managed_piper_model_dir);
        let voice = config
            .tts_local_voice
            .as_deref()
            .unwrap_or(crate::setup::piper::recommended_piper_voice().stem);
        required.push(json!({"id":format!("piper:{voice}"),"label":format!("Piper {voice}"),"installed":crate::setup::piper::validate_piper_voice_artifacts(&dir,Some(voice)).is_ok(),"size_hint":null}));
    }
    json!({"required":required})
}

fn model_download_op(
    id: u64,
    value: Value,
    ctx: &Context,
    out: &Output,
    cancel: &Arc<AtomicBool>,
) -> Value {
    #[derive(Default, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Download {
        id: String,
    }
    let p: Download = match params(value) {
        Ok(p) => p,
        Err(message) => return err(Some(id), "protocol", message, Value::Null),
    };
    let config = ctx
        .model_catalog
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get(&p.id)
        .cloned();
    let Some(config) = config else {
        return err(
            Some(id),
            "invalid",
            "Request models for the draft before downloading this id",
            Value::Null,
        );
    };
    let mut progress_callback = |fraction: Option<f32>, text: &str| {
        emit_cancellable(
            out,
            &progress(id, "downloading", json!({"fraction":fraction,"text":text})),
            cancel,
        )
    };
    if cancel.load(Ordering::Acquire) {
        return err(Some(id), "cancelled", "Download cancelled", Value::Null);
    }
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => return err(Some(id), "io", error.to_string(), Value::Null),
    };
    let result = runtime.block_on(async {
        if p.id.starts_with("sherpa:") {
            crate::setup::sherpa_model::download_sherpa_model_cancellable(&config, None, &mut progress_callback, Some(Arc::clone(cancel))).await.map(drop)
        } else if let Some(voice) = p.id.strip_prefix("piper:") {
            let voice = crate::setup::piper::get_curated_piper_voice(voice)?;
            let dir = config.tts_local_model_path.as_ref().map(shuvoice_core::expand_user_path).unwrap_or_else(crate::setup::piper::managed_piper_model_dir);
            let downloader = crate::setup::http::ReqwestDownloader::default();
            let future = crate::setup::piper::ensure_local_piper_ready(voice, &dir, false, &downloader, &shuvoice_io::process::StdCommandRunner, &mut progress_callback);
            tokio::pin!(future);
            loop {
                tokio::select! {
                    result = &mut future => break result.and_then(|result| if result.status == "ok" { Ok(()) } else { Err(result.message) }),
                    _ = tokio::time::sleep(Duration::from_millis(20)) => if cancel.load(Ordering::Acquire) { break Err("Download cancelled".into()); },
                }
            }
        } else { Err("This model is worker-managed; no standalone download adapter is implemented".into()) }
    });
    match result {
        Ok(()) => ok(id, json!({"id":p.id,"installed":true})),
        Err(message) if cancel.load(Ordering::Acquire) => {
            err(Some(id), "cancelled", message, Value::Null)
        }
        Err(message) => err(Some(id), "unavailable", message, Value::Null),
    }
}

/// Shared, line-atomic writer for responses and events from several threads.
pub type Output = Arc<Mutex<dyn Write + Send>>;

fn emit(out: &Output, message: &Value) {
    let _ = try_emit(out, message);
}

fn emit_cancellable(out: &Output, message: &Value, cancel: &AtomicBool) {
    if !try_emit(out, message) {
        cancel.store(true, Ordering::Release);
    }
}

fn try_emit(out: &Output, message: &Value) -> bool {
    let mut out = out.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    serde_json::to_writer(&mut *out, message).is_ok()
        && out.write_all(b"\n").is_ok()
        && out.flush().is_ok()
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
    if p.changes.is_empty() && !p.onboarding {
        return err(
            Some(id),
            "protocol",
            "apply requires at least one change",
            Value::Null,
        );
    }
    emit_cancellable(out, &progress(id, "validating", Value::Null), cancel);
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
    emit_cancellable(
        out,
        &progress(id, "waiting_idle", json!({"busy": []})),
        cancel,
    );
    let reservation;
    loop {
        if cancel.load(Ordering::Acquire) {
            return err(
                Some(id),
                "cancelled",
                "apply cancelled; nothing was saved",
                Value::Null,
            );
        }
        let status = (ctx.status)();
        let busy = busy_activities(&status);
        if busy != announced {
            emit_cancellable(
                out,
                &progress(id, "waiting_idle", json!({ "busy": busy })),
                cancel,
            );
            announced = busy;
        }
        if announced.is_empty() {
            emit_cancellable(out, &progress(id, "reserving", Value::Null), cancel);
            if cancel.load(Ordering::Acquire) {
                return err(
                    Some(id),
                    "cancelled",
                    "apply cancelled; nothing was saved",
                    Value::Null,
                );
            }
            match status.get("active_state").and_then(Value::as_str) {
                Some("inactive" | "failed" | "dead") => {
                    reservation = None;
                    break;
                }
                Some("active") => match (ctx.reserve)() {
                    Ok(token) => {
                        reservation = Some(Reservation(ctx, token, Instant::now()));
                        break;
                    }
                    Err(message) if message.starts_with("ERROR busy") => {}
                    Err(message) => {
                        return err(
                            Some(id),
                            "unavailable",
                            format!("cannot reserve running service: {message}; nothing was saved"),
                            Value::Null,
                        );
                    }
                },
                _ => {
                    return err(
                        Some(id),
                        "unavailable",
                        "service state is unknown; nothing was saved",
                        Value::Null,
                    );
                }
            }
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

    if cancel.load(Ordering::Acquire) {
        return err(
            Some(id),
            "cancelled",
            "apply cancelled; nothing was saved",
            Value::Null,
        );
    }
    emit_cancellable(out, &progress(id, "saving", Value::Null), cancel);
    if cancel.load(Ordering::Acquire) {
        return err(
            Some(id),
            "cancelled",
            "apply cancelled; nothing was saved",
            Value::Null,
        );
    }
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

    if p.onboarding
        && let Err(message) = (ctx.marker)()
    {
        return err(Some(id), "io", message, json!({"saved": saved}));
    }

    emit(out, &progress(id, "restarting", Value::Null));
    if reservation
        .as_ref()
        .is_some_and(|lease| lease.2.elapsed() >= Duration::from_secs(60))
    {
        return ok(
            id,
            json!({"saved":saved,"restart":{"outcome":"readiness_failed","action":"restart","message":"Save exceeded the maintenance safety budget; saved but not restarted. Apply again when idle."}}),
        );
    }
    let restart = (ctx.restart)();
    drop(reservation);
    ok(id, json!({ "saved": saved, "restart": restart }))
}

/// Serve requests from `input` until EOF. `apply` runs on a worker thread
/// (one at a time); `cancel` stops an apply that is still waiting for idle.
struct WorkerSlot {
    cancel: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Drop for WorkerSlot {
    fn drop(&mut self) {
        // Covers EOF AND an input read error; no detached apply on disconnect.
        self.cancel.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub fn serve(mut input: impl BufRead, out: Output, ctx: Arc<Context>) -> std::io::Result<()> {
    let cancel = Arc::new(AtomicBool::new(false));
    let mut slot = WorkerSlot {
        cancel: cancel.clone(),
        worker: None,
    };
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
            loop {
                sink.clear();
                let read = (&mut input).take(4096).read_until(b'\n', &mut sink)?;
                if read == 0 || sink.last() == Some(&b'\n') {
                    break;
                }
            }
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
        let running = slot.worker.as_ref().is_some_and(|w| !w.is_finished());
        match request.as_ref().map(|r| (r.v, r.op.as_str())) {
            Some((PROTOCOL_VERSION, "apply" | "model_download")) if running => {
                let id = request.as_ref().map(|r| r.id);
                emit(
                    &out,
                    &err(id, "busy", "another apply is in progress", Value::Null),
                );
            }
            Some((PROTOCOL_VERSION, "apply" | "model_download")) => {
                let Request { id, params, op, .. } = request.expect("matched Some");
                cancel.store(false, Ordering::Release);
                let (ctx, out, cancel) = (Arc::clone(&ctx), Arc::clone(&out), Arc::clone(&cancel));
                slot.worker = Some(std::thread::spawn(move || {
                    let response = if op == "apply" {
                        apply_op(id, params, &ctx, &out, &cancel)
                    } else {
                        model_download_op(id, params, &ctx, &out, &cancel)
                    };
                    emit(&out, &response);
                }));
            }
            Some((PROTOCOL_VERSION, "cancel")) => {
                cancel.store(true, Ordering::Release);
                let id = request.expect("matched Some").id;
                emit(&out, &ok(id, json!({ "cancelling": running })));
            }
            _ => {
                if !try_emit(&out, &handle(text, &ctx)) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "settings app disconnected",
                    ));
                }
            }
        }
    }
    // The app closed: cancel a pending wait, let a started restart finish.
    drop(slot);
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
    use shuvoice_core::settings::FIELDS;

    fn ctx(dir: &tempfile::TempDir) -> Context {
        Context {
            model_catalog: Mutex::new(BTreeMap::new()),
            config_path: dir.path().join("config.toml"),
            env_present: Box::new(|name| name == "OPENAI_API_KEY"),
            status: Box::new(
                || json!({ "active_state": "active", "ui_ready": true, "stt": "idle" }),
            ),
            devices: Box::new(|| Ok(json!({ "devices": [{ "index": 0, "name": "Mic" }] }))),
            restart: Box::new(|| json!({ "outcome": "ready", "action": "restart" })),
            reserve: Box::new(|| Ok(1)),
            release: Box::new(|_| {}),
            marker: Box::new(|| Ok(())),
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
        let (done, lines) = apply_direct(ctx(&dir), apply_request(5, "absent", 30));
        let phases: Vec<_> = lines
            .iter()
            .filter(|l| l["event"] == "progress")
            .map(|l| l["phase"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            phases,
            [
                "validating",
                "waiting_idle",
                "reserving",
                "saving",
                "restarting"
            ]
        );
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
        assert_eq!(events[0]["phase"], "validating");
        assert_eq!(events[2]["phase"], "waiting_idle");
        assert_eq!(events[2]["busy"], json!(["recording", "tts_paused"]));
        assert_eq!(
            events.len(),
            3,
            "busy set announced once after initial phase: {events:?}"
        );
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
            [
                json!("validating"),
                json!("waiting_idle"),
                json!("waiting_idle"),
                json!("waiting_idle"),
                json!("reserving"),
                json!("saving"),
                json!("restarting")
            ]
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
    fn output_disconnect_cancels_before_the_save_boundary() {
        struct Disconnected;
        impl Write for Disconnected {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let context = ctx(&dir);
        let out: Output = Arc::new(Mutex::new(Disconnected));
        let response = apply_op(
            1,
            apply_request(1, "absent", 30)["params"].clone(),
            &context,
            &out,
            &AtomicBool::new(false),
        );
        assert_eq!(response["error"]["kind"], "cancelled");
        assert!(!context.config_path.exists());
    }

    #[test]
    fn input_read_error_cancels_and_joins_a_pending_worker() {
        struct FailingInput(std::io::Cursor<Vec<u8>>);
        impl Read for FailingInput {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                if self.0.position() as usize == self.0.get_ref().len() {
                    return Err(std::io::Error::other("fixture disconnect"));
                }
                self.0.read(out)
            }
        }
        impl BufRead for FailingInput {
            fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
                if self.0.position() as usize == self.0.get_ref().len() {
                    return Err(std::io::Error::other("fixture disconnect"));
                }
                self.0.fill_buf()
            }
            fn consume(&mut self, amount: usize) {
                self.0.consume(amount);
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let mut context = ctx(&dir);
        context.status = Box::new(|| json!({"active_state":"active","stt":"recording"}));
        let input = FailingInput(std::io::Cursor::new(
            format!("{}\n", apply_request(1, "absent", 30)).into_bytes(),
        ));
        let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let out: Output = buffer.clone();
        let started = Instant::now();
        assert!(serve(input, out, Arc::new(context)).is_err());
        assert!(started.elapsed() < Duration::from_secs(1));
        let output = String::from_utf8(buffer.lock().unwrap().clone()).unwrap();
        assert!(output.contains("cancelled"), "{output}");
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

    #[test]
    fn race_after_idle_probe_cannot_save_until_actor_grants_reservation() {
        let dir = tempfile::tempdir().unwrap();
        let mut context = ctx(&dir);
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = attempts.clone();
        context.reserve = Box::new(move || {
            if counter.fetch_add(1, Ordering::SeqCst) == 0 {
                Err("ERROR busy: recording won race".into())
            } else {
                Ok(1)
            }
        });
        let release = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = release.clone();
        context.release = Box::new(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        let (response, _) = apply_direct(context, apply_request(1, "absent", 30));
        assert_eq!(response["ok"], true);
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(release.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failed_save_releases_reservation_and_never_restarts() {
        let dir = tempfile::tempdir().unwrap();
        let mut context = ctx(&dir);
        context.restart = Box::new(|| panic!("must not restart after failed save"));
        let releases = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = releases.clone();
        context.release = Box::new(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        let (response, _) = apply_direct(context, apply_request(1, "stale", 30));
        assert_eq!(response["error"]["kind"], "conflict");
        assert_eq!(releases.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn io_failure_during_save_releases_reservation() {
        let dir = tempfile::tempdir().unwrap();
        let mut context = ctx(&dir);
        let path = context.config_path.clone();
        context.reserve = Box::new(move || {
            std::fs::create_dir(&path).unwrap();
            Ok(1)
        });
        let released = Arc::new(AtomicBool::new(false));
        let flag = released.clone();
        context.release = Box::new(move |_| {
            flag.store(true, Ordering::Release);
        });
        context.restart = Box::new(|| panic!("must not restart after I/O save failure"));
        let (response, _) = apply_direct(context, apply_request(1, "absent", 30));
        assert_eq!(response["error"]["kind"], "io");
        assert!(released.load(Ordering::Acquire));
    }

    #[test]
    fn cancellation_after_reserving_releases_without_saving() {
        let dir = tempfile::tempdir().unwrap();
        let mut context = ctx(&dir);
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        context.reserve = Box::new(move || {
            flag.store(true, Ordering::Release);
            Ok(1)
        });
        let released = Arc::new(AtomicBool::new(false));
        let flag = released.clone();
        context.release = Box::new(move |_| {
            flag.store(true, Ordering::Release);
        });
        let out: Output = Arc::new(Mutex::new(Vec::<u8>::new()));
        let request = apply_request(1, "absent", 30);
        let response = apply_op(1, request["params"].clone(), &context, &out, &cancel);
        assert_eq!(response["error"]["kind"], "cancelled");
        assert!(released.load(Ordering::Acquire));
        assert!(!context.config_path.exists());
    }

    #[test]
    fn stopped_service_skips_reservation_and_onboarding_marks_before_restart() {
        let dir = tempfile::tempdir().unwrap();
        let mut context = ctx(&dir);
        context.status = Box::new(|| json!({"active_state":"inactive"}));
        context.reserve = Box::new(|| panic!("stopped service must not reserve"));
        let marked = Arc::new(AtomicBool::new(false));
        let flag = marked.clone();
        context.marker = Box::new(move || {
            flag.store(true, Ordering::Release);
            Ok(())
        });
        context.restart = Box::new(move || {
            assert!(marked.load(Ordering::Acquire));
            json!({"outcome":"ready"})
        });
        let (response, _) = apply_direct(
            context,
            json!({"id":1,"params":{"revision":"absent","changes":{},"onboarding":true}}),
        );
        assert_eq!(response["ok"], true, "{response}");
    }

    #[test]
    fn optional_ops_use_real_core_policy_and_feature_detection() {
        let dir = tempfile::tempdir().unwrap();
        let context = ctx(&dir);
        let features = call(&context, json!({"v":1,"id":1,"op":"hello"}));
        assert!(
            features["result"]["features"]
                .as_array()
                .unwrap()
                .contains(&json!("models"))
        );
        let preview = call(
            &context,
            json!({"v":1,"id":2,"op":"corrections_preview","params":{"text":"shove voice meets hyper land","changes":{"typing.typing_text_case":"lowercase"}}}),
        );
        assert_eq!(preview["result"]["output"], "shuvoice meets hyprland");
        assert_eq!(preview["result"]["builtins"]["shove voice"], "ShuVoice");
        let capability = call(
            &context,
            json!({"v":1,"id":3,"op":"capabilities","params":{"changes":{"asr.asr_backend":"openai_realtime"}}}),
        );
        assert_eq!(capability["result"]["vocabulary_hints"]["supported"], true);
        let models = call(
            &context,
            json!({"v":1,"id":4,"op":"models","params":{"changes":{"asr.asr_backend":"openai_realtime","tts.tts_enabled":false}}}),
        );
        assert_eq!(models["result"]["required"], json!([]));
    }

    #[test]
    fn model_download_shares_slot_with_apply_and_invalid_ids_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut context = ctx(&dir);
        context.status = Box::new(|| json!({"active_state":"active","stt":"recording"}));
        let lines = run(
            context,
            &[
                apply_request(1, "absent", 30),
                json!({"v":1,"id":2,"op":"model_download","params":{"id":"../bad"}}),
            ],
        );
        assert_eq!(
            lines.iter().find(|line| line["id"] == 2).unwrap()["error"]["kind"],
            "busy"
        );
        let context = ctx(&dir);
        let out: Output = Arc::new(Mutex::new(Vec::<u8>::new()));
        assert_eq!(
            model_download_op(
                3,
                json!({"id":"../bad"}),
                &context,
                &out,
                &Arc::new(AtomicBool::new(false))
            )["error"]["kind"],
            "invalid"
        );
    }

    #[test]
    fn model_download_installed_fixture_progress_and_cancel() {
        let dir = tempfile::tempdir().unwrap();
        let context = ctx(&dir);
        let model_dir = dir.path().join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        for name in ["tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"] {
            std::fs::write(model_dir.join(name), b"fixture").unwrap();
        }
        let inventory = call(
            &context,
            json!({"v":1,"id":1,"op":"models","params":{"changes":{"asr.sherpa_model_dir":model_dir.display().to_string(),"tts.tts_enabled":false}}}),
        );
        let model = inventory["result"]["required"][0]["id"].as_str().unwrap();
        let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let out: Output = buffer.clone();
        let response = model_download_op(
            2,
            json!({"id":model}),
            &context,
            &out,
            &Arc::new(AtomicBool::new(false)),
        );
        assert_eq!(response["result"]["installed"], true, "{response}");
        let event: Value = serde_json::from_slice(&buffer.lock().unwrap()).unwrap();
        assert_eq!(event["phase"], "downloading");
        assert_eq!(event["fraction"], 1.0);
        let response = model_download_op(
            3,
            json!({"id":model}),
            &context,
            &out,
            &Arc::new(AtomicBool::new(true)),
        );
        assert_eq!(response["error"]["kind"], "cancelled");
    }
}
