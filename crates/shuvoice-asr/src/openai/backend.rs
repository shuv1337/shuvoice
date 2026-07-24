//! OpenAI Realtime WebSocket backend.

use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use shuvoice_core::{AsrBackendKind, AsrCapabilities};
use tokio::sync::{Mutex, Notify};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;

use super::protocol::{
    OPENAI_REALTIME_SAMPLE_RATE, OPENAI_REALTIME_WS_URL_DEFAULT, OpenAiProtocolState,
    append_audio_payload, clear_input_buffer_payload, commit_payload, redact_openai_error,
    session_update_payload,
};
use crate::backend::{AsrBackend, ProgressFn};
use crate::caps::openai_realtime_caps;
use crate::config::AsrConfig;
use crate::error::{AsrError, AsrResult};
use crate::pcm;

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

struct SharedState {
    protocol: OpenAiProtocolState,
    /// Cloned **before** any `.await` on the waiter path.
    notify: Arc<Notify>,
    /// Redacted last session/transport error from the reader task.
    session_error: Option<String>,
    /// Set when the reader exits or a send fails hard.
    transport_dead: bool,
}

pub struct OpenAiRealtimeBackend {
    config: AsrConfig,
    caps: AsrCapabilities,
    state: Arc<Mutex<SharedState>>,
    write: Option<Arc<Mutex<futures_util::stream::SplitSink<WsStream, Message>>>>,
    reader_task: Option<tokio::task::JoinHandle<()>>,
    loaded: bool,
}

impl OpenAiRealtimeBackend {
    pub fn new(config: AsrConfig) -> Self {
        Self {
            config,
            caps: openai_realtime_caps(),
            state: Arc::new(Mutex::new(SharedState {
                protocol: OpenAiProtocolState::default(),
                notify: Arc::new(Notify::new()),
                session_error: None,
                transport_dead: false,
            })),
            write: None,
            reader_task: None,
            loaded: false,
        }
    }

    fn ws_url(&self) -> String {
        self.config
            .connect
            .openai_realtime_ws_url
            .clone()
            .unwrap_or_else(|| OPENAI_REALTIME_WS_URL_DEFAULT.into())
    }

    fn default_commit_timeout(&self) -> Duration {
        Duration::from_secs_f64(self.config.core.openai_realtime_commit_timeout_sec.max(0.1))
    }

    async fn mark_transport_dead(&self, reason: impl Into<String>) {
        let mut guard = self.state.lock().await;
        guard.transport_dead = true;
        guard.session_error = Some(reason.into());
        guard.notify.notify_waiters();
    }

    async fn send_json(&self, value: &serde_json::Value) -> AsrResult<()> {
        {
            let guard = self.state.lock().await;
            if guard.transport_dead {
                return Err(AsrError::transport(
                    guard
                        .session_error
                        .clone()
                        .unwrap_or_else(|| "OpenAI Realtime transport is dead".into()),
                ));
            }
        }
        let write = self
            .write
            .as_ref()
            .ok_or_else(|| AsrError::transport("OpenAI Realtime WebSocket is not connected"))?;
        let mut guard = write.lock().await;
        let text = serde_json::to_string(value)?;
        if let Err(e) = guard.send(Message::Text(text.into())).await {
            drop(guard);
            let msg = format!("ws send failed: {e}");
            self.mark_transport_dead(msg.clone()).await;
            return Err(AsrError::transport(msg));
        }
        Ok(())
    }

    async fn connect_and_start_reader(&mut self) -> AsrResult<()> {
        let api_key =
            env::var(self.config.core.openai_realtime_api_key_env.trim()).map_err(|_| {
                AsrError::startup(format!(
                    "Missing OpenAI API key environment variable: {}",
                    self.config.core.openai_realtime_api_key_env
                ))
            })?;

        let url = self.ws_url();
        let mut request = url
            .as_str()
            .into_client_request()
            .map_err(|e| AsrError::transport(format!("invalid ws url: {e}")))?;
        let headers = request.headers_mut();
        headers.insert(
            "Authorization",
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|e| AsrError::internal(format!("invalid API key header: {e}")))?,
        );
        headers.insert("OpenAI-Beta", HeaderValue::from_static("realtime=v1"));

        let connect_timeout = Duration::from_secs_f64(
            self.config
                .core
                .openai_realtime_request_timeout_sec
                .max(0.1),
        );
        let (ws, _) =
            tokio::time::timeout(connect_timeout, tokio_tungstenite::connect_async(request))
                .await
                .map_err(|_| AsrError::RemoteTimeout(connect_timeout, "connect".into()))?
                .map_err(|e| AsrError::transport(format!("ws connect failed: {e}")))?;

        let (sink, mut stream) = ws.split();
        self.write = Some(Arc::new(Mutex::new(sink)));

        {
            let mut guard = self.state.lock().await;
            guard.transport_dead = false;
            guard.session_error = None;
            guard.protocol.reset();
        }

        let state = Arc::clone(&self.state);
        self.reader_task = Some(tokio::spawn(async move {
            let mut exit_reason = "OpenAI Realtime reader exited".to_string();
            while let Some(msg) = stream.next().await {
                let msg = match msg {
                    Ok(m) => m,
                    Err(e) => {
                        exit_reason = format!("ws read failed: {e}");
                        break;
                    }
                };
                let text = match msg {
                    Message::Text(t) => t.to_string(),
                    Message::Binary(b) => String::from_utf8_lossy(&b).into_owned(),
                    Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => continue,
                    Message::Close(_) => {
                        exit_reason = "ws closed by peer".into();
                        break;
                    }
                };
                let Ok(event) = serde_json::from_str::<serde_json::Value>(&text) else {
                    continue;
                };
                let mut guard = state.lock().await;
                if event.get("type").and_then(|v| v.as_str()) == Some("error") {
                    let redacted = redact_openai_error(&event);
                    tracing::error!(error = %redacted, "OpenAI Realtime error event");
                    guard.session_error = Some(redacted);
                    // Server error does not always kill the socket; wake waiters.
                    guard.notify.notify_waiters();
                    continue;
                }
                guard.protocol.handle_event(&event);
                if guard.protocol.completed {
                    guard.notify.notify_waiters();
                }
            }
            let mut guard = state.lock().await;
            guard.transport_dead = true;
            if guard.session_error.is_none() {
                guard.session_error = Some(exit_reason);
            }
            guard.notify.notify_waiters();
        }));

        let update = session_update_payload(
            &self.config.core.openai_realtime_model,
            &self.config.core.openai_realtime_language,
        );
        self.send_json(&update).await?;
        Ok(())
    }

    async fn ensure_live(&mut self) -> AsrResult<()> {
        let dead = {
            let guard = self.state.lock().await;
            guard.transport_dead || self.write.is_none()
        };
        if !dead {
            return Ok(());
        }
        // Tear down old reader/write then reconnect (explicit recovery).
        if let Some(task) = self.reader_task.take() {
            task.abort();
        }
        self.write = None;
        self.connect_and_start_reader().await?;
        self.loaded = true;
        Ok(())
    }
}

#[async_trait]
impl AsrBackend for OpenAiRealtimeBackend {
    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }

    fn backend_id(&self) -> AsrBackendKind {
        AsrBackendKind::OpenaiRealtime
    }

    fn native_chunk_samples(&self) -> usize {
        self.config.openai_native_chunk_samples()
    }

    fn required_sample_rate_hz(&self) -> Option<u32> {
        Some(OPENAI_REALTIME_SAMPLE_RATE)
    }

    async fn load(&mut self, progress: &mut ProgressFn<'_>) -> AsrResult<()> {
        self.config.validate_openai_startup()?;
        progress(Some(0.2), "connecting OpenAI Realtime");
        // Explicit sample-rate contract for hosts.
        if self.config.core.sample_rate != OPENAI_REALTIME_SAMPLE_RATE
            && self.config.sample_rate() != OPENAI_REALTIME_SAMPLE_RATE
        {
            // Capture path uses preferred_sample_rate from caps; warn only if host
            // forced 16k into config without reading caps.
            tracing::debug!(
                config_sr = self.config.core.sample_rate,
                required = OPENAI_REALTIME_SAMPLE_RATE,
                "OpenAI Realtime requires 24 kHz capture (see capabilities.preferred_sample_rate)"
            );
        }
        self.connect_and_start_reader().await?;
        self.loaded = true;
        progress(Some(1.0), "OpenAI Realtime ready");
        Ok(())
    }

    async fn reset(&mut self) -> AsrResult<()> {
        {
            let mut guard = self.state.lock().await;
            guard.protocol.reset();
            // Keep transport_dead/session_error — host must reconnect via ensure_live.
        }
        if self.loaded && self.write.is_some() {
            let dead = self.state.lock().await.transport_dead;
            if !dead {
                // Clear remote input buffer so the next utterance starts clean.
                let _ = self.send_json(&clear_input_buffer_payload()).await;
            }
        }
        Ok(())
    }

    async fn process_chunk(&mut self, pcm_mono_f32: &[f32]) -> AsrResult<String> {
        if !self.loaded {
            return Err(AsrError::internal("OpenAI backend not loaded"));
        }
        if pcm_mono_f32.iter().any(|s| !s.is_finite()) {
            return Err(AsrError::unsupported(
                "OpenAI Realtime rejects non-finite PCM samples",
            ));
        }
        // Host must supply 24 kHz; we do not resample here.
        self.ensure_live().await?;
        let b64 = pcm::encode_pcm16_le_b64(pcm_mono_f32);
        self.send_json(&append_audio_payload(&b64)).await?;
        let mut guard = self.state.lock().await;
        if guard.transport_dead {
            let err = guard
                .session_error
                .clone()
                .unwrap_or_else(|| "OpenAI Realtime transport died".into());
            return Err(AsrError::transport(err));
        }
        // Server `error` events do not always kill the socket; surface them
        // instead of silently returning a stale partial. Taken (not cloned) so
        // one error does not poison every later call on a live session.
        if let Some(err) = guard.session_error.take() {
            return Err(AsrError::transport(err));
        }
        Ok(guard.protocol.latest_partial.clone())
    }

    async fn finish_utterance(&mut self, timeout: Option<Duration>) -> AsrResult<String> {
        if !self.loaded {
            return Err(AsrError::internal("OpenAI backend not loaded"));
        }
        self.ensure_live().await?;
        let timeout = timeout.unwrap_or_else(|| self.default_commit_timeout());

        {
            let mut guard = self.state.lock().await;
            guard.protocol.begin_commit_wait();
        }
        self.send_json(&commit_payload()).await?;

        let deadline = Instant::now() + timeout;
        loop {
            // Snapshot wait handles **then drop the mutex** before awaiting.
            let (notify, completed, latest_final, latest_partial, transport_dead, session_error) = {
                let mut guard = self.state.lock().await;
                // On a live socket, take (don't clone) the error so one server
                // `error` event fails this commit without poisoning the next
                // utterance. On a dead transport the error stays sticky until
                // ensure_live reconnects.
                let session_error = if guard.transport_dead {
                    guard.session_error.clone()
                } else {
                    guard.session_error.take()
                };
                (
                    Arc::clone(&guard.notify),
                    guard.protocol.completed,
                    guard.protocol.latest_final.clone(),
                    guard.protocol.latest_partial.clone(),
                    guard.transport_dead,
                    session_error,
                )
            };

            if completed {
                if !latest_final.is_empty() {
                    return Ok(latest_final);
                }
                return Ok(latest_partial);
            }
            if transport_dead {
                return Err(AsrError::transport(
                    session_error.unwrap_or_else(|| "OpenAI Realtime transport died".into()),
                ));
            }
            // The server rejected the commit (or the session) but kept the
            // socket open — e.g. quota exhausted, revoked key, invalid model.
            // Fail fast instead of stalling until the deadline and reporting
            // an empty transcript as success.
            if let Some(err) = session_error {
                return Err(AsrError::transport(err));
            }
            if Instant::now() >= deadline {
                tracing::warn!("OpenAI Realtime commit timed out; using best partial");
                return Ok(latest_partial);
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            let slice = remaining.min(Duration::from_millis(50));
            // Mutex is not held across this await.
            let _ = tokio::time::timeout(slice, notify.notified()).await;
        }
    }

    async fn shutdown(&mut self) -> AsrResult<()> {
        if let Some(write) = self.write.take() {
            let mut guard = write.lock().await;
            let _ = guard.close().await;
        }
        if let Some(task) = self.reader_task.take() {
            task.abort();
        }
        let mut guard = self.state.lock().await;
        guard.transport_dead = true;
        guard.protocol.reset();
        self.loaded = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[tokio::test]
    async fn open_socket_error_event_fails_commit_fast() {
        // Regression: a server `error` event that keeps the socket open must
        // fail finish_utterance immediately (so it counts for the breaker),
        // not stall the full deadline and return Ok("").
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            while let Some(Ok(msg)) = ws.next().await {
                if let Message::Text(t) = msg {
                    let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) else {
                        continue;
                    };
                    if v.get("type").and_then(|s| s.as_str()) == Some("input_audio_buffer.commit") {
                        let err = serde_json::json!({
                            "type": "error",
                            "error": {"type": "insufficient_quota", "code": "insufficient_quota"}
                        });
                        // Socket stays open after the error event.
                        let _ = ws.send(Message::Text(err.to_string().into())).await;
                    }
                }
            }
        });

        // SAFETY: test-local variable name; no concurrent env readers race it.
        unsafe { env::set_var("SHUVOICE_TEST_OPENAI_ERR_EVT_KEY", "sk-test") };
        let mut config = AsrConfig::default();
        config.core.openai_realtime_api_key_env = "SHUVOICE_TEST_OPENAI_ERR_EVT_KEY".into();
        config.connect.openai_realtime_ws_url = Some(format!("ws://{addr}"));
        let mut backend = OpenAiRealtimeBackend::new(config);
        let mut progress: Box<ProgressFn<'_>> = Box::new(|_, _| {});
        backend.load(&mut progress).await.unwrap();

        let t0 = Instant::now();
        let res = backend.finish_utterance(Some(Duration::from_secs(5))).await;
        assert!(res.is_err(), "open-socket error event must fail the commit");
        assert!(
            t0.elapsed() < Duration::from_secs(2),
            "commit must fail fast, not stall the deadline"
        );
        // The socket never closed, so the transport must still be live and the
        // surfaced error must not poison later utterances.
        let guard = backend.state.lock().await;
        assert!(!guard.transport_dead);
        assert!(
            guard.session_error.is_none(),
            "error must be taken, not sticky"
        );
        drop(guard);
        server.abort();
    }

    #[tokio::test]
    async fn commit_waiter_does_not_hold_mutex_across_notify_await() {
        // Regression: notified().await must not run while MutexGuard is live.
        let backend = OpenAiRealtimeBackend::new(AsrConfig::default());
        let state = Arc::clone(&backend.state);
        let held = Arc::new(AtomicBool::new(false));
        let held2 = Arc::clone(&held);

        // Spawn a task that waits like finish_utterance.
        let waiter = tokio::spawn(async move {
            let notify = {
                let guard = state.lock().await;
                Arc::clone(&guard.notify)
            };
            // If the mutex were still held, the notifier task below would block.
            held2.store(true, Ordering::SeqCst);
            notify.notified().await;
            held2.store(false, Ordering::SeqCst);
        });

        // Yield so waiter reaches notified().await without the lock.
        tokio::task::yield_now().await;
        assert!(
            held.load(Ordering::SeqCst),
            "waiter should be parked on notify"
        );

        {
            let guard = backend.state.lock().await;
            guard.notify.notify_waiters();
        }
        waiter.await.unwrap();
        assert!(!held.load(Ordering::SeqCst));
    }
}
