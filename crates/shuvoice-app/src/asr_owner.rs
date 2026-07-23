//! Dedicated async ASR owner task.
//!
//! Owns `Box<dyn shuvoice_asr::AsrBackend>` exclusively. The session actor never
//! calls into the backend directly and never `block_on`s model work.
//!
//! # Caps / native chunk after load
//!
//! Worker backends may rewrite capabilities during [`AsrBackend::load`] (manifest).
//! Live values live behind [`AsrOwnerHandle`]'s shared [`AsrOwnerInfo`] and are
//! refreshed after successful `load` and CPU fallback. Use
//! [`AsrOwnerHandle::capabilities`] / [`native_chunk_samples`] (or [`info`])
//! after load — do not cache pre-load snapshots across `load`.
//!
//! # Caller timeouts and head-of-line blocking
//!
//! Round-trips use a **caller-side** `tokio::time::timeout` on the oneshot reply
//! only. When that fires:
//! - the wait is abandoned and `RemoteTimeout` is returned;
//! - the owner task **keeps running** the in-flight backend op until it finishes;
//! - later mailbox requests wait behind that op (head-of-line);
//! - the oneshot is dropped so a late reply is ignored;
//! - utterance generation tokens still prevent stale commits if the session moved on.
//!
//! Most backends have no cooperative cancel. [`AsrOwnerHandle::shutdown`] enqueues
//! `Shutdown` (runs after the current op). [`AsrOwnerJoin::join_timeout`] aborts
//! the owner task if shutdown does not finish within the grace period — the only
//! hard interrupt available.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::RwLock;
use shuvoice_asr::{AsrError, AsrResult, DynAsrBackend, FallbackOutcome};
use shuvoice_core::AsrCapabilities;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::types::{DEFAULT_ASR_OP_TIMEOUT, UtteranceGen};

/// Snapshot of owner-side diagnostics (no backend borrow across await).
#[derive(Debug, Clone)]
pub struct AsrOwnerInfo {
    pub caps: AsrCapabilities,
    pub native_chunk_samples: usize,
    pub debug_step: Option<u64>,
    pub cpu_fallback_applied: bool,
    pub alive: bool,
    /// True while the owner is inside a backend await (mailbox HOL risk).
    pub op_in_flight: bool,
}

fn info_from_backend(backend: &DynAsrBackend, alive: bool, op_in_flight: bool) -> AsrOwnerInfo {
    AsrOwnerInfo {
        caps: backend.capabilities().clone(),
        native_chunk_samples: backend.native_chunk_samples().max(1),
        debug_step: backend.debug_step(),
        cpu_fallback_applied: backend.cpu_fallback_applied(),
        alive,
        op_in_flight,
    }
}

enum AsrRequest {
    Load {
        reply: oneshot::Sender<AsrResult<()>>,
    },
    Reset {
        utt_gen: UtteranceGen,
        reply: oneshot::Sender<AsrResult<UtteranceGen>>,
    },
    ProcessChunk {
        utt_gen: UtteranceGen,
        pcm: Vec<f32>,
        reply: oneshot::Sender<AsrResult<(UtteranceGen, String)>>,
    },
    ProcessUtterance {
        utt_gen: UtteranceGen,
        pcm: Vec<f32>,
        reply: oneshot::Sender<AsrResult<(UtteranceGen, String)>>,
    },
    Finish {
        utt_gen: UtteranceGen,
        timeout: Option<Duration>,
        reply: oneshot::Sender<AsrResult<(UtteranceGen, String)>>,
    },
    Fallback {
        reply: oneshot::Sender<AsrResult<FallbackOutcome>>,
    },
    Info {
        reply: oneshot::Sender<AsrOwnerInfo>,
    },
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

/// Handle used by the session actor to talk to the ASR owner.
#[derive(Clone, Debug)]
pub struct AsrOwnerHandle {
    tx: mpsc::Sender<AsrRequest>,
    utt_gen: Arc<AtomicU64>,
    alive: Arc<AtomicBool>,
    /// Live caps/native_chunk — refreshed after load / CPU fallback.
    info: Arc<RwLock<AsrOwnerInfo>>,
    default_timeout: Duration,
}

impl AsrOwnerHandle {
    /// Lock-read snapshot of live owner info (cheap; post-load when load completed).
    pub fn snapshot_info(&self) -> AsrOwnerInfo {
        self.info.read().clone()
    }

    pub fn capabilities(&self) -> AsrCapabilities {
        self.info.read().caps.clone()
    }

    pub fn native_chunk_samples(&self) -> usize {
        self.info.read().native_chunk_samples.max(1)
    }

    pub fn wants_raw_audio(&self) -> bool {
        self.info.read().caps.wants_raw_audio
    }

    pub fn finalization_mode(&self) -> shuvoice_core::FinalizationMode {
        self.info.read().caps.finalization_mode
    }

    pub fn op_in_flight(&self) -> bool {
        self.info.read().op_in_flight
    }

    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }

    pub fn current_gen(&self) -> UtteranceGen {
        self.utt_gen.load(Ordering::Acquire)
    }

    /// Bump generation so in-flight results become stale for commit guards.
    pub fn bump_gen(&self) -> UtteranceGen {
        self.utt_gen.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub async fn load(&self) -> AsrResult<()> {
        self.roundtrip_unit(|reply| AsrRequest::Load { reply })
            .await
    }

    pub async fn reset(&self) -> AsrResult<UtteranceGen> {
        let utt_gen = self.bump_gen();
        self.roundtrip(DEFAULT_ASR_OP_TIMEOUT, move |reply| AsrRequest::Reset {
            utt_gen,
            reply,
        })
        .await
    }

    pub async fn process_chunk(
        &self,
        utt_gen: UtteranceGen,
        pcm: Vec<f32>,
    ) -> AsrResult<(UtteranceGen, String)> {
        self.roundtrip(self.default_timeout, move |reply| {
            AsrRequest::ProcessChunk {
                utt_gen,
                pcm,
                reply,
            }
        })
        .await
    }

    pub async fn process_utterance(
        &self,
        utt_gen: UtteranceGen,
        pcm: Vec<f32>,
    ) -> AsrResult<(UtteranceGen, String)> {
        self.roundtrip(self.default_timeout, move |reply| {
            AsrRequest::ProcessUtterance {
                utt_gen,
                pcm,
                reply,
            }
        })
        .await
    }

    pub async fn finish_utterance(
        &self,
        utt_gen: UtteranceGen,
        timeout: Option<Duration>,
    ) -> AsrResult<(UtteranceGen, String)> {
        let base = timeout.unwrap_or(self.default_timeout);
        let op_timeout = base
            .checked_add(Duration::from_millis(250))
            .unwrap_or(self.default_timeout)
            .max(self.default_timeout);
        self.roundtrip(op_timeout, move |reply| AsrRequest::Finish {
            utt_gen,
            timeout,
            reply,
        })
        .await
    }

    pub async fn try_fallback_to_cpu(&self) -> AsrResult<FallbackOutcome> {
        self.roundtrip(self.default_timeout, |reply| AsrRequest::Fallback { reply })
            .await
    }

    /// Query live info from the owner task (always reflects backend state).
    pub async fn info(&self) -> AsrResult<AsrOwnerInfo> {
        if !self.is_alive() {
            return Err(AsrError::internal("ASR owner task is dead"));
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(AsrRequest::Info { reply: reply_tx })
            .await
            .map_err(|_| AsrError::internal("ASR owner mailbox closed"))?;
        reply_rx
            .await
            .map_err(|_| AsrError::internal("ASR owner dropped info reply"))
    }

    /// Request orderly shutdown (queued behind any in-flight backend op).
    ///
    /// Pair with [`AsrOwnerJoin::join_timeout`] for a hard abort if the current
    /// backend op does not finish.
    pub async fn shutdown(&self) {
        self.request_shutdown();
        // Best-effort brief wait for owner to observe shutdown; runtime join owns the bound.
        let _ = tokio::time::timeout(Duration::from_millis(50), async {
            while self.is_alive() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
    }

    /// Non-blocking shutdown enqueue for the session actor (never awaits backend).
    ///
    /// Marks the handle not-alive immediately so gen guards fail closed. The owner
    /// task drains the Shutdown request after any in-flight op; [`AsrOwnerJoin::join_timeout`]
    /// provides the hard abort bound.
    pub fn request_shutdown(&self) {
        let (reply_tx, _reply_rx) = oneshot::channel();
        let _ = self.tx.try_send(AsrRequest::Shutdown { reply: reply_tx });
        self.alive.store(false, Ordering::Release);
        let mut info = self.info.write();
        info.alive = false;
        info.op_in_flight = false;
    }

    async fn roundtrip_unit<F>(&self, build: F) -> AsrResult<()>
    where
        F: FnOnce(oneshot::Sender<AsrResult<()>>) -> AsrRequest,
    {
        self.roundtrip(self.default_timeout, build).await
    }

    async fn roundtrip<T, F>(&self, timeout: Duration, build: F) -> AsrResult<T>
    where
        F: FnOnce(oneshot::Sender<AsrResult<T>>) -> AsrRequest,
        T: Send + 'static,
    {
        if !self.is_alive() {
            return Err(AsrError::internal("ASR owner task is dead"));
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(build(reply_tx))
            .await
            .map_err(|_| AsrError::internal("ASR owner mailbox closed"))?;
        match tokio::time::timeout(timeout, reply_rx).await {
            Ok(Ok(res)) => res,
            Ok(Err(_)) => Err(AsrError::internal("ASR owner dropped reply")),
            Err(_) => {
                // Caller abandoned the wait. Owner continues the backend op (HOL).
                // Late oneshot send fails. Gen guards remain the commit safety net;
                // we intentionally do not bump_gen (would nuke an active utterance
                // on one slow chunk).
                warn!(
                    ?timeout,
                    "ASR owner op timed out on caller side (backend may still run; HOL until it finishes)"
                );
                Err(AsrError::RemoteTimeout(
                    timeout,
                    "ASR owner op timed out (caller abandoned reply; backend op may still run)"
                        .into(),
                ))
            }
        }
    }
}

/// Join handle for the ASR owner task.
pub struct AsrOwnerJoin {
    join: JoinHandle<()>,
    alive: Arc<AtomicBool>,
}

impl AsrOwnerJoin {
    pub async fn join(self) {
        let _ = self.join.await;
        self.alive.store(false, Ordering::Release);
    }

    /// Wait up to `grace` for a clean exit, then **abort** the owner task.
    ///
    /// Abort is the only hard interrupt when a backend op ignores shutdown.
    /// In-flight FFI/worker work may be torn down uncleanly — preferred over hang.
    pub async fn join_timeout(mut self, grace: Duration) {
        tokio::select! {
            res = &mut self.join => {
                if let Err(err) = res {
                    warn!(%err, "ASR owner task ended with join error");
                }
            }
            _ = tokio::time::sleep(grace) => {
                warn!(
                    ?grace,
                    "ASR owner did not exit within grace; aborting task (hard interrupt)"
                );
                self.join.abort();
                let _ = (&mut self.join).await;
            }
        }
        self.alive.store(false, Ordering::Release);
    }

    pub fn is_finished(&self) -> bool {
        self.join.is_finished()
    }

    /// Abort the owner task immediately (hard interrupt).
    pub fn abort(self) {
        self.join.abort();
        self.alive.store(false, Ordering::Release);
    }
}

/// Spawn the single ASR owner task.
pub fn spawn_asr_owner(
    mut backend: Box<DynAsrBackend>,
    mailbox_capacity: usize,
    default_timeout: Duration,
) -> (AsrOwnerHandle, AsrOwnerJoin) {
    let initial = info_from_backend(backend.as_ref(), true, false);
    let info = Arc::new(RwLock::new(initial));
    let info_task = Arc::clone(&info);

    let (tx, mut rx) = mpsc::channel::<AsrRequest>(mailbox_capacity.max(8));
    let utt_gen = Arc::new(AtomicU64::new(0));
    let alive = Arc::new(AtomicBool::new(true));
    let alive_task = Arc::clone(&alive);

    let join = tokio::spawn(async move {
        let set_inflight = |info: &RwLock<AsrOwnerInfo>, v: bool| {
            info.write().op_in_flight = v;
        };
        let refresh = |backend: &DynAsrBackend, info: &RwLock<AsrOwnerInfo>, alive: bool| {
            *info.write() = info_from_backend(backend, alive, false);
        };

        while let Some(req) = rx.recv().await {
            match req {
                AsrRequest::Load { reply } => {
                    set_inflight(&info_task, true);
                    let mut progress = |_f: Option<f32>, _m: &str| {};
                    let res = backend.load(&mut progress).await;
                    // Manifest/load may change caps and native chunk size.
                    if res.is_ok() {
                        refresh(backend.as_ref(), &info_task, true);
                    } else {
                        set_inflight(&info_task, false);
                    }
                    let _ = reply.send(res);
                }
                AsrRequest::Reset { utt_gen, reply } => {
                    set_inflight(&info_task, true);
                    let res = backend.reset().await.map(|_| utt_gen);
                    set_inflight(&info_task, false);
                    let _ = reply.send(res);
                }
                AsrRequest::ProcessChunk {
                    utt_gen,
                    pcm,
                    reply,
                } => {
                    set_inflight(&info_task, true);
                    let res = backend
                        .process_chunk(&pcm)
                        .await
                        .map(|text| (utt_gen, text));
                    set_inflight(&info_task, false);
                    // Reply may be ignored if caller timed out (oneshot closed).
                    let _ = reply.send(res);
                }
                AsrRequest::ProcessUtterance {
                    utt_gen,
                    pcm,
                    reply,
                } => {
                    set_inflight(&info_task, true);
                    let res = backend
                        .process_utterance(&pcm)
                        .await
                        .map(|text| (utt_gen, text));
                    set_inflight(&info_task, false);
                    let _ = reply.send(res);
                }
                AsrRequest::Finish {
                    utt_gen,
                    timeout,
                    reply,
                } => {
                    set_inflight(&info_task, true);
                    let res = backend
                        .finish_utterance(timeout)
                        .await
                        .map(|text| (utt_gen, text));
                    set_inflight(&info_task, false);
                    let _ = reply.send(res);
                }
                AsrRequest::Fallback { reply } => {
                    set_inflight(&info_task, true);
                    let res = backend.try_fallback_to_cpu().await;
                    if matches!(res, Ok(FallbackOutcome::Applied { .. })) {
                        refresh(backend.as_ref(), &info_task, true);
                    } else {
                        set_inflight(&info_task, false);
                    }
                    let _ = reply.send(res);
                }
                AsrRequest::Info { reply } => {
                    let snap =
                        info_from_backend(backend.as_ref(), true, info_task.read().op_in_flight);
                    *info_task.write() = snap.clone();
                    let _ = reply.send(snap);
                }
                AsrRequest::Shutdown { reply } => {
                    set_inflight(&info_task, true);
                    if let Err(err) = backend.shutdown().await {
                        warn!(%err, "ASR backend shutdown returned error");
                    }
                    set_inflight(&info_task, false);
                    let mut g = info_task.write();
                    g.alive = false;
                    g.op_in_flight = false;
                    let _ = reply.send(());
                    break;
                }
            }
        }
        alive_task.store(false, Ordering::Release);
        {
            let mut g = info_task.write();
            g.alive = false;
            g.op_in_flight = false;
        }
        debug!("ASR owner task exited");
    });

    (
        AsrOwnerHandle {
            tx,
            utt_gen,
            alive: Arc::clone(&alive),
            info,
            default_timeout,
        },
        AsrOwnerJoin { join, alive },
    )
}

/// Helper: true when a completion gen is still current.
#[inline]
pub fn gen_is_current(handle: &AsrOwnerHandle, utt_gen: UtteranceGen) -> bool {
    handle.current_gen() == utt_gen && handle.is_alive()
}
