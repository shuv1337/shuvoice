//! Independent ownership of bounded remote effects. Recording never awaits D-Bus.
use async_trait::async_trait;
use shuvoice_core::media::*;
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::{
    sync::{Semaphore, broadcast, watch},
    task::JoinSet,
    time::{Instant, timeout},
};

pub const ACQUISITION_BUDGET: Duration = Duration::from_millis(400);
const READ: Duration = Duration::from_millis(250);
const EFFECT: Duration = Duration::from_secs(1);
pub type MediaEventReceiver = broadcast::Receiver<MediaEvent>;
#[async_trait]
pub trait MediaTransport: Send + Sync {
    /// Join transport-owned subscription work after all endpoint tasks settle.
    async fn shutdown(&self) {}
    async fn subscribe(&self) -> Result<MediaEventReceiver, MediaError>;
    async fn discover(&self) -> Result<Vec<MediaEndpoint>, MediaError>;
    async fn snapshot(&self, endpoint: &MediaEndpoint) -> Result<MediaSnapshot, MediaError>;
    async fn owner_matches(&self, endpoint: &MediaEndpoint) -> Result<bool, MediaError>;
    async fn pause(&self, endpoint: &MediaEndpoint) -> MediaCallOutcome;
    async fn play(&self, endpoint: &MediaEndpoint) -> MediaCallOutcome;
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaDesired {
    Idle,
    Hold {
        intent: RecordingIntentId,
        deadline: Instant,
    },
    Shutdown,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaReadiness {
    Pending,
    Settled {
        intent: RecordingIntentId,
        degraded: bool,
    },
}
#[derive(Clone)]
pub struct MediaHandle {
    desired: watch::Sender<MediaDesired>,
    ready: watch::Receiver<MediaReadiness>,
}
impl MediaHandle {
    pub fn hold(&self, intent: RecordingIntentId) {
        self.desired.send_if_modified(|v| {
            if *v == MediaDesired::Shutdown {
                false
            } else {
                *v = MediaDesired::Hold {
                    intent,
                    deadline: Instant::now() + ACQUISITION_BUDGET,
                };
                true
            }
        });
    }
    pub fn release(&self, intent: RecordingIntentId) {
        self.desired.send_if_modified(|v| {
            if matches!(*v, MediaDesired::Hold { intent: current, .. } if current == intent) {
                *v = MediaDesired::Idle;
                true
            } else {
                false
            }
        });
    }
    pub fn settled(&self, intent: RecordingIntentId) -> bool {
        self.desired.is_closed()
            || matches!(*self.ready.borrow(), MediaReadiness::Settled { intent: i, .. } if i == intent)
    }
}
pub struct MediaCoordinatorJoin {
    handle: MediaHandle,
    join: tokio::task::JoinHandle<()>,
}
impl MediaCoordinatorJoin {
    pub async fn shutdown(mut self) {
        self.handle.desired.send_replace(MediaDesired::Shutdown);
        if timeout(Duration::from_secs(2), &mut self.join)
            .await
            .is_err()
        {
            self.join.abort();
            let _ = self.join.await;
        }
    }
}
pub fn spawn_media_coordinator(
    transport: Arc<dyn MediaTransport>,
) -> (MediaHandle, MediaCoordinatorJoin) {
    let (desired, mut rx) = watch::channel(MediaDesired::Idle);
    let (ready, readiness) = watch::channel(MediaReadiness::Pending);
    let handle = MediaHandle {
        desired,
        ready: readiness,
    };
    let join = tokio::spawn(async move {
        let mut tasks = JoinSet::new();
        let permits = Arc::new(Semaphore::new(8));
        let mut owned: HashMap<_, Arc<AtomicBool>> = HashMap::new();
        let mut last = None;
        let mut deadline = None;
        let mut shutdown_deadline = None;
        let mut degraded = false;
        loop {
            let state = *rx.borrow_and_update();
            if let MediaDesired::Hold {
                intent,
                deadline: admitted_deadline,
            } = state
            {
                if last != Some(intent) {
                    last = Some(intent);
                    ready.send_replace(MediaReadiness::Pending);
                    degraded = false;
                    deadline = Some(admitted_deadline);
                    if let Ok(Ok(events)) =
                        read_until(admitted_deadline, transport.subscribe()).await
                        && let Ok(Ok(endpoints)) =
                            read_until(deadline.unwrap(), transport.discover()).await
                    {
                        for endpoint in endpoints.into_iter().take(128) {
                            let key = endpoint.key();
                            if tasks.len() >= 128 || Instant::now() >= deadline.unwrap() {
                                break;
                            }
                            if let std::collections::hash_map::Entry::Vacant(entry) =
                                owned.entry(key)
                            {
                                let settled = Arc::new(AtomicBool::new(false));
                                entry.insert(settled.clone());
                                tasks.spawn(endpoint_worker(
                                    transport.clone(),
                                    endpoint,
                                    rx.clone(),
                                    events.resubscribe(),
                                    deadline.unwrap(),
                                    permits.clone(),
                                    intent,
                                    settled,
                                ));
                            }
                        }
                    } else {
                        degraded = true;
                    }
                }
                if deadline.is_some_and(|d| {
                    Instant::now() >= d || owned.values().all(|s| s.load(Ordering::Acquire))
                }) {
                    ready.send_replace(MediaReadiness::Settled {
                        intent,
                        degraded: degraded || owned.values().any(|s| !s.load(Ordering::Acquire)),
                    });
                    // Keep the admission deadline for a predecessor's in-flight
                    // Play reconciliation; readiness does not erase its budget.
                }
            }
            if state == MediaDesired::Shutdown {
                let until = *shutdown_deadline
                    .get_or_insert_with(|| Instant::now() + Duration::from_millis(1500));
                if tasks.is_empty() || Instant::now() >= until {
                    break;
                }
            }
            tokio::select! {
                changed = rx.changed() => { if changed.is_err() { break; } }
                Some(result) = tasks.join_next(), if !tasks.is_empty() => {
                    if let Ok(Completion { endpoint, origin, replay, degraded: failed }) = result {
                        owned.remove(&endpoint.key());
                        degraded |= failed;
                        // A Play already dispatched cannot be unsent. Reconcile it
                        // before acquiring the same endpoint for a newer hold.
                        let current = *rx.borrow();
                        if let MediaDesired::Hold { intent, deadline } = current
                            && (replay == Replay::AfterPlay || (replay == Replay::NewIntent && intent != origin))
                            && Instant::now() < deadline
                            && let Ok(Ok(events)) = read_until(deadline, transport.subscribe()).await {
                                let settled = Arc::new(AtomicBool::new(false));
                                owned.insert(endpoint.key(), settled.clone());
                                tasks.spawn(endpoint_worker(transport.clone(), endpoint, rx.clone(), events, deadline, permits.clone(), intent, settled));
                        }
                    }
                }
                () = tokio::time::sleep(Duration::from_millis(5)) => {}
            }
        }
        tasks.abort_all();
        while tasks.join_next().await.is_some() {}
        let _ = timeout(Duration::from_millis(100), transport.shutdown()).await;
    });
    (handle.clone(), MediaCoordinatorJoin { handle, join })
}
async fn read_until<F: std::future::Future>(
    deadline: Instant,
    f: F,
) -> Result<F::Output, tokio::time::error::Elapsed> {
    tokio::time::timeout_at(deadline.min(Instant::now() + READ), f).await
}
/// Signals are ordered observations; a fresh read can race a later signal.
/// A predecessor is allowed only until the target has first been observed.
struct PlaybackPhase {
    prior: Option<MediaPlayback>,
    target: MediaPlayback,
    observed_target: bool,
}
impl PlaybackPhase {
    fn stable(target: MediaPlayback) -> Self {
        Self {
            prior: None,
            target,
            observed_target: true,
        }
    }
    fn transition(prior: MediaPlayback, target: MediaPlayback) -> Self {
        Self {
            prior: Some(prior),
            target,
            observed_target: false,
        }
    }
    fn observe(&mut self, playback: MediaPlayback) -> bool {
        if playback == self.target {
            self.observed_target = true;
            true
        } else {
            self.prior == Some(playback) && !self.observed_target
        }
    }
}

fn drain(
    events: &mut MediaEventReceiver,
    endpoint: &MediaEndpoint,
    baseline: &MediaFingerprint,
    phase: &mut PlaybackPhase,
) -> (bool, bool) {
    let mut refresh = false;
    loop {
        match events.try_recv() {
            Ok(MediaEvent::Lost)
            | Err(
                broadcast::error::TryRecvError::Lagged(_) | broadcast::error::TryRecvError::Closed,
            ) => return (false, false),
            Ok(MediaEvent::Conflict { owner, path })
                if owner == endpoint.unique_owner && path == endpoint.object_path =>
            {
                return (false, false);
            }
            Ok(MediaEvent::Seeked { owner, path })
                if owner == endpoint.unique_owner && path == endpoint.object_path =>
            {
                // Chromium recalibrates its media position with Seeked during
                // ordinary Pause AND Play. Only tolerate this while confirming
                // our own bounded effect, still requiring unchanged identity,
                // owner and the expected playback transition. Before dispatch
                // and during a confirmed hold, a seek revokes restoration.
                if phase.prior.is_none() {
                    return (false, false);
                }
            }
            Ok(MediaEvent::Paused { owner, path })
                if owner == endpoint.unique_owner && path == endpoint.object_path =>
            {
                if !phase.observe(MediaPlayback::Paused) {
                    return (false, false);
                }
            }
            Ok(MediaEvent::Properties {
                owner,
                path,
                playback,
                fingerprint,
                can_control,
                can_pause,
                can_play,
                invalidated,
            }) if owner == endpoint.unique_owner && path == endpoint.object_path => {
                if playback.is_some_and(|p| !phase.observe(p))
                    || fingerprint.as_ref().is_some_and(|f| f != baseline)
                    || can_control == Some(false)
                    || can_play == Some(false)
                    || (phase.prior.is_none()
                        && phase.target == MediaPlayback::Playing
                        && can_pause == Some(false))
                {
                    return (false, false);
                }
                refresh |= invalidated.iter().any(|k| {
                    matches!(
                        k.as_str(),
                        "PlaybackStatus" | "Metadata" | "CanControl" | "CanPause" | "CanPlay"
                    )
                });
            }
            Err(broadcast::error::TryRecvError::Empty) => return (true, refresh),
            Ok(_) => {}
        }
    }
}

async fn events_valid(
    t: &dyn MediaTransport,
    e: &MediaEndpoint,
    events: &mut MediaEventReceiver,
    baseline: &MediaFingerprint,
    expected: MediaPlayback,
) -> bool {
    let mut phase = PlaybackPhase::stable(expected);
    // Reconcile relevant invalidations against uncached properties, then drain
    // the fenced read's signals. Persistent invalidation churn is ambiguous.
    for _ in 0..2 {
        let (valid, refresh) = drain(events, e, baseline, &mut phase);
        if !valid {
            return false;
        }
        if !refresh {
            return true;
        }
        let Ok(Ok(s)) = timeout(READ, t.snapshot(e)).await else {
            return false;
        };
        if s.fingerprint != *baseline
            || s.playback != expected
            || !s.can_control
            || !s.can_play
            || (expected == MediaPlayback::Playing && !s.can_pause)
        {
            return false;
        }
    }
    matches!(drain(events, e, baseline, &mut phase), (true, false))
}

/// Called only after a successful method reply, inside the effect's total
/// deadline. Invalidation is reconciled by each iteration's uncached read.
async fn confirm_transition(
    t: &dyn MediaTransport,
    e: &MediaEndpoint,
    events: &mut MediaEventReceiver,
    baseline: &MediaFingerprint,
    prior: MediaPlayback,
    target: MediaPlayback,
) -> bool {
    let mut phase = PlaybackPhase::transition(prior, target);
    loop {
        if !drain(events, e, baseline, &mut phase).0 {
            return false;
        }
        let observed_before_read = phase.observed_target;
        let Ok(Ok(s)) = timeout(READ, t.snapshot(e)).await else {
            return false;
        };
        // This fresh target read is itself an observation, before owner
        // validation can yield. Any queued predecessor signal is now a
        // reversal or ambiguously ordered: neither authorizes restoration.
        if s.playback == target {
            phase.observe(target);
        }
        let owner = matches!(timeout(READ, t.owner_matches(e)).await, Ok(Ok(true)));
        let (valid, refresh) = drain(events, e, baseline, &mut phase);
        if !valid || !owner || s.fingerprint != *baseline || !s.can_control || !s.can_play {
            return false;
        }
        // A target signal arriving during a read may be newer than its returned
        // predecessor value. Re-read rather than mistaking that stale value for
        // a reversal. A predecessor read begun AFTER target observation revokes.
        let raced_target = !observed_before_read && phase.observed_target && s.playback == prior;
        if !raced_target && !phase.observe(s.playback) {
            return false;
        }
        if s.playback == target && !refresh {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Replay {
    Never,
    NewIntent,
    AfterPlay,
}
struct Completion {
    endpoint: MediaEndpoint,
    origin: RecordingIntentId,
    replay: Replay,
    degraded: bool,
}
#[allow(clippy::too_many_arguments)]
async fn endpoint_worker(
    t: Arc<dyn MediaTransport>,
    e: MediaEndpoint,
    mut desired: watch::Receiver<MediaDesired>,
    mut events: MediaEventReceiver,
    deadline: Instant,
    permits: Arc<Semaphore>,
    origin: RecordingIntentId,
    settled: Arc<AtomicBool>,
) -> Completion {
    let done = |replay, degraded| Completion {
        endpoint: e.clone(),
        origin,
        replay,
        degraded,
    };
    let Ok(Ok(permit)) = tokio::time::timeout_at(deadline, permits.clone().acquire_owned()).await
    else {
        return done(Replay::NewIntent, true);
    };
    let Ok(Ok(baseline)) = read_until(deadline, t.snapshot(&e)).await else {
        return done(Replay::NewIntent, true);
    };
    if !eligible_to_pause(&baseline) {
        return done(Replay::Never, baseline.playback == MediaPlayback::Playing);
    }
    if !matches!(
        read_until(deadline, t.owner_matches(&e)).await,
        Ok(Ok(true))
    ) {
        return done(Replay::NewIntent, true);
    }
    let Ok(Ok(fresh)) = read_until(deadline, t.snapshot(&e)).await else {
        return done(Replay::NewIntent, true);
    };
    if !eligible_to_pause(&fresh)
        || fresh.fingerprint != baseline.fingerprint
        || !events_valid(
            t.as_ref(),
            &e,
            &mut events,
            &baseline.fingerprint,
            MediaPlayback::Playing,
        )
        .await
    {
        return done(Replay::Never, true);
    }
    if !matches!(*desired.borrow(), MediaDesired::Hold { intent, .. } if intent == origin)
        || Instant::now() >= deadline
    {
        return done(Replay::NewIntent, true);
    }
    let confirmed = timeout(EFFECT, async {
        if t.pause(&e).await != MediaCallOutcome::Success {
            return false;
        }
        confirm_transition(
            t.as_ref(),
            &e,
            &mut events,
            &baseline.fingerprint,
            MediaPlayback::Playing,
            MediaPlayback::Paused,
        )
        .await
    })
    .await
    .unwrap_or(false);
    if !confirmed {
        return done(Replay::Never, true);
    }
    drop(permit);
    settled.store(true, Ordering::Release);
    loop {
        if !events_valid(
            t.as_ref(),
            &e,
            &mut events,
            &baseline.fingerprint,
            MediaPlayback::Paused,
        )
        .await
        {
            return done(Replay::Never, true);
        }
        if !matches!(*desired.borrow_and_update(), MediaDesired::Hold { .. }) {
            settled.store(false, Ordering::Release);
            // Capacity contention is local, not a failed remote operation.
            // Retain the ledger until capacity arrives or a new hold takes over.
            let permit = tokio::select! {
                permit = permits.clone().acquire_owned() => permit,
                _ = desired.changed() => { settled.store(true, Ordering::Release); continue; },
            };
            let Ok(_permit) = permit else {
                return done(Replay::Never, true);
            };
            let valid = timeout(READ, async {
                let owner = t.owner_matches(&e).await.unwrap_or(false);
                t.snapshot(&e)
                    .await
                    .is_ok_and(|s| eligible_to_restore(&s, &baseline.fingerprint, true, owner))
            })
            .await
            .unwrap_or(false);
            if !valid
                || !events_valid(
                    t.as_ref(),
                    &e,
                    &mut events,
                    &baseline.fingerprint,
                    MediaPlayback::Paused,
                )
                .await
            {
                return done(Replay::Never, true);
            }
            if matches!(*desired.borrow(), MediaDesired::Hold { .. }) {
                settled.store(true, Ordering::Release);
                continue;
            }
            let played = timeout(EFFECT, async {
                if t.play(&e).await != MediaCallOutcome::Success {
                    return false;
                }
                confirm_transition(
                    t.as_ref(),
                    &e,
                    &mut events,
                    &baseline.fingerprint,
                    MediaPlayback::Paused,
                    MediaPlayback::Playing,
                )
                .await
            })
            .await
            .unwrap_or(false);
            // A reply is not a playback transition. Keep this endpoint pending
            // until Playing is freshly confirmed; never re-pause from an
            // unknown/no-op Play solely because a successor is waiting.
            return done(
                if played {
                    Replay::AfterPlay
                } else {
                    Replay::Never
                },
                !played,
            );
        }
        tokio::select! {
            _ = desired.changed() => {},
            () = tokio::time::sleep(Duration::from_millis(5)) => {},
        }
    }
}
