//! Ordered dual-lane event delivery without per-event OS threads.
//!
//! # Ordering & bounds
//!
//! Production path:
//! 1. **event_log** — diagnostic mirror for `take_events` / tests. Never popped
//!    by delivery. Independent of the bus.
//! 2. **delivery_outbox** — ordered queue awaiting ingress acceptance.
//! 3. `flush_pending` try_sends outbox→ingress FIFO (never awaits).
//! 4. One **dispatcher task** reads ingress and forwards:
//!    - essentials → `essential_rx` (await send — only the dispatcher waits)
//!    - partials → `partial_rx` (try_send, drop on full)
//!
//! The session actor never awaits delivery and never spawns threads per event.
//!
//! # Essential overflow policy
//!
//! When the outbox is full and a new essential arrives:
//! 1. Drop partials first.
//! 2. **Coalesce** replaceable essentials (`Status`, overlay lifecycle, `TtsState`,
//!    toasts, overflow counters, ASR lifecycle flags) — newest wins in place.
//! 3. Evict the **oldest replaceable** essential if still full.
//! 4. **Never silently evict** `FinalTranscript` / `InjectFinal` / `ShutdownComplete`
//!    while any replaceable essential remains.
//! 5. If the buffer holds only critical essentials and still cannot accept the
//!    new event, **reject** the incoming event, increment
//!    `essentials_overflow_rejected`, and set `reliable_delivery_degraded`.
//!    Callers must not treat the bus as reliably delivering under degradation.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::warn;

use crate::types::{
    DEFAULT_EVENT_CAPACITY, SessionEvent, event_is_critical_essential, event_is_essential,
    event_is_partial, event_is_replaceable_essential,
};

/// Default bound for the ordered ingress queue (actor → dispatcher).
pub const DEFAULT_INGRESS_CAP: usize = 512;
/// Default bound for local/pending session buffers.
pub const DEFAULT_LOCAL_CAP: usize = 256;

#[derive(Debug, Default)]
struct BusMetrics {
    essentials_enqueued: AtomicU64,
    essentials_delivered: AtomicU64,
    essentials_dropped: AtomicU64,
    essentials_coalesced: AtomicU64,
    essentials_overflow_rejected: AtomicU64,
    partials_dropped: AtomicU64,
    ingress_full_signals: AtomicU64,
    reliable_delivery_degraded: AtomicBool,
}

/// Dual-lane event bus with a single ordered dispatcher task.
#[derive(Clone)]
pub struct EventBus {
    /// Ordered ingress to the single dispatcher (try_send only from actor).
    ingress_tx: mpsc::Sender<SessionEvent>,
    /// Best-effort direct partial lane used only when ingress is full.
    partial_tx: mpsc::Sender<SessionEvent>,
    metrics: Arc<BusMetrics>,
    closed: Arc<AtomicBool>,
}

pub struct EventBusRx {
    pub essential_rx: mpsc::Receiver<SessionEvent>,
    pub partial_rx: mpsc::Receiver<SessionEvent>,
    /// Join handle for the dispatcher; abort/await on runtime shutdown.
    pub dispatcher_join: JoinHandle<()>,
}

impl EventBus {
    pub fn new(essential_cap: usize, partial_cap: usize) -> (Self, EventBusRx) {
        Self::with_ingress_cap(essential_cap, partial_cap, DEFAULT_INGRESS_CAP)
    }

    pub fn with_ingress_cap(
        essential_cap: usize,
        partial_cap: usize,
        ingress_cap: usize,
    ) -> (Self, EventBusRx) {
        let essential_cap = essential_cap.max(DEFAULT_EVENT_CAPACITY).max(64);
        let partial_cap = partial_cap.max(32);
        let ingress_cap = ingress_cap.max(64);

        let (essential_tx, essential_rx) = mpsc::channel(essential_cap);
        let (partial_tx, partial_rx) = mpsc::channel(partial_cap);
        let (ingress_tx, mut ingress_rx) = mpsc::channel::<SessionEvent>(ingress_cap);

        let metrics = Arc::new(BusMetrics::default());
        let metrics_d = Arc::clone(&metrics);
        let closed = Arc::new(AtomicBool::new(false));
        let closed_d = Arc::clone(&closed);
        let partial_tx_dispatch = partial_tx.clone();

        // Single ordered dispatcher — the only task that may await on delivery.
        let dispatcher_join = tokio::spawn(async move {
            while let Some(ev) = ingress_rx.recv().await {
                if event_is_essential(&ev) {
                    match essential_tx.send(ev).await {
                        Ok(()) => {
                            metrics_d
                                .essentials_delivered
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_) => {
                            warn!("essential observer gone; dispatcher stopping");
                            break;
                        }
                    }
                } else if partial_tx_dispatch.try_send(ev).is_err() {
                    metrics_d.partials_dropped.fetch_add(1, Ordering::Relaxed);
                }
            }
            closed_d.store(true, Ordering::Release);
        });

        (
            Self {
                ingress_tx,
                partial_tx,
                metrics,
                closed,
            },
            EventBusRx {
                essential_rx,
                partial_rx,
                dispatcher_join,
            },
        )
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    pub fn essentials_dropped(&self) -> u64 {
        self.metrics.essentials_dropped.load(Ordering::Relaxed)
    }

    pub fn essentials_coalesced(&self) -> u64 {
        self.metrics.essentials_coalesced.load(Ordering::Relaxed)
    }

    pub fn essentials_overflow_rejected(&self) -> u64 {
        self.metrics
            .essentials_overflow_rejected
            .load(Ordering::Relaxed)
    }

    /// True once a critical essential could not be accepted under saturation.
    /// While set, the bus must not be treated as providing reliable delivery.
    pub fn reliable_delivery_degraded(&self) -> bool {
        self.metrics
            .reliable_delivery_degraded
            .load(Ordering::Acquire)
    }

    pub fn partials_dropped(&self) -> u64 {
        self.metrics.partials_dropped.load(Ordering::Relaxed)
    }

    pub fn essentials_delivered(&self) -> u64 {
        self.metrics.essentials_delivered.load(Ordering::Relaxed)
    }

    pub fn essentials_enqueued(&self) -> u64 {
        self.metrics.essentials_enqueued.load(Ordering::Relaxed)
    }

    pub fn ingress_full_signals(&self) -> u64 {
        self.metrics.ingress_full_signals.load(Ordering::Relaxed)
    }

    /// Non-async emit for the session actor (never awaits, never spawns threads).
    ///
    /// Pushes into **both** buffers deliberately:
    /// - `event_log`: diagnostic mirror; **never** popped by flush/delivery
    /// - `delivery_outbox`: ordered queue drained into ingress by `flush_pending`
    pub fn emit_now(
        &self,
        event: SessionEvent,
        event_log: &mut VecDeque<SessionEvent>,
        delivery_outbox: &mut VecDeque<SessionEvent>,
        cap: usize,
    ) {
        let cap = cap.max(1);
        // Diagnostic mirror first — independent of delivery pops.
        push_event_log(event_log, cap, event.clone());
        // Delivery outbox (policy + metrics live here).
        push_delivery(delivery_outbox, cap, event, &self.metrics);
        self.flush_pending(delivery_outbox);
    }

    /// Compatibility helper when the caller only has a delivery outbox
    /// (no separate diagnostic log). Prefer the 4-buffer `emit_now`.
    pub fn emit_delivery_only(
        &self,
        event: SessionEvent,
        delivery_outbox: &mut VecDeque<SessionEvent>,
        cap: usize,
    ) {
        push_delivery(delivery_outbox, cap.max(1), event, &self.metrics);
        self.flush_pending(delivery_outbox);
    }

    /// Push pending events into the ordered ingress (FIFO, non-blocking).
    pub fn flush_pending(&self, pending: &mut VecDeque<SessionEvent>) {
        while let Some(front) = pending.front() {
            let essential = event_is_essential(front);
            if essential {
                match self.ingress_tx.try_send(front.clone()) {
                    Ok(()) => {
                        pending.pop_front();
                        self.metrics
                            .essentials_enqueued
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        self.metrics
                            .ingress_full_signals
                            .fetch_add(1, Ordering::Relaxed);
                        // Stop to preserve order — retry on next emit/tick.
                        break;
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        warn!("event ingress closed");
                        pending.pop_front();
                    }
                }
            } else {
                // Partials: try ingress for ordering with nearby essentials; if
                // ingress is full, drop rather than stall essential flush.
                let ev = front.clone();
                match self.ingress_tx.try_send(ev.clone()) {
                    Ok(()) => {
                        pending.pop_front();
                    }
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        pending.pop_front();
                        self.metrics
                            .partials_dropped
                            .fetch_add(1, Ordering::Relaxed);
                        // Best-effort out-of-band partial (may reorder vs blocked
                        // essentials — acceptable for partials only).
                        let _ = self.partial_tx.try_send(ev);
                        self.metrics
                            .ingress_full_signals
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        pending.pop_front();
                    }
                }
            }
        }
    }

    /// Backward-compatible alias used by older call sites.
    pub fn flush_local(&self, pending: &mut VecDeque<SessionEvent>) {
        self.flush_pending(pending);
    }
}

/// Same-kind coalesce target for replaceable essentials (newest wins).
fn coalesce_index(q: &VecDeque<SessionEvent>, event: &SessionEvent) -> Option<usize> {
    if !event_is_replaceable_essential(event) {
        return None;
    }
    q.iter().position(|existing| {
        use SessionEvent::*;
        matches!(
            (existing, event),
            (Status(_), Status(_))
                | (OverlayShow { .. }, OverlayShow { .. })
                | (OverlayHide, OverlayHide)
                | (TtsState { .. }, TtsState { .. })
                | (TtsError { .. }, TtsError { .. })
                | (ErrorToast { .. }, ErrorToast { .. })
                | (AudioOverflow { .. }, AudioOverflow { .. })
                | (AsrDisabled { .. }, AsrDisabled { .. })
                | (AsrRecovered, AsrRecovered)
                | (AsrThreadDead, AsrThreadDead)
                | (CudaFallbackApplied { .. }, CudaFallbackApplied { .. })
        )
    })
}

/// Delivery-outbox push with explicit essential overflow policy + metrics.
fn push_delivery(
    q: &mut VecDeque<SessionEvent>,
    cap: usize,
    event: SessionEvent,
    metrics: &BusMetrics,
) {
    // 1) Coalesce replaceable essentials even when not full (newest wins, moves to back).
    if let Some(idx) = coalesce_index(q, &event) {
        q.remove(idx);
        metrics.essentials_coalesced.fetch_add(1, Ordering::Relaxed);
        // Fall through to push_back (queue shrank by 1).
        if q.len() < cap {
            q.push_back(event);
            return;
        }
        // Extremely tight cap==0 already handled by caller (.max(1)).
    }

    if q.len() < cap {
        q.push_back(event);
        return;
    }

    // 2) Full — free space preferentially.
    // 2a) Drop a partial first.
    if let Some(idx) = q.iter().position(event_is_partial) {
        q.remove(idx);
        metrics.partials_dropped.fetch_add(1, Ordering::Relaxed);
        q.push_back(event);
        return;
    }

    // 2b) Evict oldest replaceable essential (never touch critical while these remain).
    if let Some(idx) = q.iter().position(event_is_replaceable_essential) {
        q.remove(idx);
        metrics.essentials_dropped.fetch_add(1, Ordering::Relaxed);
        q.push_back(event);
        return;
    }

    // 3) Only critical essentials remain (FinalTranscript / InjectFinal / ShutdownComplete).
    // Do not silently evict them. Reject incoming and mark delivery degraded —
    // except ShutdownComplete, which is allowed to displace the oldest critical
    // so orderly teardown can still signal completion (still marks degraded).
    if matches!(event, SessionEvent::ShutdownComplete) {
        if q.iter()
            .any(|e| matches!(e, SessionEvent::ShutdownComplete))
        {
            // Already present — coalesce in place / move to back.
            if let Some(idx) = q
                .iter()
                .position(|e| matches!(e, SessionEvent::ShutdownComplete))
            {
                q.remove(idx);
                metrics.essentials_coalesced.fetch_add(1, Ordering::Relaxed);
            }
            if q.len() < cap {
                q.push_back(event);
            } else {
                // Still full of other criticals — displace oldest non-shutdown.
                q.pop_front();
                metrics.essentials_dropped.fetch_add(1, Ordering::Relaxed);
                metrics
                    .reliable_delivery_degraded
                    .store(true, Ordering::Release);
                q.push_back(event);
            }
            return;
        }
        // Insert ShutdownComplete by displacing oldest critical.
        q.pop_front();
        metrics.essentials_dropped.fetch_add(1, Ordering::Relaxed);
        metrics
            .essentials_overflow_rejected
            .fetch_add(1, Ordering::Relaxed);
        metrics
            .reliable_delivery_degraded
            .store(true, Ordering::Release);
        q.push_back(event);
        return;
    }

    // Reject incoming (critical or replaceable) — do not silently drop buffered criticals.
    metrics
        .essentials_overflow_rejected
        .fetch_add(1, Ordering::Relaxed);
    metrics
        .reliable_delivery_degraded
        .store(true, Ordering::Release);
    if event_is_essential(&event) {
        // Count as a drop of the *incoming* essential, not a buffered critical.
        metrics.essentials_dropped.fetch_add(1, Ordering::Relaxed);
    } else {
        metrics.partials_dropped.fetch_add(1, Ordering::Relaxed);
    }
}

/// Priority-aware diagnostic log (no bus metrics; never used as delivery outbox).
pub fn push_event_log(log: &mut VecDeque<SessionEvent>, cap: usize, event: SessionEvent) {
    let cap = cap.max(1);
    // Mirror the same coalesce/overflow *shape* so take_events stays coherent,
    // but never claims bus-level reliability metrics.
    if let Some(idx) = coalesce_index(log, &event) {
        log.remove(idx);
    }
    if log.len() >= cap {
        if let Some(idx) = log.iter().position(event_is_partial) {
            log.remove(idx);
        } else if let Some(idx) = log.iter().position(event_is_replaceable_essential) {
            log.remove(idx);
        } else if matches!(event, SessionEvent::ShutdownComplete) {
            if !log
                .iter()
                .any(|e| matches!(e, SessionEvent::ShutdownComplete))
            {
                log.pop_front();
            } else {
                return;
            }
        } else if event_is_critical_essential(&event) {
            // Keep buffered criticals in the diagnostic log; drop incoming.
            return;
        } else {
            return;
        }
    }
    if log.len() < cap {
        log.push_back(event);
    }
}

#[cfg(test)]
mod overflow_policy_tests {
    use super::*;
    use crate::types::{RecordingStatus, TtsPlayerState};

    fn inject(text: &str) -> SessionEvent {
        SessionEvent::InjectFinal { text: text.into() }
    }

    #[test]
    fn replaceable_coalesce_before_critical_eviction() {
        let metrics = BusMetrics::default();
        let mut q = VecDeque::new();
        let cap = 3;
        push_delivery(&mut q, cap, inject("a"), &metrics);
        push_delivery(&mut q, cap, inject("b"), &metrics);
        push_delivery(
            &mut q,
            cap,
            SessionEvent::Status(RecordingStatus::Recording),
            &metrics,
        );
        assert_eq!(q.len(), 3);

        push_delivery(
            &mut q,
            cap,
            SessionEvent::Status(RecordingStatus::Idle),
            &metrics,
        );
        assert!(metrics.essentials_coalesced.load(Ordering::Relaxed) >= 1);
        let injects = q
            .iter()
            .filter(|e| matches!(e, SessionEvent::InjectFinal { .. }))
            .count();
        assert_eq!(
            injects, 2,
            "InjectFinal must survive Status coalesce: {q:?}"
        );
        assert!(
            q.iter()
                .any(|e| matches!(e, SessionEvent::Status(RecordingStatus::Idle))),
            "newest Status should remain: {q:?}"
        );
        assert!(!metrics.reliable_delivery_degraded.load(Ordering::Acquire));
    }

    #[test]
    fn criticals_not_silently_evicted_when_saturated() {
        let metrics = BusMetrics::default();
        let mut q = VecDeque::new();
        let cap = 2;
        push_delivery(&mut q, cap, inject("one"), &metrics);
        push_delivery(
            &mut q,
            cap,
            SessionEvent::FinalTranscript {
                text: "final".into(),
            },
            &metrics,
        );
        assert_eq!(q.len(), 2);
        assert!(q.iter().all(event_is_critical_essential));

        push_delivery(&mut q, cap, inject("two"), &metrics);
        assert!(metrics.essentials_overflow_rejected.load(Ordering::Relaxed) >= 1);
        assert!(metrics.reliable_delivery_degraded.load(Ordering::Acquire));
        assert_eq!(q.len(), 2);
        assert!(
            q.iter()
                .any(|e| matches!(e, SessionEvent::InjectFinal { text } if text == "one")),
            "{q:?}"
        );
        assert!(
            q.iter()
                .any(|e| matches!(e, SessionEvent::FinalTranscript { .. })),
            "{q:?}"
        );
        assert!(
            !q.iter()
                .any(|e| matches!(e, SessionEvent::InjectFinal { text } if text == "two")),
            "{q:?}"
        );

        let rejected_before = metrics.essentials_overflow_rejected.load(Ordering::Relaxed);
        push_delivery(
            &mut q,
            cap,
            SessionEvent::TtsState {
                state: TtsPlayerState::Playing,
                preview_text: String::new(),
            },
            &metrics,
        );
        assert!(metrics.essentials_overflow_rejected.load(Ordering::Relaxed) > rejected_before);
        assert!(q.iter().all(event_is_critical_essential));
    }

    #[test]
    fn shutdown_complete_enters_saturated_critical_buffer() {
        let metrics = BusMetrics::default();
        let mut q = VecDeque::new();
        let cap = 2;
        push_delivery(&mut q, cap, inject("a"), &metrics);
        push_delivery(&mut q, cap, inject("b"), &metrics);
        push_delivery(&mut q, cap, SessionEvent::ShutdownComplete, &metrics);
        assert!(
            q.iter()
                .any(|e| matches!(e, SessionEvent::ShutdownComplete)),
            "ShutdownComplete must be present: {q:?}"
        );
        assert!(metrics.reliable_delivery_degraded.load(Ordering::Acquire));
    }

    #[test]
    fn classification_helpers_line_up() {
        assert!(event_is_replaceable_essential(&SessionEvent::Status(
            RecordingStatus::Idle
        )));
        assert!(event_is_replaceable_essential(&SessionEvent::TtsState {
            state: TtsPlayerState::Idle,
            preview_text: String::new(),
        }));
        assert!(event_is_critical_essential(&inject("x")));
        assert!(event_is_critical_essential(&SessionEvent::ShutdownComplete));
        assert!(event_is_essential(&inject("x")));
        assert!(!event_is_critical_essential(&SessionEvent::Status(
            RecordingStatus::Idle
        )));
    }
}
