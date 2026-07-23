//! Essential event ordering, bounded dispatcher resources, overflow policy,
//! dual-buffer take_events, and wedged-session shutdown.

use std::collections::VecDeque;
use std::time::Duration;

use shuvoice_app::events::{DEFAULT_LOCAL_CAP, EventBus};
use shuvoice_app::fakes::ScriptedAsrBackend;
use shuvoice_app::types::event_is_essential;
use shuvoice_app::{Config, RecordingStatus, SessionEvent, TestHarness, TtsPlayerState};
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout};

fn essential_n(n: u64) -> SessionEvent {
    SessionEvent::AudioOverflow { dropped: n }
}

fn inject(text: &str) -> SessionEvent {
    SessionEvent::InjectFinal { text: text.into() }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn essential_events_preserve_order_under_backpressure() {
    let (bus, mut rx) = EventBus::with_ingress_cap(4, 8, 8);
    let mut log = VecDeque::new();
    let mut outbox = VecDeque::new();
    let local_cap = 64;

    const N: u64 = 200;
    for i in 0..N {
        bus.emit_now(essential_n(i), &mut log, &mut outbox, local_cap);
        bus.emit_now(
            SessionEvent::PartialTranscript {
                text: format!("p{i}"),
            },
            &mut log,
            &mut outbox,
            local_cap,
        );
        if i % 17 == 0 {
            let _ = rx.essential_rx.try_recv();
            bus.flush_pending(&mut outbox);
        }
    }

    let mut got = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while got.len() < N as usize && tokio::time::Instant::now() < deadline {
        bus.flush_pending(&mut outbox);
        match rx.essential_rx.try_recv() {
            Ok(SessionEvent::AudioOverflow { dropped }) => got.push(dropped),
            Ok(other) => panic!("unexpected essential: {other:?}"),
            Err(_) => {
                tokio::task::yield_now().await;
                sleep(Duration::from_millis(1)).await;
            }
        }
    }

    assert!(
        !got.is_empty(),
        "expected some essentials delivered, drops={}",
        bus.essentials_dropped()
    );
    for w in got.windows(2) {
        assert!(
            w[0] < w[1],
            "out of order essentials: {:?} (drops={})",
            got,
            bus.essentials_dropped()
        );
    }

    rx.dispatcher_join.abort();
    let _ = rx.dispatcher_join.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn essential_dispatcher_is_single_task_and_bounded() {
    let (bus, mut rx) = EventBus::with_ingress_cap(8, 8, 16);
    let mut log = VecDeque::new();
    let mut outbox = VecDeque::new();

    for i in 0..500u64 {
        bus.emit_now(essential_n(i), &mut log, &mut outbox, 32);
    }
    assert!(
        bus.ingress_full_signals() > 0
            || bus.essentials_dropped() > 0
            || bus.essentials_coalesced() > 0
            || outbox.is_empty(),
        "expected backpressure signal under flood"
    );

    let mut n = 0u64;
    let start = tokio::time::Instant::now();
    while start.elapsed() < Duration::from_millis(500) {
        bus.flush_pending(&mut outbox);
        while rx.essential_rx.try_recv().is_ok() {
            n += 1;
        }
        tokio::task::yield_now().await;
    }
    assert!(n > 0, "dispatcher should deliver some events");

    rx.dispatcher_join.abort();
    timeout(Duration::from_secs(2), rx.dispatcher_join)
        .await
        .expect("dispatcher join timed out")
        .ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_aborts_wedged_session_join_handle() {
    let wedged: JoinHandle<()> = tokio::spawn(async {
        loop {
            sleep(Duration::from_secs(3600)).await;
        }
    });

    let mut session_join = wedged;
    let grace = Duration::from_millis(50);
    let started = tokio::time::Instant::now();
    tokio::select! {
        res = &mut session_join => {
            panic!("wedged task exited early: {res:?}");
        }
        _ = sleep(grace) => {
            session_join.abort();
            let join_res = session_join.await;
            assert!(join_res.unwrap_err().is_cancelled());
        }
    }
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "abort path must not hang"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn harness_shutdown_completes_with_dispatcher() {
    let mut cfg = Config::default();
    cfg.tts_enabled = false;
    let h = TestHarness::basic(ScriptedAsrBackend::local_streaming(1600), cfg).await;
    let mut log = VecDeque::new();
    let mut outbox = VecDeque::new();
    for i in 0..10u64 {
        h.events
            .emit_now(essential_n(i), &mut log, &mut outbox, DEFAULT_LOCAL_CAP);
    }
    h.events.flush_pending(&mut outbox);
    sleep(Duration::from_millis(20)).await;
    timeout(Duration::from_secs(3), h.shutdown())
        .await
        .expect("harness shutdown hung");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ordered_essentials_survive_partial_flood_on_session_bus() {
    let mut cfg = Config::default();
    cfg.tts_enabled = false;
    cfg.silence_rms_threshold = 0.0;
    let mut h = TestHarness::basic(ScriptedAsrBackend::local_streaming(1600), cfg).await;

    for i in 0..50u64 {
        h.session
            .emit_for_test(SessionEvent::AudioOverflow { dropped: i });
        for j in 0..20 {
            h.session.emit_for_test(SessionEvent::PartialTranscript {
                text: format!("{i}-{j}"),
            });
        }
        let _ = h.session.tick().await;
    }

    let mut got = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while tokio::time::Instant::now() < deadline {
        let _ = h.session.tick().await;
        while let Ok(ev) = h.essential_rx.try_recv() {
            if let SessionEvent::AudioOverflow { dropped } = ev {
                got.push(dropped);
            } else if event_is_essential(&ev) {
                // ignore other essentials from setup/teardown
            }
        }
        if got.len() >= 40 {
            break;
        }
        sleep(Duration::from_millis(2)).await;
    }
    assert!(got.len() >= 10, "got only {got:?}");
    for w in got.windows(2) {
        assert!(w[0] < w[1], "order broken: {got:?}");
    }
    h.shutdown().await;
}

/// Flooding essentials must not spawn unbounded OS threads / tasks.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_thread_explosion_under_essential_flood() {
    let (bus, rx) = EventBus::with_ingress_cap(16, 16, 32);
    let mut log = VecDeque::new();
    let mut outbox = VecDeque::new();
    for i in 0..10_000u64 {
        bus.emit_now(essential_n(i), &mut log, &mut outbox, 128);
    }
    let mut dj = rx.dispatcher_join;
    dj.abort();
    timeout(Duration::from_secs(2), &mut dj)
        .await
        .expect("dispatcher join hung after flood")
        .ok();
}

/// take_events must retain diagnostics after delivery flush pops the outbox.
#[tokio::test]
async fn take_events_survives_delivery_flush() {
    let mut cfg = Config::default();
    cfg.tts_enabled = false;
    let mut h = TestHarness::basic(ScriptedAsrBackend::default(), cfg).await;

    h.session
        .emit_for_test(SessionEvent::Status(RecordingStatus::Recording));
    h.session.emit_for_test(inject("keep me"));
    h.session.emit_for_test(SessionEvent::TtsState {
        state: TtsPlayerState::Playing,
        preview_text: "x".into(),
    });
    // Flush delivery path (as tick does).
    assert!(h.session.tick().await);

    let ev = h.session.take_events();
    assert!(
        ev.iter()
            .any(|e| matches!(e, SessionEvent::InjectFinal { text } if text == "keep me")),
        "take_events lost InjectFinal after flush: {ev:?}"
    );
    assert!(
        ev.iter().any(|e| matches!(e, SessionEvent::Status(_))),
        "take_events lost Status after flush: {ev:?}"
    );
    // A second take should be empty (drained), proving isolation from outbox.
    assert!(h.session.take_events().is_empty());
    h.shutdown().await;
}
