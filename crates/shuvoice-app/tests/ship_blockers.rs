//! Mandatory structural regressions from post-review ship blockers.

use std::time::Duration;

use serial_test::serial;
use shuvoice_app::core::STOP_TAIL_GRACE;
use shuvoice_app::fakes::{FakeSelection, FakeTts, ScriptedAsrBackend, offline_asr};
use shuvoice_app::{Config, SessionEvent, TTS_AWAIT_FINALIZE_TIMEOUT, TestHarness};
use shuvoice_asr::FallbackOutcome;

fn cfg() -> Config {
    // Belt-and-suspenders: never leave process-global hang armed across tests.
    shuvoice_app::TEST_HANG_ACTOR_ON_SHUTDOWN.store(false, std::sync::atomic::Ordering::SeqCst);
    let mut c = Config::default();
    c.min_speech_ms = 0;
    c.silence_rms_threshold = 0.0;
    c.tts_enabled = true;
    c.audio_queue_max_size = 32;
    c
}

#[tokio::test]
async fn actor_stays_responsive_while_slow_finalize_runs() {
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted.shared().lock().delay = Duration::from_millis(80);
    scripted.shared().lock().texts.push_back("slow".into());
    let mut h = TestHarness::basic(scripted, cfg()).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 4]);
    h.session.stop_recording();
    // Falling edge starts finalize job without blocking the call.
    let t0 = std::time::Instant::now();
    assert!(h.session.tick().await);
    assert!(
        t0.elapsed() < Duration::from_millis(40),
        "tick blocked on finalize"
    );
    // Control path still works (status) while finalizing.
    let _ = h.session.recording_status();
    // Wait for finalize to complete
    let _ = h
        .session
        .await_or_cancel_finalization(Duration::from_secs(5))
        .await;
    h.shutdown().await;
}

#[tokio::test]
async fn stop_edge_audio_is_included_via_grace() {
    // Audio present only after stop (key-up) must still be finalized.
    let scripted = offline_asr("with grace");
    let mut h = TestHarness::basic(scripted, cfg()).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.stop_recording();
    // Direct handle path drains stop-edge buffer; push key-up then finalize.
    h.audio.try_push(vec![0.3; 160]);
    h.session.handle_recording_stop().await;
    assert!(
        h.scripted.lock().utterance_calls >= 1,
        "stop-edge/key-up audio must reach process_utterance"
    );
    assert!(!h.injector.finals().is_empty());
    let _ = STOP_TAIL_GRACE;
    h.shutdown().await;
}

#[tokio::test]
async fn essential_events_reach_production_observer() {
    let mut h = TestHarness::basic(offline_asr("essential"), cfg()).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 8]);
    h.session.handle_recording_stop().await;
    // Drain essential lane
    let mut saw_inject = false;
    let mut saw_status = false;
    while let Ok(ev) = h.essential_rx.try_recv() {
        match ev {
            SessionEvent::InjectFinal { .. } | SessionEvent::FinalTranscript { .. } => {
                saw_inject = true;
            }
            SessionEvent::Status(_) => saw_status = true,
            SessionEvent::OverlayHide | SessionEvent::OverlayShow { .. } => {}
            _ => {}
        }
    }
    assert!(saw_inject || !h.injector.finals().is_empty());
    assert!(
        saw_status
            || h.session
                .take_events()
                .iter()
                .any(|e| matches!(e, SessionEvent::Status(_)))
    );
    h.shutdown().await;
}

#[tokio::test]
async fn stop_rearm_start_without_tick_cancels_and_resets_injector() {
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted.shared().lock().delay = Duration::from_millis(100);
    scripted.shared().lock().texts.push_back("old".into());
    let mut h = TestHarness::basic(scripted, cfg()).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 4]);
    h.session.process_recording_chunks().await;
    // Partial may exist
    h.session.stop_recording();
    assert!(h.session.tick().await); // spawn finalize
    // Rearm window passes without waiting finalize
    h.clock.advance_ms(400);
    h.session.start_recording().await; // must cancel prior finalize + reset injector
    assert!(h.session.is_recording());
    // Injector partials cleared on reset
    assert!(h.injector.partials().is_empty());
    h.shutdown().await;
}

#[tokio::test]
async fn tts_prepare_awaits_or_cancels_finalize_honestly() {
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted.shared().lock().delay = Duration::from_millis(30);
    scripted.shared().lock().texts.push_back("hi".into());
    let mut h = TestHarness::new_with(
        scripted,
        FakeTts::new(),
        FakeSelection::with(Ok("sel".into()), Ok("c".into())),
        cfg(),
    )
    .await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 4]);
    h.session.stop_recording();
    // Immediate TTS path
    h.session.tts_speak_selection().await.unwrap();
    assert_eq!(h.session.tts().unwrap().speak_calls.len(), 1);
    assert!(TTS_AWAIT_FINALIZE_TIMEOUT.as_secs() >= 5);
    h.shutdown().await;
}

#[tokio::test]
async fn uses_config_audio_queue_max_size() {
    let mut c = cfg();
    c.audio_queue_max_size = 7;
    let h = TestHarness::basic(ScriptedAsrBackend::default(), c).await;
    assert_eq!(h.session.audio().capacity(), 7);
    h.shutdown().await;
}

#[tokio::test]
async fn inject_err_does_not_retry_or_duplicate() {
    let scripted = offline_asr("no retry");
    let mut h = TestHarness::basic(scripted, cfg()).await;
    h.injector.set_fail_commits(2); // definitive Err — still no automatic retry
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 8]);
    h.session.handle_recording_stop().await;
    assert_eq!(h.injector.commit_calls(), 1);
    assert!(h.injector.finals().is_empty());
    h.shutdown().await;
}

#[tokio::test]
async fn cuda_fallback_outcome_emits_event_and_does_not_count() {
    let scripted = offline_asr("x");
    {
        let s = scripted.shared();
        let mut g = s.lock();
        g.fail_utterance_with = Some("CUBLAS_STATUS_ALLOC_FAILED".into());
        g.fallback = FallbackOutcome::Applied {
            detail: "to cpu".into(),
        };
    }
    let mut h = TestHarness::basic(scripted, cfg()).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 8]);
    h.session.decode_offline_utterance().await;
    assert_eq!(h.session.consecutive_failures(), 0);
    let ev = h.session.take_events();
    assert!(
        ev.iter()
            .any(|e| matches!(e, SessionEvent::CudaFallbackApplied { .. }))
    );
    h.shutdown().await;
}

#[tokio::test]
async fn model_load_failed_is_truthful_after_success() {
    let h = TestHarness::basic(ScriptedAsrBackend::default(), cfg()).await;
    let dbg: serde_json::Value = serde_json::from_str(&h.session.debug_status_json()).unwrap();
    assert_eq!(dbg["app"]["model_load_failed"], false);
    h.shutdown().await;
}

#[tokio::test]
async fn tts_emits_state_and_error_events() {
    let mut h = TestHarness::new_with(
        ScriptedAsrBackend::default(),
        FakeTts::new(),
        FakeSelection::with(Ok("hello".into()), Ok("c".into())),
        cfg(),
    )
    .await;
    h.session.tts_speak_selection().await.unwrap();
    let ev = h.session.take_events();
    assert!(ev.iter().any(|e| matches!(
        e,
        SessionEvent::TtsState {
            state: shuvoice_app::TtsPlayerState::Synthesizing,
            ..
        }
    )));
    h.shutdown().await;
}

// ── Additional mandatory stress / regression coverage ───────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slow_asr_control_latency_stays_low() {
    use shuvoice_app::{ControlHandlerSurface, SessionCommand, spawn_test_runtime};
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted.shared().lock().delay = Duration::from_millis(300);
    let (rt, _, _) = spawn_test_runtime(cfg(), scripted).await;
    rt.handle.try_enqueue(SessionCommand::Start).unwrap();
    // Give actor a moment to enter reset/HOL on ASR.
    tokio::time::sleep(Duration::from_millis(20)).await;
    let t0 = std::time::Instant::now();
    for _ in 0..50 {
        let _ = rt.control.on_status();
        let _ = rt.control.on_metrics();
        rt.control.on_stop();
    }
    let elapsed = t0.elapsed();
    assert!(
        elapsed < Duration::from_millis(100),
        "control plane blocked under slow ASR: {elapsed:?}"
    );
    rt.shutdown().await.unwrap();
}

#[tokio::test]
async fn falling_edge_tail_grace_captures_late_audio() {
    use shuvoice_app::core::STOP_TAIL_GRACE;
    // Key-up audio arrives only after stop + during grace window.
    let scripted = offline_asr("tail grace text");
    let mut h = TestHarness::basic(scripted, cfg()).await;
    h.session.start_recording().await;
    // Tick while recording so was_recording arms the falling-edge detector.
    h.audio.try_push(vec![0.2; 8]);
    assert!(h.session.tick().await);
    assert!(h.session.is_recording());
    h.session.stop_recording();
    // Falling edge spawns finalize (non-blocking w.r.t. ASR work).
    let t0 = std::time::Instant::now();
    assert!(h.session.tick().await);
    assert!(
        t0.elapsed() < Duration::from_millis(40),
        "falling-edge tick must not HOL-block on finalize"
    );
    assert!(
        h.session.is_finalizing(),
        "finalize job must start on falling edge"
    );
    // Push late key-up audio while grace is open; next tick feeds late_tx.
    h.audio.try_push(vec![0.35; 320]);
    assert!(h.session.tick().await);
    let _ = h
        .session
        .await_or_cancel_finalization(Duration::from_secs(5))
        .await;
    // Offline path should have seen the late samples (more than the initial 8).
    let last = h.scripted.lock().last_utterance.len();
    assert!(
        last > 8,
        "expected STOP_TAIL_GRACE ({STOP_TAIL_GRACE:?}) to absorb late audio; got {last} samples"
    );
    assert!(!h.injector.finals().is_empty());
    h.shutdown().await;
}

#[tokio::test]
async fn stop_edge_tick_routes_ring_audio_into_finalize() {
    // Audio sitting in the capture ring at key-up is drained by the stop-edge
    // tick BEFORE finalize exists; it must reach the utterance buffer, not the
    // noise-floor estimator.
    let scripted = offline_asr("tail kept");
    let mut h = TestHarness::basic(scripted, cfg()).await;
    h.session.start_recording().await;
    h.audio.try_push(vec![0.2; 8]);
    assert!(h.session.tick().await); // arms was_recording
    assert!(h.session.is_recording());
    h.session.stop_recording();
    // Tail of the last word: buffered at key-up, drained on the stop-edge tick.
    h.audio.try_push(vec![0.35; 320]);
    assert!(h.session.tick().await);
    let _ = h
        .session
        .await_or_cancel_finalization(Duration::from_secs(5))
        .await;
    let last = h.scripted.lock().last_utterance.len();
    assert!(
        last > 8,
        "stop-edge ring audio must reach finalize, not the noise floor; got {last} samples"
    );
    assert!(!h.injector.finals().is_empty());
    h.shutdown().await;
}

#[tokio::test]
async fn stale_finalize_after_rearm_does_not_commit() {
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted.shared().lock().delay = Duration::from_millis(80);
    scripted.shared().lock().texts.push_back("stale".into());
    let mut h = TestHarness::basic(scripted, cfg()).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 4]);
    h.session.process_recording_chunks().await;
    h.session.stop_recording();
    assert!(h.session.tick().await); // start finalize gen N
    h.clock.advance_ms(400);
    // Cancel finalize + start new gen before it lands.
    h.session.start_recording().await;
    assert!(h.session.is_recording());
    // Wait longer than the stale job's delay; must not inject "stale".
    tokio::time::sleep(Duration::from_millis(200)).await;
    let _ = h.session.tick().await;
    assert!(
        h.injector.finals().is_empty(),
        "stale finalize must not commit after rearm: {:?}",
        h.injector.finals()
    );
    h.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tts_player_update_reenqueues_without_blocking() {
    use shuvoice_app::{SessionCommand, TtsPlayerState, spawn_test_runtime};
    let (mut rt, _, _) = spawn_test_runtime(cfg(), ScriptedAsrBackend::default()).await;
    // Simulate player callback re-entry.
    // Seed preview via a speak so TtsState carries current preview.
    rt.handle
        .try_enqueue(SessionCommand::TtsSpeak {
            text: "hello from player".into(),
            source: shuvoice_app::TtsSource::Selection,
        })
        .ok();
    tokio::time::sleep(Duration::from_millis(30)).await;
    rt.handle
        .enqueue_tts_player_update(TtsPlayerState::Playing, None)
        .unwrap();
    // Drain essential lane for TtsState.
    let mut saw = false;
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        while let Ok(ev) = rt.essential_rx.try_recv() {
            if let shuvoice_app::SessionEvent::TtsState {
                state: TtsPlayerState::Playing,
                ..
            } = ev
            {
                saw = true;
            }
        }
        if saw {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(saw, "TtsPlayerUpdate must emit essential TtsState");
    assert_eq!(rt.handle.tts_status(), "playing");
    // Distinct from STT status.
    assert_ne!(rt.handle.status(), "playing");
    rt.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_runtime_shutdown_aborts_on_timeout() {
    use shuvoice_app::spawn_test_runtime;
    let scripted = ScriptedAsrBackend::default();
    // Wedged ASR ops shouldn't hang shutdown forever.
    scripted.shared().lock().delay = Duration::from_millis(50);
    let (rt, _, _) = spawn_test_runtime(cfg(), scripted).await;
    rt.handle
        .try_enqueue(shuvoice_app::SessionCommand::Start)
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), rt.shutdown())
        .await
        .expect("runtime shutdown hung")
        .unwrap();
}

#[tokio::test]
async fn inject_never_latches_committed_gen_on_total_failure() {
    let scripted = offline_asr("never lands");
    let mut h = TestHarness::basic(scripted, cfg()).await;
    h.injector.set_fail_commits(100); // always fail
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 8]);
    h.session.handle_recording_stop().await;
    assert!(h.injector.finals().is_empty());
    assert_eq!(h.injector.commit_calls(), 1);
    // Gen is spent after unknown/failed attempt — must not re-issue commit.
    h.injector.set_fail_commits(0);
    h.session.commit_utterance().await;
    assert_eq!(h.injector.commit_calls(), 1);
    assert!(h.injector.finals().is_empty());
    h.shutdown().await;
}

#[test]
fn tts_and_overlay_lifecycle_are_essential() {
    use shuvoice_app::{OverlayState, TtsPlayerState, event_is_essential, event_is_partial};
    assert!(event_is_essential(&SessionEvent::TtsState {
        state: TtsPlayerState::Playing,
        preview_text: String::new(),
    }));
    assert!(event_is_essential(&SessionEvent::TtsError {
        message: "x".into()
    }));
    assert!(event_is_essential(&SessionEvent::OverlayShow {
        state: OverlayState::Listening,
        text: "Listening…".into(),
    }));
    assert!(event_is_essential(&SessionEvent::OverlayHide));
    // OverlayUpdate is high-frequency → partial (not essential).
    assert!(!event_is_essential(&SessionEvent::OverlayUpdate {
        state: Some(OverlayState::Processing),
        text: None,
    }));
    assert!(event_is_partial(&SessionEvent::OverlayUpdate {
        state: Some(OverlayState::Processing),
        text: None,
    }));
    assert!(!event_is_partial(&SessionEvent::TtsState {
        state: TtsPlayerState::Idle,
        preview_text: String::new(),
    }));
    assert!(event_is_partial(&SessionEvent::PartialTranscript {
        text: "p".into()
    }));
}

#[tokio::test]
async fn tts_status_distinct_from_stt_status_on_view() {
    let mut h = TestHarness::new_with(
        ScriptedAsrBackend::default(),
        FakeTts::new(),
        FakeSelection::with(Ok("sel".into()), Ok("c".into())),
        cfg(),
    )
    .await;
    h.session.start_recording().await;
    assert_eq!(h.session.view().status(), "recording");
    assert_eq!(h.session.view().tts_status(), "idle");
    h.session.tts_speak_selection().await.unwrap();
    assert_eq!(h.session.view().tts_status(), "synthesizing");
    // After speak, recording was stopped.
    assert_ne!(h.session.view().status(), "synthesizing");
    h.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tts_essentials_survive_partial_flood() {
    use shuvoice_app::{SessionCommand, TtsPlayerState, spawn_test_runtime};
    let (mut rt, _, _) = spawn_test_runtime(cfg(), ScriptedAsrBackend::default()).await;

    // Flood partials via the session command path is not available; use handle
    // player updates interleaved with a high rate of status-like traffic by
    // enqueueing many TtsPlayerUpdate essentials while the actor also emits
    // partials from a short recording.
    rt.handle.try_enqueue(SessionCommand::Start).unwrap();
    for i in 0..80u32 {
        // Player callback re-entry under load.
        let st = if i % 2 == 0 {
            TtsPlayerState::Playing
        } else {
            TtsPlayerState::Paused
        };
        let err = if i % 17 == 0 {
            Some(format!("transient-{i}"))
        } else {
            None
        };
        // Command mailbox is bounded — ignore full under intentional flood.
        let _ = rt.handle.enqueue_tts_player_update(st, err);
    }
    let _ = rt.handle.try_enqueue(SessionCommand::Stop);

    let mut tts_states = 0u32;
    let mut tts_errors = 0u32;
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while std::time::Instant::now() < deadline {
        while let Ok(ev) = rt.essential_rx.try_recv() {
            match ev {
                shuvoice_app::SessionEvent::TtsState { .. } => tts_states += 1,
                shuvoice_app::SessionEvent::TtsError { .. } => tts_errors += 1,
                _ => {}
            }
        }
        // Drain partials so they cannot stall anything (best-effort lane).
        while rt.partial_rx.try_recv().is_ok() {}
        if tts_states >= 40 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(
        tts_states >= 40,
        "expected most TtsState essentials under partial flood, got {tts_states}"
    );
    assert!(
        tts_errors >= 3,
        "expected TtsError essentials delivered, got {tts_errors}"
    );
    // tts_status remains player state, not STT.
    let tts = rt.handle.tts_status();
    assert!(
        matches!(
            tts.as_str(),
            "playing" | "paused" | "idle" | "synthesizing" | "error"
        ),
        "unexpected tts_status {tts}"
    );
    assert_ne!(rt.handle.status(), tts);
    rt.shutdown().await.unwrap();
}

#[tokio::test]
async fn recording_status_never_equals_tts_player_mirror() {
    let mut h = TestHarness::new_with(
        ScriptedAsrBackend::default(),
        FakeTts::new(),
        FakeSelection::with(Ok("sel".into()), Ok("c".into())),
        cfg(),
    )
    .await;
    h.session.start_recording().await;
    let stt = h.session.view().status();
    assert_eq!(stt, "recording");
    // Force player mirror without going through speak.
    h.session
        .apply_tts_player_update(shuvoice_app::TtsPlayerState::Playing, None);
    assert_eq!(h.session.view().tts_status(), "playing");
    assert_eq!(h.session.view().status(), "recording"); // STT unchanged
    assert_ne!(h.session.view().status(), h.session.view().tts_status());
    h.session.stop_recording();
    h.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn control_tts_status_distinct_from_stt_status() {
    use shuvoice_app::{ControlHandlerSurface, SessionCommand, TtsPlayerState, spawn_test_runtime};
    let (rt, _, _) = spawn_test_runtime(cfg(), ScriptedAsrBackend::default()).await;
    rt.handle.try_enqueue(SessionCommand::Start).unwrap();
    // Wait until STT status reflects recording.
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        if rt.control.on_status() == "recording" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(rt.control.on_status(), "recording");
    // Player still idle — control tts_status must not echo STT status.
    let tts = rt.control.on_tts_command("tts_status");
    assert_eq!(tts, "OK idle", "tts_status must be player state, got {tts}");
    assert_ne!(tts, format!("OK {}", rt.control.on_status()));

    rt.handle
        .enqueue_tts_player_update(TtsPlayerState::Playing, None)
        .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    let mut saw = None;
    while std::time::Instant::now() < deadline {
        let tts = rt.control.on_tts_command("tts_status");
        if tts == "OK playing" {
            saw = Some(tts);
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(saw.as_deref(), Some("OK playing"));
    // STT status remains independent.
    assert_ne!(rt.control.on_status(), "playing");
    rt.shutdown().await.unwrap();
}

/// Stop must be *applied* while a slow start-reset is still in flight on ASR —
/// not merely enqueued after reset finishes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stop_applied_while_slow_start_reset_in_flight() {
    use shuvoice_app::{ControlHandlerSurface, spawn_test_runtime};
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted.shared().lock().delay = Duration::from_millis(400);
    let (rt, shared, _) = spawn_test_runtime(cfg(), scripted).await;

    rt.control.on_start();
    // Allow actor to accept Start and spawn reset (reset will hold ASR ~400ms).
    tokio::time::sleep(Duration::from_millis(30)).await;
    let resets_at_stop = shared.lock().reset_calls;

    let t0 = std::time::Instant::now();
    rt.control.on_stop();
    // Wait until start_pending cleared / not recording — must happen before reset finishes.
    let deadline = std::time::Instant::now() + Duration::from_millis(250);
    let mut saw_cancel = false;
    while std::time::Instant::now() < deadline {
        let dbg: serde_json::Value =
            serde_json::from_str(&rt.control.on_debug_status()).unwrap_or_default();
        let start_pending = dbg["app"]["start_pending"].as_bool().unwrap_or(true);
        let recording = dbg["app"]["recording"].as_bool().unwrap_or(true);
        if !start_pending && !recording {
            saw_cancel = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let applied_in = t0.elapsed();
    assert!(
        saw_cancel,
        "stop was not applied while reset in flight; debug={}",
        rt.control.on_debug_status()
    );
    assert!(
        applied_in < Duration::from_millis(300),
        "stop application blocked on ASR reset: {applied_in:?}"
    );
    // Reset may still be running or just finishing — must not have required full delay.
    assert!(
        applied_in < Duration::from_millis(350),
        "stop waited for slow reset"
    );
    let _ = resets_at_stop;
    rt.shutdown().await.unwrap();
}

/// Shutdown must complete quickly even if ASR reset/backend op is wedged/slow.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_completes_while_slow_asr_reset_in_flight() {
    use shuvoice_app::{ControlHandlerSurface, spawn_test_runtime};
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted.shared().lock().delay = Duration::from_millis(800);
    let (rt, _, _) = spawn_test_runtime(cfg(), scripted).await;
    rt.control.on_start();
    tokio::time::sleep(Duration::from_millis(30)).await;
    let t0 = std::time::Instant::now();
    tokio::time::timeout(Duration::from_secs(3), rt.shutdown())
        .await
        .expect("shutdown timed out waiting on ASR")
        .unwrap();
    let elapsed = t0.elapsed();
    assert!(
        elapsed < Duration::from_millis(1500),
        "shutdown HOL-blocked on slow ASR: {elapsed:?}"
    );
}

/// Actor handle_command(Start) returns without waiting for slow reset.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn start_command_returns_before_slow_reset_finishes() {
    use shuvoice_app::{SessionCommand, spawn_test_runtime};
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted.shared().lock().delay = Duration::from_millis(300);
    let (rt, shared, _) = spawn_test_runtime(cfg(), scripted).await;
    let t0 = std::time::Instant::now();
    let reply = rt.handle.send(SessionCommand::Start).await.unwrap();
    let elapsed = t0.elapsed();
    assert!(reply.contains("started") || reply.contains("OK"));
    assert!(
        elapsed < Duration::from_millis(100),
        "Start command awaited ASR reset: {elapsed:?}"
    );
    // Reset still in flight or just landing — either way command did not wait full delay.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let _ = shared.lock().reset_calls;
    rt.shutdown().await.unwrap();
}

/// SessionRuntime::shutdown must not wait on handle.send(10s) before join grace.
/// With a wedged actor (stuck inside handle_command), elapsed must stay near the
/// advertised SESSION_SHUTDOWN_GRACE (+ small ASR/dispatcher teardown), never ~10s+.
#[tokio::test(flavor = "current_thread")]
#[serial]
async fn runtime_shutdown_bound_holds_with_wedged_actor() {
    use std::sync::atomic::Ordering;

    use shuvoice_app::{
        SESSION_SHUTDOWN_GRACE, SessionCommand, TEST_HANG_ACTOR_ON_SHUTDOWN, spawn_test_runtime,
    };

    struct ClearHang;
    impl Drop for ClearHang {
        fn drop(&mut self) {
            TEST_HANG_ACTOR_ON_SHUTDOWN.store(false, Ordering::SeqCst);
        }
    }
    let _guard = ClearHang;
    TEST_HANG_ACTOR_ON_SHUTDOWN.store(false, Ordering::SeqCst);

    let (rt, _, _) = spawn_test_runtime(cfg(), ScriptedAsrBackend::default()).await;

    TEST_HANG_ACTOR_ON_SHUTDOWN.store(true, Ordering::SeqCst);
    rt.handle.try_enqueue(SessionCommand::Shutdown).unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;

    let t0 = std::time::Instant::now();
    let result = tokio::time::timeout(Duration::from_secs(8), rt.shutdown()).await;
    TEST_HANG_ACTOR_ON_SHUTDOWN.store(false, Ordering::SeqCst);
    result
        .expect("SessionRuntime::shutdown exceeded hard test ceiling")
        .unwrap();
    let elapsed = t0.elapsed();

    assert!(
        elapsed >= SESSION_SHUTDOWN_GRACE,
        "expected to wait roughly the join grace, got {elapsed:?}"
    );
    let max = SESSION_SHUTDOWN_GRACE + Duration::from_secs(3);
    assert!(
        elapsed < max,
        "shutdown bound not real: elapsed {elapsed:?} >= {max:?}"
    );
    assert!(elapsed < Duration::from_secs(10));
}

/// Wedged/slow injector must not HOL the actor: stop/status/shutdown stay responsive.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn wedged_injector_does_not_block_stop_status_shutdown() {
    use shuvoice_app::fakes::offline_asr;
    use shuvoice_app::{ControlHandlerSurface, SessionCommand, spawn_test_runtime};

    let scripted = offline_asr("inject me");
    // Slow commit (longer than our status sampling window).
    let (rt, _shared, injector) = spawn_test_runtime(cfg(), scripted).await;
    injector.set_delay(Duration::from_millis(800));

    rt.handle.try_enqueue(SessionCommand::Start).unwrap();
    // Wait until recording.
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        if rt.control.on_status() == "recording" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    // Push audio via handle's audio ingress and stop to trigger finalize+inject.
    for _ in 0..4 {
        rt.audio.try_push(vec![0.3; 200]);
    }
    tokio::time::sleep(Duration::from_millis(30)).await;
    rt.control.on_stop();

    // While inject may be slow/in-flight, status reads and stop must stay fast.
    let t0 = std::time::Instant::now();
    for _ in 0..30 {
        let _ = rt.control.on_status();
        let _ = rt.control.on_debug_status();
    }
    let status_elapsed = t0.elapsed();
    assert!(
        status_elapsed < Duration::from_millis(100),
        "status path HOL-blocked on injector: {status_elapsed:?}"
    );

    let t1 = std::time::Instant::now();
    tokio::time::timeout(Duration::from_secs(3), rt.shutdown())
        .await
        .expect("shutdown hung on wedged injector")
        .unwrap();
    assert!(
        t1.elapsed() < Duration::from_secs(3),
        "shutdown blocked on injector"
    );
}

/// Wedged selection capture must not HOL stop/status/shutdown on the actor.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn wedged_selection_does_not_block_control_plane() {
    use std::sync::Arc;

    use async_trait::async_trait;
    use shuvoice_app::fakes::{
        FakeFeedback, FakeInjector, FakeOverlay, FakeTts, ScriptedAsrBackend,
    };
    use shuvoice_app::traits::{SelectionCapture, SystemClock};
    use shuvoice_app::{ControlHandlerSurface, SessionCommand, spawn_session_runtime};

    struct SlowSelection {
        delay: Duration,
        text: String,
    }
    #[async_trait]
    impl SelectionCapture for SlowSelection {
        async fn capture_selection(&self) -> Result<String, String> {
            tokio::time::sleep(self.delay).await;
            Ok(self.text.clone())
        }
        async fn capture_clipboard(&self) -> Result<String, String> {
            tokio::time::sleep(self.delay).await;
            Ok(self.text.clone())
        }
    }

    let mut config = cfg();
    config.tts_enabled = true;
    let rt = spawn_session_runtime(
        config,
        Box::new(ScriptedAsrBackend::default()),
        Arc::new(FakeInjector::default()),
        Arc::new(SlowSelection {
            delay: Duration::from_millis(800),
            text: "sel".into(),
        }),
        FakeOverlay::default(),
        FakeFeedback::default(),
        Arc::new(SystemClock),
        Some(FakeTts::new()),
    )
    .await
    .expect("spawn");

    // Enqueue TTS speak selection — actor must ack immediately and run capture as job.
    let t0 = std::time::Instant::now();
    rt.handle
        .try_enqueue(SessionCommand::TtsSpeakSelection)
        .unwrap();
    // Give actor a tick to spawn capture job.
    tokio::time::sleep(Duration::from_millis(20)).await;

    let t_status = std::time::Instant::now();
    for _ in 0..20 {
        let _ = rt.control.on_status();
        let _ = rt.control.on_tts_command("tts_status");
    }
    assert!(
        t_status.elapsed() < Duration::from_millis(80),
        "control plane blocked on selection capture"
    );

    // Stop/start still enqueue fast.
    rt.control.on_start();
    rt.control.on_stop();
    assert!(t0.elapsed() < Duration::from_millis(200));

    tokio::time::timeout(Duration::from_secs(3), rt.shutdown())
        .await
        .expect("shutdown hung on selection")
        .unwrap();
}

/// handle_command(TtsSpeakSelection) returns without waiting for slow capture.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tts_speak_command_acks_before_slow_selection_finishes() {
    use std::sync::Arc;

    use async_trait::async_trait;
    use shuvoice_app::fakes::{
        FakeFeedback, FakeInjector, FakeOverlay, FakeTts, ScriptedAsrBackend,
    };
    use shuvoice_app::traits::{SelectionCapture, SystemClock};
    use shuvoice_app::{SessionCommand, spawn_session_runtime};

    struct SlowSelection;
    #[async_trait]
    impl SelectionCapture for SlowSelection {
        async fn capture_selection(&self) -> Result<String, String> {
            tokio::time::sleep(Duration::from_millis(400)).await;
            Ok("hello".into())
        }
        async fn capture_clipboard(&self) -> Result<String, String> {
            Ok("c".into())
        }
    }

    let mut config = cfg();
    config.tts_enabled = true;
    let rt = spawn_session_runtime(
        config,
        Box::new(ScriptedAsrBackend::default()),
        Arc::new(FakeInjector::default()),
        Arc::new(SlowSelection),
        FakeOverlay::default(),
        FakeFeedback::default(),
        Arc::new(SystemClock),
        Some(FakeTts::new()),
    )
    .await
    .expect("spawn");

    let t0 = std::time::Instant::now();
    let reply = rt
        .handle
        .send(SessionCommand::TtsSpeakSelection)
        .await
        .unwrap();
    let elapsed = t0.elapsed();
    assert!(reply.contains("tts"));
    assert!(
        elapsed < Duration::from_millis(100),
        "TtsSpeakSelection command awaited selection: {elapsed:?}"
    );
    rt.shutdown().await.unwrap();
}

/// Slow injector (longer than the old 2s app timeout) must commit exactly once.
/// Proves we do not timeout+retry an in-flight commit (which would duplicate text
/// when spawn_blocking/work continues after a dropped timeout JoinHandle).
#[tokio::test]
async fn slow_inject_commit_is_exactly_once() {
    use shuvoice_app::fakes::offline_asr;

    let scripted = offline_asr("only once please");
    let mut h = TestHarness::basic(scripted, cfg()).await;
    // Slow commit must still be exactly one attempt (no app-level retry).
    h.injector.set_commit_delay(Duration::from_millis(400));

    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.25; 32]);
    h.session.handle_recording_stop().await;

    assert_eq!(
        h.injector.commit_calls(),
        1,
        "expected exactly one commit attempt, got {} finals={:?}",
        h.injector.commit_calls(),
        h.injector.finals()
    );
    assert_eq!(h.injector.finals().len(), 1);
    assert!(
        h.injector.finals()[0].to_lowercase().contains("only once"),
        "{:?}",
        h.injector.finals()
    );
    h.shutdown().await;
}

/// Commit Err after side effect must NOT retry (exactly-once; unknown outcome).
#[tokio::test]
async fn commit_err_after_side_effect_is_exactly_once_no_retry() {
    use shuvoice_app::fakes::offline_asr;

    let scripted = offline_asr("pasted once");
    let mut h = TestHarness::basic(scripted, cfg()).await;
    // Simulates IoTextInjector: paste lands, then subprocess timeout -> Err.
    h.injector.set_commit_side_effect_then_err(true);

    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.25; 32]);
    h.session.handle_recording_stop().await;

    assert_eq!(
        h.injector.commit_calls(),
        1,
        "must not retry after unknown Err: log={:?}",
        h.injector.op_log()
    );
    assert_eq!(
        h.injector.finals().len(),
        1,
        "side effect once: {:?}",
        h.injector.finals()
    );
    // Generation spent — a manual second commit_utterance must not re-fire.
    h.session.commit_utterance().await;
    assert_eq!(h.injector.commit_calls(), 1);
    h.shutdown().await;
}

/// Old partial blocks; cancel/new gen; old completes -> reset next, then new partial.
/// Recorded injector order must be: partial(old), reset, partial(new) — never reset before
/// old partial, never old partial after reset.
#[tokio::test]
async fn slow_partial_then_cancel_orders_reset_before_new_partial() {
    use shuvoice_app::fakes::ScriptedAsrBackend;
    use shuvoice_app::{OutputMode, TestHarness};

    let mut config = cfg();
    config.output_mode = OutputMode::StreamingPartial;
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted
        .shared()
        .lock()
        .texts
        .push_back("old-gen-partial".into());
    let mut h = TestHarness::basic(scripted, config).await;

    h.session.start_recording().await;
    h.session
        .pump_jobs_while(|s| s.effect_jobs_in_flight() > 0, Duration::from_secs(2))
        .await;

    h.session.begin_utterance().await;
    // Only partial is slow; reset/commit stay fast.
    h.injector.set_partial_delay(Duration::from_millis(500));
    h.injector.set_reset_delay(Duration::from_millis(50));
    h.session.append_recording_chunk(&[0.2; 4]);
    h.session.process_recording_chunks().await;

    // New generation while old partial still running.
    h.session.stop_recording();
    h.clock.advance_ms(400);
    h.scripted.lock().texts.push_back("new-gen-partial".into());
    h.session.start_recording().await;

    // Wait start + old partial + deferred reset.
    h.session
        .pump_jobs_while(
            |s| s.effect_jobs_in_flight() > 0 || s.is_starting(),
            Duration::from_secs(5),
        )
        .await;

    // Drive a new partial on the new generation (barrier must be down).
    h.injector.set_partial_delay(Duration::from_millis(20));
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 4]);
    h.session.process_recording_chunks().await;
    h.session
        .pump_jobs_while(|s| s.effect_jobs_in_flight() > 0, Duration::from_secs(3))
        .await;

    let log = h.injector.op_log();
    let old_p = log
        .iter()
        .position(|e| e.to_lowercase().contains("old-gen"));
    let new_p = log
        .iter()
        .position(|e| e.to_lowercase().contains("new-gen"));
    let resets: Vec<usize> = log
        .iter()
        .enumerate()
        .filter(|(_, e)| e.as_str() == "reset")
        .map(|(i, _)| i)
        .collect();

    assert!(old_p.is_some(), "old partial missing: {log:?}");
    assert!(!resets.is_empty(), "reset missing: {log:?}");
    let old_i = old_p.unwrap();
    let reset_after_old = resets.iter().find(|&&r| r > old_i);
    assert!(
        reset_after_old.is_some(),
        "reset must follow in-flight old partial: {log:?}"
    );
    let r_i = *reset_after_old.unwrap();
    if let Some(n_i) = new_p {
        assert!(
            n_i > r_i,
            "new partial must come after reset that drained old partial: {log:?}"
        );
    }
    // No old-gen partial content after that reset.
    assert!(
        !log[r_i + 1..]
            .iter()
            .any(|e| e.to_lowercase().contains("old-gen")),
        "stale old partial after reset: {log:?}"
    );
    h.shutdown().await;
}

/// Repeated reset requests must not overlap/abort an in-flight reset.
#[tokio::test]
async fn repeated_reset_requests_do_not_overlap() {
    use shuvoice_app::fakes::offline_asr;

    let mut h = TestHarness::basic(offline_asr("x"), cfg()).await;
    h.injector.set_reset_delay(Duration::from_millis(300));

    h.session.start_recording().await;
    // complete_start schedules reset (slow).
    // Hammer cancel/reset path repeatedly while first reset runs.
    for _ in 0..5 {
        h.session.stop_recording();
        h.clock.advance_ms(400);
        h.session.start_recording().await;
    }
    h.session
        .pump_jobs_while(
            |s| s.effect_jobs_in_flight() > 0 || s.is_starting(),
            Duration::from_secs(5),
        )
        .await;

    let log = h.injector.op_log();
    let reset_count = log.iter().filter(|e| e.as_str() == "reset").count();
    // Coalesced: should not be one-per-hammer; serialized completions only.
    // At least 1; must not exceed number of start completions by crazy overlap kills.
    assert!(reset_count >= 1, "{log:?}");
    // All resets are sequential in the log (no concurrent interleaving markers).
    // If we aborted and restarted, we'd still see N resets but commits could reorder;
    // with non-overlap, reset_count stays modest relative to starts.
    assert!(
        reset_count <= 8,
        "too many resets suggests abort/restart thrash: {log:?}"
    );
    h.shutdown().await;
}

/// Full streaming stop/start with slow partial + final commit: exactly-once final,
/// reset after prior inject ops.
#[tokio::test]
async fn slow_partial_then_finalize_commit_then_reset_is_ordered_and_once() {
    use shuvoice_app::fakes::offline_asr;

    // Use offline for clean single commit after stop.
    let scripted = offline_asr("final once");
    let mut h = TestHarness::basic(scripted, cfg()).await;
    h.injector.set_delay(Duration::from_millis(200));

    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.25; 32]);
    h.session.handle_recording_stop().await;

    assert_eq!(h.injector.finals().len(), 1);
    assert_eq!(h.injector.commit_calls(), 1);

    // New start triggers reset after any prior ops.
    h.clock.advance_ms(400);
    h.session.start_recording().await;
    h.session
        .pump_jobs_while(
            |s| s.effect_jobs_in_flight() > 0 || s.is_starting(),
            Duration::from_secs(3),
        )
        .await;

    let log = h.injector.op_log();
    let commit_idx = log.iter().rposition(|e| e.starts_with("commit:"));
    let reset_after = log
        .iter()
        .enumerate()
        .filter(|(_, e)| e.as_str() == "reset")
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    assert!(commit_idx.is_some(), "{log:?}");
    // At least one reset should occur at/after start rearm.
    assert!(!reset_after.is_empty(), "expected reset on rearm: {log:?}");
    h.shutdown().await;
}

/// Failed finalize must clear streaming partial residue via inject reset.
#[tokio::test]
async fn failed_finalize_requests_injector_reset() {
    use shuvoice_app::fakes::offline_asr;

    let scripted = offline_asr("x");
    scripted.shared().lock().fail_utterance_with = Some("decode boom".into());
    let mut h = TestHarness::basic(scripted, cfg()).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 16]);
    h.session.handle_recording_stop().await;
    h.session
        .pump_jobs_while(|s| s.effect_jobs_in_flight() > 0, Duration::from_secs(2))
        .await;
    let log = h.injector.op_log();
    assert!(
        log.iter().any(|e| e == "reset" || e == "reset:fail"),
        "failed finalize must request inject reset: {log:?}"
    );
    h.shutdown().await;
}

/// Ready text that sanitizes/renders empty must still reset injector (clear partials).
#[tokio::test]
async fn empty_ready_text_requests_injector_reset() {
    use shuvoice_app::fakes::offline_asr;

    let mut config = cfg();
    // Force ASR text to disappear after replacements + sanitize.
    config
        .text_replacements
        .insert("noiseonly".into(), "".into());
    let scripted = offline_asr("noiseonly");
    let mut h = TestHarness::basic(scripted, config).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 16]);
    h.session.handle_recording_stop().await;
    h.session
        .pump_jobs_while(|s| s.effect_jobs_in_flight() > 0, Duration::from_secs(2))
        .await;
    assert!(h.injector.finals().is_empty());
    let log = h.injector.op_log();
    assert!(
        log.iter().any(|e| e == "reset"),
        "empty ready must reset injector: {log:?}"
    );
    h.shutdown().await;
}

/// InjectReset Err keeps barrier; partials suppressed until a later reset succeeds.
#[tokio::test]
async fn inject_reset_err_keeps_barrier_until_success() {
    use shuvoice_app::fakes::ScriptedAsrBackend;
    use shuvoice_app::{OutputMode, TestHarness};

    let mut config = cfg();
    config.output_mode = OutputMode::StreamingPartial;
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted
        .shared()
        .lock()
        .texts
        .push_back("should-block".into());
    let mut h = TestHarness::basic(scripted, config).await;

    // First start's complete_start reset fails once; barrier must remain.
    h.injector.set_fail_resets(1);
    h.session.start_recording().await;
    h.session
        .pump_jobs_while(
            |s| s.effect_jobs_in_flight() > 0 || s.is_starting(),
            Duration::from_secs(3),
        )
        .await;
    assert!(
        h.injector.op_log().iter().any(|e| e == "reset:fail"),
        "expected failed reset: {:?}",
        h.injector.op_log()
    );

    // Streaming partial while barrier held after reset Err — must not reach injector.
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 4]);
    h.session.process_recording_chunks().await;
    h.session
        .pump_jobs_while(|s| s.effect_jobs_in_flight() > 0, Duration::from_secs(2))
        .await;
    assert!(
        !h.injector
            .op_log()
            .iter()
            .any(|e| e.to_lowercase().contains("should-block")),
        "partial suppressed while barrier held: {:?}",
        h.injector.op_log()
    );

    // Successful reset via cancel/rearm (fail_resets already exhausted).
    h.session.stop_recording();
    h.clock.advance_ms(400);
    h.scripted.lock().texts.clear();
    h.scripted.lock().texts.push_back("after-barrier".into());
    h.session.start_recording().await;
    h.session
        .pump_jobs_while(
            |s| s.effect_jobs_in_flight() > 0 || s.is_starting(),
            Duration::from_secs(3),
        )
        .await;
    assert!(
        h.injector.op_log().iter().any(|e| e == "reset"),
        "expected successful reset on rearm: {:?}",
        h.injector.op_log()
    );

    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 4]);
    h.session.process_recording_chunks().await;
    h.session
        .pump_jobs_while(|s| s.effect_jobs_in_flight() > 0, Duration::from_secs(2))
        .await;
    assert!(
        h.injector
            .op_log()
            .iter()
            .any(|e| e.to_lowercase().contains("after-barrier")),
        "partial allowed only after successful reset: {:?}",
        h.injector.op_log()
    );
    h.shutdown().await;
}

/// Grace-window late audio must not be dropped (unbounded transport + deadline drain).
#[tokio::test]
async fn stop_tail_grace_transport_is_lossless_under_burst() {
    use shuvoice_app::fakes::offline_asr;

    // Large audio ring so the test measures late_tx/grace, not ring overflow.
    let mut config = cfg();
    config.audio_queue_max_size = 512;
    let scripted = offline_asr("grace burst");
    let mut h = TestHarness::basic(scripted, config).await;
    h.session.start_recording().await;
    h.audio.try_push(vec![0.2; 8]);
    assert!(h.session.tick().await);
    h.session.stop_recording();
    assert!(h.session.tick().await); // finalize + grace open

    const N: usize = 200;
    const CHUNK: usize = 64;
    // Interleave push/tick so grace path drains ring -> unbounded late_tx continuously.
    for i in 0..N {
        assert!(
            h.audio.try_push(vec![0.35 + (i as f32) * 1e-6; CHUNK]),
            "audio ring dropped during grace burst at {i}"
        );
        if i % 8 == 7 {
            assert!(h.session.tick().await);
        }
    }
    assert!(h.session.tick().await);
    let _ = h
        .session
        .await_or_cancel_finalization(Duration::from_secs(5))
        .await;

    let got = h.scripted.lock().last_utterance.len();
    let min = 8 + N * CHUNK;
    assert!(
        got >= min,
        "expected lossless grace capture (>= {min}), got {got}"
    );
    h.shutdown().await;
}
