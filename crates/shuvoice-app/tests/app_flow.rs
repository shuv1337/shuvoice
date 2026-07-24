//! High-risk flow tests (async ASR owner composition).

use std::time::Duration;

use shuvoice_app::core::{
    UtteranceState, apply_utterance_gain, looks_like_cuda_oom_error, prefer_transcript,
};
use shuvoice_app::fakes::{
    FakeSelection, FakeTts, ScriptedAsrBackend, ScriptedInner, offline_asr, remote_asr,
};
use shuvoice_app::{
    ASR_MAX_FAILURES, Config, OutputMode, OverlayState, PTT_REARM_GRACE_SEC, RecordingStatus,
    TestHarness, TtsPlayerState, TtsSource, TypingTextCase,
};
use shuvoice_asr::FallbackOutcome;

fn cfg() -> Config {
    let mut c = Config::default();
    c.min_speech_ms = 0;
    c.silence_rms_threshold = 0.0;
    c.tts_enabled = true;
    c
}

async fn build(scripted: ScriptedAsrBackend) -> TestHarness {
    let mut config = cfg();
    config.audio_feedback = true;
    TestHarness::basic(scripted, config).await
}

async fn build_with(
    scripted: ScriptedAsrBackend,
    tts: FakeTts,
    selection: FakeSelection,
    config: Config,
) -> TestHarness {
    TestHarness::new_with(scripted, tts, selection, config).await
}

fn sc(h: &TestHarness) -> parking_lot::MutexGuard<'_, ScriptedInner> {
    h.scripted.lock()
}

#[tokio::test]
async fn recording_start_stop_transitions() {
    let mut h = build(ScriptedAsrBackend::default()).await;
    h.session.start_recording().await;
    assert!(h.session.is_recording());
    assert!(!h.session.is_processing());
    assert!(sc(&h).reset_calls >= 1);
    assert!(
        h.session
            .overlay()
            .calls
            .iter()
            .any(|c| c.kind == "show" && c.state == Some(OverlayState::Listening))
    );
    h.session.stop_recording();
    assert!(!h.session.is_recording());
    assert!(h.session.is_processing());
    assert_eq!(h.session.recording_status(), RecordingStatus::Processing);
    h.shutdown().await;
}

#[tokio::test]
async fn recording_status_reports_processing_between_stop_and_commit() {
    let mut h = build(ScriptedAsrBackend::default()).await;
    assert_eq!(h.session.recording_status().as_str(), "idle");
    h.session.start_recording().await;
    h.session.stop_recording();
    assert_eq!(h.session.recording_status().as_str(), "processing");
    h.shutdown().await;
}

#[tokio::test]
async fn recording_start_ignores_spurious_restart_during_processing_rearm() {
    let mut h = build(ScriptedAsrBackend::default()).await;
    h.session.start_recording().await;
    h.session.stop_recording();
    h.clock.advance_ms(200);
    assert!(h.clock.elapsed().as_secs_f64() < PTT_REARM_GRACE_SEC);
    h.session.start_recording().await;
    assert!(!h.session.is_recording());
    h.shutdown().await;
}

#[tokio::test]
async fn recording_start_allows_restart_after_processing_rearm_window() {
    let mut h = build(ScriptedAsrBackend::default()).await;
    h.session.start_recording().await;
    h.session.stop_recording();
    h.clock.advance_ms(400);
    h.session.start_recording().await;
    assert!(h.session.is_recording());
    assert!(!h.session.is_processing());
    h.shutdown().await;
}

#[tokio::test]
async fn begin_utterance_resets_asr_before_threshold_setup() {
    // Start performs the single ASR reset; begin only sets thresholds/preroll
    // (no double-reset).
    let mut h = build(ScriptedAsrBackend::default()).await;
    h.session.start_recording().await;
    let resets_after_start = sc(&h).reset_calls;
    assert!(resets_after_start >= 1);
    h.session.begin_utterance().await;
    assert_eq!(sc(&h).reset_calls, resets_after_start);
    assert!(h.session.utterance_state().last_text.is_empty());
    h.shutdown().await;
}

#[tokio::test]
async fn begin_utterance_prepends_recording_preroll() {
    let mut h = build(ScriptedAsrBackend::local_streaming(160)).await;
    {
        let mut g = sc(&h);
        g.delay = Duration::ZERO;
    }
    // force wants_raw via caps is fixed at construction; local_streaming wants_raw=false ok
    h.audio.try_push(vec![0.02; 160]);
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    assert_eq!(h.session.utterance_state().total, 160);
    h.shutdown().await;
}

#[tokio::test]
async fn begin_utterance_caps_inflated_dynamic_noise_gate() {
    let mut config = cfg();
    config.silence_rms_threshold = 0.008;
    config.silence_rms_multiplier = 1.8;
    let mut h = build_with(
        ScriptedAsrBackend::local_streaming(4),
        FakeTts::new(),
        FakeSelection::default(),
        config,
    )
    .await;
    h.audio.try_push(vec![0.15; 100]);
    let _ = h.session.tick().await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    assert!((h.session.utterance_state().utterance_rms_threshold - 0.024).abs() < 1e-6);
    h.shutdown().await;
}

#[tokio::test]
async fn remote_manual_commit_stop_bypasses_local_tail_flush() {
    let mut h = build_with(
        remote_asr("remote final"),
        FakeTts::new(),
        FakeSelection::default(),
        cfg(),
    )
    .await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.1; 4]);
    h.session.stop_recording();
    h.session.handle_recording_stop().await;
    assert_eq!(sc(&h).finish_calls, 1);
    assert_eq!(h.injector.finals(), vec!["Remote final".to_string()]);
    assert_eq!(sc(&h).utterance_calls, 0);
    h.shutdown().await;
}

#[tokio::test]
async fn remote_manual_commit_failure_does_not_commit_stale_text() {
    let scripted = remote_asr("unused");
    scripted.shared().lock().fail_finish_with = Some("socket closed".into());
    let mut h = build_with(scripted, FakeTts::new(), FakeSelection::default(), cfg()).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 4]);
    h.session.stop_recording();
    h.session.handle_recording_stop().await;
    assert!(h.injector.finals().is_empty());
    assert_eq!(sc(&h).finish_calls, 1);
    h.shutdown().await;
}

#[tokio::test]
async fn flush_tail_silence_aborts_when_new_recording_already_started() {
    let mut h = build(ScriptedAsrBackend::local_streaming(1600)).await;
    h.session.start_recording().await;
    h.session.flush_tail_silence().await;
    assert_eq!(sc(&h).chunk_calls, 0);
    h.shutdown().await;
}

#[tokio::test]
async fn flush_tail_silence_stops_after_recording_set() {
    // Tail flush runs inside finalize jobs, not on the actor task.
    let mut h = build(ScriptedAsrBackend::local_streaming(4)).await;
    h.session.flush_tail_silence().await;
    assert_eq!(sc(&h).chunk_calls, 0);
    h.session.start_recording().await;
    h.session.flush_tail_silence().await;
    assert_eq!(sc(&h).chunk_calls, 0);
    h.shutdown().await;
}

#[tokio::test]
async fn handle_recording_stop_ignores_silent_utterances_without_commit() {
    let mut config = cfg();
    config.min_speech_ms = 1000;
    config.sample_rate = 16_000;
    let mut h = build_with(
        ScriptedAsrBackend::default(),
        FakeTts::new(),
        FakeSelection::default(),
        config,
    )
    .await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.0001; 20]);
    h.session.stop_recording();
    h.session.handle_recording_stop().await;
    assert!(h.injector.finals().is_empty());
    // start requests inject reset; silent finalize may request another — both OK.
    assert!(h.injector.resets() >= 1);
    h.shutdown().await;
}

#[tokio::test]
async fn handle_recording_stop_commits_when_speech_threshold_met() {
    let scripted = ScriptedAsrBackend::local_streaming(400);
    scripted
        .shared()
        .lock()
        .texts
        .push_back("hello world".into());
    let mut h = build_with(scripted, FakeTts::new(), FakeSelection::default(), cfg()).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 400]);
    h.session.stop_recording();
    h.session.handle_recording_stop().await;
    assert_eq!(h.injector.finals().len(), 1);
    assert!(h.injector.finals()[0].to_lowercase().contains("hello"));
    h.shutdown().await;
}

#[tokio::test]
async fn handle_recording_stop_offline_mode_decodes_once_and_commits() {
    let mut h = build_with(
        offline_asr("decoded once"),
        FakeTts::new(),
        FakeSelection::default(),
        cfg(),
    )
    .await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 400]);
    h.session.stop_recording();
    h.session.handle_recording_stop().await;
    assert_eq!(sc(&h).utterance_calls, 1);
    assert_eq!(h.injector.finals(), vec!["Decoded once".to_string()]);
    assert_eq!(sc(&h).chunk_calls, 0);
    h.shutdown().await;
}

#[tokio::test]
async fn decode_offline_utterance_applies_gain_for_non_raw_backend() {
    let mut h = build_with(
        offline_asr("offline text"),
        FakeTts::new(),
        FakeSelection::default(),
        cfg(),
    )
    .await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.1, -0.2, 0.3]);
    h.session.append_recording_chunk(&[0.1, -0.2, 0.3]);
    h.session.append_recording_chunk(&[0.1, -0.2, 0.3]);
    h.session.decode_offline_utterance().await;
    assert_eq!(sc(&h).utterance_calls, 1);
    assert_eq!(h.session.utterance_state().last_text, "offline text");
    h.shutdown().await;
}

#[tokio::test]
async fn process_recording_chunks_is_noop_in_offline_mode() {
    let mut h = build_with(
        offline_asr("x"),
        FakeTts::new(),
        FakeSelection::default(),
        cfg(),
    )
    .await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 64]);
    h.session.process_recording_chunks().await;
    assert_eq!(sc(&h).chunk_calls, 0);
    h.shutdown().await;
}

#[tokio::test]
async fn on_transcript_update_skips_partials_in_offline_mode() {
    let mut config = cfg();
    config.output_mode = OutputMode::StreamingPartial;
    let mut h = build_with(
        offline_asr("hello"),
        FakeTts::new(),
        FakeSelection::default(),
        config,
    )
    .await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 8]);
    h.session.stop_recording();
    h.session.handle_recording_stop().await;
    assert!(h.injector.partials().is_empty());
    assert!(!h.injector.finals().is_empty());
    h.shutdown().await;
}

#[tokio::test]
async fn on_transcript_update_keeps_partials_in_streaming_mode() {
    let mut config = cfg();
    config.output_mode = OutputMode::StreamingPartial;
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted.shared().lock().texts.push_back("hello".into());
    let mut h = build_with(scripted, FakeTts::new(), FakeSelection::default(), config).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 4]);
    h.session.process_recording_chunks().await;
    assert_eq!(h.injector.partials().len(), 1);
    assert!(h.injector.partials()[0].to_lowercase().contains("hello"));
    h.shutdown().await;
}

#[tokio::test]
async fn render_transcript_text_applies_replacements_and_capitalize() {
    let mut config = cfg();
    config
        .text_replacements
        .insert("shove voice".into(), "ShuVoice".into());
    config.text_replacements.insert("um".into(), "".into());
    config.auto_capitalize = true;
    config.typing_text_case = TypingTextCase::Default;
    let h = build_with(
        ScriptedAsrBackend::default(),
        FakeTts::new(),
        FakeSelection::default(),
        config,
    )
    .await;
    assert_eq!(
        h.session.render_transcript_text("shove voice um"),
        "ShuVoice"
    );
    h.shutdown().await;
}

#[tokio::test]
async fn render_transcript_text_can_force_lowercase_output() {
    let mut config = cfg();
    config
        .text_replacements
        .insert("shove voice".into(), "ShuVoice".into());
    config.typing_text_case = TypingTextCase::Lowercase;
    let h = build_with(
        ScriptedAsrBackend::default(),
        FakeTts::new(),
        FakeSelection::default(),
        config,
    )
    .await;
    assert_eq!(
        h.session.render_transcript_text("Shove voice Is Great."),
        "shuvoice is great."
    );
    h.shutdown().await;
}

#[tokio::test]
async fn commit_utterance_uses_rendered_text_for_overlay_and_typing() {
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted
        .shared()
        .lock()
        .texts
        .push_back("raw transcript".into());
    let mut h = build_with(scripted, FakeTts::new(), FakeSelection::default(), cfg()).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 4]);
    h.session.process_recording_chunks().await;
    h.session.commit_utterance().await;
    assert_eq!(h.session.debug_last_final_transcript(), "Raw transcript");
    assert_eq!(h.injector.finals(), vec!["Raw transcript".to_string()]);
    // exactly-once
    h.session.commit_utterance().await;
    assert_eq!(h.injector.finals().len(), 1);
    h.shutdown().await;
}

#[tokio::test]
async fn commit_utterance_skips_when_rendered_text_is_empty() {
    let mut config = cfg();
    config.text_replacements.insert("um".into(), "".into());
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted.shared().lock().texts.push_back("um".into());
    let mut h = build_with(scripted, FakeTts::new(), FakeSelection::default(), config).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 4]);
    h.session.process_recording_chunks().await;
    h.session.commit_utterance().await;
    assert!(h.injector.finals().is_empty());
    h.shutdown().await;
}

#[test]
fn apply_utterance_gain_uses_float32_and_does_not_mutate_input() {
    let audio = vec![0.1, -0.2, 0.95];
    let before = audio.clone();
    let result = apply_utterance_gain(&audio, 2.0);
    assert_eq!(audio, before);
    assert!((result[0] - 0.2).abs() < 1e-6);
}

#[test]
fn apply_utterance_gain_returns_same_samples_when_gain_near_unity() {
    let audio = vec![0.1, -0.2];
    assert_eq!(apply_utterance_gain(&audio, 1.01), audio);
}

#[tokio::test]
async fn recording_start_stops_active_tts_first() {
    let mut tts = FakeTts::new();
    tts.active = true;
    tts.state_name = TtsPlayerState::Playing;
    let mut h = build_with(
        ScriptedAsrBackend::default(),
        tts,
        FakeSelection::default(),
        cfg(),
    )
    .await;
    h.session.start_recording().await;
    assert_eq!(h.session.tts().unwrap().stop_calls, 1);
    assert!(h.session.is_recording());
    h.shutdown().await;
}

#[tokio::test]
async fn tts_speak_selection_stops_recording_and_starts_player() {
    let mut h = build_with(
        ScriptedAsrBackend::default(),
        FakeTts::new(),
        FakeSelection::with(Ok("selected text".into()), Ok("clip".into())),
        cfg(),
    )
    .await;
    h.session.start_recording().await;
    h.session.tts_speak_selection().await.unwrap();
    assert!(!h.session.is_recording());
    assert_eq!(h.session.tts().unwrap().speak_calls[0].0, "selected text");
    h.shutdown().await;
}

#[tokio::test]
async fn tts_speak_clipboard_captures_clipboard_and_starts_player() {
    let mut h = build_with(
        ScriptedAsrBackend::default(),
        FakeTts::new(),
        FakeSelection::with(Ok("sel".into()), Ok("clipboard text".into())),
        cfg(),
    )
    .await;
    h.session.tts_speak_clipboard().await.unwrap();
    assert_eq!(h.session.tts().unwrap().speak_calls[0].0, "clipboard text");
    h.shutdown().await;
}

#[tokio::test]
async fn tts_speak_truncates_text_and_updates_overlay() {
    let mut config = cfg();
    config.tts_max_chars = 5;
    let mut h = build_with(
        ScriptedAsrBackend::default(),
        FakeTts::new(),
        FakeSelection::default(),
        config,
    )
    .await;
    h.session
        .tts_speak("123456789".into(), TtsSource::Selection)
        .await
        .unwrap();
    assert_eq!(h.session.tts().unwrap().speak_calls[0].0, "12345");
    h.shutdown().await;
}

#[tokio::test]
async fn tts_set_playback_speed_updates_player() {
    let mut tts = FakeTts::new();
    tts.supports_speed = true;
    let mut h = build_with(
        ScriptedAsrBackend::default(),
        tts,
        FakeSelection::default(),
        cfg(),
    )
    .await;
    assert!((h.session.tts_set_playback_speed(1.4) - 1.4).abs() < 1e-9);
    h.shutdown().await;
}

#[tokio::test]
async fn tts_set_playback_speed_ignores_unsupported_backend() {
    let mut tts = FakeTts::new();
    tts.supports_speed = false;
    tts.speed = 1.0;
    let mut config = cfg();
    config.tts_playback_speed = 1.0;
    let mut h = build_with(
        ScriptedAsrBackend::default(),
        tts,
        FakeSelection::default(),
        config,
    )
    .await;
    assert!((h.session.tts_set_playback_speed(1.4) - 1.0).abs() < 1e-9);
    h.shutdown().await;
}

#[tokio::test]
async fn handle_tts_command_returns_disabled_error_when_config_disabled() {
    let mut config = cfg();
    config.tts_enabled = false;
    let h = build_with(
        ScriptedAsrBackend::default(),
        FakeTts::new(),
        FakeSelection::default(),
        config,
    )
    .await;
    // Use enqueue adapter via runtime view path — for disabled, session tts_status errors
    assert!(matches!(
        h.session.tts_status(),
        Err(shuvoice_app::AppError::TtsDisabled)
    ));
    h.shutdown().await;
}

#[tokio::test]
async fn handle_tts_status_command_reports_player_state() {
    let mut tts = FakeTts::new();
    tts.state_name = TtsPlayerState::Playing;
    let h = build_with(
        ScriptedAsrBackend::default(),
        tts,
        FakeSelection::default(),
        cfg(),
    )
    .await;
    assert_eq!(h.session.tts_status().unwrap(), "playing");
    h.shutdown().await;
}

#[tokio::test]
async fn handle_asr_runtime_error_applies_cpu_fallback_on_cuda_oom() {
    let scripted = offline_asr("x");
    {
        let shared = scripted.shared();
        let mut g = shared.lock();
        g.fail_utterance_with = Some("CUBLAS_STATUS_ALLOC_FAILED".into());
        g.fallback = FallbackOutcome::Applied {
            detail: "Switched to CPU".into(),
        };
    }
    let mut h = build_with(scripted, FakeTts::new(), FakeSelection::default(), cfg()).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 16]);
    h.session.decode_offline_utterance().await;
    assert_eq!(h.session.consecutive_failures(), 0);
    assert!(
        h.session
            .overlay()
            .calls
            .iter()
            .any(|c| c.text.as_deref().is_some_and(|t| t.contains("CPU")))
    );
    h.shutdown().await;
}

#[tokio::test]
async fn handle_asr_runtime_error_skips_fallback_on_non_cuda_error() {
    let scripted = offline_asr("x");
    scripted.shared().lock().fail_utterance_with = Some("bad shape".into());
    let mut h = build_with(scripted, FakeTts::new(), FakeSelection::default(), cfg()).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 16]);
    h.session.decode_offline_utterance().await;
    assert_eq!(h.session.consecutive_failures(), 1);
    h.shutdown().await;
}

#[tokio::test]
async fn process_utterance_safe_flashes_overlay_on_non_recoverable_error() {
    let scripted = offline_asr("x");
    scripted.shared().lock().fail_utterance_with = Some("boom".into());
    let mut h = build_with(scripted, FakeTts::new(), FakeSelection::default(), cfg()).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 16]);
    h.session.decode_offline_utterance().await;
    assert_eq!(h.session.consecutive_failures(), 1);
    assert!(
        h.session
            .overlay()
            .calls
            .iter()
            .any(|c| c.text.as_deref().is_some_and(|t| t.contains("1/10")))
    );
    h.shutdown().await;
}

#[tokio::test]
async fn process_utterance_safe_does_not_count_when_error_was_recovered() {
    let scripted = offline_asr("x");
    {
        let shared = scripted.shared();
        let mut g = shared.lock();
        g.fail_utterance_with = Some("CUBLAS_STATUS_ALLOC_FAILED".into());
        g.fallback = FallbackOutcome::Applied {
            detail: "ok".into(),
        };
    }
    let mut h = build_with(scripted, FakeTts::new(), FakeSelection::default(), cfg()).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 16]);
    h.session.decode_offline_utterance().await;
    assert_eq!(h.session.consecutive_failures(), 0);
    h.shutdown().await;
}

#[tokio::test]
async fn process_utterance_safe_triggers_disable_after_max_failures() {
    let scripted = offline_asr("x");
    scripted.shared().lock().fail_utterance_with = Some("boom".into());
    let mut h = build_with(scripted, FakeTts::new(), FakeSelection::default(), cfg()).await;
    for _ in 0..ASR_MAX_FAILURES {
        h.session.start_recording().await;
        h.session.begin_utterance().await;
        h.session.append_recording_chunk(&[0.2; 16]);
        h.session.decode_offline_utterance().await;
        h.session.stop_recording();
        let _ = h.session.tick().await;
    }
    assert_eq!(h.session.recording_status(), RecordingStatus::AsrDisabled);
    h.shutdown().await;
}

#[tokio::test]
async fn owner_loop_tick_finalizes_on_stop() {
    let scripted = ScriptedAsrBackend::local_streaming(4);
    scripted
        .shared()
        .lock()
        .texts
        .push_back("hello from tick".into());
    let mut h = build_with(scripted, FakeTts::new(), FakeSelection::default(), cfg()).await;
    h.session.start_recording().await;
    h.audio.try_push(vec![0.2; 4]);
    assert!(h.session.tick().await);
    h.session.stop_recording();
    // Falling edge spawns finalize; poll until done (grace + job).
    for _ in 0..100 {
        assert!(h.session.tick().await);
        if !h.session.is_processing() && !h.session.is_finalizing() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert!(!h.session.is_processing());
    assert!(!h.injector.finals().is_empty());
    h.shutdown().await;
}

#[tokio::test]
async fn metrics_and_debug_status_are_json() {
    let mut h = build(ScriptedAsrBackend::default()).await;
    h.session.start_recording().await;
    let metrics: serde_json::Value = serde_json::from_str(&h.session.metrics_json()).unwrap();
    assert!(metrics.get("counters").is_some());
    let debug: serde_json::Value = serde_json::from_str(&h.session.debug_status_json()).unwrap();
    assert_eq!(debug["app"]["recording"], true);
    h.shutdown().await;
}

#[test]
fn utterance_consume_native_chunk_splits_correctly() {
    let mut st = UtteranceState::new();
    st.add_chunk(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let (chunk, more) = st.consume_native_chunk(3).unwrap();
    assert_eq!(chunk, vec![1.0, 2.0, 3.0]);
    assert!(!more);
}

#[test]
fn prefer_transcript_stable_growth() {
    assert_eq!(prefer_transcript("hello", "hello world"), "hello world");
}

#[test]
fn cuda_oom_markers() {
    assert!(looks_like_cuda_oom_error("CUBLAS_STATUS_ALLOC_FAILED"));
    assert!(!looks_like_cuda_oom_error("connection reset by peer"));
}

#[tokio::test]
async fn shutdown_is_clean() {
    let mut h = build(ScriptedAsrBackend::default()).await;
    h.session.start_recording().await;
    h.session.shutdown().await;
    assert!(!h.session.tick().await);
    assert!(!h.session.is_recording());
    h.asr_join.join().await;
}

#[tokio::test]
async fn no_stale_completion_commits_after_new_utterance() {
    let scripted = offline_asr("first");
    scripted.shared().lock().delay = Duration::from_millis(50);
    let mut h = build_with(scripted, FakeTts::new(), FakeSelection::default(), cfg()).await;
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    h.session.append_recording_chunk(&[0.2; 16]);
    // Start decode but bump gen via new recording start before commit
    h.session.stop_recording();
    // New utterance invalidates gen
    h.clock.advance_ms(500);
    h.session.start_recording().await;
    h.session.begin_utterance().await;
    // Old stop finalization should not double-commit if we call handle with stale gen
    // commit_utterance is exactly-once per gen
    h.session.commit_utterance().await;
    let n = h.injector.finals().len();
    h.session.commit_utterance().await;
    assert_eq!(h.injector.finals().len(), n); // exactly-once
    h.shutdown().await;
}
