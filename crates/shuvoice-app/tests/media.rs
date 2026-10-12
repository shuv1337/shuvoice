use async_trait::async_trait;
use parking_lot::Mutex;
use shuvoice_app::media::*;
use shuvoice_core::media::*;
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::sync::broadcast;

struct Fake {
    snapshot: Mutex<MediaSnapshot>,
    events: broadcast::Sender<MediaEvent>,
    pauses: AtomicUsize,
    plays: AtomicUsize,
    pause_delay: Duration,
    play_delay: Duration,
    play_transition_delay: Duration,
    play_transition_at: Mutex<Option<tokio::time::Instant>>,
    play_outcome: MediaCallOutcome,
    outcome: MediaCallOutcome,
    no_op: bool,
    empty: bool,
    unavailable: bool,
    batch: bool,
    transition_delay: Duration,
    transition_at: Mutex<Option<tokio::time::Instant>>,
    read_delays: Mutex<std::collections::VecDeque<Duration>>,
    owner_delays: Mutex<std::collections::VecDeque<Duration>>,
    owner_calls: AtomicUsize,
    pause_invalidation_only: bool,
    transition_seek: bool,
}
impl Fake {
    fn new(identity: bool, pause_delay: Duration) -> Self {
        Self {
            snapshot: Mutex::new(MediaSnapshot {
                playback: MediaPlayback::Playing,
                can_control: true,
                can_pause: true,
                can_play: true,
                fingerprint: MediaFingerprint {
                    title: identity.then(|| "test track".into()),
                    ..Default::default()
                },
            }),
            events: broadcast::channel(16).0,
            pauses: AtomicUsize::new(0),
            plays: AtomicUsize::new(0),
            pause_delay,
            play_delay: Duration::ZERO,
            play_transition_delay: Duration::ZERO,
            play_transition_at: Mutex::new(None),
            play_outcome: MediaCallOutcome::Success,
            outcome: MediaCallOutcome::Success,
            no_op: false,
            empty: false,
            unavailable: false,
            batch: false,
            transition_delay: Duration::ZERO,
            transition_at: Mutex::new(None),
            read_delays: Mutex::default(),
            owner_delays: Mutex::default(),
            owner_calls: AtomicUsize::new(0),
            pause_invalidation_only: false,
            transition_seek: false,
        }
    }
    fn endpoint() -> MediaEndpoint {
        MediaEndpoint {
            bus_epoch: 1,
            well_known_name: "org.mpris.MediaPlayer2.fake".into(),
            unique_owner: ":1.2".into(),
            object_path: "/org/mpris/MediaPlayer2".into(),
        }
    }
    fn conflict(&self) {
        let e = Self::endpoint();
        let _ = self.events.send(MediaEvent::Conflict {
            owner: e.unique_owner,
            path: e.object_path,
        });
    }
    fn seeked(&self) {
        let e = Self::endpoint();
        let _ = self.events.send(MediaEvent::Seeked {
            owner: e.unique_owner,
            path: e.object_path,
        });
    }
    fn properties(&self, playback: Option<MediaPlayback>, invalidate_metadata: bool) {
        let e = Self::endpoint();
        let _ = self.events.send(MediaEvent::Properties {
            owner: e.unique_owner,
            path: e.object_path,
            playback,
            fingerprint: None,
            can_control: None,
            can_pause: None,
            can_play: None,
            invalidated: if invalidate_metadata {
                vec!["Metadata".into()]
            } else {
                vec![]
            },
        });
    }
}
#[async_trait]
impl MediaTransport for Fake {
    async fn subscribe(&self) -> Result<MediaEventReceiver, MediaError> {
        if self.unavailable {
            return Err(MediaError::Unavailable);
        }
        Ok(self.events.subscribe())
    }
    async fn discover(&self) -> Result<Vec<MediaEndpoint>, MediaError> {
        if self.empty {
            return Ok(vec![]);
        }
        Ok(vec![Self::endpoint(), Self::endpoint()])
    }
    async fn snapshot(&self, _: &MediaEndpoint) -> Result<MediaSnapshot, MediaError> {
        let delay = self.read_delays.lock().pop_front();
        if let Some(delay) = delay {
            tokio::time::sleep(delay).await;
        }
        if self
            .transition_at
            .lock()
            .take_if(|at| tokio::time::Instant::now() >= *at)
            .is_some()
        {
            self.snapshot.lock().playback = MediaPlayback::Paused;
        }
        if self
            .play_transition_at
            .lock()
            .take_if(|at| tokio::time::Instant::now() >= *at)
            .is_some()
        {
            self.snapshot.lock().playback = MediaPlayback::Playing;
        }
        Ok(self.snapshot.lock().clone())
    }
    async fn owner_matches(&self, _: &MediaEndpoint) -> Result<bool, MediaError> {
        self.owner_calls.fetch_add(1, Ordering::SeqCst);
        let delay = self.owner_delays.lock().pop_front();
        if let Some(delay) = delay {
            tokio::time::sleep(delay).await;
        }
        Ok(true)
    }
    async fn pause(&self, e: &MediaEndpoint) -> MediaCallOutcome {
        self.pauses.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(self.pause_delay).await;
        if self.no_op {
            return self.outcome;
        }
        if !self.transition_delay.is_zero() {
            *self.transition_at.lock() = Some(tokio::time::Instant::now() + self.transition_delay);
            return self.outcome;
        }
        self.snapshot.lock().playback = MediaPlayback::Paused;
        if self.transition_seek {
            self.seeked();
        }
        if self.pause_invalidation_only {
            let _ = self.events.send(MediaEvent::Properties {
                owner: e.unique_owner.clone(),
                path: e.object_path.clone(),
                playback: None,
                fingerprint: None,
                can_control: None,
                can_pause: None,
                can_play: None,
                invalidated: vec!["PlaybackStatus".into()],
            });
            return self.outcome;
        }
        if self.batch {
            let _ = self.events.send(MediaEvent::Properties {
                owner: e.unique_owner.clone(),
                path: e.object_path.clone(),
                playback: Some(MediaPlayback::Paused),
                fingerprint: Some(self.snapshot.lock().fingerprint.clone()),
                can_control: Some(true),
                can_pause: Some(true),
                can_play: Some(true),
                invalidated: vec!["Metadata".into(), "Volume".into()],
            });
            return self.outcome;
        }
        let _ = self.events.send(MediaEvent::Paused {
            owner: e.unique_owner.clone(),
            path: e.object_path.clone(),
        });
        self.outcome
    }
    async fn play(&self, _: &MediaEndpoint) -> MediaCallOutcome {
        self.plays.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(self.play_delay).await;
        if self.transition_seek {
            self.seeked();
        }
        if self.play_transition_delay.is_zero() {
            self.snapshot.lock().playback = MediaPlayback::Playing;
        } else {
            *self.play_transition_at.lock() =
                Some(tokio::time::Instant::now() + self.play_transition_delay);
        }
        self.play_outcome
    }
}
async fn advance(ms: u64) {
    for _ in 0..ms {
        tokio::time::advance(Duration::from_millis(1)).await;
        tokio::task::yield_now().await;
    }
}

#[tokio::test(start_paused = true)]
async fn chromium_transition_seek_resumes_but_seek_during_hold_revokes() {
    for manual_seek in [false, true] {
        let mut f = Fake::new(true, Duration::ZERO);
        f.transition_seek = true;
        let fake = Arc::new(f);
        let (h, join) = spawn_media_coordinator(fake.clone());
        h.hold(RecordingIntentId(1));
        advance(30).await;
        assert!(h.settled(RecordingIntentId(1)));
        assert_eq!(fake.pauses.load(Ordering::SeqCst), 1);
        if manual_seek {
            fake.seeked();
            advance(10).await;
        }
        h.release(RecordingIntentId(1));
        advance(30).await;
        assert_eq!(fake.plays.load(Ordering::SeqCst), usize::from(!manual_seek));
        assert_eq!(
            fake.snapshot.lock().playback,
            if manual_seek {
                MediaPlayback::Paused
            } else {
                MediaPlayback::Playing
            }
        );
        join.shutdown().await;
    }
}

#[tokio::test(start_paused = true)]
async fn chromium_play_seek_keeps_successor_reconciliation() {
    let mut f = Fake::new(true, Duration::ZERO);
    f.transition_seek = true;
    f.play_delay = Duration::from_millis(20);
    let fake = Arc::new(f);
    let (h, join) = spawn_media_coordinator(fake.clone());
    h.hold(RecordingIntentId(1));
    advance(30).await;
    h.release(RecordingIntentId(1));
    advance(5).await;
    assert_eq!(fake.plays.load(Ordering::SeqCst), 1);
    h.hold(RecordingIntentId(2));
    advance(50).await;
    assert_eq!(fake.pauses.load(Ordering::SeqCst), 2);
    assert_eq!(fake.snapshot.lock().playback, MediaPlayback::Paused);
    h.release(RecordingIntentId(2));
    advance(50).await;
    assert_eq!(fake.plays.load(Ordering::SeqCst), 2);
    assert_eq!(fake.snapshot.lock().playback, MediaPlayback::Playing);
    join.shutdown().await;
}

#[tokio::test(start_paused = true)]
async fn target_snapshot_precedes_owner_await_and_revokes_intervening_manual_resume() {
    let mut f = Fake::new(true, Duration::ZERO);
    f.pause_invalidation_only = true;
    f.owner_delays
        .lock()
        .extend([Duration::ZERO, Duration::from_millis(50)]);
    let fake = Arc::new(f);
    let (h, join) = spawn_media_coordinator(fake.clone());
    h.hold(RecordingIntentId(1));
    advance(10).await;
    assert_eq!(fake.pauses.load(Ordering::SeqCst), 1);
    assert_eq!(
        fake.owner_calls.load(Ordering::SeqCst),
        2,
        "confirmation read returned; owner validation is blocked"
    );
    assert_eq!(fake.snapshot.lock().playback, MediaPlayback::Paused);
    assert!(!h.settled(RecordingIntentId(1)));
    // No Paused signal announced our Pause; its fresh snapshot was the first
    // target observation. Manual resume/re-pause during the following await
    // must revoke entitlement even though the eventual snapshot is identical.
    fake.properties(Some(MediaPlayback::Playing), false);
    fake.properties(Some(MediaPlayback::Paused), false);
    advance(60).await;
    assert!(h.settled(RecordingIntentId(1)));
    h.release(RecordingIntentId(1));
    advance(30).await;
    assert_eq!(fake.plays.load(Ordering::SeqCst), 0);
    join.shutdown().await;
}

#[tokio::test(start_paused = true)]
async fn delayed_pause_with_metadata_invalidation_keeps_pretransition_playing_valid() {
    for reversal in [false, true] {
        let mut f = Fake::new(true, Duration::ZERO);
        f.transition_delay = Duration::from_millis(20);
        let fake = Arc::new(f);
        let (h, join) = spawn_media_coordinator(fake.clone());
        h.hold(RecordingIntentId(1));
        advance(5).await;
        assert_eq!(fake.pauses.load(Ordering::SeqCst), 1);
        fake.properties(None, true);
        // Once Paused is observed, a subsequent Playing is a real conflict,
        // even if both signals arrive before the next fresh read.
        if reversal {
            fake.properties(Some(MediaPlayback::Paused), false);
            fake.properties(Some(MediaPlayback::Playing), false);
        }
        advance(10).await;
        if !reversal {
            assert!(!h.settled(RecordingIntentId(1)));
        }
        advance(30).await;
        assert!(h.settled(RecordingIntentId(1)));
        // Materialize the independently scheduled remote transition even when
        // the coordinator correctly abandoned a conflicting operation.
        fake.snapshot(&Fake::endpoint()).await.unwrap();
        h.release(RecordingIntentId(1));
        advance(30).await;
        assert_eq!(fake.plays.load(Ordering::SeqCst), usize::from(!reversal));
        join.shutdown().await;
    }
}

#[tokio::test(start_paused = true)]
async fn successor_waits_for_play_transition_after_successful_reply_then_repauses() {
    let mut f = Fake::new(true, Duration::ZERO);
    f.play_delay = Duration::from_millis(20);
    f.play_transition_delay = Duration::from_millis(80);
    let fake = Arc::new(f);
    let (h, join) = spawn_media_coordinator(fake.clone());
    h.hold(RecordingIntentId(1));
    advance(30).await;
    h.release(RecordingIntentId(1));
    advance(5).await;
    assert_eq!(fake.plays.load(Ordering::SeqCst), 1);
    h.hold(RecordingIntentId(2));
    advance(30).await;
    assert!(
        fake.play_transition_at.lock().is_some(),
        "Play has replied but has not transitioned"
    );
    assert_eq!(fake.snapshot.lock().playback, MediaPlayback::Paused);
    assert!(
        !h.settled(RecordingIntentId(2)),
        "Paused after Play reply is not settlement"
    );
    assert_eq!(fake.pauses.load(Ordering::SeqCst), 1);
    advance(100).await;
    assert_eq!(fake.pauses.load(Ordering::SeqCst), 2);
    assert_eq!(fake.snapshot.lock().playback, MediaPlayback::Paused);
    assert!(h.settled(RecordingIntentId(2)));
    join.shutdown().await;
}

#[tokio::test(start_paused = true)]
async fn unconfirmed_or_unknown_play_never_blindly_retries_or_repauses() {
    for unknown in [false, true] {
        let mut f = Fake::new(true, Duration::ZERO);
        f.play_delay = Duration::from_millis(20);
        f.play_transition_delay = Duration::from_secs(10);
        if unknown {
            f.play_outcome = MediaCallOutcome::Unknown;
        }
        let fake = Arc::new(f);
        let (h, join) = spawn_media_coordinator(fake.clone());
        h.hold(RecordingIntentId(1));
        advance(30).await;
        h.release(RecordingIntentId(1));
        advance(5).await;
        h.hold(RecordingIntentId(2));
        advance(1200).await;
        assert!(h.settled(RecordingIntentId(2)));
        assert_eq!(fake.pauses.load(Ordering::SeqCst), 1);
        assert_eq!(fake.plays.load(Ordering::SeqCst), 1);
        join.shutdown().await;
    }
}

#[tokio::test(start_paused = true)]
async fn acquisition_settles_early_for_empty_failure_and_confirmed_player() {
    for case in 0..3 {
        let mut f = Fake::new(true, Duration::ZERO);
        f.empty = case == 0;
        f.unavailable = case == 1;
        let fake = Arc::new(f);
        let (h, join) = spawn_media_coordinator(fake);
        h.hold(RecordingIntentId(1));
        advance(20).await;
        assert!(
            h.settled(RecordingIntentId(1)),
            "case {case} consumed full budget"
        );
        join.shutdown().await;
    }
}

#[tokio::test(start_paused = true)]
async fn delayed_transition_and_batched_benign_changes_preserve_restore() {
    for batch in [false, true] {
        let mut f = Fake::new(true, Duration::ZERO);
        f.batch = batch;
        if !batch {
            f.transition_delay = Duration::from_millis(20);
        }
        let fake = Arc::new(f);
        let (h, join) = spawn_media_coordinator(fake.clone());
        h.hold(RecordingIntentId(1));
        advance(10).await;
        if !batch {
            assert!(!h.settled(RecordingIntentId(1)));
        }
        advance(30).await;
        assert!(h.settled(RecordingIntentId(1)));
        h.release(RecordingIntentId(1));
        advance(30).await;
        assert_eq!(fake.plays.load(Ordering::SeqCst), 1);
        join.shutdown().await;
    }
}

#[tokio::test(start_paused = true)]
async fn noop_expires_and_conflict_during_confirmation_cannot_authorize_restore() {
    for conflict in [false, true] {
        let mut f = Fake::new(true, Duration::ZERO);
        f.no_op = !conflict;
        f.transition_delay = Duration::from_millis(100);
        let fake = Arc::new(f);
        let (h, join) = spawn_media_coordinator(fake.clone());
        h.hold(RecordingIntentId(1));
        advance(20).await;
        assert!(!h.settled(RecordingIntentId(1)));
        if conflict {
            fake.conflict();
        }
        advance(1100).await;
        assert!(h.settled(RecordingIntentId(1)));
        h.release(RecordingIntentId(1));
        advance(20).await;
        assert_eq!(fake.pauses.load(Ordering::SeqCst), 1);
        assert_eq!(fake.plays.load(Ordering::SeqCst), 0);
        join.shutdown().await;
    }
}

#[tokio::test(start_paused = true)]
async fn undispatched_expired_predecessor_is_reacquired_for_new_intent() {
    let f = Fake::new(true, Duration::ZERO);
    f.read_delays.lock().push_back(Duration::from_millis(300));
    let fake = Arc::new(f);
    let (h, join) = spawn_media_coordinator(fake.clone());
    h.hold(RecordingIntentId(1));
    advance(100).await;
    h.hold(RecordingIntentId(2));
    advance(180).await;
    assert_eq!(fake.pauses.load(Ordering::SeqCst), 1);
    assert!(h.settled(RecordingIntentId(2)));
    h.release(RecordingIntentId(2));
    advance(30).await;
    assert_eq!(fake.plays.load(Ordering::SeqCst), 1);
    join.shutdown().await;
}

struct Many {
    players: Vec<Fake>,
    events: broadcast::Sender<MediaEvent>,
}
impl Many {
    fn endpoint(i: usize) -> MediaEndpoint {
        let mut e = Fake::endpoint();
        e.unique_owner = format!(":1.{i}");
        e.well_known_name = format!("org.mpris.MediaPlayer2.fake{i}");
        e
    }
    fn player(&self, e: &MediaEndpoint) -> &Fake {
        &self.players[e
            .unique_owner
            .strip_prefix(":1.")
            .unwrap()
            .parse::<usize>()
            .unwrap()]
    }
}
#[async_trait]
impl MediaTransport for Many {
    async fn subscribe(&self) -> Result<MediaEventReceiver, MediaError> {
        Ok(self.events.subscribe())
    }
    async fn discover(&self) -> Result<Vec<MediaEndpoint>, MediaError> {
        Ok((0..self.players.len()).map(Self::endpoint).collect())
    }
    async fn snapshot(&self, e: &MediaEndpoint) -> Result<MediaSnapshot, MediaError> {
        self.player(e).snapshot(e).await
    }
    async fn owner_matches(&self, _: &MediaEndpoint) -> Result<bool, MediaError> {
        Ok(true)
    }
    async fn pause(&self, e: &MediaEndpoint) -> MediaCallOutcome {
        self.player(e).pause(e).await
    }
    async fn play(&self, e: &MediaEndpoint) -> MediaCallOutcome {
        self.player(e).play(e).await
    }
}
#[tokio::test(start_paused = true)]
async fn local_restore_contention_keeps_all_nine_confirmed_pauses() {
    let (events, _) = broadcast::channel(256);
    let players = (0..9)
        .map(|_| {
            let mut f = Fake::new(true, Duration::ZERO);
            f.play_delay = Duration::from_millis(350);
            f.events = events.clone();
            f
        })
        .collect();
    let transport = Arc::new(Many { players, events });
    let (h, join) = spawn_media_coordinator(transport.clone());
    h.hold(RecordingIntentId(1));
    advance(50).await;
    assert!(
        transport
            .players
            .iter()
            .all(|p| p.pauses.load(Ordering::SeqCst) == 1)
    );
    h.release(RecordingIntentId(1));
    advance(800).await;
    assert!(
        transport
            .players
            .iter()
            .all(|p| p.plays.load(Ordering::SeqCst) == 1)
    );
    join.shutdown().await;
}

#[tokio::test(start_paused = true)]
async fn short_tap_starts_early_and_start_observing_dead_asr_releases_hold() {
    use shuvoice_app::fakes::{
        FakeClock, FakeFeedback, FakeInjector, FakeOverlay, FakeSelection, FakeTts,
        ScriptedAsrBackend,
    };
    use shuvoice_app::{AudioIngress, Config, Session, SessionDeps, spawn_asr_owner};
    for dead in [false, true] {
        let fake = Arc::new(Fake::new(true, Duration::ZERO));
        let (media, coordinator) = spawn_media_coordinator(fake.clone());
        let scripted = ScriptedAsrBackend::offline_instant(160, "short speech");
        let (asr, owner) = spawn_asr_owner(Box::new(scripted), 32, Duration::from_secs(5));
        asr.load().await.unwrap();
        let (audio, ring) = AudioIngress::new(64);
        let injector = Arc::new(FakeInjector::default());
        let deps = SessionDeps::new(
            asr.clone(),
            ring,
            injector.clone(),
            Arc::new(FakeSelection::default()),
            FakeOverlay::default(),
            FakeFeedback::default(),
            Arc::new(FakeClock::new()),
        )
        .with_tts(FakeTts::new())
        .with_media(media);
        let mut config = Config::default();
        config.pause_media_during_recording = true;
        config.min_speech_ms = 0;
        config.silence_rms_threshold = 0.0;
        let mut session = Session::new(config, deps);
        session.request_start_recording();
        advance(20).await;
        session.tick().await;
        assert!(
            session.is_recording(),
            "quick acquisition must not discard a short utterance"
        );
        audio.try_push(vec![0.2; 160]);
        session.tick().await;
        if dead {
            asr.shutdown().await;
            assert!(!asr.is_alive());
            // Admission notices death before the next tick. It must do the
            // complete teardown rather than setting a flag that suppresses it.
            session.request_start_recording();
            advance(20).await;
            assert!(!session.is_recording());
            assert!(!session.is_starting());
            assert_eq!(fake.plays.load(Ordering::SeqCst), 1);
            session.tick().await;
            advance(20).await;
            assert_eq!(fake.plays.load(Ordering::SeqCst), 1);
        } else {
            session.stop_recording();
            session.handle_recording_stop().await;
            advance(20).await;
            assert_eq!(injector.finals().len(), 1);
            assert_eq!(fake.plays.load(Ordering::SeqCst), 1);
        }
        session.shutdown().await;
        coordinator.shutdown().await;
        owner.join_timeout(Duration::from_secs(1)).await;
    }
}

#[tokio::test(start_paused = true)]
async fn identityless_is_never_paused_and_aliases_are_deduplicated() {
    for identity in [false, true] {
        let fake = Arc::new(Fake::new(identity, Duration::ZERO));
        let (h, join) = spawn_media_coordinator(fake.clone());
        h.hold(RecordingIntentId(1));
        advance(410).await;
        assert!(h.settled(RecordingIntentId(1)));
        assert_eq!(fake.pauses.load(Ordering::SeqCst), usize::from(identity));
        h.release(RecordingIntentId(1));
        advance(20).await;
        assert_eq!(fake.plays.load(Ordering::SeqCst), usize::from(identity));
        join.shutdown().await;
    }
}

#[tokio::test(start_paused = true)]
async fn late_pause_is_restored_on_idle_but_transferred_to_new_hold() {
    for new_hold in [false, true] {
        let fake = Arc::new(Fake::new(true, Duration::from_millis(200)));
        let (h, join) = spawn_media_coordinator(fake.clone());
        h.hold(RecordingIntentId(1));
        advance(10).await;
        assert_eq!(fake.pauses.load(Ordering::SeqCst), 1);
        h.release(RecordingIntentId(1));
        if new_hold {
            h.hold(RecordingIntentId(2));
            h.release(RecordingIntentId(1));
        }
        advance(250).await;
        assert_eq!(fake.plays.load(Ordering::SeqCst), usize::from(!new_hold));
        h.release(RecordingIntentId(2));
        advance(30).await;
        assert_eq!(fake.plays.load(Ordering::SeqCst), 1);
        join.shutdown().await;
    }
}

#[tokio::test(start_paused = true)]
async fn conflicts_and_event_loss_revoke_and_never_fight_user_resume() {
    for lost in [false, true] {
        let fake = Arc::new(Fake::new(true, Duration::ZERO));
        let (h, join) = spawn_media_coordinator(fake.clone());
        h.hold(RecordingIntentId(1));
        advance(30).await;
        if lost {
            let _ = fake.events.send(MediaEvent::Lost);
        } else {
            fake.conflict();
        }
        advance(30).await;
        h.release(RecordingIntentId(1));
        advance(30).await;
        assert_eq!(fake.pauses.load(Ordering::SeqCst), 1);
        assert_eq!(fake.plays.load(Ordering::SeqCst), 0);
        join.shutdown().await;
    }
}

#[tokio::test(start_paused = true)]
async fn unknown_pause_and_timeout_never_authorize_play() {
    for delay in [Duration::ZERO, Duration::from_secs(10)] {
        let mut f = Fake::new(true, delay);
        f.outcome = MediaCallOutcome::Unknown;
        let fake = Arc::new(f);
        let (h, join) = spawn_media_coordinator(fake.clone());
        h.hold(RecordingIntentId(1));
        advance(410).await;
        assert!(h.settled(RecordingIntentId(1)));
        h.release(RecordingIntentId(1));
        advance(1100).await;
        join.shutdown().await;
        assert_eq!(fake.plays.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test(start_paused = true)]
async fn shutdown_joins_pending_effect_and_stale_handles_cannot_reopen_it() {
    let fake = Arc::new(Fake::new(true, Duration::from_secs(10)));
    let (h, join) = spawn_media_coordinator(fake.clone());
    h.hold(RecordingIntentId(1));
    advance(10).await;
    assert_eq!(fake.pauses.load(Ordering::SeqCst), 1);
    let start = tokio::time::Instant::now();
    join.shutdown().await;
    assert!(start.elapsed() <= Duration::from_secs(2));
    h.hold(RecordingIntentId(2));
    advance(20).await;
    assert_eq!(fake.pauses.load(Ordering::SeqCst), 1);
    assert_eq!(fake.plays.load(Ordering::SeqCst), 0);
}

#[tokio::test(start_paused = true)]
async fn ineligible_noop_and_changed_metadata_cannot_create_restore_entitlement() {
    for case in 0..6 {
        let mut f = Fake::new(true, Duration::ZERO);
        match case {
            0 => f.snapshot.lock().playback = MediaPlayback::Paused,
            1 => f.snapshot.lock().playback = MediaPlayback::Stopped,
            2 => f.snapshot.lock().can_pause = false,
            3 => f.snapshot.lock().can_control = false,
            4 => f.no_op = true,
            _ => {}
        }
        let fake = Arc::new(f);
        let (h, join) = spawn_media_coordinator(fake.clone());
        h.hold(RecordingIntentId(1));
        advance(30).await;
        if case == 5 {
            fake.snapshot.lock().fingerprint.title = Some("different track".into());
        }
        h.release(RecordingIntentId(1));
        advance(30).await;
        assert_eq!(fake.pauses.load(Ordering::SeqCst), usize::from(case >= 4));
        assert_eq!(fake.plays.load(Ordering::SeqCst), 0);
        join.shutdown().await;
    }
}

#[tokio::test(start_paused = true)]
async fn already_dispatched_play_settles_before_new_pause() {
    let mut f = Fake::new(true, Duration::ZERO);
    f.play_delay = Duration::from_millis(100);
    let fake = Arc::new(f);
    let (h, join) = spawn_media_coordinator(fake.clone());
    h.hold(RecordingIntentId(1));
    advance(30).await;
    h.release(RecordingIntentId(1));
    advance(10).await;
    assert_eq!(fake.plays.load(Ordering::SeqCst), 1);
    h.hold(RecordingIntentId(2));
    advance(30).await;
    assert_eq!(fake.pauses.load(Ordering::SeqCst), 1);
    advance(100).await;
    assert_eq!(fake.pauses.load(Ordering::SeqCst), 2);
    join.shutdown().await;
}

#[tokio::test(start_paused = true)]
async fn post_reset_speech_survives_media_wait_and_short_tap_closes_before_decode() {
    use shuvoice_app::fakes::{
        FakeClock, FakeFeedback, FakeInjector, FakeOverlay, FakeSelection, FakeTts,
        ScriptedAsrBackend,
    };
    use shuvoice_app::{AudioIngress, Config, Session, SessionDeps, spawn_asr_owner};
    let fake = Arc::new(Fake::new(true, Duration::from_millis(200)));
    let (media, coordinator) = spawn_media_coordinator(fake.clone());
    let scripted = ScriptedAsrBackend::offline_instant(160, "test");
    let shared = scripted.shared();
    let (asr, owner) = spawn_asr_owner(Box::new(scripted), 32, Duration::from_secs(5));
    asr.load().await.unwrap();
    let (audio, ring) = AudioIngress::new(64);
    let deps = SessionDeps::new(
        asr,
        ring,
        Arc::new(FakeInjector::default()),
        Arc::new(FakeSelection::default()),
        FakeOverlay::default(),
        FakeFeedback::default(),
        Arc::new(FakeClock::new()),
    )
    .with_tts(FakeTts::new())
    .with_media(media);
    let mut config = Config::default();
    config.pause_media_during_recording = true;
    config.min_speech_ms = 0;
    config.silence_rms_threshold = 0.0;
    let mut session = Session::new(config, deps);
    session.request_start_recording();
    advance(20).await;
    session.tick().await;
    assert!(!session.is_recording());
    assert!(shared.lock().reset_calls > 0);
    audio.try_push(vec![0.1; 160]);
    session.tick().await;
    audio.try_push(vec![0.2; 160]);
    session.tick().await;
    advance(400).await;
    // The direct pump starts synchronously and arms the capture edge without a tick.
    session
        .pump_jobs_while(|s| s.is_starting(), Duration::from_secs(1))
        .await;
    assert!(session.is_recording());
    assert_eq!(session.utterance_state().total, 320);
    shared.lock().delay = Duration::from_secs(2);
    session.stop_recording();
    session.tick().await;
    advance(100).await;
    session.tick().await;
    assert_eq!(fake.plays.load(Ordering::SeqCst), 0);
    audio.try_push(vec![0.3; 160]);
    session.tick().await;
    advance(21).await;
    session.tick().await;
    advance(20).await;
    assert_eq!(fake.plays.load(Ordering::SeqCst), 1);
    assert!(session.is_processing());
    audio.try_push(vec![0.9; 160]);
    session.tick().await;
    advance(2100).await;
    session.tick().await;
    assert_eq!(
        shared.lock().last_utterance.len(),
        480,
        "post-close PCM must not enter the old utterance"
    );
    session.shutdown().await;
    coordinator.shutdown().await;
    owner.join_timeout(Duration::from_secs(1)).await;
}

#[tokio::test(start_paused = true)]
async fn runtime_shutdown_joins_late_pause_and_restoration_after_actor_exit() {
    use shuvoice_app::fakes::{
        FakeClock, FakeFeedback, FakeInjector, FakeOverlay, FakeSelection, FakeTts,
        ScriptedAsrBackend,
    };
    use shuvoice_app::{
        Config, RuntimeServices, SessionCommand, spawn_session_runtime_with_services,
    };
    let fake = Arc::new(Fake::new(true, Duration::from_millis(200)));
    let mut config = Config::default();
    config.pause_media_during_recording = true;
    let runtime = spawn_session_runtime_with_services(
        config,
        Box::new(ScriptedAsrBackend::offline_instant(160, "test")),
        Arc::new(FakeInjector::default()),
        Arc::new(FakeSelection::default()),
        FakeOverlay::default(),
        FakeFeedback::default(),
        Arc::new(FakeClock::new()),
        Some(FakeTts::new()),
        RuntimeServices {
            transcripts: None,
            media: Some(fake.clone()),
        },
    )
    .await
    .unwrap();
    runtime.handle.send(SessionCommand::Start).await.unwrap();
    advance(10).await;
    assert_eq!(fake.pauses.load(Ordering::SeqCst), 1);
    let start = tokio::time::Instant::now();
    runtime.shutdown().await.unwrap();
    assert!(start.elapsed() < Duration::from_secs(3));
    assert_eq!(fake.plays.load(Ordering::SeqCst), 1);
    advance(100).await;
    assert_eq!(fake.plays.load(Ordering::SeqCst), 1);
}
