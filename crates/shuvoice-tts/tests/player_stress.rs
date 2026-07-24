//! Stress / contract tests for player generation, queue, carry, and speed restart.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream;
use parking_lot::Mutex;
use shuvoice_tts::{
    AudioEncoding, BackendId, Capabilities, FakeAudioOutputFactory, PLAYER_QUEUE_CAPACITY,
    PlayerState, SynthesisRequest, SynthesisStream, TtsBackend, TtsError, TtsPlayer, VoiceInfo,
    chunk_to_samples,
};
use tokio_util::sync::CancellationToken;

struct ControllableBackend {
    sample_rate: u32,
    chunks: Mutex<Vec<Bytes>>,
    delay_per_chunk: Mutex<Duration>,
    requests: Mutex<Vec<SynthesisRequest>>,
    flood_count: Mutex<usize>,
}

impl ControllableBackend {
    fn pcm_chunks(parts: &[&[i16]]) -> Self {
        let mut chunks = Vec::new();
        for part in parts {
            let mut bytes = Vec::with_capacity(part.len() * 2);
            for s in *part {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            chunks.push(Bytes::from(bytes));
        }
        Self {
            sample_rate: 24_000,
            chunks: Mutex::new(chunks),
            delay_per_chunk: Mutex::new(Duration::from_millis(1)),
            requests: Mutex::new(Vec::new()),
            flood_count: Mutex::new(0),
        }
    }
}

#[async_trait]
impl TtsBackend for ControllableBackend {
    fn id(&self) -> BackendId {
        BackendId::Kokoro
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_speed_control: true,
            ..Capabilities::default()
        }
    }
    fn sample_rate_hz(&self) -> u32 {
        self.sample_rate
    }
    fn dependency_errors(&self) -> Vec<String> {
        Vec::new()
    }
    async fn synthesize_stream(
        &self,
        request: SynthesisRequest,
        cancel: CancellationToken,
    ) -> Result<SynthesisStream, TtsError> {
        self.requests.lock().push(request);
        let mut chunks = self.chunks.lock().clone();
        let flood = *self.flood_count.lock();
        if flood > 0 {
            let base = chunks
                .first()
                .cloned()
                .unwrap_or_else(|| Bytes::from(vec![1, 0]));
            chunks = vec![base; flood];
        }
        let delay = *self.delay_per_chunk.lock();
        let stream = stream::unfold(
            (chunks, 0usize, cancel, delay),
            |(chunks, idx, cancel, delay)| async move {
                if cancel.is_cancelled() || idx >= chunks.len() {
                    return None;
                }
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                let item = Ok(chunks[idx].clone());
                Some((item, (chunks, idx + 1, cancel, delay)))
            },
        );
        Ok(SynthesisStream {
            sample_rate_hz: self.sample_rate,
            encoding: AudioEncoding::PcmS16Le,
            chunks: Box::pin(stream),
        })
    }
    async fn list_voices(&self) -> Result<Vec<VoiceInfo>, TtsError> {
        Ok(vec![VoiceInfo::new("v", "V")])
    }
}

async fn wait_until(mut pred: impl FnMut() -> bool, timeout: Duration) {
    let start = std::time::Instant::now();
    while !pred() {
        if start.elapsed() > timeout {
            panic!("timeout waiting for condition");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generation_cancel_on_interrupt_keeps_latest_audio() {
    let backend = Arc::new(ControllableBackend::pcm_chunks(&[&[1, 2]]));
    let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
    *factory.write_delay.lock() = Duration::from_millis(30);

    let player = TtsPlayer::builder(backend.clone(), factory.clone()).build();

    *backend.flood_count.lock() = 40;
    *backend.chunks.lock() = {
        let mut b = Vec::new();
        for s in [1i16, 1] {
            b.extend_from_slice(&s.to_le_bytes());
        }
        vec![Bytes::from(b)]
    };
    player.speak("first", "v", "m").unwrap();
    wait_until(|| player.is_active(), Duration::from_secs(2)).await;

    *backend.flood_count.lock() = 0;
    *backend.chunks.lock() = {
        let mut b = Vec::new();
        for s in [42i16, 43, 44] {
            b.extend_from_slice(&s.to_le_bytes());
        }
        vec![Bytes::from(b)]
    };
    let interrupted = player.speak("second", "v", "m").unwrap();
    assert!(interrupted, "expected first generation to be interrupted");

    wait_until(
        || player.state() == PlayerState::Idle,
        Duration::from_secs(3),
    )
    .await;

    let written = factory.snapshot();
    assert!(
        written.windows(3).any(|w| w == [42, 43, 44]),
        "expected second generation PCM in output, got {written:?}"
    );
    assert_eq!(player.state(), PlayerState::Idle);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn odd_byte_carry_across_synth_chunks() {
    let s0 = 0x0201i16;
    let s1 = 0x0403i16;
    let s2 = 0x0605i16;
    let b0 = s0.to_le_bytes();
    let b1 = s1.to_le_bytes();
    let b2 = s2.to_le_bytes();

    let chunk_a = Bytes::from(vec![b0[0]]);
    let chunk_b = Bytes::from(vec![b0[1], b1[0], b1[1], b2[0]]);
    let chunk_c = Bytes::from(vec![b2[1]]);

    let (s, c) = chunk_to_samples(&chunk_a, &[]);
    assert!(s.is_empty());
    let (s, c) = chunk_to_samples(&chunk_b, &c);
    assert_eq!(s, vec![s0, s1]);
    let (s, c) = chunk_to_samples(&chunk_c, &c);
    assert_eq!(s, vec![s2]);
    assert!(c.is_empty());

    let backend = Arc::new(ControllableBackend {
        sample_rate: 24_000,
        chunks: Mutex::new(vec![chunk_a, chunk_b, chunk_c]),
        delay_per_chunk: Mutex::new(Duration::from_millis(1)),
        requests: Mutex::new(Vec::new()),
        flood_count: Mutex::new(0),
    });
    let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
    let player = TtsPlayer::builder(backend, factory.clone()).build();
    player.speak("carry", "v", "m").unwrap();
    wait_until(
        || player.state() == PlayerState::Idle,
        Duration::from_secs(2),
    )
    .await;
    assert_eq!(factory.snapshot(), vec![s0, s1, s2]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queue_saturation_does_not_deadlock() {
    let backend = Arc::new(ControllableBackend::pcm_chunks(&[&[7, 8]]));
    *backend.flood_count.lock() = PLAYER_QUEUE_CAPACITY * 3;
    *backend.delay_per_chunk.lock() = Duration::ZERO;

    let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
    *factory.write_delay.lock() = Duration::from_millis(5);

    let player = TtsPlayer::builder(backend, factory.clone()).build();
    player.speak("flood", "v", "m").unwrap();
    wait_until(
        || player.state() == PlayerState::Idle,
        Duration::from_secs(15),
    )
    .await;

    let written = factory.snapshot();
    assert!(!written.is_empty(), "expected some PCM despite saturation");
    assert!(written.iter().all(|&s| s == 7 || s == 8));
    assert_eq!(player.state(), PlayerState::Idle);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn speed_change_while_active_requires_explicit_restart() {
    // Many tiny chunks so the first generation stays active long enough.
    let samples: Vec<i16> = (1..=32).collect();
    let backend = Arc::new(ControllableBackend::pcm_chunks(&[&samples]));
    *backend.delay_per_chunk.lock() = Duration::from_millis(5);
    // Split into many chunks for longer synth.
    {
        let mut parts = Vec::new();
        for s in &samples {
            let mut b = Vec::new();
            b.extend_from_slice(&s.to_le_bytes());
            parts.push(Bytes::from(b));
        }
        *backend.chunks.lock() = parts;
    }

    let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
    *factory.write_delay.lock() = Duration::from_millis(20);

    let player = TtsPlayer::builder(backend.clone(), factory)
        .playback_speed(1.0)
        .build();
    player.speak("speed", "v", "m").unwrap();
    wait_until(
        || !backend.requests.lock().is_empty(),
        Duration::from_secs(2),
    )
    .await;
    assert_eq!(backend.requests.lock()[0].playback_speed, 1.0);
    wait_until(|| player.is_active(), Duration::from_secs(2)).await;

    let updated = player.set_playback_speed(1.5);
    assert_eq!(updated, 1.5);
    let reqs_before = backend.requests.lock().len();
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(
        backend.requests.lock().len(),
        reqs_before,
        "set_playback_speed must not start a new synthesis by itself"
    );

    assert!(player.restart());
    wait_until(
        || backend.requests.lock().len() > reqs_before,
        Duration::from_secs(2),
    )
    .await;
    wait_until(
        || player.state() == PlayerState::Idle,
        Duration::from_secs(5),
    )
    .await;

    let reqs = backend.requests.lock().clone();
    assert!(reqs.len() > reqs_before);
    assert_eq!(reqs.first().unwrap().playback_speed, 1.0);
    assert_eq!(reqs.last().unwrap().playback_speed, 1.5);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stop_from_paused_returns_idle() {
    let samples: Vec<i16> = (1..=64).collect();
    let backend = Arc::new(ControllableBackend::pcm_chunks(&[&samples]));
    *backend.delay_per_chunk.lock() = Duration::ZERO;
    {
        let mut parts = Vec::new();
        for s in &samples {
            let mut b = Vec::new();
            b.extend_from_slice(&s.to_le_bytes());
            parts.push(Bytes::from(b));
        }
        *backend.chunks.lock() = parts;
    }
    let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
    *factory.write_delay.lock() = Duration::from_millis(30);

    let player = TtsPlayer::builder(backend, factory).build();
    player.speak("pause-stop", "v", "m").unwrap();
    wait_until(|| player.is_active(), Duration::from_secs(2)).await;

    // Prefer pausing from Playing; if still synthesizing, wait a bit.
    for _ in 0..50 {
        if player.state() == PlayerState::Playing {
            break;
        }
        if player.state() == PlayerState::Idle {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    if player.state() == PlayerState::Playing {
        assert!(player.pause());
        assert_eq!(player.state(), PlayerState::Paused);
        assert!(player.stop());
    } else if player.is_active() {
        // Still synthesizing — stop should work from active set.
        assert!(player.stop());
    } else {
        // Completed already — stop returns false.
        assert!(!player.stop());
    }
    assert_eq!(player.state(), PlayerState::Idle);
    assert!(!player.is_active());
}

/// Prove saturated/blocking AudioOutput cannot starve a single-threaded Tokio runtime.
///
/// Playback runs on a dedicated OS thread; a concurrent timer on the current-thread
/// runtime must still fire while write_samples blocks for hundreds of milliseconds.
#[tokio::test(flavor = "current_thread")]
async fn blocking_playback_does_not_starve_current_thread_runtime() {
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

    let backend = Arc::new(ControllableBackend::pcm_chunks(&[&[1, 2, 3, 4]]));
    // Several chunks so multiple blocking writes occur.
    {
        let mut parts = Vec::new();
        for s in [1i16, 2, 3, 4, 5, 6, 7, 8] {
            let mut b = Vec::new();
            b.extend_from_slice(&s.to_le_bytes());
            parts.push(Bytes::from(b));
        }
        *backend.chunks.lock() = parts;
    }
    *backend.delay_per_chunk.lock() = Duration::ZERO;

    let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
    // Each write blocks 150ms on the playback OS thread — long enough that a
    // starved current-thread runtime would miss a 30ms timer.
    *factory.write_delay.lock() = Duration::from_millis(150);

    let player = TtsPlayer::builder(backend, factory).build();
    player.speak("starve-check", "v", "m").unwrap();

    let progressed = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&progressed);
    let probe = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        flag.store(true, AtomicOrdering::SeqCst);
    });

    // While playback is blocked in write_samples on another thread, the
    // current-thread runtime must still drive `probe` to completion.
    let start = std::time::Instant::now();
    probe.await.expect("probe task join");
    let elapsed = start.elapsed();

    assert!(
        progressed.load(AtomicOrdering::SeqCst),
        "runtime probe never ran — playback likely blocked the Tokio worker"
    );
    assert!(
        elapsed < Duration::from_millis(120),
        "runtime probe took {elapsed:?}; expected << write_delay if non-blocking. \
         Blocking playback on the Tokio worker would serialize behind 150ms writes."
    );

    // Cleanup.
    let _ = player.stop();
    // Allow playback thread to finish after interrupt.
    tokio::time::sleep(Duration::from_millis(50)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_while_paused_reaps_workers() {
    let samples: Vec<i16> = (1..=32).collect();
    let backend = Arc::new(ControllableBackend::pcm_chunks(&[&samples]));
    {
        let mut parts = Vec::new();
        for s in &samples {
            let mut b = Vec::new();
            b.extend_from_slice(&s.to_le_bytes());
            parts.push(Bytes::from(b));
        }
        *backend.chunks.lock() = parts;
    }
    let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
    *factory.write_delay.lock() = Duration::from_millis(40);

    let player = TtsPlayer::builder(backend, factory).build();
    player.speak("drop-paused", "v", "m").unwrap();
    wait_until(|| player.is_active(), Duration::from_secs(2)).await;
    // Best-effort pause
    let _ = player.pause();
    // Drop must not hang.
    let start = std::time::Instant::now();
    drop(player);
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "Drop of paused TtsPlayer hung for {:?}",
        start.elapsed()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stop_uninterruptible_sink_respects_join_deadline() {
    let backend = Arc::new(ControllableBackend::pcm_chunks(&[&[1, 2, 3, 4]]));
    *backend.chunks.lock() = {
        let mut parts = Vec::new();
        for _ in 0..8 {
            parts.push(Bytes::from(1i16.to_le_bytes().to_vec()));
        }
        parts
    };
    let factory = Arc::new(FakeAudioOutputFactory::new(24_000));
    // Long uninterruptible write — interrupt is ignored.
    *factory.write_delay.lock() = Duration::from_secs(5);
    factory
        .uninterruptible_flag
        .store(true, std::sync::atomic::Ordering::SeqCst);

    let player = TtsPlayer::builder(backend, factory).build();
    player.speak("wedge", "v", "m").unwrap();
    wait_until(|| player.is_active(), Duration::from_secs(2)).await;

    let start = std::time::Instant::now();
    let _ = player.stop();
    let elapsed = start.elapsed();
    // Must return near WORKER_JOIN_DEADLINE, not wait full 5s write.
    assert!(
        elapsed < Duration::from_millis(1500),
        "stop hung for {elapsed:?} under uninterruptible sink"
    );
}

#[test]
fn redact_for_ui_strips_urls_and_paths() {
    use shuvoice_tts::redact_for_ui;
    let msg = "failed http://example.com/v1/x and /home/user/secret.onnx boom";
    let red = redact_for_ui(msg);
    assert!(!red.contains("example.com"), "{red}");
    assert!(!red.contains("/home/user"), "{red}");
    assert!(red.contains("redacted"), "{red}");
}
