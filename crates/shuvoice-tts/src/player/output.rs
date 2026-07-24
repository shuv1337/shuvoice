//! Audio output abstraction.
//!
//! All methods take `&self` with interior mutability so the playback OS thread
//! can write while `stop()` concurrently calls [`AudioOutput::interrupt`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use crate::error::TtsError;

/// Blocking/synchronous sink used exclusively by the **playback OS thread**.
///
/// Implementations may block (e.g. CPAL ring backpressure). The player never
/// calls these methods on a Tokio worker thread.
pub trait AudioOutput: Send + Sync {
    fn write_samples(&self, samples: &[i16]) -> Result<(), TtsError>;
    fn close(&self) -> Result<(), TtsError>;

    /// Unblock any in-flight [`write_samples`] as quickly as possible.
    fn interrupt(&self) {}
}

/// Factory invoked each time the player needs a fresh output stream.
pub trait AudioOutputFactory: Send + Sync {
    fn open(&self, sample_rate_hz: u32) -> Result<Arc<dyn AudioOutput>, TtsError>;
}

/// Records PCM for tests; never touches a real device.
#[derive(Debug, Clone)]
pub struct FakeAudioOutput {
    sample_rate_hz: u32,
    written: Arc<Mutex<Vec<i16>>>,
    fail_times: Arc<Mutex<u32>>,
    closed: Arc<AtomicBool>,
    write_delay: Arc<Mutex<std::time::Duration>>,
    interrupt: Arc<AtomicBool>,
    /// When true, write_delay ignores interrupt (tests join deadline).
    pub uninterruptible: Arc<AtomicBool>,
}

impl FakeAudioOutput {
    pub fn new(sample_rate_hz: u32) -> Self {
        Self {
            sample_rate_hz,
            written: Arc::new(Mutex::new(Vec::new())),
            fail_times: Arc::new(Mutex::new(0)),
            closed: Arc::new(AtomicBool::new(false)),
            write_delay: Arc::new(Mutex::new(std::time::Duration::ZERO)),
            interrupt: Arc::new(AtomicBool::new(false)),
            uninterruptible: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn with_shared(sample_rate_hz: u32, written: Arc<Mutex<Vec<i16>>>) -> Self {
        Self {
            sample_rate_hz,
            written,
            fail_times: Arc::new(Mutex::new(0)),
            closed: Arc::new(AtomicBool::new(false)),
            write_delay: Arc::new(Mutex::new(std::time::Duration::ZERO)),
            interrupt: Arc::new(AtomicBool::new(false)),
            uninterruptible: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    pub fn written_handle(&self) -> Arc<Mutex<Vec<i16>>> {
        Arc::clone(&self.written)
    }

    pub fn set_fail_times(&self, times: u32) {
        *self.fail_times.lock() = times;
    }

    pub fn set_write_delay(&self, delay: std::time::Duration) {
        *self.write_delay.lock() = delay;
    }

    pub fn snapshot(&self) -> Vec<i16> {
        self.written.lock().clone()
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    fn is_interrupted(&self) -> bool {
        if self.uninterruptible.load(Ordering::SeqCst) {
            // Ignore interrupt(); only hard close after write finishes.
            return false;
        }
        self.interrupt.load(Ordering::SeqCst) || self.closed.load(Ordering::SeqCst)
    }
}

impl AudioOutput for FakeAudioOutput {
    fn write_samples(&self, samples: &[i16]) -> Result<(), TtsError> {
        {
            let mut fails = self.fail_times.lock();
            if *fails > 0 {
                *fails -= 1;
                return Err(TtsError::audio("transient fake PortAudio error"));
            }
        }
        if self.is_interrupted() {
            return Err(TtsError::audio("fake output interrupted"));
        }
        let delay = *self.write_delay.lock();
        if !delay.is_zero() {
            // Sleep in short slices so stop()/interrupt() can unblock promptly.
            let slice = std::time::Duration::from_millis(5);
            let mut remaining = delay;
            while !remaining.is_zero() {
                if self.is_interrupted() {
                    return Err(TtsError::audio("fake output interrupted"));
                }
                let step = remaining.min(slice);
                std::thread::sleep(step);
                remaining = remaining.saturating_sub(step);
            }
        }
        if self.is_interrupted() {
            return Err(TtsError::audio("fake output interrupted"));
        }
        self.written.lock().extend_from_slice(samples);
        Ok(())
    }

    fn close(&self) -> Result<(), TtsError> {
        self.interrupt.store(true, Ordering::SeqCst);
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn interrupt(&self) {
        self.interrupt.store(true, Ordering::SeqCst);
    }
}

/// Factory that clones a shared sample buffer for each opened stream.
#[derive(Debug, Clone)]
pub struct FakeAudioOutputFactory {
    pub sample_rate_hz: u32,
    pub written: Arc<Mutex<Vec<i16>>>,
    pub fail_times: Arc<Mutex<u32>>,
    pub open_count: Arc<Mutex<u32>>,
    pub write_delay: Arc<Mutex<std::time::Duration>>,
    pub uninterruptible_flag: Arc<AtomicBool>,
}

impl Default for FakeAudioOutputFactory {
    fn default() -> Self {
        Self {
            sample_rate_hz: 24_000,
            written: Arc::new(Mutex::new(Vec::new())),
            fail_times: Arc::new(Mutex::new(0)),
            open_count: Arc::new(Mutex::new(0)),
            write_delay: Arc::new(Mutex::new(std::time::Duration::ZERO)),
            uninterruptible_flag: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl FakeAudioOutputFactory {
    pub fn new(sample_rate_hz: u32) -> Self {
        Self {
            sample_rate_hz,
            ..Self::default()
        }
    }

    pub fn snapshot(&self) -> Vec<i16> {
        self.written.lock().clone()
    }
}

impl AudioOutputFactory for FakeAudioOutputFactory {
    fn open(&self, sample_rate_hz: u32) -> Result<Arc<dyn AudioOutput>, TtsError> {
        *self.open_count.lock() += 1;
        let rate = if sample_rate_hz == 0 {
            self.sample_rate_hz
        } else {
            sample_rate_hz
        };
        let out = FakeAudioOutput {
            sample_rate_hz: rate,
            written: Arc::clone(&self.written),
            fail_times: Arc::clone(&self.fail_times),
            closed: Arc::new(AtomicBool::new(false)),
            write_delay: Arc::clone(&self.write_delay),
            interrupt: Arc::new(AtomicBool::new(false)),
            uninterruptible: Arc::clone(&self.uninterruptible_flag),
        };
        Ok(Arc::new(out))
    }
}

/// Discarding output (useful when only exercising synth).
#[derive(Debug, Default)]
pub struct NullAudioOutput;

impl AudioOutput for NullAudioOutput {
    fn write_samples(&self, _samples: &[i16]) -> Result<(), TtsError> {
        Ok(())
    }
    fn close(&self) -> Result<(), TtsError> {
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct NullAudioOutputFactory;

impl AudioOutputFactory for NullAudioOutputFactory {
    fn open(&self, _sample_rate_hz: u32) -> Result<Arc<dyn AudioOutput>, TtsError> {
        Ok(Arc::new(NullAudioOutput))
    }
}
