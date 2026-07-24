//! Production audio path: non-blocking push from cpal callbacks into a bounded ring.
//!
//! # CPAL callback contract
//!
//! [`AudioRing::try_push`] / [`AudioIngress::try_push`] must **never wait**.
//! They use `parking_lot::Mutex::try_lock` only. On contention the chunk is
//! dropped and counted — the capture callback returns immediately.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::types::DEFAULT_AUDIO_CAPACITY;

/// Bounded mono PCM ring shared by the capture callback and session actor.
#[derive(Debug)]
pub struct AudioRing {
    buf: Mutex<VecDeque<Vec<f32>>>,
    capacity: usize,
    /// Chunks dropped because the ring was full (oldest evicted) or lock contended.
    dropped: AtomicU64,
    /// Subset of `dropped` caused by `try_lock` failure in the callback path.
    contention_drops: AtomicU64,
}

impl AudioRing {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: Mutex::new(VecDeque::with_capacity(capacity.max(1))),
            capacity: capacity.max(1),
            dropped: AtomicU64::new(0),
            contention_drops: AtomicU64::new(0),
        }
    }

    /// Non-blocking push for real-time callbacks.
    ///
    /// - Never calls blocking `lock()`.
    /// - On lock contention: drops **this** chunk, increments counters, returns `false`.
    /// - On full ring: drops **oldest** chunk(s), keeps newest, returns `false`.
    pub fn try_push(&self, chunk: Vec<f32>) -> bool {
        if chunk.is_empty() {
            return true;
        }
        let Some(mut g) = self.buf.try_lock() else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            self.contention_drops.fetch_add(1, Ordering::Relaxed);
            return false;
        };
        let mut kept = true;
        while g.len() >= self.capacity {
            g.pop_front();
            self.dropped.fetch_add(1, Ordering::Relaxed);
            kept = false;
        }
        g.push_back(chunk);
        kept
    }

    /// Drain all pending chunks. Session owner path — blocking lock is OK here.
    pub fn drain(&self) -> Vec<Vec<f32>> {
        self.buf.lock().drain(..).collect()
    }

    pub fn depth(&self) -> usize {
        self.buf.try_lock().map(|g| g.len()).unwrap_or(0)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn contention_drops(&self) -> u64 {
        self.contention_drops.load(Ordering::Relaxed)
    }

    pub fn clear(&self) {
        if let Some(mut g) = self.buf.try_lock() {
            g.clear();
        } else {
            // Owner fallback: rare; prefer not to stall callbacks.
            self.buf.lock().clear();
        }
    }
}

/// Callback-facing handle. **Sole production audio ingress.**
#[derive(Clone, Debug)]
pub struct AudioIngress {
    ring: Arc<AudioRing>,
}

impl AudioIngress {
    pub fn new(capacity: usize) -> (Self, Arc<AudioRing>) {
        let ring = Arc::new(AudioRing::new(capacity));
        (
            Self {
                ring: Arc::clone(&ring),
            },
            ring,
        )
    }

    pub fn from_ring(ring: Arc<AudioRing>) -> Self {
        Self { ring }
    }

    /// Never waits. Safe for cpal callbacks (`try_lock` only).
    pub fn try_push(&self, chunk: Vec<f32>) -> bool {
        self.ring.try_push(chunk)
    }

    pub fn ring(&self) -> Arc<AudioRing> {
        Arc::clone(&self.ring)
    }
}

impl Default for AudioIngress {
    fn default() -> Self {
        Self::new(DEFAULT_AUDIO_CAPACITY).0
    }
}
