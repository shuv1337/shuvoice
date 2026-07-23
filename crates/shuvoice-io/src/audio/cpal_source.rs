//! Optional cpal-backed audio capture (cpal 0.18 API).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, Stream, StreamConfig};

use crate::audio::{
    AudioConfig, AudioQueue, IntegerDecimator, apply_input_gain, integer_ratio,
    prefer_pulse_pipewire_index,
};
use crate::error::AudioError;

/// cpal input capture with bounded drop-oldest queue and optional integer decimation.
pub struct CpalAudioCapture {
    cfg: AudioConfig,
    queue: Arc<Mutex<AudioQueue>>,
    stream: Option<Stream>,
    resolved_device_name: Option<String>,
    resampling: bool,
    callback_lock_fails: Arc<AtomicU64>,
}

impl CpalAudioCapture {
    #[must_use]
    pub fn new(cfg: AudioConfig) -> Self {
        let qsize = cfg.queue_max_size.max(1);
        Self {
            cfg,
            queue: Arc::new(Mutex::new(AudioQueue::new(qsize))),
            stream: None,
            resolved_device_name: None,
            resampling: false,
            callback_lock_fails: Arc::new(AtomicU64::new(0)),
        }
    }

    #[must_use]
    pub fn resolved_device_name(&self) -> Option<&str> {
        self.resolved_device_name.as_deref()
    }

    #[must_use]
    pub fn is_resampling(&self) -> bool {
        self.resampling
    }

    pub fn dropped_chunks(&self) -> u64 {
        self.queue.lock().map(|q| q.dropped_chunks).unwrap_or(0)
    }

    pub fn queue_depth(&self) -> usize {
        self.queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Number of times the audio callback dropped a frame due to a contended queue lock.
    pub fn callback_lock_fails(&self) -> u64 {
        self.callback_lock_fails.load(Ordering::Relaxed)
    }

    pub fn start(&mut self) -> Result<(), AudioError> {
        self.stop();
        let host = cpal::default_host();
        let device = select_device(&host, self.cfg.device_name.as_deref())?;
        self.resolved_device_name = device_name(&device);

        match self.open_stream(&device, self.cfg.sample_rate, false) {
            Ok(stream) => {
                stream
                    .play()
                    .map_err(|e| AudioError::Stream(e.to_string()))?;
                self.stream = Some(stream);
                self.resampling = false;
                tracing::info!(
                    "Audio capture started at {} Hz (device={:?}, gain={:.2})",
                    self.cfg.sample_rate,
                    self.resolved_device_name,
                    self.cfg.input_gain
                );
                Ok(())
            }
            Err(err) => {
                tracing::warn!(
                    "Failed at {} Hz ({err}), falling back to {} Hz",
                    self.cfg.sample_rate,
                    self.cfg.fallback_sample_rate
                );
                let ratio = integer_ratio(self.cfg.fallback_sample_rate, self.cfg.sample_rate)?;
                let stream = self.open_stream(&device, self.cfg.fallback_sample_rate, true)?;
                stream
                    .play()
                    .map_err(|e| AudioError::Stream(e.to_string()))?;
                self.stream = Some(stream);
                self.resampling = true;
                tracing::info!(
                    "Audio capture started at {} Hz (resampling to {} Hz, ratio={}x, device={:?}, gain={:.2})",
                    self.cfg.fallback_sample_rate,
                    self.cfg.sample_rate,
                    ratio,
                    self.resolved_device_name,
                    self.cfg.input_gain
                );
                Ok(())
            }
        }
    }

    pub fn stop(&mut self) {
        if let Some(stream) = self.stream.take() {
            let _ = stream.pause();
            drop(stream);
        }
        if let Ok(q) = self.queue.lock()
            && q.dropped_chunks > 0
        {
            tracing::info!("Audio dropped chunks total: {}", q.dropped_chunks);
        }
    }

    pub fn drain_pending_chunks(&self) -> Vec<Vec<f32>> {
        self.queue.lock().map(|mut q| q.drain()).unwrap_or_default()
    }

    pub fn clear(&self) {
        if let Ok(mut q) = self.queue.lock() {
            q.clear();
        }
    }

    pub fn get_chunk(&self, timeout: Duration) -> Option<Vec<f32>> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Ok(mut q) = self.queue.lock()
                && let Some(chunk) = q.pop()
            {
                return Some(chunk);
            }
            if std::time::Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    fn open_stream(
        &self,
        device: &cpal::Device,
        sample_rate: u32,
        resample: bool,
    ) -> Result<Stream, AudioError> {
        let channels = device
            .default_input_config()
            .map(|c| c.channels().max(1))
            .unwrap_or(1);

        // Prefer device native channel count; we downmix to mono in the callback.
        let config = StreamConfig {
            channels,
            sample_rate,
            buffer_size: BufferSize::Default,
        };

        let queue = Arc::clone(&self.queue);
        let gain = self.cfg.input_gain;
        let target_chunk = self.cfg.chunk_samples.max(1);
        let ratio = if resample {
            integer_ratio(self.cfg.fallback_sample_rate, self.cfg.sample_rate)? as usize
        } else {
            1
        };
        let decimator = Arc::new(Mutex::new(IntegerDecimator::new(ratio)));
        let pending = Arc::new(Mutex::new(Vec::<f32>::new()));
        let lock_fails = Arc::clone(&self.callback_lock_fails);
        let ch = channels as usize;

        let err_fn = |err| tracing::warn!("Audio status: {err}");

        device
            .build_input_stream(
                config,
                move |data: &[f32], _| {
                    // Downmix interleaved multi-channel → mono.
                    let mono = if ch <= 1 {
                        data.to_vec()
                    } else {
                        let frames = data.len() / ch;
                        let mut out = Vec::with_capacity(frames);
                        for i in 0..frames {
                            let mut sum = 0.0f32;
                            for c in 0..ch {
                                sum += data[i * ch + c];
                            }
                            out.push(sum / ch as f32);
                        }
                        out
                    };

                    let mut mono = mono;
                    if ratio > 1 {
                        let Ok(mut dec) = decimator.try_lock() else {
                            lock_fails.fetch_add(1, Ordering::Relaxed);
                            return;
                        };
                        mono = dec.push(&mono);
                        if mono.is_empty() {
                            return;
                        }
                    }
                    apply_input_gain(&mut mono, gain);

                    // Assemble fixed-size chunks.
                    let Ok(mut pend) = pending.try_lock() else {
                        lock_fails.fetch_add(1, Ordering::Relaxed);
                        return;
                    };
                    pend.extend_from_slice(&mono);
                    while pend.len() >= target_chunk {
                        let chunk: Vec<f32> = pend.drain(..target_chunk).collect();
                        match queue.try_lock() {
                            Ok(mut q) => q.push(chunk),
                            Err(_) => {
                                lock_fails.fetch_add(1, Ordering::Relaxed);
                                // Drop this chunk rather than block the audio callback.
                            }
                        }
                    }
                },
                err_fn,
                None,
            )
            .map_err(|e| AudioError::Stream(e.to_string()))
    }
}

impl Drop for CpalAudioCapture {
    fn drop(&mut self) {
        self.stop();
    }
}

fn device_name(device: &cpal::Device) -> Option<String> {
    device.description().ok().map(|d| d.name().to_string())
}

fn select_device(host: &cpal::Host, explicit: Option<&str>) -> Result<cpal::Device, AudioError> {
    if let Some(name) = explicit {
        for dev in host
            .input_devices()
            .map_err(|e| AudioError::Device(e.to_string()))?
        {
            let dev_name = device_name(&dev).unwrap_or_default();
            if dev_name == name {
                return Ok(dev);
            }
        }
        return Err(AudioError::Device(format!(
            "input device not found: {name}"
        )));
    }

    let devices: Vec<cpal::Device> = host
        .input_devices()
        .map_err(|e| AudioError::Device(e.to_string()))?
        .collect();

    let indexed: Vec<(usize, String, u16)> = devices
        .iter()
        .enumerate()
        .filter_map(|(idx, d)| {
            let name = device_name(d)?;
            let max_in = d.default_input_config().map(|c| c.channels()).unwrap_or(0);
            Some((idx, name, max_in))
        })
        .collect();

    if let Some(idx) =
        prefer_pulse_pipewire_index(indexed.iter().map(|(i, n, c)| (*i, n.as_str(), *c)))
    {
        return Ok(devices[idx].clone());
    }

    host.default_input_device()
        .ok_or_else(|| AudioError::Device("no default input device".into()))
}
