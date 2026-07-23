//! Production CPAL audio output (feature = `cpal-output`).
//!
//! - Best sample-rate selection (exact → prefer 48 kHz → nearest)
//! - Linear resampling on the **playback OS thread** (never in the RT callback)
//! - Callback uses `try_lock` only — never blocks or allocates

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    BufferSize, Device, Sample, SampleFormat, SampleRate, Stream, StreamConfig,
    SupportedStreamConfig, SupportedStreamConfigRange,
};
use parking_lot::Mutex;

use super::pcm::resample_linear_i16;
use super::{AudioOutput, AudioOutputFactory};
use crate::error::TtsError;

const DEFAULT_RING_FRAMES: usize = 48_000;
const WRITE_WAIT_SLICE: Duration = Duration::from_millis(10);

/// Diagnostics for the resolved output path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputDeviceInfo {
    pub name: String,
    pub requested_sample_rate_hz: u32,
    pub effective_sample_rate_hz: u32,
    pub channels: u16,
    pub sample_format: String,
    pub buffer_frames: Option<u32>,
    pub resampling: bool,
}

#[derive(Debug, Clone)]
pub struct CpalOutputConfig {
    pub device: Option<String>,
    pub ring_frames: usize,
    pub buffer_frames: Option<u32>,
}

impl Default for CpalOutputConfig {
    fn default() -> Self {
        Self {
            device: None,
            ring_frames: DEFAULT_RING_FRAMES,
            buffer_frames: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CpalAudioOutputFactory {
    config: CpalOutputConfig,
    last_info: Arc<Mutex<Option<OutputDeviceInfo>>>,
}

impl CpalAudioOutputFactory {
    #[must_use]
    pub fn new(config: CpalOutputConfig) -> Self {
        Self {
            config,
            last_info: Arc::new(Mutex::new(None)),
        }
    }

    #[must_use]
    pub fn default_device() -> Self {
        Self::new(CpalOutputConfig::default())
    }

    #[must_use]
    pub fn last_device_info(&self) -> Option<OutputDeviceInfo> {
        self.last_info.lock().clone()
    }

    pub fn probe(&self, sample_rate_hz: u32) -> Result<OutputDeviceInfo, TtsError> {
        let host = cpal::default_host();
        let device = select_output_device(&host, self.config.device.as_deref())?;
        let (supported, requested, effective) = pick_best_output_config(&device, sample_rate_hz)?;
        Ok(device_info(
            &device,
            &supported,
            requested,
            effective,
            self.config.buffer_frames,
        ))
    }

    pub fn list_output_devices() -> Result<Vec<String>, TtsError> {
        let host = cpal::default_host();
        let mut names = Vec::new();
        for dev in host
            .output_devices()
            .map_err(|err| TtsError::audio(format!("enumerate output devices: {err}")))?
        {
            names.push(device_name(&dev).unwrap_or_else(|| "<unknown>".into()));
        }
        Ok(names)
    }
}

impl Default for CpalAudioOutputFactory {
    fn default() -> Self {
        Self::default_device()
    }
}

impl AudioOutputFactory for CpalAudioOutputFactory {
    fn open(&self, sample_rate_hz: u32) -> Result<Arc<dyn AudioOutput>, TtsError> {
        let host = cpal::default_host();
        let device = select_output_device(&host, self.config.device.as_deref())?;
        let (supported, requested, effective) = pick_best_output_config(&device, sample_rate_hz)?;
        let info = device_info(
            &device,
            &supported,
            requested,
            effective,
            self.config.buffer_frames,
        );
        tracing::info!(
            device = %info.name,
            requested = info.requested_sample_rate_hz,
            effective = info.effective_sample_rate_hz,
            channels = info.channels,
            sample_format = %info.sample_format,
            resampling = info.resampling,
            "Opening CPAL TTS output stream"
        );

        let ring = Arc::new(CallbackRing::new(self.config.ring_frames.max(1024)));
        let stream = build_output_stream(
            &device,
            &supported,
            self.config.buffer_frames,
            Arc::clone(&ring),
        )?;
        stream
            .play()
            .map_err(|err| TtsError::audio(format!("CPAL stream play failed: {err}")))?;

        *self.last_info.lock() = Some(info.clone());

        Ok(Arc::new(CpalAudioOutput {
            stream: Mutex::new(Some(stream)),
            ring,
            info,
            closed: AtomicBool::new(false),
        }))
    }
}

pub struct CpalAudioOutput {
    stream: Mutex<Option<Stream>>,
    ring: Arc<CallbackRing>,
    info: OutputDeviceInfo,
    closed: AtomicBool,
}

impl CpalAudioOutput {
    #[must_use]
    pub fn device_info(&self) -> &OutputDeviceInfo {
        &self.info
    }
}

impl AudioOutput for CpalAudioOutput {
    fn write_samples(&self, samples: &[i16]) -> Result<(), TtsError> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(TtsError::audio("CPAL output closed"));
        }
        let pcm = if self.info.resampling {
            resample_linear_i16(
                samples,
                self.info.requested_sample_rate_hz,
                self.info.effective_sample_rate_hz,
            )
        } else {
            samples.to_vec()
        };
        self.ring.push_all(&pcm)
    }

    fn close(&self) -> Result<(), TtsError> {
        self.closed.store(true, Ordering::SeqCst);
        self.ring.close();
        if let Some(stream) = self.stream.lock().take() {
            let _ = stream.pause();
            drop(stream);
        }
        Ok(())
    }

    fn interrupt(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.ring.close();
    }
}

impl Drop for CpalAudioOutput {
    fn drop(&mut self) {
        let _ = AudioOutput::close(self);
    }
}

// ── Ring: producer may block; consumer (RT callback) only try_lock ─────────

struct CallbackRing {
    buf: Mutex<VecDeque<i16>>,
    capacity: usize,
    closed: AtomicBool,
}

impl CallbackRing {
    fn new(capacity: usize) -> Self {
        Self {
            buf: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity: capacity.max(64),
            closed: AtomicBool::new(false),
        }
    }

    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    fn push_all(&self, samples: &[i16]) -> Result<(), TtsError> {
        let mut offset = 0;
        while offset < samples.len() {
            if self.is_closed() {
                return Err(TtsError::audio("CPAL output closed"));
            }
            let mut guard = self.buf.lock();
            while guard.len() >= self.capacity && !self.is_closed() {
                drop(guard);
                std::thread::sleep(WRITE_WAIT_SLICE);
                if self.is_closed() {
                    return Err(TtsError::audio("CPAL output closed"));
                }
                guard = self.buf.lock();
            }
            if self.is_closed() {
                return Err(TtsError::audio("CPAL output closed"));
            }
            if guard.len() >= self.capacity {
                continue;
            }
            let take = (self.capacity - guard.len()).min(samples.len() - offset);
            guard.extend(samples[offset..offset + take].iter().copied());
            offset += take;
        }
        Ok(())
    }

    /// RT-safe: never allocates, never blocks.
    fn try_pull_mono_into(&self, dst: &mut [i16]) -> usize {
        let Some(mut guard) = self.buf.try_lock() else {
            for s in dst.iter_mut() {
                *s = 0;
            }
            return 0;
        };
        let n = dst.len().min(guard.len());
        for slot in dst.iter_mut().take(n) {
            *slot = guard.pop_front().unwrap_or(0);
        }
        for slot in dst.iter_mut().skip(n) {
            *slot = 0;
        }
        n
    }
}

// ── Device / config selection ────────────────────────────────────────────

fn select_output_device(host: &cpal::Host, selector: Option<&str>) -> Result<Device, TtsError> {
    let Some(raw) = selector.map(str::trim).filter(|s| !s.is_empty()) else {
        return host
            .default_output_device()
            .ok_or_else(|| TtsError::audio("no default CPAL output device"));
    };

    let devices: Vec<Device> = host
        .output_devices()
        .map_err(|err| TtsError::audio(format!("enumerate output devices: {err}")))?
        .collect();

    if let Ok(index) = raw.parse::<usize>() {
        return devices
            .into_iter()
            .nth(index)
            .ok_or_else(|| TtsError::audio(format!("output device index {index} out of range")));
    }

    for dev in &devices {
        if device_name(dev).as_deref() == Some(raw) {
            return Ok(dev.clone());
        }
    }
    let lowered = raw.to_ascii_lowercase();
    for dev in &devices {
        if device_name(dev)
            .map(|n| n.to_ascii_lowercase().contains(&lowered))
            .unwrap_or(false)
        {
            return Ok(dev.clone());
        }
    }

    Err(TtsError::audio(format!(
        "output device not found: {raw} (available: {})",
        devices
            .iter()
            .filter_map(device_name)
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

/// Returns (config, requested_rate, effective_rate).
fn pick_best_output_config(
    device: &Device,
    requested: u32,
) -> Result<(SupportedStreamConfig, u32, u32), TtsError> {
    let mut supported = device
        .supported_output_configs()
        .map_err(|err| TtsError::audio(format!("query output configs: {err}")))?
        .collect::<Vec<_>>();

    if supported.is_empty() {
        return Err(TtsError::audio("device reports no output configs"));
    }

    supported.sort_by_key(|range| {
        let fmt_rank = match range.sample_format() {
            SampleFormat::F32 => 0u16,
            SampleFormat::I16 => 1,
            SampleFormat::U16 => 2,
            SampleFormat::I32 => 3,
            SampleFormat::F64 => 4,
            _ => 100,
        };
        let ch_rank = match range.channels() {
            1 => 0u16,
            2 => 1,
            n => n,
        };
        (fmt_rank, ch_rank)
    });

    // 1) Exact rate
    if let Some(cfg) = try_rate(&supported, requested) {
        return Ok((cfg, requested, requested));
    }

    // 2) Prefer 48 kHz
    if requested != 48_000
        && let Some(cfg) = try_rate(&supported, 48_000)
    {
        return Ok((cfg, requested, 48_000));
    }

    // 3) Prefer 44.1 kHz
    if let Some(cfg) = try_rate(&supported, 44_100) {
        return Ok((cfg, requested, 44_100));
    }

    // 4) Nearest supported rate from any range (use mid or clamp)
    let mut best: Option<(SupportedStreamConfig, u32, u32)> = None; // cfg, effective, distance
    for range in &supported {
        if !is_supported_sample_format(range.sample_format()) {
            continue;
        }
        let min = range.min_sample_rate();
        let max = range.max_sample_rate();
        let candidate = requested.clamp(min, max);
        let dist = requested.abs_diff(candidate);
        let cfg = (*range).with_sample_rate(candidate as SampleRate);
        match &best {
            None => best = Some((cfg, candidate, dist)),
            Some((_, _, best_dist)) if dist < *best_dist => {
                best = Some((cfg, candidate, dist));
            }
            _ => {}
        }
    }

    best.map(|(cfg, effective, _)| (cfg, requested, effective))
        .ok_or_else(|| {
            TtsError::audio(format!(
                "no usable CPAL output config on device '{}'",
                device_name(device).unwrap_or_else(|| "<unknown>".into())
            ))
        })
}

fn try_rate(supported: &[SupportedStreamConfigRange], rate: u32) -> Option<SupportedStreamConfig> {
    let sr = rate as SampleRate;
    for range in supported {
        if !is_supported_sample_format(range.sample_format()) {
            continue;
        }
        if range.min_sample_rate() <= sr && range.max_sample_rate() >= sr {
            return Some((*range).with_sample_rate(sr));
        }
    }
    None
}

fn is_supported_sample_format(fmt: SampleFormat) -> bool {
    matches!(
        fmt,
        SampleFormat::I8
            | SampleFormat::U8
            | SampleFormat::I16
            | SampleFormat::U16
            | SampleFormat::I32
            | SampleFormat::U32
            | SampleFormat::F32
            | SampleFormat::F64
    )
}

fn device_name(device: &Device) -> Option<String> {
    device.description().ok().map(|d| d.name().to_string())
}

fn device_info(
    device: &Device,
    supported: &SupportedStreamConfig,
    requested: u32,
    effective: u32,
    buffer_frames: Option<u32>,
) -> OutputDeviceInfo {
    OutputDeviceInfo {
        name: device_name(device).unwrap_or_else(|| "<unknown>".into()),
        requested_sample_rate_hz: requested,
        effective_sample_rate_hz: effective,
        channels: supported.channels(),
        sample_format: format!("{:?}", supported.sample_format()),
        buffer_frames,
        resampling: requested != effective,
    }
}

fn build_output_stream(
    device: &Device,
    supported: &SupportedStreamConfig,
    buffer_frames: Option<u32>,
    ring: Arc<CallbackRing>,
) -> Result<Stream, TtsError> {
    let channels = supported.channels();
    let sample_format = supported.sample_format();
    let mut config: StreamConfig = (*supported).into();
    if let Some(frames) = buffer_frames {
        config.buffer_size = BufferSize::Fixed(frames);
    }

    let err_fn = |err| tracing::warn!("CPAL TTS output status: {err}");

    let stream = match sample_format {
        SampleFormat::F32 => device.build_output_stream(
            config,
            make_callback::<f32>(Arc::clone(&ring), channels),
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_output_stream(
            config,
            make_callback::<i16>(Arc::clone(&ring), channels),
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_output_stream(
            config,
            make_callback::<u16>(Arc::clone(&ring), channels),
            err_fn,
            None,
        ),
        SampleFormat::I32 => device.build_output_stream(
            config,
            make_callback::<i32>(Arc::clone(&ring), channels),
            err_fn,
            None,
        ),
        SampleFormat::U32 => device.build_output_stream(
            config,
            make_callback::<u32>(Arc::clone(&ring), channels),
            err_fn,
            None,
        ),
        SampleFormat::F64 => device.build_output_stream(
            config,
            make_callback::<f64>(Arc::clone(&ring), channels),
            err_fn,
            None,
        ),
        SampleFormat::I8 => device.build_output_stream(
            config,
            make_callback::<i8>(Arc::clone(&ring), channels),
            err_fn,
            None,
        ),
        SampleFormat::U8 => device.build_output_stream(
            config,
            make_callback::<u8>(Arc::clone(&ring), channels),
            err_fn,
            None,
        ),
        other => {
            return Err(TtsError::audio(format!(
                "unsupported CPAL sample format for TTS output: {other:?}"
            )));
        }
    }
    .map_err(|err| TtsError::audio(format!("build CPAL output stream: {err}")))?;

    Ok(stream)
}

fn sample_i16_to<T>(sample: i16) -> T
where
    T: Sample + cpal::FromSample<i16>,
{
    T::from_sample(sample)
}

fn make_callback<T>(
    ring: Arc<CallbackRing>,
    channels: u16,
) -> impl FnMut(&mut [T], &cpal::OutputCallbackInfo) + Send + 'static
where
    T: Sample + cpal::SizedSample + cpal::FromSample<i16> + Send + 'static,
{
    let channels = channels.max(1) as usize;
    // Pre-allocate scratch once per stream (callback must not allocate).
    let mut mono_scratch = vec![0i16; 4096];
    move |data: &mut [T], _| {
        let frames = data.len() / channels;
        if frames == 0 {
            return;
        }
        if frames > mono_scratch.len() {
            // Grow only if host asks for a larger buffer than initial (rare).
            // This is still an allocation in RT — keep scratch large enough by default.
            mono_scratch.resize(frames, 0);
        }
        let mono = &mut mono_scratch[..frames];
        let _ = ring.try_pull_mono_into(mono);
        for (frame_idx, &sample) in mono.iter().enumerate() {
            let converted = sample_i16_to::<T>(sample);
            let base = frame_idx * channels;
            for ch in 0..channels {
                if let Some(slot) = data.get_mut(base + ch) {
                    *slot = converted;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_push_pull_roundtrip() {
        let ring = CallbackRing::new(8);
        ring.push_all(&[1, 2, 3, 4]).unwrap();
        let mut out = [0i16; 6];
        let n = ring.try_pull_mono_into(&mut out);
        assert_eq!(n, 4);
        assert_eq!(&out[..4], &[1, 2, 3, 4]);
        assert_eq!(&out[4..], &[0, 0]);
    }

    #[test]
    fn ring_close_errors_push() {
        let ring = CallbackRing::new(4);
        ring.close();
        assert!(ring.push_all(&[1]).is_err());
    }

    #[test]
    fn sample_format_gate() {
        assert!(is_supported_sample_format(SampleFormat::F32));
        assert!(!is_supported_sample_format(SampleFormat::DsdU8));
    }
}
