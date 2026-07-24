//! Live offline Parakeet smoke (ignored unless model + feature present).
#![cfg(feature = "sherpa")]

use std::path::PathBuf;

use shuvoice_asr::{
    AsrBackend, AsrBackendKind, PARAKEET_TDT_V3_INT8_MODEL_NAME, SherpaBackend, SherpaDecodeMode,
    is_model_dir_complete, test_config,
};

fn parakeet_dir() -> PathBuf {
    if let Ok(p) = std::env::var("SHUVOICE_PARAKEET_MODEL_DIR") {
        return PathBuf::from(p);
    }
    dirs_data()
        .join("models")
        .join("sherpa")
        .join(PARAKEET_TDT_V3_INT8_MODEL_NAME)
}

fn dirs_data() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME")
        && !xdg.trim().is_empty()
    {
        return PathBuf::from(xdg).join("shuvoice");
    }
    PathBuf::from(std::env::var_os("HOME").unwrap()).join(".local/share/shuvoice")
}

fn downsample_to_16k(sample_rate: u32, samples: &[f32]) -> Vec<f32> {
    if sample_rate == 16_000 {
        return samples.to_vec();
    }
    assert!(
        sample_rate.is_multiple_of(16_000),
        "sample rate {sample_rate} not integer multiple of 16000"
    );
    let ratio = (sample_rate / 16_000) as usize;
    samples.iter().step_by(ratio).copied().collect()
}

fn read_wav_mono_f32(path: &std::path::Path) -> (u32, Vec<f32>) {
    // Minimal WAV reader for PCM16 mono/stereo.
    let bytes = std::fs::read(path).expect("read wav");
    assert!(&bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE");
    let mut offset = 12usize;
    let mut channels = 1u16;
    let mut sample_rate = 16000u32;
    let mut data = Vec::new();
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let start = offset + 8;
        let end = start + size;
        if id == b"fmt " {
            channels = u16::from_le_bytes(bytes[start + 2..start + 4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(bytes[start + 4..start + 8].try_into().unwrap());
            let bits = u16::from_le_bytes(bytes[start + 14..start + 16].try_into().unwrap());
            assert_eq!(bits, 16, "only pcm16 supported in smoke");
        } else if id == b"data" {
            data = bytes[start..end.min(bytes.len())].to_vec();
        }
        offset = end + (size % 2); // word align
    }
    let mut samples = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let s = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(s as f32 / 32768.0);
    }
    if channels == 2 {
        samples = samples
            .chunks_exact(2)
            .map(|c| (c[0] + c[1]) * 0.5)
            .collect();
    }
    (sample_rate, samples)
}

#[tokio::test]
async fn parakeet_offline_cpu_smoke() {
    let model = parakeet_dir();
    if !is_model_dir_complete(&model) {
        eprintln!("skip: parakeet model not complete at {}", model.display());
        return;
    }
    let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audio-sample.wav");
    if !wav.is_file() {
        eprintln!("skip: audio sample missing at {}", wav.display());
        return;
    }

    let (sr, samples) = read_wav_mono_f32(&wav);
    let samples = downsample_to_16k(sr, &samples);

    let mut cfg = test_config(AsrBackendKind::Sherpa);
    cfg.core.sherpa_model_name = PARAKEET_TDT_V3_INT8_MODEL_NAME.into();
    cfg.core.sherpa_model_dir = Some(model.display().to_string());
    cfg.core.sherpa_decode_mode = SherpaDecodeMode::OfflineInstant;
    cfg.core.sherpa_provider = shuvoice_asr::Provider::Cpu;
    cfg.core.instant_mode = true;
    cfg.core.validate().unwrap();

    let mut backend = SherpaBackend::new(cfg);
    let mut progress = |f: Option<f32>, m: &str| {
        eprintln!("load progress {f:?}: {m}");
    };
    backend.load(&mut progress).await.expect("load parakeet");
    let text = backend
        .process_utterance(&samples)
        .await
        .expect("process_utterance");
    eprintln!("PARAKEET_SMOKE_TRANSCRIPT={text:?}");
    // Non-empty is the bar for smoke; content depends on the sample clip.
    assert!(
        !text.trim().is_empty(),
        "expected non-empty transcript from audio-sample.wav"
    );
    backend.shutdown().await.unwrap();
}
