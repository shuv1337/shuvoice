//! Capture sample-rate contract: the active ASR backend's preferred rate wins.

use shuvoice_app::fakes::{ScriptedAsrBackend, remote_asr};
use shuvoice_app::{
    Config, TestHarness, effective_audio_chunk_samples, effective_audio_sample_rate,
};

fn base_cfg() -> Config {
    let mut c = Config::default();
    c.sample_rate = 16_000; // stock config
    c.chunk_ms = 100;
    c.min_speech_ms = 80;
    c.recording_preroll_ms = 200;
    c.silence_rms_threshold = 0.0;
    c.tts_enabled = false;
    c
}

#[test]
fn helper_prefers_asr_rate_over_config() {
    assert_eq!(effective_audio_sample_rate(16_000, Some(24_000)), 24_000);
    assert_eq!(effective_audio_sample_rate(16_000, None), 16_000);
    assert_eq!(effective_audio_sample_rate(0, Some(24_000)), 24_000);
    assert_eq!(effective_audio_chunk_samples(24_000, 100), 2_400);
    assert_eq!(effective_audio_chunk_samples(16_000, 100), 1_600);
}

#[tokio::test]
async fn openai_remote_session_uses_24khz_not_config_16k() {
    let scripted = remote_asr("ok");
    let h = TestHarness::basic(scripted, base_cfg()).await;
    assert_eq!(h.session.config_sample_rate(), 16_000);
    assert_eq!(h.session.asr_preferred_sample_rate(), Some(24_000));
    assert_eq!(h.session.effective_sample_rate(), 24_000);
    assert_eq!(h.session.audio_chunk_samples(), 2_400);
    // min_speech_ms=80 at 24k => 1920 samples
    assert_eq!(h.session.min_speech_samples(), 24_000 * 80 / 1000);
    let dbg: serde_json::Value = serde_json::from_str(&h.session.debug_status_json()).unwrap();
    assert_eq!(dbg["audio"]["effective_sample_rate"], 24_000);
    assert_eq!(dbg["audio"]["config_sample_rate"], 16_000);
    assert_eq!(dbg["audio"]["asr_preferred_sample_rate"], 24_000);
    assert_eq!(dbg["audio"]["audio_chunk_samples"], 2_400);
    h.shutdown().await;
}

#[tokio::test]
async fn sherpa_session_stays_16k_when_caps_say_16k() {
    let h = TestHarness::basic(ScriptedAsrBackend::local_streaming(1600), base_cfg()).await;
    assert_eq!(h.session.effective_sample_rate(), 16_000);
    assert_eq!(h.session.audio_chunk_samples(), 1_600);
    assert_eq!(h.session.min_speech_samples(), 16_000 * 80 / 1000);
    h.shutdown().await;
}

#[tokio::test]
async fn sync_audio_params_is_idempotent_for_openai_caps() {
    let mut h = TestHarness::basic(remote_asr("x"), base_cfg()).await;
    assert_eq!(h.session.effective_sample_rate(), 24_000);
    h.session.sync_audio_params_from_asr();
    assert_eq!(h.session.effective_sample_rate(), 24_000);
    assert_eq!(h.session.audio_chunk_samples(), 2_400);
    h.shutdown().await;
}

#[test]
fn pre_spawn_normalization_contract() {
    // Hosts that open the capture device before spawn must normalize using the
    // same helper the session uses after load (preferred ASR rate wins).
    let mut cfg = base_cfg();
    let preferred = Some(24_000u32);
    let rate = effective_audio_sample_rate(cfg.sample_rate, preferred);
    cfg.sample_rate = rate;
    assert_eq!(cfg.sample_rate, 24_000);
    assert_eq!(
        effective_audio_chunk_samples(cfg.sample_rate, cfg.chunk_ms),
        2_400
    );
}
