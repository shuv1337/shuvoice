//! Headless composition harness: real ControlServer + fake ASR.
//!
//! No microphone, display, or network required.

use std::time::Duration;

use serial_test::serial;
use shuvoice_app::fakes::remote_asr;
use shuvoice_cli::compose::HeadlessHarness;
use shuvoice_control::ControlCommand;
use shuvoice_core::{AsrBackendKind, Config};
use tempfile::TempDir;

struct EnvGuard {
    key: &'static str,
    prev: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &std::path::Path) -> Self {
        let prev = std::env::var_os(key);
        // SAFETY: serial test mutates process env and restores on drop.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: paired restore of process env mutated in EnvGuard::set.
        unsafe {
            match &self.prev {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn harness_control_roundtrip_and_24k_effective_rate() {
    let tmp = TempDir::new().unwrap();
    let runtime = tmp.path().join("run");
    std::fs::create_dir_all(&runtime).unwrap();
    let _guard = EnvGuard::set("XDG_RUNTIME_DIR", &runtime);

    let sock = runtime.join("shuvoice").join("compose-harness.sock");
    std::fs::create_dir_all(sock.parent().unwrap()).unwrap();

    let mut cfg = Config::default();
    cfg.asr_backend = AsrBackendKind::OpenaiRealtime;
    cfg.tts_enabled = false;
    cfg.audio_feedback = false;
    cfg.chunk_ms = 100;
    cfg.sample_rate = 16_000;
    cfg.control_socket = Some(sock.to_string_lossy().into_owned());
    let _ = cfg.validate();

    let harness = HeadlessHarness::spawn(cfg, Box::new(remote_asr("done")), sock)
        .await
        .expect("spawn");

    assert_eq!(harness.effective_sample_rate(), 24_000);
    assert_eq!(harness.audio_chunk_samples(), 2_400);

    let start = harness.send_control(ControlCommand::Start).unwrap();
    assert!(start.contains("OK"), "{start}");
    let status = harness.send_control(ControlCommand::Status).unwrap();
    assert!(status.starts_with("OK "), "{status}");
    let stop = harness.send_control(ControlCommand::Stop).unwrap();
    assert!(stop.contains("OK"), "{stop}");

    // Brief settle so actor applies stop before shutdown.
    tokio::time::sleep(Duration::from_millis(20)).await;
    harness.shutdown().await.expect("shutdown");
}

#[test]
fn feature_flags_desktop_is_default_packaged_surface() {
    // This test binary is built with the package default features unless
    // `--no-default-features` is passed. Document the packaged desktop set.
    const DESKTOP: bool = cfg!(feature = "desktop");
    const AUDIO: bool = cfg!(feature = "audio");
    const SHERPA: bool = cfg!(feature = "asr-sherpa");
    const OPENAI: bool = cfg!(feature = "asr-openai");
    const UI: bool = cfg!(feature = "ui");
    const TTS: bool = cfg!(feature = "tts");
    const TTS_WORKER: bool = cfg!(feature = "tts-worker");
    const OK: bool = !DESKTOP || (AUDIO && SHERPA && OPENAI && UI && TTS && TTS_WORKER);
    const {
        assert!(OK);
    }
}
