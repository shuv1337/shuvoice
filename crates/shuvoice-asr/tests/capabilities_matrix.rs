use shuvoice_asr::{
    AsrBackendKind, ExpectedChunking, FinalizationMode, create_backend, dependency_errors,
    dependency_errors_for, moonshine_caps, nemo_caps, openai_realtime_caps, sherpa_offline_caps,
    sherpa_streaming_caps, test_config,
};

#[test]
fn wants_raw_audio_matrix() {
    assert!(nemo_caps().wants_raw_audio);
    assert!(moonshine_caps().wants_raw_audio);
    assert!(openai_realtime_caps().wants_raw_audio);
    assert!(!sherpa_streaming_caps().wants_raw_audio);
    assert!(!sherpa_offline_caps().wants_raw_audio);
}

#[test]
fn finalization_modes() {
    assert_eq!(
        openai_realtime_caps().finalization_mode,
        FinalizationMode::RemoteManualCommit
    );
    assert_eq!(
        sherpa_offline_caps().finalization_mode,
        FinalizationMode::OfflineInstant
    );
    assert_eq!(
        sherpa_streaming_caps().finalization_mode,
        FinalizationMode::LocalStreaming
    );
    assert_eq!(
        moonshine_caps().expected_chunking,
        ExpectedChunking::Windowed
    );
}

#[test]
fn backend_id_parse() {
    assert_eq!(
        "openai_realtime".parse::<AsrBackendKind>().unwrap(),
        AsrBackendKind::OpenaiRealtime
    );
    assert_eq!(
        "SHERPA".parse::<AsrBackendKind>().unwrap(),
        AsrBackendKind::Sherpa
    );
    assert!("nope".parse::<AsrBackendKind>().is_err());
}

#[test]
fn nemo_create_returns_worker_client() {
    let cfg = test_config(AsrBackendKind::Nemo);
    let b = create_backend(cfg).unwrap();
    assert_eq!(b.backend_id(), AsrBackendKind::Nemo);
}

#[test]
fn dependency_errors_mention_bridge_for_nemo() {
    let errs = dependency_errors(AsrBackendKind::Nemo);
    assert!(errs.iter().any(|e| e.contains("worker")));
}

#[test]
fn dependency_errors_empty_when_worker_configured() {
    let mut cfg = test_config(AsrBackendKind::Nemo);
    cfg.connect.worker_command = Some(vec!["python".into(), "worker.py".into()]);
    assert!(dependency_errors_for(AsrBackendKind::Nemo, Some(&cfg)).is_empty());
    assert!(!dependency_errors(AsrBackendKind::Nemo).is_empty());
}
