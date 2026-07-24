#![cfg(feature = "cpal-output")]

use shuvoice_tts::{CpalAudioOutputFactory, CpalOutputConfig};

#[test]
fn factory_defaults_and_list_devices() {
    let factory = CpalAudioOutputFactory::default_device();
    assert!(factory.last_device_info().is_none());

    // Enumeration should not panic; empty is allowed on headless CI.
    let listed = CpalAudioOutputFactory::list_output_devices();
    assert!(listed.is_ok(), "list_output_devices error: {listed:?}");
}

#[test]
fn probe_reports_structured_error_or_info() {
    let factory = CpalAudioOutputFactory::new(CpalOutputConfig {
        device: Some("definitely-not-a-real-device-xyz".into()),
        ..CpalOutputConfig::default()
    });
    let err = factory.probe(24_000).unwrap_err();
    assert!(
        err.to_string().contains("not found") || err.to_string().contains("device"),
        "unexpected: {err}"
    );
}

#[test]
fn default_probe_is_host_dependent() {
    let factory = CpalAudioOutputFactory::default_device();
    // On machines with audio, probe succeeds; on headless CI it may fail.
    // Either outcome is acceptable — we only require a typed Result.
    let _ = factory.probe(24_000);
}
