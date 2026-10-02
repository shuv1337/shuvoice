use shuvoice_app::fakes::ScriptedAsrBackend;
use shuvoice_app::{Config, SessionCommand, TestHarness};

fn token(response: &str) -> u64 {
    response
        .split_whitespace()
        .find_map(|part| part.strip_prefix("token=").and_then(|n| n.parse().ok()))
        .expect("reservation token")
}

#[tokio::test]
async fn reservation_serializes_start_toggle_tts_and_expires() {
    let mut h = TestHarness::basic(ScriptedAsrBackend::default(), Config::default()).await;
    let first = token(
        &h.session
            .handle_command(SessionCommand::MaintenanceReserve)
            .await
            .unwrap(),
    );
    for cmd in [
        SessionCommand::Start,
        SessionCommand::Toggle,
        SessionCommand::TtsSpeakSelection,
        SessionCommand::TtsSpeakClipboard,
        SessionCommand::TtsResume,
        SessionCommand::TtsRestart,
    ] {
        assert!(
            h.session
                .handle_command(cmd)
                .await
                .unwrap()
                .starts_with("ERROR busy")
        );
    }
    assert!(!h.session.is_recording());
    h.clock.advance_ms(120_001);
    let second = token(
        &h.session
            .handle_command(SessionCommand::MaintenanceReserve)
            .await
            .unwrap(),
    );
    assert!(
        h.session
            .handle_command(SessionCommand::MaintenanceRelease(first))
            .await
            .unwrap()
            .starts_with("ERROR maintenance token")
    );
    assert_eq!(
        h.session
            .handle_command(SessionCommand::MaintenanceRelease(second))
            .await
            .unwrap(),
        "OK released"
    );
    h.session
        .handle_command(SessionCommand::Start)
        .await
        .unwrap();
    assert!(
        h.session
            .handle_command(SessionCommand::MaintenanceReserve)
            .await
            .unwrap()
            .starts_with("ERROR busy")
    );
    h.shutdown().await;
}

#[tokio::test]
async fn disconnected_reservation_acknowledgment_releases_the_grant() {
    let mut h = TestHarness::basic(ScriptedAsrBackend::default(), Config::default()).await;
    let (reply, receiver) = std::sync::mpsc::channel();
    drop(receiver);
    h.session
        .handle_command(SessionCommand::ControlRequest {
            command: Box::new(SessionCommand::MaintenanceReserve),
            reply,
        })
        .await
        .unwrap();
    assert!(
        h.session
            .handle_command(SessionCommand::MaintenanceReserve)
            .await
            .unwrap()
            .starts_with("OK reserved")
    );
    h.shutdown().await;
}

#[tokio::test]
async fn serialized_tts_preserves_legacy_acknowledgments_outside_maintenance() {
    let mut h = TestHarness::basic(ScriptedAsrBackend::default(), Config::default()).await;
    for (command, expected) in [
        (SessionCommand::TtsResume, "OK tts resumed"),
        (SessionCommand::TtsRestart, "OK tts restarted"),
        (SessionCommand::TtsTogglePause, "OK tts toggled"),
    ] {
        let (reply, receiver) = std::sync::mpsc::channel();
        let _ = h
            .session
            .handle_command(SessionCommand::ControlRequest {
                command: Box::new(command),
                reply,
            })
            .await;
        assert_eq!(receiver.try_recv().unwrap(), expected);
    }
    h.shutdown().await;
}
