#![cfg(feature = "mpris")]
use shuvoice_core::media::*;
use shuvoice_io::media::MprisClient;
use std::{
    collections::HashMap,
    io::{BufRead, BufReader},
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use zbus::zvariant::{OwnedValue, Str};

struct PrivateBus {
    child: Child,
    address: String,
}
impl PrivateBus {
    fn start() -> Option<Self> {
        let mut child = match Command::new("dbus-daemon")
            .args(["--session", "--nofork", "--print-address=1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                assert!(
                    std::env::var_os("CI").is_none(),
                    "CI requires dbus-daemon for private MPRIS tests"
                );
                eprintln!("SKIP private MPRIS test: dbus-daemon is unavailable");
                return None;
            }
            Err(e) => panic!("private bus spawn: {e}"),
        };
        let mut address = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut address)
            .unwrap();
        assert!(address.starts_with("unix:"));
        Some(Self {
            child,
            address: address.trim().into(),
        })
    }
}
impl Drop for PrivateBus {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
struct Player {
    pauses: Arc<AtomicUsize>,
    plays: Arc<AtomicUsize>,
}
#[zbus::interface(name = "org.mpris.MediaPlayer2.Player")]
impl Player {
    fn pause(&self) {
        self.pauses.fetch_add(1, Ordering::SeqCst);
    }
    fn play(&self) {
        self.plays.fetch_add(1, Ordering::SeqCst);
    }
    #[zbus(property)]
    fn playback_status(&self) -> &str {
        "Playing"
    }
    #[zbus(property)]
    fn can_control(&self) -> bool {
        true
    }
    #[zbus(property)]
    fn can_pause(&self) -> bool {
        true
    }
    #[zbus(property)]
    fn can_play(&self) -> bool {
        true
    }
    #[zbus(property)]
    fn metadata(&self) -> HashMap<String, OwnedValue> {
        HashMap::from([(
            "xesam:title".into(),
            OwnedValue::from(Str::from("private test track")),
        )])
    }
}

#[tokio::test]
async fn explicit_private_bus_deduplicates_aliases_checks_owner_and_reads_uncached() {
    let Some(bus) = PrivateBus::start() else {
        return;
    };
    let pauses = Arc::new(AtomicUsize::new(0));
    let plays = Arc::new(AtomicUsize::new(0));
    let service = zbus::connection::Builder::address(bus.address.as_str())
        .unwrap()
        .name("org.mpris.MediaPlayer2.fake")
        .unwrap()
        .serve_at(
            "/org/mpris/MediaPlayer2",
            Player {
                pauses: pauses.clone(),
                plays: plays.clone(),
            },
        )
        .unwrap()
        .build()
        .await
        .unwrap();
    service
        .request_name("org.mpris.MediaPlayer2.alias")
        .await
        .unwrap();
    let synthetic = zbus::connection::Builder::address(bus.address.as_str())
        .unwrap()
        .name("org.mpris.MediaPlayer2.playerctld.instance")
        .unwrap()
        .build()
        .await
        .unwrap();
    let client = MprisClient::for_address(bus.address.clone());
    assert_eq!(client.discover().await.err(), Some(MediaError::Unavailable));
    let mut events = client.subscribe().await.unwrap();
    let endpoints = client.discover().await.unwrap();
    assert_eq!(endpoints.len(), 1);
    let endpoint = &endpoints[0];
    assert!(client.owner_matches(endpoint).await.unwrap());
    assert!(eligible_to_pause(&client.snapshot(endpoint).await.unwrap()));
    assert_eq!(client.pause(endpoint).await, MediaCallOutcome::Success);
    // A method success is not a pause confirmation: this fake deliberately no-ops.
    assert_eq!(
        client.snapshot(endpoint).await.unwrap().playback,
        MediaPlayback::Playing
    );
    assert_eq!(pauses.load(Ordering::SeqCst), 1);
    assert_eq!(plays.load(Ordering::SeqCst), 0);
    let invalidated = vec!["Metadata"];
    service
        .emit_signal(
            None::<&str>,
            "/org/mpris/MediaPlayer2",
            "org.freedesktop.DBus.Properties",
            "PropertiesChanged",
            &(
                "org.mpris.MediaPlayer2.Player",
                HashMap::<String, OwnedValue>::new(),
                invalidated,
            ),
        )
        .await
        .unwrap();
    // Fresh snapshots fence the signal stream: no extra scheduling grace is
    // needed before policy drains all preceding invalidations.
    client.snapshot(endpoint).await.unwrap();
    let event = events.try_recv().unwrap();
    assert!(
        matches!(event, MediaEvent::Properties { owner, invalidated, .. } if owner == endpoint.unique_owner && invalidated == ["Metadata"])
    );
    let metadata = HashMap::from([(
        "xesam:title".to_owned(),
        OwnedValue::from(Str::from("private test track")),
    )]);
    let batch = HashMap::from([
        (
            "PlaybackStatus".to_owned(),
            OwnedValue::from(Str::from("Paused")),
        ),
        ("Volume".to_owned(), OwnedValue::from(0.5_f64)),
        ("CanPlay".to_owned(), OwnedValue::from(true)),
        (
            "Metadata".to_owned(),
            OwnedValue::try_from(zbus::zvariant::Value::from(metadata)).unwrap(),
        ),
    ]);
    service
        .emit_signal(
            None::<&str>,
            "/org/mpris/MediaPlayer2",
            "org.freedesktop.DBus.Properties",
            "PropertiesChanged",
            &("org.mpris.MediaPlayer2.Player", batch, Vec::<String>::new()),
        )
        .await
        .unwrap();
    client.snapshot(endpoint).await.unwrap();
    assert!(
        matches!(events.try_recv().unwrap(), MediaEvent::Properties { playback: Some(MediaPlayback::Paused), fingerprint: Some(f), can_play: Some(true), .. } if f.title.as_deref() == Some("private test track"))
    );
    // Spoofing a player signal preserves the actual sender identity for policy filtering.
    synthetic
        .emit_signal(
            None::<&str>,
            "/org/mpris/MediaPlayer2",
            "org.mpris.MediaPlayer2.Player",
            "Seeked",
            &(1_i64,),
        )
        .await
        .unwrap();
    let event = tokio::time::timeout(Duration::from_secs(1), events.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(event, MediaEvent::Conflict { owner, .. } if owner != endpoint.unique_owner));
    service
        .release_name(endpoint.well_known_name.as_str())
        .await
        .unwrap();
    synthetic
        .request_name(endpoint.well_known_name.as_str())
        .await
        .unwrap();
    assert!(!client.owner_matches(endpoint).await.unwrap());
    let mut wrong_epoch = endpoint.clone();
    wrong_epoch.bus_epoch += 1;
    assert_eq!(
        client.snapshot(&wrong_epoch).await.err(),
        Some(MediaError::Lost)
    );
    assert_eq!(client.play(&wrong_epoch).await, MediaCallOutcome::Unknown);
    assert_eq!(plays.load(Ordering::SeqCst), 0);
    client.shutdown().await;
    let _new_subscription = client.subscribe().await.unwrap();
    assert_eq!(
        client.snapshot(endpoint).await.err(),
        Some(MediaError::Lost)
    );
    assert_eq!(client.play(endpoint).await, MediaCallOutcome::Unknown);
    assert!(
        client
            .discover()
            .await
            .unwrap()
            .iter()
            .all(|e| e.bus_epoch > endpoint.bus_epoch)
    );
    client.shutdown().await;
}
