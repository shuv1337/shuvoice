//! Lazy native MPRIS transport. Tests must supply a private bus address.
use futures_util::StreamExt;
use shuvoice_core::media::*;
use std::collections::{HashMap, HashSet};
use tokio::sync::{Mutex, broadcast, watch};
use zbus::{Connection, MatchRule, MessageStream, message::Type, zvariant::OwnedValue};

const PLAYER: &str = "org.mpris.MediaPlayer2.Player";
const PATH: &str = "/org/mpris/MediaPlayer2";
struct Bus {
    connection: Connection,
    epoch: u64,
    events: broadcast::Sender<MediaEvent>,
    monitor: tokio::task::JoinHandle<()>,
    fence: watch::Receiver<u64>,
    next_fence: std::sync::atomic::AtomicU64,
}
impl Drop for Bus {
    fn drop(&mut self) {
        self.monitor.abort();
    }
}
#[derive(Default)]
pub struct MprisClient {
    address: Option<String>,
    bus: Mutex<Option<Bus>>,
    epoch: std::sync::atomic::AtomicU64,
}
impl MprisClient {
    pub fn new() -> Self {
        Self::default()
    }
    pub async fn shutdown(&self) {
        if let Some(mut bus) = self.bus.lock().await.take() {
            bus.monitor.abort();
            let _ = (&mut bus.monitor).await;
        }
    }
    /// Explicit-address constructor: never falls back to the user's session bus.
    pub fn for_address(address: String) -> Self {
        Self {
            address: Some(address),
            ..Self::default()
        }
    }
    pub async fn subscribe(&self) -> Result<broadcast::Receiver<MediaEvent>, MediaError> {
        let mut bus = self.bus.lock().await;
        if bus.as_ref().is_some_and(|b| b.monitor.is_finished()) {
            *bus = None;
        }
        if bus.is_none() {
            let connection = match &self.address {
                Some(address) => {
                    zbus::connection::Builder::address(address.as_str())
                        .map_err(|_| MediaError::Unavailable)?
                        .build()
                        .await
                }
                None => Connection::session().await,
            }
            .map_err(|_| MediaError::Unavailable)?;
            let rule = MatchRule::builder().msg_type(Type::Signal).build();
            let mut stream = MessageStream::for_match_rule(rule, &connection, Some(256))
                .await
                .map_err(|_| MediaError::Unavailable)?;
            let (events, _) = broadcast::channel(256);
            let (fence_tx, fence) = watch::channel(0_u64);
            let own_name = connection
                .unique_name()
                .ok_or(MediaError::Unavailable)?
                .to_string();
            let tx = events.clone();
            let monitor = tokio::spawn(async move {
                while let Some(message) = stream.next().await {
                    let Ok(message) = message else {
                        break;
                    };
                    let h = message.header();
                    let owner = h.sender().map(ToString::to_string).unwrap_or_default();
                    let path = h.path().map(ToString::to_string).unwrap_or_default();
                    let interface = h.interface().map(|v| v.as_str());
                    let member = h.member().map(|v| v.as_str());
                    // A unicast marker on the same ordered incoming stream
                    // fences every signal preceding our fresh read's barrier.
                    if owner == own_name
                        && path == "/org/shuvoice/MediaBarrier"
                        && interface == Some("org.shuvoice.MediaBarrier")
                        && member == Some("Fence")
                    {
                        if let Ok((id,)) = message.body().deserialize::<(u64,)>() {
                            fence_tx.send_modify(|value| *value = (*value).max(id));
                        }
                        continue;
                    }
                    let event = if interface == Some("org.freedesktop.DBus")
                        && member == Some("NameOwnerChanged")
                        && owner == "org.freedesktop.DBus"
                    {
                        if let Ok((name, old, _new)) =
                            message.body().deserialize::<(String, String, String)>()
                        {
                            if name.starts_with("org.mpris.MediaPlayer2.") && !old.is_empty() {
                                Some(MediaEvent::Conflict {
                                    owner: old,
                                    path: PATH.into(),
                                })
                            } else {
                                None
                            }
                        } else {
                            Some(MediaEvent::Lost)
                        }
                    } else if path == PATH && interface == Some(PLAYER) && member == Some("Seeked")
                    {
                        Some(MediaEvent::Conflict { owner, path })
                    } else if path == PATH
                        && interface == Some("org.freedesktop.DBus.Properties")
                        && member == Some("PropertiesChanged")
                    {
                        match message
                            .body()
                            .deserialize::<(String, HashMap<String, OwnedValue>, Vec<String>)>()
                        {
                            Ok((interface, changed, invalidated)) if interface == PLAYER => {
                                let boolean = |name: &str| {
                                    changed
                                        .get(name)
                                        .map(|v| bool::try_from(v).unwrap_or(false))
                                };
                                Some(MediaEvent::Properties {
                                    owner,
                                    path,
                                    playback: changed.get("PlaybackStatus").map(playback),
                                    fingerprint: changed
                                        .contains_key("Metadata")
                                        .then(|| fingerprint(&changed)),
                                    can_control: boolean("CanControl"),
                                    can_pause: boolean("CanPause"),
                                    can_play: boolean("CanPlay"),
                                    invalidated,
                                })
                            }
                            Ok(_) => None,
                            Err(_) => Some(MediaEvent::Lost),
                        }
                    } else {
                        None
                    };
                    if let Some(event) = event {
                        let _ = tx.send(event);
                    }
                }
                let _ = tx.send(MediaEvent::Lost);
            });
            let epoch = self
                .epoch
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            *bus = Some(Bus {
                connection,
                epoch,
                events,
                monitor,
                fence,
                next_fence: std::sync::atomic::AtomicU64::new(1),
            });
        }
        Ok(bus.as_ref().unwrap().events.subscribe())
    }
    async fn fence(&self, epoch: u64) -> Result<(), MediaError> {
        let (connection, id, mut receiver) = {
            let bus = self.bus.lock().await;
            let b = bus
                .as_ref()
                .filter(|b| b.epoch == epoch && !b.monitor.is_finished())
                .ok_or(MediaError::Lost)?;
            (
                b.connection.clone(),
                b.next_fence
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                b.fence.clone(),
            )
        };
        connection
            .emit_signal(
                connection.unique_name(),
                "/org/shuvoice/MediaBarrier",
                "org.shuvoice.MediaBarrier",
                "Fence",
                &(id,),
            )
            .await
            .map_err(|_| MediaError::Lost)?;
        receiver
            .wait_for(|seen| *seen >= id)
            .await
            .map_err(|_| MediaError::Lost)?;
        Ok(())
    }
    async fn connection(&self, epoch: Option<u64>) -> Result<(Connection, u64), MediaError> {
        let bus = self.bus.lock().await;
        let b = bus.as_ref().ok_or(MediaError::Unavailable)?;
        if b.monitor.is_finished() || epoch.is_some_and(|e| e != b.epoch) {
            return Err(MediaError::Lost);
        }
        Ok((b.connection.clone(), b.epoch))
    }
    pub async fn discover(&self) -> Result<Vec<MediaEndpoint>, MediaError> {
        let (connection, epoch) = self.connection(None).await?;
        let dbus = zbus::fdo::DBusProxy::new(&connection)
            .await
            .map_err(|_| MediaError::Unavailable)?;
        let names = dbus
            .list_names()
            .await
            .map_err(|_| MediaError::Unavailable)?;
        let mut seen = HashSet::new();
        let mut endpoints = Vec::new();
        for name in names {
            let s = name.as_str();
            if !s.starts_with("org.mpris.MediaPlayer2.")
                || s.strip_prefix("org.mpris.MediaPlayer2.")
                    .is_some_and(|s| s == "playerctld" || s.starts_with("playerctld."))
            {
                continue;
            }
            let Ok(owner) = dbus.get_name_owner(name.as_ref()).await else {
                continue;
            };
            if seen.insert(owner.to_string()) {
                endpoints.push(MediaEndpoint {
                    bus_epoch: epoch,
                    well_known_name: s.into(),
                    unique_owner: owner.to_string(),
                    object_path: PATH.into(),
                });
            }
        }
        Ok(endpoints)
    }
    pub async fn owner_matches(&self, e: &MediaEndpoint) -> Result<bool, MediaError> {
        let (c, _) = self.connection(Some(e.bus_epoch)).await?;
        let p = zbus::fdo::DBusProxy::new(&c)
            .await
            .map_err(|_| MediaError::Unavailable)?;
        let name = e
            .well_known_name
            .as_str()
            .try_into()
            .map_err(|_| MediaError::Invalid)?;
        Ok(p.get_name_owner(name)
            .await
            .is_ok_and(|o| o.as_str() == e.unique_owner))
    }
    pub async fn snapshot(&self, e: &MediaEndpoint) -> Result<MediaSnapshot, MediaError> {
        let (c, _) = self.connection(Some(e.bus_epoch)).await?;
        let reply = c
            .call_method(
                Some(e.unique_owner.as_str()),
                e.object_path.as_str(),
                Some("org.freedesktop.DBus.Properties"),
                "GetAll",
                &(PLAYER,),
            )
            .await
            .map_err(|_| MediaError::Unavailable)?;
        let p: HashMap<String, OwnedValue> = reply
            .body()
            .deserialize()
            .map_err(|_| MediaError::Invalid)?;
        self.fence(e.bus_epoch).await?;
        let boolean = |name: &str| {
            p.get(name)
                .and_then(|v| bool::try_from(v).ok())
                .unwrap_or(false)
        };
        let playback = match p
            .get("PlaybackStatus")
            .and_then(|v| <&str>::try_from(v).ok())
        {
            Some("Playing") => MediaPlayback::Playing,
            Some("Paused") => MediaPlayback::Paused,
            Some("Stopped") => MediaPlayback::Stopped,
            _ => MediaPlayback::Unknown,
        };
        Ok(MediaSnapshot {
            playback,
            can_control: boolean("CanControl"),
            can_pause: boolean("CanPause"),
            can_play: boolean("CanPlay"),
            fingerprint: fingerprint(&p),
        })
    }
    async fn effect(&self, e: &MediaEndpoint, method: &str) -> MediaCallOutcome {
        let Ok((c, _)) = self.connection(Some(e.bus_epoch)).await else {
            return MediaCallOutcome::Unknown;
        };
        match c
            .call_method(
                Some(e.unique_owner.as_str()),
                e.object_path.as_str(),
                Some(PLAYER),
                method,
                &(),
            )
            .await
        {
            Ok(_) => MediaCallOutcome::Success,
            Err(zbus::Error::MethodError(..)) => MediaCallOutcome::Rejected,
            Err(_) => MediaCallOutcome::Unknown,
        }
    }
    pub async fn pause(&self, e: &MediaEndpoint) -> MediaCallOutcome {
        self.effect(e, "Pause").await
    }
    pub async fn play(&self, e: &MediaEndpoint) -> MediaCallOutcome {
        self.effect(e, "Play").await
    }
}

fn playback(value: &OwnedValue) -> MediaPlayback {
    match <&str>::try_from(value).ok() {
        Some("Playing") => MediaPlayback::Playing,
        Some("Paused") => MediaPlayback::Paused,
        Some("Stopped") => MediaPlayback::Stopped,
        _ => MediaPlayback::Unknown,
    }
}

fn fingerprint(p: &HashMap<String, OwnedValue>) -> MediaFingerprint {
    let metadata = p
        .get("Metadata")
        .and_then(|v| v.try_clone().ok())
        .and_then(|v| HashMap::<String, OwnedValue>::try_from(v).ok())
        .unwrap_or_default();
    let text = |key: &str| {
        metadata
            .get(key)
            .and_then(|v| <&str>::try_from(v).ok())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    };
    let track_id = metadata
        .get("mpris:trackid")
        .and_then(|v| <&zbus::zvariant::ObjectPath>::try_from(v).ok())
        .map(ToString::to_string);
    let artist = metadata
        .get("xesam:artist")
        .and_then(|v| v.try_clone().ok())
        .and_then(|v| Vec::<String>::try_from(v).ok())
        .unwrap_or_default()
        .into_iter()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect();
    MediaFingerprint {
        track_id,
        url: text("xesam:url"),
        title: text("xesam:title"),
        album: text("xesam:album"),
        artist,
        length: metadata
            .get("mpris:length")
            .and_then(|v| i64::try_from(v).ok())
            .filter(|n| *n >= 0),
    }
}
