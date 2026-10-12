//! Ephemeral transport-independent media ownership values. Never log fingerprints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RecordingIntentId(pub u64);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MediaEndpoint {
    pub bus_epoch: u64,
    pub well_known_name: String,
    pub unique_owner: String,
    pub object_path: String,
}
impl MediaEndpoint {
    pub fn key(&self) -> (u64, String, String) {
        (
            self.bus_epoch,
            self.unique_owner.clone(),
            self.object_path.clone(),
        )
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaPlayback {
    Playing,
    Paused,
    Stopped,
    Unknown,
}
#[derive(Clone, Default, PartialEq, Eq)]
pub struct MediaFingerprint {
    pub track_id: Option<String>,
    pub url: Option<String>,
    pub title: Option<String>,
    pub artist: Vec<String>,
    pub album: Option<String>,
    pub length: Option<i64>,
}
impl MediaFingerprint {
    pub fn usable(&self) -> bool {
        self.track_id.as_ref().is_some_and(|id| {
            id.starts_with('/')
                && id != "/"
                && id != "/org/mpris/MediaPlayer2/TrackList/NoTrack"
                && id[1..].split('/').all(|part| {
                    !part.is_empty() && part.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                })
        }) || [&self.url, &self.title, &self.album]
            .into_iter()
            .any(|v| v.as_ref().is_some_and(|s| !s.trim().is_empty()))
            || self.artist.iter().any(|s| !s.trim().is_empty())
    }
}
#[derive(Clone)]
pub struct MediaSnapshot {
    pub playback: MediaPlayback,
    pub can_control: bool,
    pub can_pause: bool,
    pub can_play: bool,
    pub fingerprint: MediaFingerprint,
}
pub fn eligible_to_pause(s: &MediaSnapshot) -> bool {
    s.playback == MediaPlayback::Playing
        && s.can_control
        && s.can_pause
        && s.can_play
        && s.fingerprint.usable()
}
pub fn eligible_to_restore(
    s: &MediaSnapshot,
    baseline: &MediaFingerprint,
    entitled: bool,
    owner_matches: bool,
) -> bool {
    entitled
        && owner_matches
        && baseline.usable()
        && s.fingerprint == *baseline
        && s.playback == MediaPlayback::Paused
        && s.can_control
        && s.can_play
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaCallOutcome {
    Success,
    Rejected,
    Unknown,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaError {
    Unavailable,
    Invalid,
    Lost,
}
#[derive(Clone)]
pub enum MediaEvent {
    Properties {
        owner: String,
        path: String,
        playback: Option<MediaPlayback>,
        fingerprint: Option<MediaFingerprint>,
        can_control: Option<bool>,
        can_pause: Option<bool>,
        can_play: Option<bool>,
        invalidated: Vec<String>,
    },
    Paused {
        owner: String,
        path: String,
    },
    /// Some players also emit this when recalibrating position on Pause/Play.
    Seeked {
        owner: String,
        path: String,
    },
    Conflict {
        owner: String,
        path: String,
    },
    Lost,
}
