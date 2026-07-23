//! Pure selection/visibility helpers for the GTK wizard (no GTK types).

use crate::wizard::{
    ASR_BACKENDS, KEYBIND_PRESETS, SHERPA_PROFILE_OPTIONS, TTS_BACKENDS,
    TTS_PLAYBACK_SPEED_PRESET_IDS, WizardVm, tts_playback_speed_preset_id,
};

/// Which TTS sub-controls should be visible for a backend + local setup mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TtsControlVisibility {
    pub voice_entry: bool,
    pub kokoro_url: bool,
    pub local_setup_mode: bool,
    pub local_path: bool,
    pub local_auto_voice: bool,
    pub melotts_device: bool,
}

impl TtsControlVisibility {
    pub fn for_backend(backend: &str, local_setup_mode: &str) -> Self {
        let b = backend.trim().to_ascii_lowercase();
        let auto = local_setup_mode.trim().eq_ignore_ascii_case("automatic");
        match b.as_str() {
            "kokoro" => Self {
                voice_entry: true,
                kokoro_url: true,
                local_setup_mode: false,
                local_path: false,
                local_auto_voice: false,
                melotts_device: false,
            },
            "local" => Self {
                voice_entry: !auto,
                kokoro_url: false,
                local_setup_mode: true,
                local_path: !auto,
                local_auto_voice: auto,
                melotts_device: false,
            },
            "melotts" => Self {
                voice_entry: true,
                kokoro_url: false,
                local_setup_mode: false,
                local_path: false,
                local_auto_voice: false,
                melotts_device: true,
            },
            // elevenlabs / openai / unknown: voice id entry only
            _ => Self {
                voice_entry: true,
                kokoro_url: false,
                local_setup_mode: false,
                local_path: false,
                local_auto_voice: false,
                melotts_device: false,
            },
        }
    }
}

#[must_use]
pub fn sherpa_controls_visible(asr_backend: &str) -> bool {
    asr_backend.trim().eq_ignore_ascii_case("sherpa")
}

#[must_use]
pub fn keybind_auto_add_sensitive(keybind_id: &str) -> bool {
    KEYBIND_PRESETS
        .iter()
        .find(|p| p.id == keybind_id)
        .and_then(|p| p.hypr_key_spec)
        .is_some()
}

#[must_use]
pub fn speed_preset_index(speed: f64) -> u32 {
    let id = tts_playback_speed_preset_id(speed);
    TTS_PLAYBACK_SPEED_PRESET_IDS
        .iter()
        .position(|p| *p == id)
        .unwrap_or(2) as u32 // 1.25 default is index 2
}

#[must_use]
pub fn asr_backend_index(backend: &str) -> u32 {
    ASR_BACKENDS
        .iter()
        .position(|(id, ..)| *id == backend)
        .unwrap_or(0) as u32
}

#[must_use]
pub fn tts_backend_index(backend: &str) -> u32 {
    TTS_BACKENDS
        .iter()
        .position(|(id, ..)| *id == backend)
        .unwrap_or(0) as u32
}

#[must_use]
pub fn sherpa_profile_index(profile_id: &str) -> u32 {
    SHERPA_PROFILE_OPTIONS
        .iter()
        .position(|(id, ..)| *id == profile_id)
        .unwrap_or(1) as u32 // instant is recommended default
}

#[must_use]
pub fn keybind_index(keybind_id: &str) -> u32 {
    KEYBIND_PRESETS
        .iter()
        .position(|p| p.id == keybind_id)
        .unwrap_or(0) as u32
}

/// After a failed finish, the Done page should allow Back/retry.
#[must_use]
pub fn should_show_done_back(finish_in_progress: bool, show_launch: bool) -> bool {
    !finish_in_progress && !show_launch
}

/// Initial VM-driven visibility snapshot for tests.
#[must_use]
pub fn tts_visibility_from_vm(vm: &WizardVm) -> TtsControlVisibility {
    TtsControlVisibility::for_backend(&vm.tts_backend, &vm.tts_local_setup_mode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wizard::DEFAULT_TTS_PLAYBACK_SPEED;

    #[test]
    fn speed_default_is_1_25_index() {
        assert_eq!(speed_preset_index(DEFAULT_TTS_PLAYBACK_SPEED), 2);
        assert_eq!(speed_preset_index(1.25), 2);
        assert_eq!(TTS_PLAYBACK_SPEED_PRESET_IDS[2], "1.25");
    }

    #[test]
    fn sherpa_visibility() {
        assert!(sherpa_controls_visible("sherpa"));
        assert!(!sherpa_controls_visible("nemo"));
        assert!(!sherpa_controls_visible("moonshine"));
    }

    #[test]
    fn custom_keybind_disables_auto_add() {
        assert!(keybind_auto_add_sensitive("right_ctrl"));
        assert!(!keybind_auto_add_sensitive("custom"));
    }

    #[test]
    fn tts_visibility_kokoro_local_melotts() {
        let k = TtsControlVisibility::for_backend("kokoro", "automatic");
        assert!(k.kokoro_url && k.voice_entry && !k.local_path);

        let local_auto = TtsControlVisibility::for_backend("local", "automatic");
        assert!(local_auto.local_setup_mode && local_auto.local_auto_voice);
        assert!(!local_auto.local_path && !local_auto.voice_entry);

        let local_manual = TtsControlVisibility::for_backend("local", "manual");
        assert!(local_manual.local_path && local_manual.voice_entry);
        assert!(!local_manual.local_auto_voice);

        let melo = TtsControlVisibility::for_backend("melotts", "automatic");
        assert!(melo.melotts_device && melo.voice_entry);
    }

    #[test]
    fn done_back_visibility() {
        assert!(should_show_done_back(false, false));
        assert!(!should_show_done_back(true, false));
        assert!(!should_show_done_back(false, true));
    }

    #[test]
    fn default_vm_indices() {
        let vm = WizardVm::new(false);
        assert_eq!(asr_backend_index(&vm.asr_backend), 0); // sherpa first
        assert_eq!(speed_preset_index(vm.tts_playback_speed), 2);
        assert!(sherpa_controls_visible(&vm.asr_backend));
        assert!(keybind_auto_add_sensitive(&vm.keybind));
    }
}
