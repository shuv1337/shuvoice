//! Waybar JSON payload formatting helpers.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Sanitize a CSS class token.
pub fn sanitize_class(value: &str) -> String {
    let mut out = String::new();
    for ch in value.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else {
            out.push('-');
        }
    }
    let cleaned = out.trim_matches('-').to_string();
    if cleaned.is_empty() {
        "unknown".into()
    } else {
        // collapse consecutive dashes lightly
        let mut collapsed = String::new();
        let mut prev_dash = false;
        for ch in cleaned.chars() {
            if ch == '-' {
                if !prev_dash {
                    collapsed.push(ch);
                }
                prev_dash = true;
            } else {
                collapsed.push(ch);
                prev_dash = false;
            }
        }
        collapsed
    }
}

/// Build Waybar custom-module JSON object.
pub fn build_waybar_payload(
    state: &str,
    config_lines: Option<&[String]>,
    service_state: Option<&str>,
    control_error: Option<&str>,
    action_error: Option<&str>,
) -> Value {
    let (mut base_state, mut reason) = if let Some((b, r)) = state.split_once(':') {
        (b.to_string(), r.to_string())
    } else {
        (state.to_string(), String::new())
    };

    let known = [
        "recording",
        "processing",
        "idle",
        "starting",
        "stopped",
        "error",
    ];
    if !known.contains(&base_state.as_str()) {
        if reason.is_empty() {
            reason = "unknown_state".into();
        }
        base_state = "error".into();
    }

    let icon = match base_state.as_str() {
        "recording" => "",
        "processing" => "",
        "idle" => "",
        "starting" => "",
        "stopped" => "",
        _ => "",
    };
    let label = match base_state.as_str() {
        "recording" => "Recording",
        "processing" => "Processing",
        "idle" => "Ready",
        "starting" => "Starting",
        "stopped" => "Stopped",
        _ => "Error",
    };

    let mut lines = vec![format!("ShuVoice: {label}")];
    if !reason.is_empty() {
        lines.push(format!("Reason: {reason}"));
    }
    if let Some(ss) = service_state
        && ss != "unknown"
    {
        lines.push(format!("Service: {ss}"));
    }
    if let Some(ce) = control_error
        && (base_state == "starting" || base_state == "error")
    {
        lines.push(format!("Control: {ce}"));
    }
    if let Some(ae) = action_error {
        lines.push(format!("Action: {ae}"));
    }
    if let Some(cfg) = config_lines
        && !cfg.is_empty()
    {
        lines.push(String::new());
        lines.extend(cfg.iter().cloned());
    }
    lines.push(String::new());
    lines.push("Left click: toggle recording".into());
    lines.push("Middle click: toggle service".into());
    lines.push("Right click: open action menu".into());

    json!({
        "text": icon,
        "alt": base_state,
        "class": sanitize_class(&base_state),
        "tooltip": lines.join("\n"),
    })
}

/// Tooltip config lines from a simplified config view.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WaybarConfigInfo {
    pub asr_backend: String,
    pub instant_mode: bool,
    pub model_label: Option<String>,
    pub device_label: String,
    pub tts_enabled: bool,
    pub tts_backend_label: Option<String>,
    pub tts_voice_label: Option<String>,
    pub tts_speed_label: Option<String>,
    pub overlay_debug_mode: bool,
}

pub fn config_info_lines(info: &WaybarConfigInfo) -> Vec<String> {
    let backend_label = match info.asr_backend.as_str() {
        "nemo" => "NeMo (NVIDIA)".into(),
        "sherpa" => "Sherpa-ONNX".into(),
        "moonshine" => "Moonshine-ONNX".into(),
        other => other.to_string(),
    };
    let mut lines = vec![format!("Backend:  {backend_label}")];
    if info.instant_mode {
        lines.push("Profile:  Instant".into());
    }
    if let Some(model) = &info.model_label {
        lines.push(format!("Model:    {model}"));
    }
    lines.push(format!("Device:   {}", info.device_label));
    if info.tts_enabled {
        lines.push(format!(
            "TTS:      {}",
            info.tts_backend_label.as_deref().unwrap_or("unknown")
        ));
        lines.push(format!(
            "Voice:    {}",
            info.tts_voice_label.as_deref().unwrap_or("default")
        ));
        if let Some(speed) = &info.tts_speed_label {
            lines.push(format!("Speed:    {speed} (default synth)"));
        }
    } else {
        lines.push("TTS:      Disabled".into());
    }
    lines.push(format!(
        "Debug:    {}",
        if info.overlay_debug_mode {
            "Overlay on"
        } else {
            "Overlay off"
        }
    ));
    lines
}

/// Human labels for TTS backends (legacy waybar map + passthrough).
pub fn tts_backend_label(backend: &str) -> String {
    match backend.trim().to_ascii_lowercase().as_str() {
        "elevenlabs" => "ElevenLabs".into(),
        "openai" => "OpenAI".into(),
        "local" => "Local Piper".into(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_payload_idle() {
        let v = build_waybar_payload("idle", None, None, None, None);
        assert_eq!(v["alt"], "idle");
        assert_eq!(v["class"], "idle");
        assert_eq!(v["text"], "");
        let tip = v["tooltip"].as_str().unwrap();
        assert!(tip.contains("ShuVoice: Ready"));
        assert!(tip.contains("Left click: toggle recording"));
    }

    #[test]
    fn build_payload_error_reason_sanitized() {
        let v = build_waybar_payload("error:service_failed", None, Some("failed"), None, None);
        assert_eq!(v["alt"], "error");
        assert_eq!(v["class"], "error");
        let tip = v["tooltip"].as_str().unwrap();
        assert!(tip.contains("Reason: service_failed"));
        assert!(tip.contains("Service: failed"));
    }

    #[test]
    fn unknown_state_becomes_error() {
        let v = build_waybar_payload("weird", None, None, None, None);
        assert_eq!(v["alt"], "error");
        assert!(v["tooltip"].as_str().unwrap().contains("unknown_state"));
    }

    #[test]
    fn config_info_includes_debug() {
        let lines = config_info_lines(&WaybarConfigInfo {
            asr_backend: "sherpa".into(),
            instant_mode: true,
            model_label: Some("parakeet".into()),
            device_label: "CPU".into(),
            tts_enabled: false,
            tts_backend_label: None,
            tts_voice_label: None,
            tts_speed_label: None,
            overlay_debug_mode: true,
        });
        assert!(lines.iter().any(|l| l.contains("Sherpa-ONNX")));
        assert!(lines.iter().any(|l| l.contains("Instant")));
        assert!(lines.iter().any(|l| l.contains("Overlay on")));
        assert!(lines.iter().any(|l| l.contains("Disabled")));
    }

    #[test]
    fn sanitize_class_strips_junk() {
        assert_eq!(sanitize_class("Error:X Y"), "error-x-y");
    }
}
