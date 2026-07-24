//! Waybar JSON + tooltip formatting.

use std::collections::BTreeMap;

use regex::Regex;
use serde_json::{Value, json};
use std::sync::OnceLock;

fn class_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^a-z0-9_-]+").expect("static"))
}

/// Sanitize a CSS-ish class token.
#[must_use]
pub fn sanitize_class(value: &str) -> String {
    let cleaned = class_re()
        .replace_all(&value.to_ascii_lowercase(), "-")
        .trim_matches('-')
        .to_string();
    if cleaned.is_empty() {
        "unknown".into()
    } else {
        cleaned
    }
}

/// Minimal config projection for tooltip lines (avoids full core Config coupling).
#[derive(Debug, Clone, Default)]
pub struct WaybarConfigInfo {
    pub asr_backend: String,
    pub instant_mode: bool,
    pub model_label: Option<String>,
    pub device_label: Option<String>,
    pub tts_enabled: bool,
    pub tts_backend_label: Option<String>,
    pub tts_voice_label: Option<String>,
    pub tts_speed_label: Option<String>,
    pub overlay_debug_mode: bool,
}

/// Build tooltip info lines from config projection.
#[must_use]
pub fn config_info_lines(cfg: &WaybarConfigInfo) -> Vec<String> {
    let backend_label = match cfg.asr_backend.as_str() {
        "nemo" => "NeMo (NVIDIA)".into(),
        "sherpa" => "Sherpa-ONNX".into(),
        "moonshine" => "Moonshine-ONNX".into(),
        other => other.to_string(),
    };
    let mut lines = vec![format!("Backend:  {backend_label}")];
    if cfg.instant_mode {
        lines.push("Profile:  Instant".into());
    }
    if let Some(model) = &cfg.model_label {
        lines.push(format!("Model:    {model}"));
    }
    if let Some(device) = &cfg.device_label {
        lines.push(format!("Device:   {device}"));
    }
    if cfg.tts_enabled {
        lines.push(format!(
            "TTS:      {}",
            cfg.tts_backend_label.as_deref().unwrap_or("unknown")
        ));
        if let Some(voice) = &cfg.tts_voice_label {
            lines.push(format!("Voice:    {voice}"));
        }
        if let Some(speed) = &cfg.tts_speed_label {
            lines.push(format!("Speed:    {speed} (default synth)"));
        }
    } else {
        lines.push("TTS:      Disabled".into());
    }
    lines.push(format!(
        "Debug:    {}",
        if cfg.overlay_debug_mode {
            "Overlay on"
        } else {
            "Overlay off"
        }
    ));
    lines
}

/// Build the Waybar custom-module JSON object.
#[must_use]
pub fn build_waybar_payload(
    state: &str,
    config_lines: Option<&[String]>,
    service_state: Option<&str>,
    control_error: Option<&str>,
    action_error: Option<&str>,
) -> Value {
    let (mut base_state, mut reason) = match state.split_once(':') {
        Some((b, r)) => (b.to_string(), r.to_string()),
        None => (state.to_string(), String::new()),
    };

    let mut icons = BTreeMap::new();
    icons.insert("recording", "\u{f130}"); // 
    icons.insert("processing", "\u{f252}"); // 
    icons.insert("idle", "\u{f130}");
    icons.insert("starting", "\u{f252}");
    icons.insert("stopped", "\u{f131}"); // 
    icons.insert("error", "\u{f071}"); // 

    let mut labels = BTreeMap::new();
    labels.insert("recording", "Recording");
    labels.insert("processing", "Processing");
    labels.insert("idle", "Ready");
    labels.insert("starting", "Starting");
    labels.insert("stopped", "Stopped");
    labels.insert("error", "Error");

    if !labels.contains_key(base_state.as_str()) {
        if reason.is_empty() {
            reason = "unknown_state".into();
        }
        base_state = "error".into();
    }

    let mut lines = vec![format!(
        "ShuVoice: {}",
        labels.get(base_state.as_str()).copied().unwrap_or("Error")
    )];
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
    if let Some(cfg_lines) = config_lines {
        lines.push(String::new());
        lines.extend(cfg_lines.iter().cloned());
    }
    lines.push(String::new());
    lines.push("Left click: toggle recording".into());
    lines.push("Middle click: toggle service".into());
    lines.push("Right click: open action menu".into());

    let class_name = sanitize_class(&base_state);
    let text = icons
        .get(base_state.as_str())
        .copied()
        .unwrap_or("\u{f071}");

    json!({
        "text": text,
        "alt": base_state,
        "class": class_name,
        "tooltip": lines.join("\n"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_payload() {
        let p = build_waybar_payload("recording", None, None, None, None);
        assert_eq!(p["alt"], "recording");
        assert_eq!(p["class"], "recording");
        assert!(
            p["tooltip"]
                .as_str()
                .unwrap()
                .contains("ShuVoice: Recording")
        );
    }

    #[test]
    fn error_reason_sanitized_class() {
        let p = build_waybar_payload(
            "error:asr_thread_dead",
            None,
            Some("failed"),
            Some("Control socket not found"),
            Some("systemctl restart failed"),
        );
        assert_eq!(p["alt"], "error");
        assert_eq!(p["class"], "error");
        let tip = p["tooltip"].as_str().unwrap();
        assert!(tip.contains("Reason: asr_thread_dead"));
        assert!(tip.contains("Service: failed"));
        assert!(tip.contains("Control: Control socket not found"));
    }

    #[test]
    fn config_lines_include_debug() {
        let cfg = WaybarConfigInfo {
            asr_backend: "sherpa".into(),
            overlay_debug_mode: true,
            tts_enabled: false,
            ..Default::default()
        };
        let lines = config_info_lines(&cfg);
        assert!(lines.iter().any(|l| l.contains("Sherpa-ONNX")));
        assert!(lines.iter().any(|l| l.contains("Overlay on")));
        assert!(lines.iter().any(|l| l.contains("TTS:      Disabled")));
    }
}
