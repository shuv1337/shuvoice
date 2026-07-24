//! Hyprland control-command matching and keybind detection.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::process::{CommandRunner, RunOptions, StdCommandRunner, argv};
use std::sync::Arc;

/// Control-command argv patterns used by Hyprland binds.
pub fn control_command_patterns(command: &str) -> &'static [&'static str] {
    match command {
        "start" => &["--control start", " control start"],
        "tts_speak" => &["--control tts_speak", " control tts_speak"],
        "tts_speak_clipboard" => &[
            "--control tts_speak_clipboard",
            " control tts_speak_clipboard",
        ],
        _ => &[],
    }
}

fn description_patterns(command: &str) -> &'static [&'static str] {
    match command {
        "start" => &["shuvoice start"],
        "tts_speak" => &["shuvoice tts speak", "shuvoice tts_speak"],
        "tts_speak_clipboard" => &[
            "shuvoice tts speak clipboard",
            "shuvoice tts_speak_clipboard",
        ],
        _ => &[],
    }
}

const TRACKED_COMMANDS: &[&str] = &["start", "tts_speak", "tts_speak_clipboard"];

fn pattern_has_command_boundary(text_lc: &str, pattern: &str) -> bool {
    let mut start = 0;
    while let Some(rel) = text_lc[start..].find(pattern) {
        let idx = start + rel;
        let end = idx + pattern.len();
        if end >= text_lc.len() {
            return true;
        }
        let next = text_lc.as_bytes()[end] as char;
        if next == ' ' || next == '\t' || text_lc[end..].starts_with("--") {
            return true;
        }
        start = end;
    }
    false
}

fn pattern_has_description_boundary(text_lc: &str, pattern: &str) -> bool {
    let Some(idx) = text_lc.find(pattern) else {
        return false;
    };
    let end = idx + pattern.len();
    if end >= text_lc.len() {
        return true;
    }
    let next = text_lc[end..].chars().next().unwrap_or('\0');
    !(next.is_ascii_alphanumeric() || next == '_')
}

/// Return true when `token` appears as a complete ShuVoice control subcommand.
#[must_use]
pub fn matches_control_command_token(command_lc: &str, token: &str) -> bool {
    for prefix in ["--control ", " control "] {
        let pattern = format!("{prefix}{token}");
        if pattern_has_command_boundary(command_lc, &pattern) {
            return true;
        }
    }
    false
}

/// Match a Hyprland bind `arg` against a ShuVoice control command.
#[must_use]
pub fn matches_shuvoice_command(arg: &str, command: &str) -> bool {
    let arg_lc = arg.to_ascii_lowercase();
    if !arg_lc.contains("shuvoice") {
        return false;
    }
    control_command_patterns(command)
        .iter()
        .any(|pattern| pattern_has_command_boundary(&arg_lc, pattern))
}

/// Match a Hyprland bind description against a ShuVoice control command.
#[must_use]
pub fn matches_shuvoice_description(description: &str, command: &str) -> bool {
    let desc_lc = description.to_ascii_lowercase();
    if !desc_lc.contains("shuvoice") {
        return false;
    }

    let own = description_patterns(command);
    let mut best_own = "";
    for pattern in own {
        if pattern_has_description_boundary(&desc_lc, pattern) && pattern.len() > best_own.len() {
            best_own = pattern;
        }
    }
    if best_own.is_empty() {
        return false;
    }

    for other_command in TRACKED_COMMANDS {
        if *other_command == command {
            continue;
        }
        for other_pattern in description_patterns(other_command) {
            if other_pattern.len() <= best_own.len() {
                continue;
            }
            if pattern_has_description_boundary(&desc_lc, other_pattern) {
                return false;
            }
        }
    }
    true
}

fn format_bind(bind: &Value) -> Option<String> {
    let key = bind.get("key")?.as_str()?.trim();
    if key.is_empty() {
        return None;
    }
    let modmask = bind.get("modmask").and_then(Value::as_i64).unwrap_or(0);
    let mut mods = Vec::new();
    if modmask & 64 != 0 {
        mods.push("Super");
    }
    if modmask & 4 != 0 {
        mods.push("Ctrl");
    }
    if modmask & 8 != 0 {
        mods.push("Alt");
    }
    if modmask & 1 != 0 {
        mods.push("Shift");
    }
    if mods.is_empty() {
        Some(key.to_string())
    } else {
        Some(format!("{} + {key}", mods.join(" + ")))
    }
}

struct BindCache {
    at: Instant,
    value: HashMap<String, Option<String>>,
}

fn bind_cache() -> &'static Mutex<Option<BindCache>> {
    static CACHE: OnceLock<Mutex<Option<BindCache>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

/// Clear cached keybind detection results.
pub fn clear_keybind_cache() {
    *bind_cache().lock().unwrap() = None;
}

fn detect_keybinds_uncached(runner: &dyn CommandRunner) -> HashMap<String, Option<String>> {
    let mut detected: HashMap<String, Option<String>> = TRACKED_COMMANDS
        .iter()
        .map(|c| ((*c).to_string(), None))
        .collect();

    let args = argv(["hyprctl", "binds", "-j"]);
    let opts = RunOptions {
        timeout: Duration::from_secs(1),
        check: false,
        ..RunOptions::default()
    };
    let Ok(out) = runner.run(&args, &opts) else {
        return detected;
    };
    if !out.success {
        return detected;
    }
    let Ok(binds) = serde_json::from_slice::<Vec<Value>>(&out.stdout) else {
        return detected;
    };

    for bind in binds {
        let arg = bind.get("arg").and_then(Value::as_str).unwrap_or("");
        let description = bind
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("");
        let is_release = bind
            .get("release")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        for command in TRACKED_COMMANDS {
            if detected.get(*command).and_then(|v| v.as_ref()).is_some() {
                continue;
            }
            let matched = matches_shuvoice_command(arg, command)
                || matches_shuvoice_description(description, command);
            if !matched {
                continue;
            }
            if is_release {
                continue;
            }
            if let Some(formatted) = format_bind(&bind) {
                detected.insert((*command).to_string(), Some(formatted));
            }
        }
        if detected.values().all(|v| v.is_some()) {
            break;
        }
    }
    detected
}

/// Detect active ShuVoice Hyprland keybinds with short-lived caching.
pub fn detect_keybinds(
    runner: Option<Arc<dyn CommandRunner>>,
    ttl: Duration,
) -> HashMap<String, Option<String>> {
    let mut guard = bind_cache().lock().unwrap();
    let now = Instant::now();
    if let Some(cache) = guard.as_ref()
        && now.duration_since(cache.at) <= ttl
    {
        return cache.value.clone();
    }
    let runner = runner.unwrap_or_else(|| Arc::new(StdCommandRunner));
    let value = detect_keybinds_uncached(runner.as_ref());
    *guard = Some(BindCache {
        at: now,
        value: value.clone(),
    });
    value
}

/// Detect a specific ShuVoice keybind.
pub fn detect_keybind(
    command: &str,
    runner: Option<Arc<dyn CommandRunner>>,
    ttl: Duration,
) -> Option<String> {
    detect_keybinds(runner, ttl).get(command).cloned().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::ScriptedRunner;
    use std::sync::Arc;

    #[test]
    fn token_distinguishes_clipboard_variant() {
        let clipboard = "shuvoice control tts_speak_clipboard";
        let selection = "shuvoice control tts_speak";
        assert!(matches_control_command_token(
            clipboard,
            "tts_speak_clipboard"
        ));
        assert!(!matches_control_command_token(clipboard, "tts_speak"));
        assert!(matches_control_command_token(selection, "tts_speak"));
        assert!(!matches_control_command_token(
            selection,
            "tts_speak_clipboard"
        ));
    }

    #[test]
    fn description_distinguishes_clipboard() {
        let clipboard = "ShuVoice TTS speak clipboard";
        let selection = "ShuVoice TTS speak";
        assert!(matches_shuvoice_description(
            clipboard,
            "tts_speak_clipboard"
        ));
        assert!(!matches_shuvoice_description(clipboard, "tts_speak"));
        assert!(matches_shuvoice_description(selection, "tts_speak"));
        assert!(!matches_shuvoice_description(
            selection,
            "tts_speak_clipboard"
        ));
    }

    #[test]
    fn legacy_control_flag_style() {
        let arg = "shuvoice --control tts_speak";
        assert!(matches_shuvoice_command(arg, "tts_speak"));
        assert!(!matches_shuvoice_command(arg, "tts_speak_clipboard"));
    }

    #[test]
    fn description_prefers_longer_command() {
        let clipboard = "ShuVoice tts_speak_clipboard hotkey";
        assert!(matches_shuvoice_description(
            clipboard,
            "tts_speak_clipboard"
        ));
        assert!(!matches_shuvoice_description(clipboard, "tts_speak"));
    }

    #[test]
    fn detect_keybind_skips_release_and_caches() {
        clear_keybind_cache();
        let r = ScriptedRunner::new();
        let payload = r#"[
          {"key":"V","modmask":64,"arg":"shuvoice --control start","release":true,"description":"ShuVoice start"},
          {"key":"V","modmask":64,"arg":"shuvoice --control start","release":false,"description":"ShuVoice start"},
          {"key":"S","modmask":68,"arg":"shuvoice --control tts_speak","release":false,"description":"ShuVoice tts speak"}
        ]"#;
        r.push_ok(payload.as_bytes());
        let runner: Arc<dyn CommandRunner> = Arc::new(r.clone());
        let start = detect_keybind("start", Some(Arc::clone(&runner)), Duration::from_secs(10));
        let tts = detect_keybind("tts_speak", Some(runner), Duration::from_secs(10));
        assert_eq!(start.as_deref(), Some("Super + V"));
        assert_eq!(tts.as_deref(), Some("Super + Ctrl + S"));
        assert_eq!(r.calls().len(), 1);
        clear_keybind_cache();
    }

    #[test]
    fn detect_does_not_collide_tts_variants() {
        clear_keybind_cache();
        let r = ScriptedRunner::new();
        let payload = r#"[
          {"key":"S","modmask":68,"arg":"shuvoice control tts_speak","release":false},
          {"key":"S","modmask":69,"arg":"shuvoice control tts_speak_clipboard","release":false}
        ]"#;
        r.push_ok(payload.as_bytes());
        let map = detect_keybinds(Some(Arc::new(r)), Duration::from_secs(10));
        assert_eq!(map["tts_speak"].as_deref(), Some("Super + Ctrl + S"));
        assert_eq!(
            map["tts_speak_clipboard"].as_deref(),
            Some("Super + Ctrl + Shift + S")
        );
        clear_keybind_cache();
    }
}
