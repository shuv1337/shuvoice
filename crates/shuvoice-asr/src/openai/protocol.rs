//! Pure protocol state machine for OpenAI Realtime transcription.
//!
//! Strict event handling for the OpenAI Realtime transcription protocol.

use std::collections::HashMap;

use serde_json::{Value, json};

pub const OPENAI_REALTIME_SAMPLE_RATE: u32 = 24_000;
pub const OPENAI_REALTIME_WS_URL_DEFAULT: &str =
    "wss://api.openai.com/v1/realtime?intent=transcription";

/// Build `transcription_session.update` payload.
pub fn session_update_payload(model: &str, language: &str) -> Value {
    let mut transcription = json!({ "model": model });
    if !language.trim().is_empty() {
        transcription["language"] = json!(language);
    }
    json!({
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "input_audio_transcription": transcription,
            "turn_detection": Value::Null,
            "input_audio_noise_reduction": { "type": "near_field" },
        }
    })
}

pub fn append_audio_payload(pcm16_b64: &str) -> Value {
    json!({
        "type": "input_audio_buffer.append",
        "audio": pcm16_b64,
    })
}

pub fn commit_payload() -> Value {
    json!({ "type": "input_audio_buffer.commit" })
}

pub fn clear_input_buffer_payload() -> Value {
    json!({ "type": "input_audio_buffer.clear" })
}

/// Redact secrets from OpenAI error payloads before storing/logging.
pub fn redact_openai_error(event: &Value) -> String {
    let mut cloned = event.clone();
    if let Some(obj) = cloned.as_object_mut() {
        for key in ["api_key", "authorization", "token", "secret"] {
            if obj.contains_key(key) {
                obj.insert(key.to_string(), json!("[redacted]"));
            }
        }
        if let Some(err) = obj.get_mut("error").and_then(|e| e.as_object_mut()) {
            for key in ["message", "code", "type"] {
                let _ = key;
            }
            // Keep message but strip bearer-looking substrings.
            if let Some(msg) = err.get("message").and_then(|m| m.as_str()) {
                let redacted = msg
                    .split_whitespace()
                    .map(|w| {
                        if w.starts_with("sk-") || w.len() > 40 {
                            "[redacted]"
                        } else {
                            w
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                err.insert("message".into(), json!(redacted));
            }
        }
    }
    cloned.to_string()
}

/// Tracks partial/final text by item id (Python backend fields).
#[derive(Debug, Default, Clone)]
pub struct OpenAiProtocolState {
    pub partial_by_item: HashMap<String, String>,
    pub completed_by_item: HashMap<String, String>,
    pub current_item_id: Option<String>,
    pub latest_partial: String,
    pub latest_final: String,
    pub completed: bool,
}

impl OpenAiProtocolState {
    pub fn reset(&mut self) {
        self.partial_by_item.clear();
        self.completed_by_item.clear();
        self.current_item_id = None;
        self.latest_partial.clear();
        self.latest_final.clear();
        self.completed = false;
    }

    pub fn begin_commit_wait(&mut self) {
        self.completed = false;
        self.latest_final.clear();
    }

    pub fn handle_event(&mut self, event: &Value) {
        let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match event_type {
            "conversation.item.input_audio_transcription.delta" => self.handle_delta(event),
            "conversation.item.input_audio_transcription.completed" => self.handle_completed(event),
            "input_audio_buffer.committed" => self.handle_committed(event),
            _ => {}
        }
    }

    fn event_item_id(event: &Value) -> Option<String> {
        if let Some(id) = event.get("item_id").and_then(|v| v.as_str())
            && !id.is_empty()
        {
            return Some(id.to_owned());
        }
        event
            .pointer("/item/id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    }

    fn handle_delta(&mut self, event: &Value) {
        let Some(item_id) = Self::event_item_id(event) else {
            return;
        };
        let delta = event
            .get("delta")
            .or_else(|| event.get("transcript"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if delta.is_empty() {
            return;
        }
        if self.current_item_id.is_none() {
            self.current_item_id = Some(item_id.clone());
        }
        let entry = self.partial_by_item.entry(item_id.clone()).or_default();
        entry.push_str(delta);
        if self.current_item_id.as_deref() == Some(item_id.as_str()) {
            self.latest_partial = entry.clone();
        }
    }

    fn handle_completed(&mut self, event: &Value) {
        let Some(item_id) = Self::event_item_id(event) else {
            return;
        };
        let transcript = event
            .get("transcript")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();
        self.completed_by_item
            .insert(item_id.clone(), transcript.clone());
        if self.current_item_id.as_deref() == Some(item_id.as_str()) {
            self.latest_final = if transcript.is_empty() {
                self.partial_by_item
                    .get(&item_id)
                    .cloned()
                    .unwrap_or_default()
            } else {
                transcript
            };
            self.completed = true;
        }
    }

    fn handle_committed(&mut self, event: &Value) {
        let Some(item_id) = Self::event_item_id(event) else {
            return;
        };
        self.current_item_id = Some(item_id.clone());
        if let Some(completed) = self.completed_by_item.get(&item_id) {
            self.latest_final = if completed.is_empty() {
                self.partial_by_item
                    .get(&item_id)
                    .cloned()
                    .unwrap_or_default()
            } else {
                completed.clone()
            };
            self.completed = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn session_update_shape() {
        let v = session_update_payload("gpt-4o-transcribe", "en");
        assert_eq!(v["type"], "transcription_session.update");
        assert_eq!(v["session"]["input_audio_format"], "pcm16");
        assert!(v["session"]["turn_detection"].is_null());
    }

    #[test]
    fn delta_and_completed_track_current_item_only() {
        let mut st = OpenAiProtocolState::default();
        st.handle_event(&json!({
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "a",
            "delta": "hello"
        }));
        assert_eq!(st.latest_partial, "hello");
        st.handle_event(&json!({
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "b",
            "delta": "NOPE"
        }));
        assert_eq!(st.latest_partial, "hello");
        st.handle_event(&json!({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "a",
            "transcript": "hello world"
        }));
        assert!(st.completed);
        assert_eq!(st.latest_final, "hello world");
    }

    #[test]
    fn late_completion_other_item_ignored() {
        let mut st = OpenAiProtocolState::default();
        st.handle_event(&json!({
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "a",
            "delta": "x"
        }));
        st.handle_event(&json!({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "other",
            "transcript": "zzz"
        }));
        assert!(!st.completed);
        assert!(st.latest_final.is_empty());
    }

    #[test]
    fn committed_promotes_stored_completion() {
        let mut st = OpenAiProtocolState::default();
        st.completed_by_item
            .insert("item-1".into(), "final text".into());
        st.handle_event(&json!({
            "type": "input_audio_buffer.committed",
            "item_id": "item-1"
        }));
        assert!(st.completed);
        assert_eq!(st.latest_final, "final text");
        assert_eq!(st.current_item_id.as_deref(), Some("item-1"));
    }

    #[test]
    fn reset_clears_state() {
        let mut st = OpenAiProtocolState {
            latest_partial: "x".into(),
            completed: true,
            ..Default::default()
        };
        st.reset();
        assert!(st.latest_partial.is_empty());
        assert!(!st.completed);
    }

    #[test]
    fn redact_strips_sk_tokens() {
        let ev = json!({"type":"error","error":{"message":"bad sk-abc123XYZ key"}});
        let r = redact_openai_error(&ev);
        assert!(!r.contains("sk-abc"));
        assert!(r.contains("[redacted]"));
    }

    #[test]
    fn clear_buffer_payload() {
        assert_eq!(
            clear_input_buffer_payload()["type"],
            "input_audio_buffer.clear"
        );
    }
}
