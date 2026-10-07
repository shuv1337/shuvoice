//! Opt-in transcript log record (`[vocabulary].transcript_log`).
//!
//! One JSON object per finalized utterance, written as JSON lines to
//! [`transcript_log_path`]. The log exists to build a corpus for tuning
//! `text_replacements`, so it keeps every stage of the render pipeline.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::postprocess::{RenderTrace, ReplacementHit};
use crate::types::{AsrBackendKind, TypingTextCase};
use crate::xdg::state_dir;

/// Rotate the active file to `<name>.1` once it reaches this size.
pub const TRANSCRIPT_LOG_MAX_BYTES: u64 = 16 * 1024 * 1024;

/// `$XDG_STATE_HOME/shuvoice/transcripts.jsonl`.
pub fn transcript_log_path() -> PathBuf {
    state_dir().join("transcripts.jsonl")
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptRecord {
    /// RFC 3339 UTC timestamp with millisecond precision.
    pub ts: String,
    pub utt_gen: u64,
    pub asr_backend: String,
    pub asr_model: String,
    /// Case policy latched for this utterance (`start_alt` flips it).
    pub text_case: String,
    /// Captured audio up to the stop edge (excludes the stop tail grace).
    pub audio_ms: u64,
    /// ASR output before any post-processing.
    pub raw: String,
    /// After `text_replacements`, before case policy.
    pub replaced: String,
    /// Text handed to the injector; empty when nothing was typed.
    pub output: String,
    pub replacements: Vec<ReplacementHit>,
    /// Config revision active for this utterance, when known.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub config_revision: Option<String>,
}

impl TranscriptRecord {
    pub fn new(
        config: &Config,
        utt_gen: u64,
        text_case: TypingTextCase,
        audio_ms: u64,
        raw: &str,
        trace: RenderTrace,
        output: &str,
    ) -> Self {
        Self {
            ts: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            utt_gen,
            asr_backend: config.asr_backend.as_str().to_string(),
            asr_model: active_asr_model(config).to_string(),
            text_case: text_case.as_str().to_string(),
            audio_ms,
            raw: raw.to_string(),
            replaced: trace.replaced,
            output: output.to_string(),
            replacements: trace.hits,
            config_revision: config.loaded_config_revision.clone(),
        }
    }
}

fn active_asr_model(config: &Config) -> &str {
    match config.asr_backend {
        AsrBackendKind::Sherpa => &config.sherpa_model_name,
        AsrBackendKind::Nemo => &config.model_name,
        AsrBackendKind::Moonshine => &config.moonshine_model_name,
        AsrBackendKind::OpenaiRealtime => &config.openai_realtime_model,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postprocess::{
        RenderOptions, compile_text_replacements, render_transcript_text_traced,
    };
    use std::collections::BTreeMap;

    #[test]
    fn record_serializes_all_stages() {
        let mut replacements = BTreeMap::new();
        replacements.insert("tail scale".into(), "Tailscale".into());
        let options = RenderOptions {
            text_case: TypingTextCase::Lowercase,
            auto_capitalize: true,
            replacements: compile_text_replacements(&replacements),
        };
        let raw = "Connect over tail scale";
        let trace = render_transcript_text_traced(raw, &options);
        let output = trace.rendered.clone();
        let config = Config::default();
        let record = TranscriptRecord::new(
            &config,
            7,
            TypingTextCase::Lowercase,
            1250,
            raw,
            trace,
            &output,
        );
        let json: serde_json::Value = serde_json::to_value(&record).unwrap();
        assert_eq!(json["raw"], "Connect over tail scale");
        assert_eq!(json["replaced"], "Connect over Tailscale");
        assert_eq!(json["output"], "connect over tailscale");
        assert_eq!(json["text_case"], "lowercase");
        assert_eq!(json["asr_backend"], "sherpa");
        assert_eq!(json["asr_model"], config.sherpa_model_name);
        assert_eq!(json["replacements"][0]["rule"], "tail scale");
        assert!(json.get("config_revision").is_none());
        let back: TranscriptRecord = serde_json::from_value(json).unwrap();
        assert_eq!(back, record);
    }
}
