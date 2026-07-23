//! OpenAI TTS backend.

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use super::http_util::{build_client, classify_status, map_reqwest_error};
use super::{
    SynthesisStream, TtsBackend, ensure_text, parse_pcm_sample_rate, positive_finite_speed,
};
use crate::error::TtsError;
use crate::speed::{TTS_PLAYBACK_SPEED_MAX, TTS_PLAYBACK_SPEED_MIN};
use crate::types::{
    AudioEncoding, BackendId, Capabilities, DEFAULT_OPENAI_TTS_MODEL_ID,
    DEFAULT_OPENAI_TTS_VOICE_ID, SynthesisRequest, VoiceInfo,
};

const API_BASE: &str = "https://api.openai.com/v1";
const PROVIDER_SPEED_MIN: f64 = 0.25;
const PROVIDER_SPEED_MAX: f64 = 4.0;

/// Built-in OpenAI voice catalogue.
pub fn openai_voices() -> Vec<VoiceInfo> {
    vec![
        VoiceInfo::with_description("alloy", "Alloy", "Balanced neutral voice"),
        VoiceInfo::with_description("ash", "Ash", "Clear and steady voice"),
        VoiceInfo::with_description("coral", "Coral", "Warm expressive voice"),
        VoiceInfo::with_description("echo", "Echo", "Bright conversational voice"),
        VoiceInfo::with_description("fable", "Fable", "Narrative story-like voice"),
        VoiceInfo::with_description("onyx", "Onyx", "Deep resonant voice"),
        VoiceInfo::with_description("nova", "Nova", "Upbeat modern voice"),
        VoiceInfo::with_description("sage", "Sage", "Calm measured voice"),
        VoiceInfo::with_description("shimmer", "Shimmer", "Light energetic voice"),
    ]
}

/// Runtime configuration for OpenAI TTS.
#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    pub api_key_env: String,
    pub output_format: String,
    pub max_chars: usize,
    pub request_timeout: std::time::Duration,
    pub default_voice_id: String,
    pub default_model_id: String,
    pub base_url: String,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            api_key_env: "OPENAI_API_KEY".into(),
            output_format: "pcm_24000".into(),
            max_chars: 5000,
            request_timeout: std::time::Duration::from_secs(30),
            default_voice_id: DEFAULT_OPENAI_TTS_VOICE_ID.into(),
            default_model_id: DEFAULT_OPENAI_TTS_MODEL_ID.into(),
            base_url: API_BASE.into(),
        }
    }
}

/// OpenAI text-to-speech backend.
pub struct OpenAiTtsBackend {
    config: OpenAiConfig,
    client: reqwest::Client,
}

impl OpenAiTtsBackend {
    pub fn new(config: OpenAiConfig) -> Result<Self, TtsError> {
        let client = build_client(config.request_timeout)?;
        Ok(Self { config, client })
    }

    fn api_key(&self) -> Result<String, TtsError> {
        let env_name = self.config.api_key_env.trim();
        let key = std::env::var(env_name).unwrap_or_default();
        let key = key.trim();
        if key.is_empty() {
            return Err(TtsError::MissingApiKey(env_name.to_string()));
        }
        Ok(key.to_string())
    }

    fn response_format(&self) -> Result<&'static str, TtsError> {
        match self
            .config
            .output_format
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "pcm" | "pcm_24000" => Ok("pcm"),
            other => Err(TtsError::config(format!(
                "OpenAI TTS requires raw PCM output; set [tts].tts_output_format to \
                 \"pcm_24000\" (or \"pcm\"), got {other:?}"
            ))),
        }
    }

    fn native_speed(request: &SynthesisRequest) -> Result<f64, TtsError> {
        let speed = positive_finite_speed(request.playback_speed, "OpenAI")?;
        let native = speed.clamp(PROVIDER_SPEED_MIN, PROVIDER_SPEED_MAX);
        let native = (native * 100.0).round() / 100.0;
        if (native - speed).abs() >= 1e-6 {
            tracing::info!(requested = speed, native, "OpenAI TTS speed clamped");
        }
        Ok(native)
    }
}

#[async_trait]
impl TtsBackend for OpenAiTtsBackend {
    fn id(&self) -> BackendId {
        BackendId::OpenAi
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_streaming: true,
            supports_voice_list: true,
            requires_api_key: true,
            supports_speed_control: true,
            speed_min: Some(TTS_PLAYBACK_SPEED_MIN),
            speed_max: Some(TTS_PLAYBACK_SPEED_MAX),
        }
    }

    fn sample_rate_hz(&self) -> u32 {
        parse_pcm_sample_rate(&self.config.output_format, 24_000)
    }

    fn dependency_errors(&self) -> Vec<String> {
        let env_name = self.config.api_key_env.trim();
        if std::env::var(env_name)
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
        {
            return Vec::new();
        }
        vec![format!(
            "Missing {env_name} environment variable \
             (or configure [tts].tts_api_key_env and set that variable)"
        )]
    }

    async fn synthesize_stream(
        &self,
        request: SynthesisRequest,
        cancel: CancellationToken,
    ) -> Result<SynthesisStream, TtsError> {
        let text_value = ensure_text(&request.text, self.config.max_chars)?;
        let voice = if request.voice_id.trim().is_empty() {
            self.config.default_voice_id.clone()
        } else {
            request.voice_id.trim().to_string()
        };
        let model = if request.model_id.trim().is_empty() {
            self.config.default_model_id.clone()
        } else {
            request.model_id.trim().to_string()
        };
        let api_key = self.api_key()?;
        let response_format = self.response_format()?;
        let native_speed = Self::native_speed(&request)?;

        tracing::info!(
            voice = %voice,
            model = %model,
            speed = request.playback_speed,
            native_speed,
            "OpenAI TTS request"
        );

        let body = json!({
            "model": model,
            "voice": voice,
            "input": text_value,
            "response_format": response_format,
            "speed": native_speed,
        });

        let url = format!(
            "{}/audio/speech",
            self.config.base_url.trim_end_matches('/')
        );
        let client = self.client.clone();
        let response = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(TtsError::Cancelled),
            result = client
                .post(url)
                .header("Content-Type", "application/json")
                .header("Accept", "application/octet-stream")
                .bearer_auth(api_key)
                .json(&body)
                .send() => result.map_err(|e| map_reqwest_error("OpenAI", e))?,
        };

        if !response.status().is_success() {
            return Err(classify_status("OpenAI", response.status()));
        }

        let mut byte_stream = response.bytes_stream();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, TtsError>>(16);
        let cancel_worker = cancel.clone();
        tokio::spawn(async move {
            loop {
                let next = tokio::select! {
                    biased;
                    _ = cancel_worker.cancelled() => {
                        let _ = tx.send(Err(TtsError::Cancelled)).await;
                        return;
                    }
                    item = byte_stream.next() => item,
                };
                match next {
                    Some(Ok(chunk)) if !chunk.is_empty() => {
                        if tx.send(Ok(chunk)).await.is_err() {
                            return;
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(err)) => {
                        let _ = tx.send(Err(map_reqwest_error("OpenAI", err))).await;
                        return;
                    }
                    None => break,
                }
            }
        });
        let stream = async_stream::stream! {
            let mut rx = rx;
            while let Some(item) = rx.recv().await {
                yield item;
            }
        };
        Ok(SynthesisStream {
            sample_rate_hz: self.sample_rate_hz(),
            encoding: AudioEncoding::PcmS16Le,
            chunks: Box::pin(stream),
        })
    }

    async fn list_voices(&self) -> Result<Vec<VoiceInfo>, TtsError> {
        Ok(openai_voices())
    }
}
