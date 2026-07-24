//! ElevenLabs streaming TTS backend.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::StreamExt;
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use super::http_util::{build_client, classify_status, map_reqwest_error};
use super::{
    SynthesisStream, TtsBackend, ensure_text, parse_pcm_sample_rate, positive_finite_speed,
};
use crate::error::TtsError;
use crate::speed::{TTS_PLAYBACK_SPEED_MAX, TTS_PLAYBACK_SPEED_MIN};
use crate::types::{
    AudioEncoding, BackendId, Capabilities, SynthesisRequest, VOICE_CACHE_TTL_SECS, VoiceInfo,
};

const API_BASE: &str = "https://api.elevenlabs.io/v1";

/// Runtime configuration for ElevenLabs.
#[derive(Debug, Clone)]
pub struct ElevenLabsConfig {
    pub api_key_env: String,
    pub output_format: String,
    pub max_chars: usize,
    pub request_timeout: Duration,
    pub default_voice_id: String,
    pub default_model_id: String,
    /// Override base URL (tests).
    pub base_url: String,
}

impl Default for ElevenLabsConfig {
    fn default() -> Self {
        Self {
            api_key_env: "ELEVENLABS_API_KEY".into(),
            output_format: "pcm_24000".into(),
            max_chars: 5000,
            request_timeout: Duration::from_secs(30),
            default_voice_id: crate::types::DEFAULT_ELEVENLABS_TTS_VOICE_ID.into(),
            default_model_id: crate::types::DEFAULT_ELEVENLABS_TTS_MODEL_ID.into(),
            base_url: API_BASE.into(),
        }
    }
}

struct VoiceCache {
    voices: Vec<VoiceInfo>,
    expires_at: Instant,
}

/// Streaming ElevenLabs backend using reqwest.
pub struct ElevenLabsTtsBackend {
    config: ElevenLabsConfig,
    client: reqwest::Client,
    cache: Mutex<Option<VoiceCache>>,
}

impl ElevenLabsTtsBackend {
    pub fn new(config: ElevenLabsConfig) -> Result<Self, TtsError> {
        let client = build_client(config.request_timeout)?;
        Ok(Self {
            config,
            client,
            cache: Mutex::new(None),
        })
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

    fn native_speed(request: &SynthesisRequest) -> Result<f64, TtsError> {
        positive_finite_speed(request.playback_speed, "ElevenLabs")
    }

    /// The playback path treats the response body as raw PCM s16le, so only
    /// `pcm_<rate>` output formats are valid; compressed formats (mp3, opus,
    /// ulaw) would be reinterpreted as samples and play as noise.
    fn pcm_output_format(&self) -> Result<&str, TtsError> {
        let format = self.config.output_format.trim();
        let is_pcm = format
            .strip_prefix("pcm_")
            .is_some_and(|rate| rate.parse::<u32>().is_ok());
        if !is_pcm {
            return Err(TtsError::config(format!(
                "ElevenLabs backend streams raw PCM only; set [tts].tts_output_format to a \
                 pcm_* format such as \"pcm_24000\", got {format:?}"
            )));
        }
        Ok(format)
    }
}

#[async_trait]
impl TtsBackend for ElevenLabsTtsBackend {
    fn id(&self) -> BackendId {
        BackendId::ElevenLabs
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
        // Match Python: checks ELEVENLABS_API_KEY specifically for the static dep check
        // when the default env name is used; otherwise check configured env.
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
        let native_speed = Self::native_speed(&request)?;
        let output_format = self.pcm_output_format()?;

        tracing::info!(
            voice = %voice,
            model = %model,
            speed = request.playback_speed,
            native_speed,
            "ElevenLabs TTS request"
        );

        let url = format!(
            "{}/text-to-speech/{}/stream?output_format={}",
            self.config.base_url.trim_end_matches('/'),
            urlencoding_encode(&voice),
            output_format
        );

        let body = json!({
            "text": text_value,
            "model_id": model,
            "voice_settings": {
                "speed": native_speed,
            },
        });

        let client = self.client.clone();
        let response = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(TtsError::Cancelled),
            result = client
                .post(&url)
                .header("Content-Type", "application/json")
                .header("Accept", "application/octet-stream")
                .header("xi-api-key", api_key)
                .json(&body)
                .send() => result.map_err(|err| map_reqwest_error("ElevenLabs", err))?,
        };

        if !response.status().is_success() {
            return Err(classify_status("ElevenLabs", response.status()));
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
                        let _ = tx.send(Err(map_reqwest_error("ElevenLabs", err))).await;
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
        {
            let guard = self.cache.lock();
            if let Some(cache) = guard.as_ref()
                && Instant::now() < cache.expires_at
            {
                return Ok(cache.voices.clone());
            }
        }

        let api_key = self.api_key()?;
        let url = format!("{}/voices", self.config.base_url.trim_end_matches('/'));
        let response = self
            .client
            .get(&url)
            .header("Accept", "application/json")
            .header("xi-api-key", api_key)
            .send()
            .await
            .map_err(|err| map_reqwest_error("ElevenLabs", err))?;

        if !response.status().is_success() {
            return Err(classify_status("ElevenLabs", response.status()));
        }

        let payload: Value = response.json().await.map_err(|err| {
            TtsError::backend(format!("Invalid ElevenLabs voice list response: {err}"))
        })?;

        let mut voices = Vec::new();
        if let Some(arr) = payload.get("voices").and_then(|v| v.as_array()) {
            for raw in arr {
                let id = raw
                    .get("voice_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if id.is_empty() {
                    continue;
                }
                let name = raw
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .unwrap_or(id.as_str())
                    .to_string();
                let description = raw
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                voices.push(VoiceInfo {
                    id,
                    name,
                    description,
                });
            }
        }

        *self.cache.lock() = Some(VoiceCache {
            voices: voices.clone(),
            expires_at: Instant::now() + Duration::from_secs(VOICE_CACHE_TTL_SECS),
        });

        Ok(voices)
    }
}

fn urlencoding_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}
