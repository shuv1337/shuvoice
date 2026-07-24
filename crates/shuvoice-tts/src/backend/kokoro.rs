//! Kokoro TTS backend (local self-hosted, OpenAI-compatible API).

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
use crate::mp3::{decode_mp3_to_pcm, pcm_samples_to_le_bytes};
use crate::speed::{TTS_PLAYBACK_SPEED_MAX, TTS_PLAYBACK_SPEED_MIN};
use crate::types::{
    AudioEncoding, BackendId, Capabilities, DEFAULT_KOKORO_TTS_BASE_URL,
    DEFAULT_KOKORO_TTS_MODEL_ID, DEFAULT_KOKORO_TTS_VOICE_ID, SynthesisRequest,
    VOICE_CACHE_TTL_SECS, VoiceInfo,
};

const PROVIDER_SPEED_MIN: f64 = 0.5;
const PROVIDER_SPEED_MAX: f64 = 2.0;

/// Runtime configuration for Kokoro.
#[derive(Debug, Clone)]
pub struct KokoroConfig {
    pub base_url: String,
    pub output_format: String,
    pub max_chars: usize,
    pub request_timeout: Duration,
    pub default_voice_id: String,
    pub default_model_id: String,
}

impl Default for KokoroConfig {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_KOKORO_TTS_BASE_URL.into(),
            output_format: "pcm_24000".into(),
            max_chars: 5000,
            request_timeout: Duration::from_secs(30),
            default_voice_id: DEFAULT_KOKORO_TTS_VOICE_ID.into(),
            default_model_id: DEFAULT_KOKORO_TTS_MODEL_ID.into(),
        }
    }
}

struct VoiceCache {
    voices: Vec<VoiceInfo>,
    expires_at: Instant,
}

/// Kokoro text-to-speech backend.
pub struct KokoroTtsBackend {
    config: KokoroConfig,
    client: reqwest::Client,
    cache: Mutex<Option<VoiceCache>>,
}

impl KokoroTtsBackend {
    pub fn new(config: KokoroConfig) -> Result<Self, TtsError> {
        let client = build_client(config.request_timeout)?;
        Ok(Self {
            config,
            client,
            cache: Mutex::new(None),
        })
    }

    fn base_url(&self) -> &str {
        self.config.base_url.trim_end_matches('/')
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
            "mp3" => Ok("mp3"),
            other => Err(TtsError::config(format!(
                "Kokoro TTS requires a supported output format; set [tts].tts_output_format to \
                 \"pcm_24000\" (or \"pcm\" or \"mp3\"), got {other:?}"
            ))),
        }
    }

    fn native_speed(request: &SynthesisRequest) -> Result<f64, TtsError> {
        let speed = positive_finite_speed(request.playback_speed, "Kokoro")?;
        let native = speed.clamp(PROVIDER_SPEED_MIN, PROVIDER_SPEED_MAX);
        let native = (native * 100.0).round() / 100.0;
        if (native - speed).abs() >= 1e-6 {
            tracing::info!(requested = speed, native, "Kokoro TTS speed clamped");
        }
        Ok(native)
    }
}

#[async_trait]
impl TtsBackend for KokoroTtsBackend {
    fn id(&self) -> BackendId {
        BackendId::Kokoro
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_streaming: true,
            supports_voice_list: true,
            requires_api_key: false,
            supports_speed_control: true,
            speed_min: Some(TTS_PLAYBACK_SPEED_MIN),
            speed_max: Some(TTS_PLAYBACK_SPEED_MAX),
        }
    }

    fn sample_rate_hz(&self) -> u32 {
        parse_pcm_sample_rate(&self.config.output_format, 24_000)
    }

    fn dependency_errors(&self) -> Vec<String> {
        Vec::new()
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
        let response_format = self.response_format()?;
        let native_speed = Self::native_speed(&request)?;

        tracing::info!(
            voice = %voice,
            model = %model,
            speed = request.playback_speed,
            native_speed,
            "Kokoro TTS request"
        );

        let body = json!({
            "model": model,
            "voice": voice,
            "input": text_value,
            "response_format": response_format,
            "speed": native_speed,
        });

        let url = format!("{}/audio/speech", self.base_url());
        let client = self.client.clone();
        let response = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(TtsError::Cancelled),
            result = client
                .post(url)
                .header("Content-Type", "application/json")
                .header("Accept", "application/octet-stream")
                .header("Authorization", "Bearer sk-local")
                .json(&body)
                .send() => result.map_err(|e| map_reqwest_error("Kokoro", e))?,
        };

        if !response.status().is_success() {
            return Err(classify_status("Kokoro", response.status()));
        }

        // MP3 responses are fully buffered and decoded to PCM so the player
        // always receives s16le mono.
        if response_format == "mp3" {
            let bytes = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(TtsError::Cancelled),
                result = response.bytes() => result.map_err(|e| map_reqwest_error("Kokoro", e))?,
            };
            let decoded = decode_mp3_to_pcm(&bytes)?;
            let decoded_rate = decoded.sample_rate_hz;
            if decoded_rate == 0 {
                return Err(TtsError::decode(
                    "Kokoro MP3 decoded to invalid sample rate 0",
                ));
            }
            // Prefer decoded rate; reject pathological mismatch only if config asked for
            // an explicit pcm_N and decoded is wildly different? Trust decoder.
            let pcm = pcm_samples_to_le_bytes(&decoded.samples);
            let stream = async_stream::stream! {
                if cancel.is_cancelled() {
                    yield Err(TtsError::Cancelled);
                } else {
                    yield Ok(pcm);
                }
            };
            return Ok(SynthesisStream {
                sample_rate_hz: decoded_rate,
                encoding: AudioEncoding::PcmS16Le,
                chunks: Box::pin(stream),
            });
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
                        let _ = tx.send(Err(map_reqwest_error("Kokoro", err))).await;
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

        let url = format!("{}/audio/voices", self.base_url());
        let response = self
            .client
            .get(url)
            .header("Accept", "application/json")
            .header("Authorization", "Bearer sk-local")
            .send()
            .await
            .map_err(|e| map_reqwest_error("Kokoro", e))?;

        if !response.status().is_success() {
            return Err(classify_status("Kokoro", response.status()));
        }

        let payload: Value = response.json().await.map_err(|err| {
            TtsError::backend(format!("Invalid Kokoro voice list response: {err}"))
        })?;

        let mut voices = Vec::new();
        let raw_voices = payload
            .get("voices")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        for raw in raw_voices {
            if let Some(s) = raw.as_str() {
                let id = s.trim();
                if id.is_empty() {
                    continue;
                }
                voices.push(VoiceInfo::new(id, id));
                continue;
            }
            if let Some(obj) = raw.as_object() {
                let id = obj
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if id.is_empty() {
                    continue;
                }
                let name = obj
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .unwrap_or(id.as_str())
                    .to_string();
                let description = obj
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
