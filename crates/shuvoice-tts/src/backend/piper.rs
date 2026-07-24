//! Local Piper-backed TTS backend.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use super::{SynthesisStream, TtsBackend, ensure_text, positive_finite_speed};
use crate::error::TtsError;
use crate::speed::{TTS_PLAYBACK_SPEED_MAX, TTS_PLAYBACK_SPEED_MIN};
use crate::types::{
    AudioEncoding, BackendId, Capabilities, DEFAULT_LOCAL_TTS_VOICE_ID, SynthesisRequest, VoiceInfo,
};

const DEFAULT_PIPER_SAMPLE_RATE_HZ: u32 = 22_050;
const AUTO_VOICE_IDS: &[&str] = &["default", "auto"];

/// Runtime configuration for local Piper TTS.
#[derive(Debug, Clone)]
pub struct PiperConfig {
    pub model_path: PathBuf,
    pub default_voice_id: String,
    pub local_voice: Option<String>,
    pub max_chars: usize,
    pub request_timeout: Duration,
    /// Optional explicit binary path; otherwise searches PATH for piper/piper-tts.
    pub piper_binary: Option<PathBuf>,
}

/// Local TTS backend using the Piper CLI.
pub struct PiperTtsBackend {
    config: PiperConfig,
    piper_binary: PathBuf,
    model_path: PathBuf,
    voices: Vec<VoiceInfo>,
}

impl PiperTtsBackend {
    pub fn new(config: PiperConfig) -> Result<Self, TtsError> {
        let piper_binary = match &config.piper_binary {
            Some(path) => path.clone(),
            None => find_piper_binary().ok_or_else(|| {
                TtsError::process(
                    "Missing piper binary for local TTS backend. \
                     Install Piper (piper or piper-tts) and set [tts].tts_local_model_path.",
                )
            })?,
        };
        let model_path = validate_model_path(&config.model_path)?;
        let voices = discover_voices(&model_path);
        if voices.is_empty() {
            return Err(TtsError::config(format!(
                "No .onnx model files found under local TTS path: {}",
                model_path.display()
            )));
        }
        Ok(Self {
            config,
            piper_binary,
            model_path,
            voices,
        })
    }

    fn normalize_voice_id(value: &str) -> Option<String> {
        let voice_id = value.trim();
        if voice_id.is_empty() {
            return None;
        }
        if AUTO_VOICE_IDS
            .iter()
            .any(|auto| voice_id.eq_ignore_ascii_case(auto))
        {
            return None;
        }
        Some(voice_id.to_string())
    }

    fn resolve_model_file(&self, voice_id: &str) -> Result<PathBuf, TtsError> {
        if self.model_path.is_file() {
            return Ok(self.model_path.clone());
        }

        let mut requested = Self::normalize_voice_id(voice_id);
        if requested.is_none()
            && let Some(local) = &self.config.local_voice
        {
            requested = Self::normalize_voice_id(local);
        }
        if requested.is_none() {
            requested = Self::normalize_voice_id(&self.config.default_voice_id);
        }

        if let Some(requested_voice) = requested {
            let requested_file = self.model_path.join(format!("{requested_voice}.onnx"));
            if requested_file.is_file() {
                return Ok(requested_file);
            }
            return Err(TtsError::backend(format!(
                "Requested local TTS voice '{requested_voice}' not found in {}",
                self.model_path.display()
            )));
        }

        let mut models: Vec<_> = std::fs::read_dir(&self.model_path)
            .map_err(|err| TtsError::io(err.to_string()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|ext| ext.to_str())
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
            })
            .collect();
        models.sort();
        models.into_iter().next().ok_or_else(|| {
            TtsError::config(format!(
                "No .onnx model files found under local TTS path: {}",
                self.model_path.display()
            ))
        })
    }

    /// Piper length-scale is inverse duration control.
    pub fn length_scale_for_speed(speed: f64) -> Result<f64, TtsError> {
        let speed_value = positive_finite_speed(speed, "Local Piper")?;
        Ok(((1.0 / speed_value) * 10_000.0).round() / 10_000.0)
    }
}

#[async_trait]
impl TtsBackend for PiperTtsBackend {
    fn id(&self) -> BackendId {
        BackendId::Local
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
        match self.resolve_model_file(&self.config.default_voice_id) {
            Ok(model_file) => {
                piper_sample_rate_from_sidecar(&model_file).unwrap_or(DEFAULT_PIPER_SAMPLE_RATE_HZ)
            }
            Err(_) => DEFAULT_PIPER_SAMPLE_RATE_HZ,
        }
    }

    fn dependency_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if find_piper_binary().is_none() && self.config.piper_binary.is_none() {
            errors.push(
                "Missing piper binary for local TTS backend. \
                 Install Piper (piper or piper-tts) and set [tts].tts_local_model_path."
                    .into(),
            );
        }
        errors
    }

    async fn synthesize_stream(
        &self,
        request: SynthesisRequest,
        cancel: CancellationToken,
    ) -> Result<SynthesisStream, TtsError> {
        let text_value = ensure_text(&request.text, self.config.max_chars)?;
        let model_file = self.resolve_model_file(&request.voice_id)?;
        let length_scale = Self::length_scale_for_speed(request.playback_speed)?;
        let sample_rate =
            piper_sample_rate_from_sidecar(&model_file).unwrap_or(DEFAULT_PIPER_SAMPLE_RATE_HZ);

        let mut effective_voice = Self::normalize_voice_id(&request.voice_id)
            .unwrap_or_else(|| model_file.file_stem().unwrap().to_string_lossy().into());
        if effective_voice == DEFAULT_LOCAL_TTS_VOICE_ID {
            effective_voice = model_file.file_stem().unwrap().to_string_lossy().into();
        }

        tracing::info!(
            voice = %effective_voice,
            speed = request.playback_speed,
            length_scale,
            model = %model_file.file_name().unwrap_or_default().to_string_lossy(),
            sample_rate_hz = sample_rate,
            "Local Piper TTS request"
        );

        let timeout = self
            .config
            .request_timeout
            .mul_f64(4.0)
            .max(Duration::from_secs(1));

        let binary = self.piper_binary.clone();
        let length_scale_arg = format!("{length_scale:.4}");

        let mut child = Command::new(&binary)
            .arg("--model")
            .arg(&model_file)
            .arg("--output_raw")
            .arg("--length-scale")
            .arg(&length_scale_arg)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|err| {
                TtsError::process(format!("Failed to start Piper local TTS process: {err}"))
            })?;

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| TtsError::process("Piper stdin missing"))?;
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| TtsError::process("Piper stdout missing"))?;
        let mut stderr = child
            .stderr
            .take()
            .ok_or_else(|| TtsError::process("Piper stderr missing"))?;

        stdin.write_all(text_value.as_bytes()).await?;
        drop(stdin);

        let stderr_task = tokio::spawn(async move {
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            loop {
                match stderr.read(&mut tmp).await {
                    Ok(0) => break,
                    Ok(n) => buf.extend_from_slice(&tmp[..n]),
                    Err(_) => break,
                }
            }
            buf
        });

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, TtsError>>(8);
        let cancel_worker = cancel.clone();
        let op_deadline = tokio::time::Instant::now() + timeout;
        tokio::spawn(async move {
            let mut buf = [0u8; 4096];
            let send_err = |tx: &tokio::sync::mpsc::Sender<Result<Bytes, TtsError>>,
                            err: TtsError| {
                let _ = tx.try_send(Err(err));
            };

            loop {
                if cancel_worker.is_cancelled() {
                    let _ = child.kill().await;
                    send_err(&tx, TtsError::Cancelled);
                    return;
                }

                let read = tokio::select! {
                    biased;
                    _ = cancel_worker.cancelled() => {
                        let _ = child.kill().await;
                        send_err(&tx, TtsError::Cancelled);
                        return;
                    }
                    _ = tokio::time::sleep_until(op_deadline) => {
                        let _ = child.kill().await;
                        send_err(&tx, TtsError::timed_out("Local TTS synthesis timed out"));
                        return;
                    }
                    result = stdout.read(&mut buf) => result,
                };

                match read {
                    Ok(0) => break,
                    Ok(n) => {
                        if tx
                            .send(Ok(Bytes::copy_from_slice(&buf[..n])))
                            .await
                            .is_err()
                        {
                            let _ = child.kill().await;
                            return;
                        }
                    }
                    Err(err) => {
                        let _ = child.kill().await;
                        send_err(&tx, TtsError::from(err));
                        return;
                    }
                }
            }

            let status = tokio::select! {
                biased;
                _ = cancel_worker.cancelled() => {
                    let _ = child.kill().await;
                    send_err(&tx, TtsError::Cancelled);
                    return;
                }
                result = tokio::time::timeout(timeout, child.wait()) => result,
            };

            let status = match status {
                Ok(Ok(status)) => status,
                Ok(Err(err)) => {
                    send_err(&tx, TtsError::from(err));
                    return;
                }
                Err(_) => {
                    let _ = child.kill().await;
                    send_err(&tx, TtsError::timed_out("Local TTS synthesis timed out"));
                    return;
                }
            };

            let stderr_bytes = stderr_task.await.unwrap_or_default();
            if !status.success() {
                let stderr_text = String::from_utf8_lossy(&stderr_bytes).trim().to_string();
                let err = if stderr_text.is_empty() {
                    TtsError::process(format!(
                        "Local TTS synthesis failed with exit code {:?}",
                        status.code()
                    ))
                } else {
                    TtsError::process(format!("Local TTS synthesis failed: {stderr_text}"))
                };
                send_err(&tx, err);
            }
        });

        let stream = async_stream::stream! {
            let mut rx = rx;
            while let Some(item) = rx.recv().await {
                yield item;
            }
        };
        Ok(SynthesisStream {
            sample_rate_hz: sample_rate,
            encoding: AudioEncoding::PcmS16Le,
            chunks: Box::pin(stream),
        })
    }

    async fn list_voices(&self) -> Result<Vec<VoiceInfo>, TtsError> {
        Ok(self.voices.clone())
    }
}

/// Locate `piper` or `piper-tts` on PATH.
pub fn find_piper_binary() -> Option<PathBuf> {
    for name in ["piper", "piper-tts"] {
        if let Ok(path) = which::which(name) {
            return Some(path);
        }
    }
    None
}

/// Read sample rate from Piper `.onnx.json` sidecar.
pub fn piper_sample_rate_from_sidecar(model_file: &Path) -> Option<u32> {
    // Python uses `model_file.name + ".json"` i.e. `foo.onnx.json`.
    let sidecar = model_file.with_file_name(format!(
        "{}.json",
        model_file.file_name()?.to_string_lossy()
    ));
    if !sidecar.is_file() {
        return None;
    }
    let text = std::fs::read_to_string(&sidecar).ok()?;
    let payload: serde_json::Value = serde_json::from_str(&text).ok()?;
    let candidates = [
        payload
            .get("audio")
            .and_then(|a| a.get("sample_rate"))
            .cloned(),
        payload.get("sample_rate").cloned(),
        payload.get("sampleRate").cloned(),
    ];
    for value in candidates.into_iter().flatten() {
        if let Some(rate) = value.as_u64()
            && rate > 0
            && rate <= u64::from(u32::MAX)
        {
            return Some(rate as u32);
        }
        if let Some(rate) = value.as_i64()
            && rate > 0
            && rate <= i64::from(u32::MAX)
        {
            return Some(rate as u32);
        }
        if let Some(s) = value.as_str()
            && let Ok(rate) = s.parse::<u32>()
            && rate > 0
        {
            return Some(rate);
        }
    }
    None
}

fn validate_model_path(model_path: &Path) -> Result<PathBuf, TtsError> {
    let path = expand_user(model_path);
    if path.is_file() {
        let is_onnx = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("onnx"));
        if !is_onnx {
            return Err(TtsError::config(format!(
                "Local TTS model path must point to a .onnx file: {}",
                path.display()
            )));
        }
        return Ok(path);
    }
    if !path.exists() {
        return Err(TtsError::config(format!(
            "Local TTS model path does not exist: {}",
            path.display()
        )));
    }
    if !path.is_dir() {
        return Err(TtsError::config(format!(
            "Local TTS model path must be a file or directory: {}",
            path.display()
        )));
    }
    let has_onnx = std::fs::read_dir(&path)
        .map_err(|err| TtsError::io(err.to_string()))?
        .filter_map(|e| e.ok())
        .any(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
        });
    if !has_onnx {
        return Err(TtsError::config(format!(
            "No .onnx model files found under local TTS path: {}",
            path.display()
        )));
    }
    Ok(path)
}

fn discover_voices(model_path: &Path) -> Vec<VoiceInfo> {
    if model_path.is_file() {
        let stem = model_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "voice".into());
        return vec![VoiceInfo::with_description(
            stem.clone(),
            stem,
            model_path.display().to_string(),
        )];
    }
    let mut voices = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model_path) {
        let mut paths: Vec<_> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|ext| ext.to_str())
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
            })
            .collect();
        paths.sort();
        for candidate in paths {
            let stem = candidate
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            voices.push(VoiceInfo::with_description(
                stem.clone(),
                stem,
                candidate.display().to_string(),
            ));
        }
    }
    voices
}

fn expand_user(path: &Path) -> PathBuf {
    let raw = path.as_os_str().to_string_lossy();
    if let Some(rest) = raw.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    if raw == "~"
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home);
    }
    path.to_path_buf()
}
