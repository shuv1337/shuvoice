//! Curated Local Piper voice download and validation.

use std::path::{Path, PathBuf};

use shuvoice_core::{data_dir, expand_user_path};
use shuvoice_io::process::CommandRunner;

use super::http::{self, HttpDownloader, ProgressFn, publish_paired_files, stage_dir};

const MAX_VOICE_BYTES: u64 = 512 * 1024 * 1024; // 512 MiB per file

#[derive(Debug, Clone)]
pub struct PiperVoiceOption {
    pub id: &'static str,
    pub label: &'static str,
    pub stem: &'static str,
    pub language: &'static str,
    pub quality: &'static str,
    pub description: &'static str,
    pub model_url: &'static str,
    pub sidecar_url: &'static str,
}

pub const CURATED_PIPER_VOICES: &[PiperVoiceOption] = &[
    PiperVoiceOption {
        id: "en_US-amy-medium",
        label: "US English — Amy (medium, recommended)",
        stem: "en_US-amy-medium",
        language: "en-US",
        quality: "medium",
        description: "Balanced default voice. Good quality with fast local inference.",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true",
        sidecar_url: "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true",
    },
    PiperVoiceOption {
        id: "en_US-lessac-medium",
        label: "US English — Lessac (medium)",
        stem: "en_US-lessac-medium",
        language: "en-US",
        quality: "medium",
        description: "Popular clean US voice. Similar size to Amy with slightly different tone.",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true",
        sidecar_url: "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true",
    },
    PiperVoiceOption {
        id: "en_US-ryan-medium",
        label: "US English — Ryan (medium)",
        stem: "en_US-ryan-medium",
        language: "en-US",
        quality: "medium",
        description: "Medium-quality male US voice. Good alternative to Amy/Lessac.",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx?download=true",
        sidecar_url: "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json?download=true",
    },
    PiperVoiceOption {
        id: "en_US-lessac-high",
        label: "US English — Lessac (high)",
        stem: "en_US-lessac-high",
        language: "en-US",
        quality: "high",
        description: "Higher-quality Lessac voice. Larger download and slower inference.",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/en_US-lessac-high.onnx?download=true",
        sidecar_url: "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/en_US-lessac-high.onnx.json?download=true",
    },
    PiperVoiceOption {
        id: "en_US-ljspeech-high",
        label: "US English — LJSpeech (high)",
        stem: "en_US-ljspeech-high",
        language: "en-US",
        quality: "high",
        description: "Higher-quality female US voice trained on LJSpeech.",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx?download=true",
        sidecar_url: "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json?download=true",
    },
    PiperVoiceOption {
        id: "en_US-ryan-high",
        label: "US English — Ryan (high)",
        stem: "en_US-ryan-high",
        language: "en-US",
        quality: "high",
        description: "Higher-quality male US voice. Largest of the curated set.",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx?download=true",
        sidecar_url: "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json?download=true",
    },
];

pub fn recommended_piper_voice() -> &'static PiperVoiceOption {
    &CURATED_PIPER_VOICES[0]
}

pub fn get_curated_piper_voice(id: &str) -> Result<&'static PiperVoiceOption, String> {
    let needle = id.trim();
    CURATED_PIPER_VOICES
        .iter()
        .find(|v| v.id.eq_ignore_ascii_case(needle) || v.stem.eq_ignore_ascii_case(needle))
        .ok_or_else(|| {
            let known = CURATED_PIPER_VOICES
                .iter()
                .map(|v| v.id)
                .collect::<Vec<_>>()
                .join(", ");
            format!("Unknown curated Piper voice '{id}'. Known: {known}")
        })
}

pub fn managed_piper_model_dir() -> PathBuf {
    data_dir().join("models").join("piper")
}

pub fn find_piper_binary() -> Option<String> {
    for name in ["piper", "piper-tts"] {
        if which::which(name).is_ok() {
            return Some(name.into());
        }
    }
    None
}

/// Ordered **alternative** install commands (first success wins). Arch AUR only.
pub fn piper_install_commands() -> Vec<Vec<String>> {
    vec![
        vec![
            "yay".into(),
            "-S".into(),
            "--needed".into(),
            "piper-tts".into(),
        ],
        vec![
            "paru".into(),
            "-S".into(),
            "--needed".into(),
            "piper-tts".into(),
        ],
    ]
}

/// Human install hints for Arch and non-Arch hosts.
pub fn piper_install_hints() -> Vec<String> {
    let mut hints = Vec::new();
    if which::which("yay").is_ok() {
        hints.push("Arch (AUR): yay -S --needed piper-tts".into());
    }
    if which::which("paru").is_ok() {
        hints.push("Arch (AUR): paru -S --needed piper-tts".into());
    }
    if hints.is_empty() {
        hints.push(
            "Non-Arch: install the upstream Piper CLI (piper-tts) from your distro packages, \
             Flatpak, or https://github.com/rhasspy/piper/releases and ensure `piper` or \
             `piper-tts` is on PATH"
                .into(),
        );
    } else {
        hints.push(
            "Manual / non-Arch: install upstream Piper CLI and ensure `piper` or `piper-tts` is in PATH"
                .into(),
        );
    }
    hints
}

pub fn attempt_piper_auto_install(runner: &dyn CommandRunner) -> bool {
    use shuvoice_io::process::RunOptions;
    // Alternatives: try each package manager until binary appears.
    for cmd in piper_install_commands() {
        if which::which(&cmd[0]).is_err() {
            continue;
        }
        let opts = RunOptions {
            check: false,
            timeout: std::time::Duration::from_secs(600),
            ..RunOptions::default()
        };
        if runner.run(&cmd, &opts).map(|o| o.success).unwrap_or(false)
            && find_piper_binary().is_some()
        {
            return true;
        }
    }
    find_piper_binary().is_some()
}

pub fn validate_piper_voice_artifacts(
    model_dir: &Path,
    voice_stem: Option<&str>,
) -> Result<(PathBuf, Option<u32>), String> {
    let model_dir = expand_user_path(model_dir);
    if !model_dir.exists() {
        return Err(format!("model dir missing: {}", model_dir.display()));
    }
    let model_file = if model_dir.is_file() {
        model_dir.clone()
    } else if let Some(stem) = voice_stem {
        let p = model_dir.join(format!("{stem}.onnx"));
        if !p.is_file() {
            return Err(format!("missing voice model: {}", p.display()));
        }
        p
    } else {
        let mut models: Vec<_> = std::fs::read_dir(&model_dir)
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("onnx"))
            })
            .collect();
        models.sort();
        models
            .into_iter()
            .next()
            .ok_or_else(|| format!("no .onnx models in {}", model_dir.display()))?
    };
    // Piper sidecars are `foo.onnx.json`; managed voices require the sidecar.
    let sidecar = PathBuf::from(format!("{}.json", model_file.display()));
    if !sidecar.is_file() {
        return Err(format!(
            "Missing Piper sidecar metadata: {}",
            sidecar.display()
        ));
    }
    let text = std::fs::read_to_string(&sidecar).map_err(|e| e.to_string())?;
    let parsed = serde_json::from_str::<serde_json::Value>(&text)
        .map_err(|e| format!("Invalid Piper sidecar metadata {}: {e}", sidecar.display()))?;
    let rate = parsed
        .get("audio")
        .and_then(|a| a.get("sample_rate"))
        .or_else(|| parsed.get("sample_rate"))
        .or_else(|| parsed.get("sampleRate"))
        .and_then(|x| x.as_u64())
        .map(|u| u as u32);
    Ok((model_file, rate))
}

#[derive(Debug, Clone)]
pub struct PiperSetupResult {
    pub status: &'static str,
    pub message: String,
    pub binary_name: Option<String>,
    pub model_dir: PathBuf,
    pub voice_stem: String,
    pub sample_rate_hz: Option<u32>,
}

pub async fn ensure_local_piper_ready(
    voice: &PiperVoiceOption,
    model_dir: &Path,
    auto_install_missing: bool,
    downloader: &dyn HttpDownloader,
    runner: &dyn CommandRunner,
    progress: &mut ProgressFn<'_>,
) -> Result<PiperSetupResult, String> {
    let model_dir = expand_user_path(model_dir);
    let mut binary = find_piper_binary();
    if binary.is_none() && auto_install_missing {
        progress(None, "Installing Local Piper runtime…");
        if attempt_piper_auto_install(runner) {
            binary = find_piper_binary();
        }
    }
    if binary.is_none() {
        let hints = piper_install_hints().join("; ");
        return Ok(PiperSetupResult {
            status: "skipped_missing_deps",
            message: format!(
                "Local Piper binary not found (piper/piper-tts). Install piper-tts. Hints: {hints}"
            ),
            binary_name: None,
            model_dir: model_dir.clone(),
            voice_stem: voice.stem.into(),
            sample_rate_hz: None,
        });
    }

    std::fs::create_dir_all(&model_dir).map_err(|e| e.to_string())?;
    let model_path = model_dir.join(format!("{}.onnx", voice.stem));
    let sidecar_path = model_dir.join(format!("{}.onnx.json", voice.stem));

    if model_path.is_file() && sidecar_path.is_file() {
        let rate = validate_piper_voice_artifacts(&model_dir, Some(voice.stem))?.1;
        return Ok(PiperSetupResult {
            status: "ok",
            message: format!("Local Piper voice already present ({})", voice.stem),
            binary_name: binary,
            model_dir: model_dir.clone(),
            voice_stem: voice.stem.into(),
            sample_rate_hz: rate,
        });
    }

    // Unique temp staging (pid + nanos); cleaned on success or failure.
    let stage = stage_dir(&model_dir, &format!("piper-{}", voice.stem))?;
    let stage_model = stage.join(format!("{}.onnx", voice.stem));
    let stage_side = stage.join(format!("{}.onnx.json", voice.stem));

    let result = async {
        // Validate curated HTTPS URLs up front (defense in depth; downloader also checks).
        http::validate_download_url(voice.model_url)?;
        http::validate_download_url(voice.sidecar_url)?;

        progress(Some(0.0), &format!("Downloading {} model…", voice.stem));
        downloader
            .download_to_file(voice.model_url, &stage_model, MAX_VOICE_BYTES, progress)
            .await?;
        let meta = std::fs::metadata(&stage_model).map_err(|e| e.to_string())?;
        if meta.len() < 1024 {
            return Err(format!("downloaded model too small ({} bytes)", meta.len()));
        }
        progress(Some(0.6), &format!("Downloading {} sidecar…", voice.stem));
        downloader
            .download_to_file(voice.sidecar_url, &stage_side, MAX_VOICE_BYTES, progress)
            .await?;
        let side_meta = std::fs::metadata(&stage_side).map_err(|e| e.to_string())?;
        if side_meta.len() < 8 {
            return Err("downloaded sidecar too small".into());
        }
        // Transactional paired publish with rollback + fsync.
        publish_paired_files(&stage_model, &stage_side, &model_path, &sidecar_path)?;
        Ok(())
    }
    .await;

    let _ = std::fs::remove_dir_all(&stage);
    result?;

    let rate = validate_piper_voice_artifacts(&model_dir, Some(voice.stem))?.1;
    progress(Some(1.0), "Local Piper voice ready");
    Ok(PiperSetupResult {
        status: "ok",
        message: format!("Local Piper voice installed ({})", voice.stem),
        binary_name: binary,
        model_dir,
        voice_stem: voice.stem.into(),
        sample_rate_hz: rate,
    })
}
