//! Preflight runtime dependency checks.

use shuvoice_core::{AsrBackendKind, Config, TtsBackendKind, expand_user_path};

use crate::commands::audio::validate_configured_input;
use crate::error::{EXIT_FAILURE, EXIT_SUCCESS, ExitStatus};
use crate::setup::layer_shell::layer_shell_detail;
use crate::setup::melotts::{
    format_melotts_report, melotts_missing_dependencies, melotts_venv_dir,
};
use crate::setup::piper::{find_piper_binary, validate_piper_voice_artifacts};
use crate::setup::sherpa_model::{
    asr_dependency_errors_for, format_sherpa_provider_line, is_complete_sherpa_dir,
    sherpa_model_dir, sherpa_runtime_errors,
};
use crate::setup::{
    check_kokoro_voices, kokoro_base_url, openai_asr_key_status, tts_api_key_env_status,
};

struct Check {
    name: &'static str,
    ok: bool,
    detail: String,
}

pub async fn run_preflight(config: &Config) -> ExitStatus {
    let checks = collect_checks(config).await;
    println!("ShuVoice preflight checks");
    println!("{}", "=".repeat(24));
    for check in &checks {
        let mark = if check.ok { "PASS" } else { "FAIL" };
        println!("[{mark}] {}: {}", check.name, check.detail);
    }
    let ready = checks.iter().all(|c| c.ok);
    println!();
    println!("Result: {}", if ready { "READY" } else { "NOT READY" });
    if ready {
        ExitStatus::code(EXIT_SUCCESS)
    } else {
        ExitStatus::code(EXIT_FAILURE)
    }
}

async fn collect_checks(config: &Config) -> Vec<Check> {
    let mut checks = Vec::new();

    checks.push(Check {
        name: "Rust runtime",
        ok: true,
        detail: format!("shuvoice-cli {}", env!("CARGO_PKG_VERSION")),
    });

    checks.push(check_binary("wtype binary", "wtype"));
    checks.push(check_binary("wl-copy binary", "wl-copy"));
    if config.preserve_clipboard || config.tts_enabled {
        checks.push(check_binary("wl-paste binary", "wl-paste"));
    } else {
        checks.push(Check {
            name: "wl-paste binary",
            ok: true,
            detail: "skipped (preserve_clipboard=false, tts_enabled=false)".into(),
        });
    }

    checks.push(check_layer_shell());
    checks.push(Check {
        name: "Output mode",
        ok: true,
        detail: config.output_mode.as_str().to_string(),
    });

    checks.push(check_asr(config));
    checks.extend(check_asr_runtime(config));
    checks.push(check_sherpa_model(config));
    checks.push(check_tts(config).await);
    match tts_api_key_env_status(config) {
        Ok(detail) => checks.push(Check {
            name: "TTS API key",
            ok: true,
            detail,
        }),
        Err(detail) => checks.push(Check {
            name: "TTS API key",
            ok: false,
            detail,
        }),
    }
    match openai_asr_key_status(config) {
        Ok(detail) if detail != "n/a" => checks.push(Check {
            name: "OpenAI ASR API key",
            ok: true,
            detail,
        }),
        Err(detail) => checks.push(Check {
            name: "OpenAI ASR API key",
            ok: false,
            detail,
        }),
        _ => {}
    }

    checks.push(check_audio_input(config));

    checks
}

fn check_binary(name: &'static str, bin: &str) -> Check {
    match which::which(bin) {
        Ok(path) => Check {
            name,
            ok: true,
            detail: path.display().to_string(),
        },
        Err(_) => Check {
            name,
            ok: false,
            detail: format!("{bin} not found in PATH"),
        },
    }
}

fn check_layer_shell() -> Check {
    match layer_shell_detail() {
        Ok(detail) => Check {
            name: "gtk4-layer-shell library",
            ok: true,
            detail,
        },
        Err(detail) => Check {
            name: "gtk4-layer-shell library",
            ok: false,
            detail,
        },
    }
}

fn check_asr(config: &Config) -> Check {
    // Share production worker discovery so setup venv + bundled workers → READY.
    let errors = asr_dependency_errors_for(config);
    if errors.is_empty() {
        let detail = match config.asr_backend {
            AsrBackendKind::Sherpa => format!(
                "sherpa feature OK (requested_provider={}, model={})",
                config.sherpa_provider.as_str(),
                config.sherpa_model_name,
            ),
            AsrBackendKind::OpenaiRealtime => format!(
                "openai_realtime feature OK (model={})",
                config.openai_realtime_model
            ),
            other => format!("{} deps OK (discovered worker runtime)", other.as_str()),
        };
        Check {
            name: "ASR dependencies",
            ok: true,
            detail,
        }
    } else {
        Check {
            name: "ASR dependencies",
            ok: false,
            detail: errors.join("; "),
        }
    }
}

/// Required Sherpa model artifacts must be present for READY.
fn check_sherpa_model(config: &Config) -> Check {
    if config.asr_backend != AsrBackendKind::Sherpa {
        return Check {
            name: "ASR model artifacts",
            ok: true,
            detail: format!("n/a ({})", config.asr_backend.as_str()),
        };
    }
    let dir = sherpa_model_dir(config);
    if is_complete_sherpa_dir(&dir) {
        Check {
            name: "ASR model artifacts",
            ok: true,
            detail: format!("present ({})", dir.display()),
        }
    } else {
        Check {
            name: "ASR model artifacts",
            ok: false,
            detail: format!(
                "missing required Sherpa model at {} (run: shuvoice model download / shuvoice setup)",
                dir.display()
            ),
        }
    }
}

fn check_asr_runtime(config: &Config) -> Vec<Check> {
    let mut out = Vec::new();
    if config.asr_backend != AsrBackendKind::Sherpa {
        return out;
    }
    // Always surface honest provider line in the detail when failing or passing.
    let provider_line = format_sherpa_provider_line(config)
        .trim_start_matches("[INFO] ")
        .to_string();
    let errs = sherpa_runtime_errors(config);
    if errs.is_empty() {
        let decode = config
            .resolved_sherpa_decode_mode()
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| config.sherpa_decode_mode.as_str().to_string());
        out.push(Check {
            name: "ASR runtime compatibility",
            ok: true,
            detail: format!(
                "{provider_line}; sherpa_decode_mode={decode}, parakeet_streaming_gate=ok"
            ),
        });
    } else {
        out.push(Check {
            name: "ASR runtime compatibility",
            ok: false,
            detail: format!("{provider_line}; {}", errs.join(" | ")),
        });
    }
    out
}

fn check_audio_input(config: &Config) -> Check {
    match validate_configured_input(&config.audio_device) {
        Ok(detail) => Check {
            name: "Audio input device",
            ok: true,
            detail,
        },
        Err(detail) => Check {
            name: "Audio input device",
            ok: false,
            detail,
        },
    }
}

async fn check_tts(config: &Config) -> Check {
    if !config.tts_enabled {
        return Check {
            name: "TTS dependencies",
            ok: true,
            detail: "disabled".into(),
        };
    }
    match config.tts_backend {
        TtsBackendKind::Local => {
            let bin = find_piper_binary();
            let dir = config.tts_local_model_path.as_ref().map(expand_user_path);
            let voice = config
                .tts_local_voice
                .as_deref()
                .or(Some(config.tts_default_voice_id.as_str()));
            match (bin, dir) {
                (Some(b), Some(d)) => match validate_piper_voice_artifacts(&d, voice) {
                    Ok((_, rate)) => Check {
                        name: "TTS dependencies",
                        ok: true,
                        detail: format!(
                            "local piper OK (bin={b}, path={}, rate={rate:?})",
                            d.display()
                        ),
                    },
                    Err(e) => Check {
                        name: "TTS dependencies",
                        ok: false,
                        detail: e,
                    },
                },
                (None, _) => Check {
                    name: "TTS dependencies",
                    ok: false,
                    detail: "piper/piper-tts binary not found in PATH".into(),
                },
                (_, None) => Check {
                    name: "TTS dependencies",
                    ok: false,
                    detail: "tts_local_model_path is not configured".into(),
                },
            }
        }
        TtsBackendKind::Melotts => {
            let venv = melotts_venv_dir(config);
            let missing = melotts_missing_dependencies(&venv);
            Check {
                name: "TTS dependencies",
                ok: missing.is_empty(),
                detail: format_melotts_report(&venv, &missing),
            }
        }
        TtsBackendKind::Kokoro => {
            let base = kokoro_base_url(config);
            match check_kokoro_voices(&base).await {
                Ok(n) => Check {
                    name: "TTS dependencies",
                    ok: true,
                    detail: format!("kokoro OK ({n} usable voices, base_url={base})"),
                },
                Err(e) => Check {
                    name: "TTS dependencies",
                    ok: false,
                    detail: e,
                },
            }
        }
        TtsBackendKind::Elevenlabs | TtsBackendKind::Openai => Check {
            name: "TTS dependencies",
            ok: true,
            detail: format!(
                "{} provider selected (API key checked separately)",
                config.tts_backend.as_str()
            ),
        },
    }
}
