//! One-shot setup workflow: deps, model artifacts, TTS automation, preflight.

use std::path::PathBuf;
use std::sync::Arc;

use shuvoice_core::{AsrBackendKind, Config, TtsBackendKind, expand_user_path, is_parakeet_model};

use crate::commands::{model, preflight};
use crate::error::{EXIT_DEPENDENCY, EXIT_FAILURE, EXIT_SUCCESS, ExitStatus};
use crate::setup::install::{
    HostProfile, asr_feature_guidance, asr_install_pipelines, run_install_pipelines,
    worker_venv_dir_for_backend, worker_venv_python_ready,
};
use crate::setup::melotts::{
    format_melotts_report, melotts_install_commands, melotts_missing_dependencies,
    melotts_venv_dir, melotts_venv_valid, run_melotts_install,
};
use crate::setup::piper::{
    ensure_local_piper_ready, find_piper_binary, get_curated_piper_voice, managed_piper_model_dir,
    piper_install_hints, recommended_piper_voice, validate_piper_voice_artifacts,
};
use crate::setup::sherpa_model::{
    asr_dependency_errors_for, format_sherpa_provider_line, model_status_line,
    sherpa_runtime_errors,
};
use crate::setup::{SetupContext, persist_local_tts_selection, run_blocking_setup};

pub struct SetupOptions {
    pub install_missing: bool,
    pub skip_model_download: bool,
    pub skip_preflight: bool,
    pub tts_local_voice: Option<String>,
    pub tts_local_model_dir: Option<PathBuf>,
    pub non_interactive: bool,
}

pub async fn run_setup(config: &Config, opts: SetupOptions) -> ExitStatus {
    run_setup_with_ctx(config, opts, &SetupContext::default()).await
}

pub async fn run_setup_with_ctx(
    config: &Config,
    opts: SetupOptions,
    ctx: &SetupContext,
) -> ExitStatus {
    let mut config = config.clone();

    println!("ShuVoice setup");
    println!("{}", "=".repeat(13));
    println!("ASR backend: {}", config.asr_backend.as_str());
    println!("TTS backend: {}", config.tts_backend.as_str());
    if config.asr_backend == AsrBackendKind::OpenaiRealtime {
        println!("OpenAI Realtime model: {}", config.openai_realtime_model);
        println!("OpenAI API key env: {}", config.openai_realtime_api_key_env);
    }

    println!("Model status: {}", model_status_line(&config));

    // ── ASR dependencies ────────────────────────────────────────────
    let mut missing = asr_dependency_errors_for(&config);
    let worker_backend = matches!(
        config.asr_backend,
        AsrBackendKind::Nemo | AsrBackendKind::Moonshine
    );

    if !missing.is_empty() || worker_backend {
        if !missing.is_empty() {
            println!();
            println!("[FAIL] Backend dependencies");
            for m in &missing {
                println!("  - {m}");
            }
        } else {
            println!();
            println!(
                "[INFO] Optional worker backend selected ({})",
                config.asr_backend.as_str()
            );
            for line in asr_feature_guidance(config.asr_backend) {
                println!("  {line}");
            }
        }

        if opts.install_missing {
            println!();
            println!("Automatic install requested.");
            let host = HostProfile::detect();
            let pipelines = asr_install_pipelines(config.asr_backend, &host);
            if pipelines.is_empty() {
                for line in asr_feature_guidance(config.asr_backend) {
                    println!("  {line}");
                }
                // Native Sherpa/OpenAI: feature rebuild cannot be auto-installed via packages.
                if !missing.is_empty() && !worker_backend {
                    println!();
                    println!("Setup incomplete: missing backend dependencies remain.");
                    return ExitStatus::code(EXIT_DEPENDENCY);
                }
            } else {
                let runner = Arc::clone(&ctx.runner);
                let pipelines_owned = pipelines.clone();
                match run_blocking_setup(move || {
                    run_install_pipelines(&pipelines_owned, runner.as_ref())
                })
                .await
                {
                    Ok(n) => println!("Install pipeline steps completed: {n}"),
                    Err(e) => println!("{e}"),
                }
            }
            missing = asr_dependency_errors_for(&config);
        }

        if !missing.is_empty() && !worker_backend {
            println!();
            println!("Setup incomplete: missing backend dependencies remain.");
            return ExitStatus::code(EXIT_DEPENDENCY);
        }
    }

    // Honest worker outcomes: never PASS then later exit 78 for the same section.
    if worker_backend {
        let venv_dir = worker_venv_dir_for_backend(config.asr_backend);
        let ready = venv_dir
            .as_ref()
            .is_some_and(|d| worker_venv_python_ready(d));
        println!();
        if ready {
            println!(
                "[PASS] Optional worker venv ready ({})",
                venv_dir.unwrap().display()
            );
        } else if opts.install_missing {
            println!("[FAIL] Optional worker venv");
            if let Some(d) = &venv_dir {
                println!("  Expected: {}", d.display());
            }
            println!("Setup incomplete: worker venv install did not produce a usable python.");
            return ExitStatus::code(EXIT_DEPENDENCY);
        } else {
            println!("[INFO] Optional worker venv not installed (guidance only; not a PASS)");
            if let Some(d) = &venv_dir {
                println!("  Target: {}", d.display());
            }
        }
    } else if missing.is_empty() {
        println!();
        println!("[PASS] Backend dependencies");
    }

    // ── Runtime compatibility (CUDA fail-closed + Parakeet gate) ────
    let startup_errors = sherpa_runtime_errors(&config);
    if config.asr_backend == AsrBackendKind::Sherpa {
        let decode = config
            .resolved_sherpa_decode_mode()
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| config.sherpa_decode_mode.as_str().to_string());
        println!("[INFO] Sherpa decode mode: {decode}");
        println!("{}", format_sherpa_provider_line(&config));
        let parakeet = is_parakeet_model(&config.sherpa_model_name);
        println!(
            "[INFO] Sherpa Parakeet model: {}",
            if parakeet { "yes" } else { "no" }
        );
        if parakeet {
            println!(
                "[INFO] Sherpa Parakeet runnable: {}",
                if startup_errors.is_empty() {
                    "yes"
                } else {
                    "no"
                }
            );
        }
    }
    if !startup_errors.is_empty() {
        println!();
        println!("[FAIL] Backend runtime compatibility");
        for e in &startup_errors {
            println!("  - {e}");
        }
        return ExitStatus::code(EXIT_DEPENDENCY);
    }
    println!("[PASS] Backend runtime compatibility");

    // ── Model download ──────────────────────────────────────────────
    if !opts.skip_model_download {
        match model::download_model_impl(&config, ctx.sherpa_archive_url_override.clone()).await {
            Ok(msg) => println!("{msg}"),
            Err(err) => {
                eprintln!("ERROR: model download failed: {err}");
                if err.contains("worker") || err.contains("feature") {
                    return ExitStatus::code(EXIT_DEPENDENCY);
                }
                return ExitStatus::code(EXIT_FAILURE);
            }
        }
    } else {
        println!("Model download: skipped (--skip-model-download).");
    }

    // ── Local Piper ─────────────────────────────────────────────────
    if config.tts_backend == TtsBackendKind::Local {
        let code = run_local_tts_setup(&mut config, &opts, ctx).await;
        if code.code != EXIT_SUCCESS {
            return code;
        }
    }

    // ── MeloTTS ─────────────────────────────────────────────────────
    if config.tts_backend == TtsBackendKind::Melotts {
        let code = run_melotts_setup(&config, &opts, ctx).await;
        if code.code != EXIT_SUCCESS {
            return code;
        }
    }

    // ── Kokoro ──────────────────────────────────────────────────────
    if config.tts_backend == TtsBackendKind::Kokoro {
        println!();
        println!("Kokoro TTS:");
        println!("  Base URL: {}", config.tts_kokoro_base_url);
        println!(
            "  No local install step required; preflight will verify endpoint connectivity and usable voices."
        );
    }

    // ── Preflight ───────────────────────────────────────────────────
    if opts.skip_preflight {
        println!("Preflight: skipped (--skip-preflight).");
        println!();
        println!("Setup complete.");
        return ExitStatus::code(EXIT_SUCCESS);
    }

    println!();
    println!("Running preflight checks...");
    println!();
    let status = preflight::run_preflight(&config).await;
    if status.code != EXIT_SUCCESS {
        return status;
    }
    println!();
    println!("Setup complete.");
    ExitStatus::code(EXIT_SUCCESS)
}

async fn run_local_tts_setup(
    config: &mut Config,
    opts: &SetupOptions,
    ctx: &SetupContext,
) -> ExitStatus {
    println!();
    println!("Local TTS backend: local");

    let binary = find_piper_binary();
    if binary.is_none() {
        println!("[FAIL] Local Piper runtime");
        for hint in piper_install_hints() {
            println!("  * {hint}");
        }
        if opts.install_missing {
            println!("Automatic install requested for Local Piper.");
            let runner = Arc::clone(&ctx.runner);
            let ok = run_blocking_setup(move || {
                Ok(crate::setup::piper::attempt_piper_auto_install(
                    runner.as_ref(),
                ))
            })
            .await
            .unwrap_or(false);
            if ok {
                println!("Local Piper install: complete.");
            } else {
                println!("Local Piper install: failed or no supported installer available.");
            }
        }
        if find_piper_binary().is_none() {
            println!();
            println!("Setup incomplete: Local Piper runtime is still missing.");
            return ExitStatus::code(EXIT_DEPENDENCY);
        }
    }
    println!("[PASS] Local Piper runtime");

    let target_dir = opts
        .tts_local_model_dir
        .clone()
        .map(expand_user_path)
        .or_else(|| config.tts_local_model_path.as_ref().map(expand_user_path))
        .unwrap_or_else(managed_piper_model_dir);

    let artifacts_ok = validate_piper_voice_artifacts(
        &target_dir,
        config
            .tts_local_voice
            .as_deref()
            .or(Some(config.tts_default_voice_id.as_str())),
    )
    .is_ok();

    if opts.skip_model_download {
        println!("Local Piper voice download: skipped (--skip-model-download).");
        if !artifacts_ok {
            println!();
            println!("Setup incomplete: Local Piper voice artifacts are missing.");
            return ExitStatus::code(EXIT_DEPENDENCY);
        }
        return ExitStatus::code(EXIT_SUCCESS);
    }

    let needs_download = opts.tts_local_voice.is_some()
        || opts.tts_local_model_dir.is_some()
        || config.tts_local_model_path.is_none()
        || !artifacts_ok;

    if !needs_download {
        println!(
            "Local Piper voice download: skipped (configured voice artifacts already present)."
        );
        return ExitStatus::code(EXIT_SUCCESS);
    }

    let voice_id = opts.tts_local_voice.clone();
    let non_interactive = opts.non_interactive;
    let current_voice = config.tts_local_voice.clone();
    let default_voice = config.tts_default_voice_id.clone();
    let voice = match run_blocking_setup(move || {
        choose_voice_owned(voice_id, current_voice, default_voice, non_interactive)
    })
    .await
    {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ERROR: {e}");
            return ExitStatus::code(EXIT_FAILURE);
        }
    };
    println!("Selected Local Piper voice: {}", voice.label);
    println!("Managed voice directory: {}", target_dir.display());

    let mut progress = |frac: Option<f32>, msg: &str| {
        if let Some(f) = frac {
            println!("[{:3.0}%] {msg}", f * 100.0);
        } else {
            println!("{msg}");
        }
    };

    match ensure_local_piper_ready(
        voice,
        &target_dir,
        opts.install_missing,
        ctx.downloader.as_ref(),
        ctx.runner.as_ref(),
        &mut progress,
    )
    .await
    {
        Ok(result) if result.status == "ok" => {
            if let Err(e) =
                persist_local_tts_selection(config, &result.model_dir, &result.voice_stem)
            {
                eprintln!("ERROR: failed to persist TTS selection: {e}");
                return ExitStatus::code(EXIT_FAILURE);
            }
            println!("Local Piper setup: {}", result.message);
            ExitStatus::code(EXIT_SUCCESS)
        }
        Ok(result) if result.status == "skipped_missing_deps" => {
            println!("Local Piper setup: {}", result.message);
            println!();
            println!("Setup incomplete: Local Piper dependencies remain missing.");
            // Config must remain unchanged on failed Piper (no persist call).
            ExitStatus::code(EXIT_DEPENDENCY)
        }
        Ok(result) => {
            println!("Local Piper setup: {}", result.message);
            ExitStatus::code(EXIT_FAILURE)
        }
        Err(e) => {
            println!("Local Piper setup failed: {e}");
            ExitStatus::code(EXIT_FAILURE)
        }
    }
}

fn choose_voice_owned(
    explicit: Option<String>,
    current_voice: Option<String>,
    default_voice_id: String,
    non_interactive: bool,
) -> Result<&'static crate::setup::piper::PiperVoiceOption, String> {
    if let Some(id) = explicit {
        return get_curated_piper_voice(&id);
    }
    let current = current_voice
        .as_deref()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            let d = default_voice_id.as_str();
            if d.is_empty() { None } else { Some(d) }
        });
    if non_interactive {
        if let Some(id) = current
            && let Ok(v) = get_curated_piper_voice(id)
        {
            return Ok(v);
        }
        return Ok(recommended_piper_voice());
    }
    // Interactive: print list and read stdin (Enter = default). Runs off Tokio.
    let options = crate::setup::piper::CURATED_PIPER_VOICES;
    let default = current
        .and_then(|id| get_curated_piper_voice(id).ok())
        .unwrap_or_else(recommended_piper_voice);
    println!();
    println!("Choose a Local Piper voice:");
    for (i, opt) in options.iter().enumerate() {
        let marker = if opt.id == default.id {
            " (default)"
        } else {
            ""
        };
        println!("  {}. {}{marker}", i + 1, opt.label);
        println!("     {}", opt.description);
    }
    print!("Select voice [1-{}] (Enter for default): ", options.len());
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(_) => {
            let t = line.trim();
            if t.is_empty() {
                return Ok(default);
            }
            if let Ok(n) = t.parse::<usize>()
                && (1..=options.len()).contains(&n)
            {
                return Ok(&options[n - 1]);
            }
            Err("Invalid selection.".into())
        }
        Err(_) => Ok(default),
    }
}

async fn run_melotts_setup(config: &Config, opts: &SetupOptions, ctx: &SetupContext) -> ExitStatus {
    println!();
    println!("TTS backend: melotts");
    let venv = melotts_venv_dir(config);
    let mut missing = melotts_missing_dependencies(&venv);
    println!("{}", format_melotts_report(&venv, &missing));

    if !missing.is_empty() && opts.install_missing {
        let already_valid = melotts_venv_valid(&venv);
        println!("Automatic install requested for MeloTTS.");
        let runner = Arc::clone(&ctx.runner);
        let venv_c = venv.clone();
        match run_blocking_setup(move || {
            run_melotts_install(&venv_c, runner.as_ref(), already_valid)
        })
        .await
        {
            Ok(()) => {
                missing = melotts_missing_dependencies(&venv);
                println!("{}", format_melotts_report(&venv, &missing));
            }
            Err(e) => {
                println!("MeloTTS install failed: {e}");
                missing = melotts_missing_dependencies(&venv);
            }
        }
    }

    if !missing.is_empty() {
        println!();
        println!("[FAIL] MeloTTS backend");
        println!("Setup incomplete: MeloTTS venv is not ready.");
        println!("Install hints:");
        for cmd in melotts_install_commands(&venv) {
            println!("  $ {}", cmd.join(" "));
        }
        return ExitStatus::code(EXIT_DEPENDENCY);
    }
    println!();
    println!("[PASS] MeloTTS backend");
    ExitStatus::code(EXIT_SUCCESS)
}
