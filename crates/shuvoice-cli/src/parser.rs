//! Clap CLI tree preserving ShuVoice's public command contract.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::config::RuntimeOverrides;
use crate::control::ControlCmd;

pub const LEGACY_FLAG_WARNING: &str =
    "This legacy flag is deprecated and will be removed in a future release.";

#[derive(Debug, Clone, Parser)]
#[command(
    name = "shuvoice",
    about = "Streaming speech-to-text overlay for Hyprland",
    disable_version_flag = true
)]
pub struct Cli {
    /// Enable debug logging
    #[arg(short = 'v', long = "verbose", global = true)]
    pub verbose: bool,

    /// [legacy] Equivalent to `shuvoice model download`
    #[arg(long = "download-model", help = format!("[legacy] Equivalent to `shuvoice model download`. {LEGACY_FLAG_WARNING}"))]
    pub download_model: bool,

    /// [legacy] Equivalent to `shuvoice preflight`
    #[arg(long = "preflight", help = format!("[legacy] Equivalent to `shuvoice preflight`. {LEGACY_FLAG_WARNING}"))]
    pub legacy_preflight: bool,

    /// [legacy] Equivalent to `shuvoice audio list-devices`
    #[arg(long = "list-audio-devices", help = format!("[legacy] Equivalent to `shuvoice audio list-devices`. {LEGACY_FLAG_WARNING}"))]
    pub list_audio_devices: bool,

    /// [legacy] Equivalent to `shuvoice wizard`
    #[arg(long = "wizard", help = format!("[legacy] Equivalent to `shuvoice wizard`. {LEGACY_FLAG_WARNING}"))]
    pub legacy_wizard: bool,

    /// [legacy] Equivalent to `shuvoice control <cmd>`
    #[arg(long = "control", value_enum, help = format!("[legacy] Equivalent to `shuvoice control <cmd>`. {LEGACY_FLAG_WARNING}"))]
    pub legacy_control: Option<ControlCmd>,

    /// When sending stop/toggle, wait up to this many seconds for post-stop processing to finish (0 disables wait).
    #[arg(long = "control-wait-sec", default_value_t = 2.0)]
    pub control_wait_sec: f64,

    #[command(flatten)]
    pub overrides: RuntimeOverrideArgs,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Clone, Args, Default)]
pub struct RuntimeOverrideArgs {
    #[arg(long = "asr-backend", value_parser = ["nemo", "sherpa", "moonshine", "openai_realtime"])]
    pub asr_backend: Option<String>,
    #[arg(long = "device")]
    pub device: Option<String>,
    #[arg(long = "right-context", value_parser = parse_right_context)]
    pub right_context: Option<i64>,
    #[arg(long = "sherpa-model-dir")]
    pub sherpa_model_dir: Option<String>,
    #[arg(long = "sherpa-model-name")]
    pub sherpa_model_name: Option<String>,
    #[arg(long = "sherpa-provider", value_parser = ["cpu", "cuda"])]
    pub sherpa_provider: Option<String>,
    #[arg(long = "sherpa-num-threads")]
    pub sherpa_num_threads: Option<i64>,
    #[arg(long = "sherpa-chunk-ms")]
    pub sherpa_chunk_ms: Option<i64>,
    #[arg(long = "moonshine-model-name")]
    pub moonshine_model_name: Option<String>,
    #[arg(long = "moonshine-model-dir")]
    pub moonshine_model_dir: Option<String>,
    #[arg(long = "moonshine-model-precision")]
    pub moonshine_model_precision: Option<String>,
    #[arg(long = "moonshine-chunk-ms")]
    pub moonshine_chunk_ms: Option<i64>,
    #[arg(long = "moonshine-max-window-sec")]
    pub moonshine_max_window_sec: Option<f64>,
    #[arg(long = "moonshine-max-tokens")]
    pub moonshine_max_tokens: Option<i64>,
    #[arg(long = "moonshine-provider", value_parser = ["cpu", "cuda"])]
    pub moonshine_provider: Option<String>,
    #[arg(long = "moonshine-onnx-threads")]
    pub moonshine_onnx_threads: Option<i64>,
    #[arg(long = "audio-device")]
    pub audio_device: Option<String>,
    #[arg(long = "input-gain")]
    pub input_gain: Option<f64>,
    #[arg(long = "output-mode", value_parser = ["final_only", "streaming_partial"])]
    pub output_mode: Option<String>,
    #[arg(long = "control-socket")]
    pub control_socket: Option<String>,
}

fn parse_right_context(s: &str) -> Result<i64, String> {
    let v: i64 = s.parse().map_err(|e| format!("{e}"))?;
    if matches!(v, 0 | 1 | 6 | 13) {
        Ok(v)
    } else {
        Err("right-context must be one of: 0, 1, 6, 13".into())
    }
}

impl RuntimeOverrideArgs {
    pub fn to_overrides(&self) -> RuntimeOverrides {
        RuntimeOverrides {
            asr_backend: self.asr_backend.clone(),
            device: self.device.clone(),
            right_context: self.right_context,
            sherpa_model_dir: self.sherpa_model_dir.clone(),
            sherpa_model_name: self.sherpa_model_name.clone(),
            sherpa_provider: self.sherpa_provider.clone(),
            sherpa_num_threads: self.sherpa_num_threads,
            sherpa_chunk_ms: self.sherpa_chunk_ms,
            moonshine_model_name: self.moonshine_model_name.clone(),
            moonshine_model_dir: self.moonshine_model_dir.clone(),
            moonshine_model_precision: self.moonshine_model_precision.clone(),
            moonshine_chunk_ms: self.moonshine_chunk_ms,
            moonshine_max_window_sec: self.moonshine_max_window_sec,
            moonshine_max_tokens: self.moonshine_max_tokens,
            moonshine_provider: self.moonshine_provider.clone(),
            moonshine_onnx_threads: self.moonshine_onnx_threads,
            audio_device: self.audio_device.clone(),
            input_gain: self.input_gain,
            output_mode: self.output_mode.clone(),
            control_socket: self.control_socket.clone(),
        }
    }

    pub fn merge_into(&self, base: &mut RuntimeOverrides) {
        let o = self.to_overrides();
        if o.asr_backend.is_some() {
            base.asr_backend = o.asr_backend;
        }
        if o.device.is_some() {
            base.device = o.device;
        }
        if o.right_context.is_some() {
            base.right_context = o.right_context;
        }
        if o.sherpa_model_dir.is_some() {
            base.sherpa_model_dir = o.sherpa_model_dir;
        }
        if o.sherpa_model_name.is_some() {
            base.sherpa_model_name = o.sherpa_model_name;
        }
        if o.sherpa_provider.is_some() {
            base.sherpa_provider = o.sherpa_provider;
        }
        if o.sherpa_num_threads.is_some() {
            base.sherpa_num_threads = o.sherpa_num_threads;
        }
        if o.sherpa_chunk_ms.is_some() {
            base.sherpa_chunk_ms = o.sherpa_chunk_ms;
        }
        if o.moonshine_model_name.is_some() {
            base.moonshine_model_name = o.moonshine_model_name;
        }
        if o.moonshine_model_dir.is_some() {
            base.moonshine_model_dir = o.moonshine_model_dir;
        }
        if o.moonshine_model_precision.is_some() {
            base.moonshine_model_precision = o.moonshine_model_precision;
        }
        if o.moonshine_chunk_ms.is_some() {
            base.moonshine_chunk_ms = o.moonshine_chunk_ms;
        }
        if o.moonshine_max_window_sec.is_some() {
            base.moonshine_max_window_sec = o.moonshine_max_window_sec;
        }
        if o.moonshine_max_tokens.is_some() {
            base.moonshine_max_tokens = o.moonshine_max_tokens;
        }
        if o.moonshine_provider.is_some() {
            base.moonshine_provider = o.moonshine_provider;
        }
        if o.moonshine_onnx_threads.is_some() {
            base.moonshine_onnx_threads = o.moonshine_onnx_threads;
        }
        if o.audio_device.is_some() {
            base.audio_device = o.audio_device;
        }
        if o.input_gain.is_some() {
            base.input_gain = o.input_gain;
        }
        if o.output_mode.is_some() {
            base.output_mode = o.output_mode;
        }
        if o.control_socket.is_some() {
            base.control_socket = o.control_socket;
        }
    }
}

#[derive(Debug, Clone, Subcommand)]
pub enum Command {
    /// Run the speech-to-text overlay
    Run {
        #[command(flatten)]
        overrides: RuntimeOverrideArgs,
    },
    /// Send control command to running instance
    Control {
        #[arg(value_enum)]
        control_command: ControlCmd,
        #[arg(long = "control-wait-sec", default_value_t = 2.0)]
        control_wait_sec: f64,
        #[arg(long = "control-socket")]
        control_socket: Option<String>,
    },
    /// Run dependency and runtime checks
    Preflight {
        #[command(flatten)]
        overrides: RuntimeOverrideArgs,
    },
    /// Bootstrap backend dependencies, model artifacts, and preflight checks
    Setup {
        #[command(flatten)]
        overrides: RuntimeOverrideArgs,
        #[arg(long = "install-missing")]
        install_missing: bool,
        #[arg(long = "skip-model-download")]
        skip_model_download: bool,
        #[arg(long = "skip-preflight")]
        skip_preflight: bool,
        #[arg(long = "tts-local-voice")]
        tts_local_voice: Option<String>,
        #[arg(long = "tts-local-model-dir")]
        tts_local_model_dir: Option<PathBuf>,
        #[arg(long = "non-interactive")]
        non_interactive: bool,
    },
    /// Launch the setup wizard
    Wizard,
    /// Inspect and validate config
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Model management commands
    Model {
        #[command(flatten)]
        overrides: RuntimeOverrideArgs,
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Audio utility commands
    Audio {
        #[command(subcommand)]
        command: AudioCommand,
    },
    /// Show runtime diagnostics
    Diagnostics {
        #[command(flatten)]
        overrides: RuntimeOverrideArgs,
        #[arg(long = "json")]
        json: bool,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum ConfigCommand {
    /// Print merged effective config
    Effective,
    /// Print active config file path
    Path,
    /// Validate active config
    Validate,
    /// Set supported config keys
    Set {
        #[arg(value_enum)]
        key: ConfigSetKey,
        #[arg(value_enum)]
        value: ConfigSetValue,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ConfigSetKey {
    #[value(name = "typing_final_injection_mode")]
    TypingFinalInjectionMode,
    #[value(name = "typing_text_case")]
    TypingTextCase,
    #[value(name = "overlay_debug_mode")]
    OverlayDebugMode,
}

impl ConfigSetKey {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TypingFinalInjectionMode => "typing_final_injection_mode",
            Self::TypingTextCase => "typing_text_case",
            Self::OverlayDebugMode => "overlay_debug_mode",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ConfigSetValue {
    Auto,
    Clipboard,
    Direct,
    Default,
    Lowercase,
    True,
    False,
}

impl ConfigSetValue {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Clipboard => "clipboard",
            Self::Direct => "direct",
            Self::Default => "default",
            Self::Lowercase => "lowercase",
            Self::True => "true",
            Self::False => "false",
        }
    }
}

#[derive(Debug, Clone, Subcommand)]
pub enum ModelCommand {
    /// Download model artifacts for active backend
    Download,
}

#[derive(Debug, Clone, Subcommand)]
pub enum AudioCommand {
    /// List audio input devices
    #[command(name = "list-devices")]
    ListDevices,
}

/// Resolved internal route after legacy flag mapping.
#[derive(Debug, Clone)]
pub enum ResolvedCommand {
    Run {
        overrides: RuntimeOverrides,
    },
    Control {
        command: ControlCmd,
        wait_sec: f64,
        socket: Option<String>,
        overrides: RuntimeOverrides,
    },
    Preflight {
        overrides: RuntimeOverrides,
    },
    Setup {
        overrides: RuntimeOverrides,
        install_missing: bool,
        skip_model_download: bool,
        skip_preflight: bool,
        tts_local_voice: Option<String>,
        tts_local_model_dir: Option<PathBuf>,
        non_interactive: bool,
    },
    Wizard,
    ConfigPath,
    ConfigValidate,
    ConfigEffective,
    ConfigSet {
        key: ConfigSetKey,
        value: ConfigSetValue,
    },
    ModelDownload {
        overrides: RuntimeOverrides,
    },
    AudioListDevices,
    Diagnostics {
        overrides: RuntimeOverrides,
        json: bool,
    },
}

pub fn resolve_command(cli: &Cli) -> Result<(ResolvedCommand, Vec<String>), String> {
    let mut warnings = Vec::new();
    let mut root_overrides = cli.overrides.to_overrides();

    if let Some(command) = &cli.command {
        match command {
            Command::Run { overrides } => {
                overrides.merge_into(&mut root_overrides);
                return Ok((
                    ResolvedCommand::Run {
                        overrides: root_overrides,
                    },
                    warnings,
                ));
            }
            Command::Control {
                control_command,
                control_wait_sec,
                control_socket,
            } => {
                if let Some(sock) = control_socket {
                    root_overrides.control_socket = Some(sock.clone());
                }
                return Ok((
                    ResolvedCommand::Control {
                        command: *control_command,
                        wait_sec: *control_wait_sec,
                        socket: control_socket
                            .clone()
                            .or(root_overrides.control_socket.clone()),
                        overrides: root_overrides,
                    },
                    warnings,
                ));
            }
            Command::Preflight { overrides } => {
                overrides.merge_into(&mut root_overrides);
                return Ok((
                    ResolvedCommand::Preflight {
                        overrides: root_overrides,
                    },
                    warnings,
                ));
            }
            Command::Setup {
                overrides,
                install_missing,
                skip_model_download,
                skip_preflight,
                tts_local_voice,
                tts_local_model_dir,
                non_interactive,
            } => {
                overrides.merge_into(&mut root_overrides);
                return Ok((
                    ResolvedCommand::Setup {
                        overrides: root_overrides,
                        install_missing: *install_missing,
                        skip_model_download: *skip_model_download,
                        skip_preflight: *skip_preflight,
                        tts_local_voice: tts_local_voice.clone(),
                        tts_local_model_dir: tts_local_model_dir.clone(),
                        non_interactive: *non_interactive,
                    },
                    warnings,
                ));
            }
            Command::Wizard => return Ok((ResolvedCommand::Wizard, warnings)),
            Command::Config { command } => {
                let resolved = match command {
                    ConfigCommand::Effective => ResolvedCommand::ConfigEffective,
                    ConfigCommand::Path => ResolvedCommand::ConfigPath,
                    ConfigCommand::Validate => ResolvedCommand::ConfigValidate,
                    ConfigCommand::Set { key, value } => ResolvedCommand::ConfigSet {
                        key: *key,
                        value: *value,
                    },
                };
                return Ok((resolved, warnings));
            }
            Command::Model {
                overrides,
                command: ModelCommand::Download,
            } => {
                overrides.merge_into(&mut root_overrides);
                return Ok((
                    ResolvedCommand::ModelDownload {
                        overrides: root_overrides,
                    },
                    warnings,
                ));
            }
            Command::Audio {
                command: AudioCommand::ListDevices,
            } => return Ok((ResolvedCommand::AudioListDevices, warnings)),
            Command::Diagnostics { overrides, json } => {
                overrides.merge_into(&mut root_overrides);
                return Ok((
                    ResolvedCommand::Diagnostics {
                        overrides: root_overrides,
                        json: *json,
                    },
                    warnings,
                ));
            }
        }
    }

    // Legacy path.
    let legacy_flags = [
        cli.legacy_preflight,
        cli.legacy_wizard,
        cli.download_model,
        cli.list_audio_devices,
        cli.legacy_control.is_some(),
    ];
    if legacy_flags.iter().filter(|v| **v).count() > 1 {
        return Err("legacy flags are mutually exclusive".into());
    }

    if let Some(cmd) = cli.legacy_control {
        warnings.push("`--control` is deprecated; use `shuvoice control <cmd>`".into());
        return Ok((
            ResolvedCommand::Control {
                command: cmd,
                wait_sec: cli.control_wait_sec,
                socket: root_overrides.control_socket.clone(),
                overrides: root_overrides,
            },
            warnings,
        ));
    }
    if cli.legacy_preflight {
        warnings.push("`--preflight` is deprecated; use `shuvoice preflight`".into());
        return Ok((
            ResolvedCommand::Preflight {
                overrides: root_overrides,
            },
            warnings,
        ));
    }
    if cli.legacy_wizard {
        warnings.push("`--wizard` is deprecated; use `shuvoice wizard`".into());
        return Ok((ResolvedCommand::Wizard, warnings));
    }
    if cli.download_model {
        warnings.push("`--download-model` is deprecated; use `shuvoice model download`".into());
        return Ok((
            ResolvedCommand::ModelDownload {
                overrides: root_overrides,
            },
            warnings,
        ));
    }
    if cli.list_audio_devices {
        warnings
            .push("`--list-audio-devices` is deprecated; use `shuvoice audio list-devices`".into());
        return Ok((ResolvedCommand::AudioListDevices, warnings));
    }

    Ok((
        ResolvedCommand::Run {
            overrides: root_overrides,
        },
        warnings,
    ))
}
