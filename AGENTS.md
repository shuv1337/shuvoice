# AGENTS.md — ShuVoice Developer & Agent Reference

> **Purpose**: Single source of truth for AI coding agents working in this
> repository. Covers backend configuration, model locations, build artifacts,
> prerequisites, and known gotchas.
>
> **Agents: read this file first** before changing ASR backends, runtime
> configuration, service setup, or dependency management.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Environment](#environment)
- [Service Management](#service-management)
- [Runtime Configuration](#runtime-configuration)
- [ASR Backends](#asr-backends)
  - [NeMo](#nemo-backend)
  - [Sherpa ONNX](#sherpa-onnx-backend)
  - [Moonshine](#moonshine-backend)
- [TTS Backends](#tts-backends)
  - [ElevenLabs](#elevenlabs-backend)
  - [OpenAI](#openai-backend)
  - [Local (Piper)](#local-piper)
  - [MeloTTS](#melotts)
  - [Kokoro](#kokoro-backend)
- [Model Locations](#model-locations)
- [Build Artifacts](#build-artifacts)
- [System Prerequisites](#system-prerequisites)
- [Complete Fresh Install (Clean Slate Test Workflow)](#complete-fresh-install-clean-slate-test-workflow)
- [Known Issues](#known-issues)
- [Maintaining This File](#maintaining-this-file)

---

## Project Overview

ShuVoice is a streaming speech-to-text overlay for Hyprland/Wayland with
pluggable ASR backends. The user holds push-to-talk, speaks, and transcribed
text is typed into the focused window via direct typing or clipboard paste
(depending on typing mode).

**Repo**: `git@github.com:shuv1337/shuvoice.git`  
**License**: MIT

---

## Environment

| Component | Value |
|---|---|
| OS | Linux (Wayland/Hyprland target) |
| Python (venv) | 3.12+ (recommended) |
| Python (system) | 3.12+ |
| Virtual env | `.venv/` |
| GPU | Optional (recommended for NeMo / Sherpa CUDA) |
| CUDA | CUDA 12.x-compatible runtime required by many prebuilt GPU wheels |
| Package manager | `uv` (Python), `pacman` (system) |

### Important version notes

- **PyTorch** on Arch is commonly provided by `python-pytorch-cuda`.
- If system CUDA is newer than wheel-linked CUDA, Sherpa CUDA may require
  CUDA compatibility libraries and patched RUNPATH (see
  [Sherpa GPU](#sherpa-gpu-cuda-support)).

---

## Service Management

ShuVoice is typically run as a **user systemd service**.

```bash
# User service unit location
~/.config/systemd/user/shuvoice.service

# Common commands
systemctl --user start shuvoice.service
systemctl --user stop shuvoice.service
systemctl --user restart shuvoice.service
systemctl --user status shuvoice.service
journalctl --user -u shuvoice.service -f
journalctl --user -u shuvoice.service --no-pager
```

### Service unit template

Matches `packaging/systemd/user/shuvoice.service`:

```ini
[Unit]
Description=ShuVoice speech-to-text overlay
After=graphical-session.target
BindsTo=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
# Packaged install path. For repo/venv workflows, override with:
#   systemctl --user edit shuvoice.service
#   [Service]
#   ExecStart=
#   ExecStart=%h/.venv/bin/shuvoice
ExecStart=/usr/bin/shuvoice
Restart=on-failure
RestartSec=2
RestartPreventExitStatus=78
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=graphical-session.target
```

### Wizard → service auto-restart

`shuvoice wizard` runs as a separate process from the long-running
`shuvoice.service`. After the wizard finishes successfully, the CLI
automatically restarts `shuvoice.service` if it is active, so config
changes (TTS voice, ASR backend, keybinds, etc.) take effect without a
manual restart.

- Implemented in `shuvoice/cli/commands/wizard.py::maybe_restart_running_service`.
- Called from `shuvoice/cli/commands/run.py::run_wizard_command` only when
  the wizard reports `completed = True`.
- No-op when the service is not active or `systemctl --user` is unavailable
  (keeps standalone/headless usage clean).
- On restart failure, prints an actionable warning with the manual
  `systemctl --user restart shuvoice.service` command.

### Wizard TTS defaults (provider / voice / speed)

The wizard TTS page persists the following defaults to `[tts]` in `config.toml`:

- `tts_backend` (ElevenLabs, OpenAI, Local Piper, MeloTTS, Kokoro)
- `tts_default_voice_id` (provider-specific voice)
- `tts_playback_speed` (default synthesis speed, 0.5×–2.0×)

The default playback speed is exposed as a dropdown on the TTS page with
presets: 0.75×, 1.0× (default), 1.25×, 1.5×, 1.75×, 2.0×. The selection is
also applied to the Kokoro “Speak sample” preview so you can audition the
exact speed you picked before finishing the wizard. All backends respect
`tts_playback_speed` at runtime — OpenAI and Kokoro use provider-native
`speed`, ElevenLabs/Local/MeloTTS use the player's resampling path.

- Presets and mapping helpers live in
  `shuvoice/wizard_state.py::TTS_PLAYBACK_SPEED_PRESETS` and
  `tts_playback_speed_preset_id`.
- Validation/normalization reuses `shuvoice/tts_speed.py::validate_tts_playback_speed`
  (0.5×–2.0×, rejects non-finite / non-numeric values).

---

## Runtime Configuration

**Config file**: `~/.config/shuvoice/config.toml`

**Automatic local env file**: `~/.config/shuvoice/local.dev`
- Loaded automatically on `shuvoice` CLI startup.
- Supports `KEY=value` and `export KEY=value` lines.
- Intended for local API keys (for example `ELEVENLABS_API_KEY` or `OPENAI_API_KEY`).
- Existing process environment variables take precedence by default.

Top-level schema marker: `config_version = 1` (legacy unversioned files are treated as v0 and auto-migrated).

Config sections map to `shuvoice/config.py::Config`:
`[audio]`, `[asr]`, `[overlay]`, `[control]`, `[tts]`, `[typing]`, `[streaming]`, `[feedback]`.
Nested table: `[typing.text_replacements]` for custom phrase corrections.

**Example config**: `examples/config.toml`.

### Switching backends

Set `[asr].asr_backend` and restart service/application.
Only keys for the active backend need to be present.

Optional low-latency profile: set `[asr].instant_mode = true`.
This applies backend-specific tuning at runtime:
- NeMo: forces `right_context = 0`
- Sherpa (streaming mode): caps `sherpa_chunk_ms` to `80`
- Sherpa (offline_instant mode): uses one-shot utterance decode on key release
- Moonshine: forces `moonshine_model_name = "moonshine/tiny"`, caps
  `moonshine_max_window_sec` to `3.0`, caps `moonshine_max_tokens` to `48`

### TTS trigger + selection behavior

- Primary command: `shuvoice control tts_speak`
- Recommended Hyprland bind: `SUPER + CTRL + S`
- Selection capture order: `wl-paste --primary --no-newline` first, then
  clipboard fallback (`wl-paste --no-newline`)
- STT and TTS are mutually exclusive at runtime (starting one stops the other)
- TTS overlay exposes runtime pause/resume, restart, stop, voice selection,
  and provider-backed speed controls (0.5×–2.0×)
- Changing speed while speaking restarts the current utterance from the beginning
  at the new synthesis speed

### Audio gain tuning (app-side)

Applied only for backends that do **not** request raw audio.

| Key | Default | Notes |
|---|---:|---|
| `auto_gain_target_peak` | `0.15` | Target RMS peak for utterance gain |
| `auto_gain_max` | `10.0` | Upper cap for utterance gain |
| `auto_gain_settle_chunks` | `2` | Speech chunks required before gain updates |
| `recording_preroll_ms` | `200` | Audio retained around push-to-talk start so very short prompts do not lose their first syllable during control/ASR reset latency |

### Typing text replacements

ShuVoice includes built-in brand corrections for common ASR variants of
`ShuVoice` and `Hyprland` (for example: `shove voice`, `shu voice`,
`show voice`, `hyper land`, `hyperland`, `high per land`).

Use `[typing.text_replacements]` to add or override replacements. Matches are
case-insensitive and applied to whole words/phrases only (longest first).
Empty values delete the matched word.

```toml
[typing.text_replacements]
"speech to text" = "speech-to-text"
"um" = ""
```

### Final text injection mode (clipboard vs direct)

| Key | Default | Notes |
|---|---:|---|
| `typing_final_injection_mode` | `auto` | `auto`, `clipboard`, `direct`. In `auto`, ShuVoice uses clipboard paste by default. On XWayland windows it prefers `xdotool`-driven `Ctrl+V`; otherwise it detects known clipboard watchers (`wl-paste --watch`, `wl-clip-persist`, `elephant`) and may switch to direct typing to avoid clipboard-history pollution/races. |
| `typing_text_case` | `default` | `default` or `lowercase`. `lowercase` forces final committed STT output to lowercase for informal conversation/chat workflows. |
| `typing_clipboard_settle_delay_ms` | `40` | Delay between `wl-copy` and simulated `Ctrl+V` in clipboard mode to reduce paste timing races. |
| `use_clipboard_for_final` | `true` (legacy) | Soft-deprecated compatibility flag. If `typing_final_injection_mode` is absent, this maps to `auto` (`true`, safer watcher-aware behavior) or `direct` (`false`). |
| `preserve_clipboard` | `false` | Capture/restore clipboard around final commit in clipboard mode; direct mode does not touch clipboard. |

### Debug overlay mode

| Key | Default | Notes |
|---|---:|---|
| `overlay_debug_mode` | `false` | Shows a multi-line diagnostics section under the normal STT caption overlay with live runtime state, audio/ASR counters, and recent in-process logs. |
| `overlay_debug_max_lines` | `12` | Maximum number of recent log lines rendered inside the debug overlay and returned by `shuvoice diagnostics`. |

Additional control/diagnostics surface:
- `shuvoice control debug_status` returns JSON with app state, audio/ASR stats, metrics, and recent logs.
- `shuvoice diagnostics` now includes `debug_status` alongside `status` and `metrics`.

---

## ASR Backends

### Gain / preprocessing behavior

| Backend | `wants_raw_audio` | App auto-gain | Notes |
|---|---|---|---|
| NeMo | `true` | Bypassed | Backend handles normalization |
| Moonshine | `true` | Bypassed | Backend handles normalization |
| Sherpa | `false` | Enabled | App-side gain helps quiet inputs |

### NeMo Backend

**Status**: ✅ Production-ready  
**Backend key**: `asr_backend = "nemo"`  
**Module**: `shuvoice/asr_nemo.py`

#### Config

```toml
[asr]
asr_backend = "nemo"
model_name = "nvidia/nemotron-speech-streaming-en-0.6b"
right_context = 13
device = "cuda"
use_cuda_graph_decoder = false
```

#### Config keys

| Key | Default | Notes |
|---|---|---|
| `model_name` | `nvidia/nemotron-speech-streaming-en-0.6b` | Hugging Face model ID |
| `right_context` | `13` | 0–13; higher accuracy, higher latency |
| `device` | `cuda` | `cuda` or `cpu` |
| `use_cuda_graph_decoder` | `false` | Keep false unless validated |

#### Dependencies

```bash
uv sync --extra asr-nemo
# or: uv add torch nemo-toolkit[asr]
```

#### Characteristics

- Low latency to first token
- Smooth incremental transcript updates
- Best quality among current backends

---

### Sherpa ONNX Backend

**Status**: ⚠️ Functional, CUDA support may require source build  
**Backend key**: `asr_backend = "sherpa"`  
**Module**: `shuvoice/asr_sherpa.py`

#### Config (CPU)

```toml
[asr]
asr_backend = "sherpa"
sherpa_model_name = "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
sherpa_model_dir = "/path/to/sherpa-model-dir"
sherpa_provider = "cpu"
sherpa_num_threads = 2
sherpa_chunk_ms = 100
```

#### Config (GPU)

```toml
[asr]
asr_backend = "sherpa"
sherpa_model_name = "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
sherpa_model_dir = "/path/to/sherpa-model-dir"
sherpa_provider = "cuda"
sherpa_num_threads = 2
sherpa_chunk_ms = 100
```

#### Config keys

| Key | Default | Notes |
|---|---:|---|
| `sherpa_model_name` | `sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` | Archive/model name used for auto-download when `sherpa_model_dir` is unset |
| `sherpa_model_dir` | *none* | If unset, ShuVoice auto-downloads `sherpa_model_name` to `~/.local/share/shuvoice/models/sherpa/<sherpa_model_name>/` |
| `sherpa_decode_mode` | `auto` | `auto`, `streaming`, or `offline_instant`. `auto` resolves to `offline_instant` for Parakeet + `instant_mode=true`, otherwise `streaming`. |
| `sherpa_enable_parakeet_streaming` | `false` | Safety gate for Parakeet streaming path. Must be `true` to allow Parakeet with `sherpa_decode_mode = "streaming"`. |
| `sherpa_provider` | `cpu` | `cpu` or `cuda` |
| `sherpa_num_threads` | `2` | CPU threads |
| `sherpa_chunk_ms` | `100` | Streaming chunk duration (ignored in `offline_instant` mode) |
| `sherpa_offline_max_utterance_sec` | `60.0` | Hard cap on per-utterance audio length passed to the offline_instant decoder. Audio over the cap is truncated to the trailing window (most recent N seconds) before decode. Prevents stuck PTT / runaway sessions from triggering CUDA or CPU OOMs. Set to `0` to disable. |

Parakeet TDT v3 note (Sherpa runtime):

```toml
[asr]
asr_backend = "sherpa"
sherpa_model_name = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
instant_mode = true
sherpa_decode_mode = "offline_instant"
```

Parakeet is supported via Sherpa offline instant mode by default.

To opt into Parakeet streaming, explicitly enable both:

```toml
[asr]
sherpa_decode_mode = "streaming"
sherpa_enable_parakeet_streaming = true
```

When enabled, ShuVoice initializes Sherpa online transducer with
`model_type="nemo_transducer"` for Parakeet models.

#### Model directory structure

```
tokens.txt
encoder.onnx  (or encoder*.onnx)
decoder.onnx  (or decoder*.onnx)
joiner.onnx   (or joiner*.onnx)
```

#### Dependencies

```bash
# Repo/venv workflows
uv sync --extra asr-sherpa
# or: uv add sherpa-onnx

# Arch/AUR packaged runtime (recommended provider)
yay -S --needed python-sherpa-onnx-bin
# provides=('python-sherpa-onnx') for shuvoice-git dependency resolution
```

`shuvoice setup --install-missing` default install behavior:
- On CUDA-detected hosts, prefers a CUDA-capable Sherpa path first
  (`python-sherpa-onnx` source provider before `python-sherpa-onnx-bin`).
- In venv workflows, prefers `uv pip install ...` and falls back to
  `python -m pip install ...`.
- For Sherpa CUDA in a venv, setup now also installs the required CUDA compat
  pip libs (`nvidia-*-cu12`), patches RUNPATH on the Sherpa runtime libs, and
  links exact sonames into `site-packages/sherpa_onnx/lib/` so CUDA hosts with
  newer system toolkits can still load the provider out of the box.

#### Sherpa GPU (CUDA) support

> ⚠️ Prebuilt wheels often target CUDA 12.x. If your system CUDA/toolkit stack
> does not provide compatible shared libs, GPU mode may fail without compat libs.

Typical rebuild flow:

```bash
cd $REPO_ROOT/build/sherpa-onnx
git checkout v<VERSION>
export SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_C_FLAGS=-Wno-error=format-security -DCMAKE_CXX_FLAGS=-Wno-error=format-security"
$REPO_ROOT/.venv/bin/python setup.py bdist_wheel
uv pip install dist/sherpa_onnx-*.whl --force-reinstall --no-deps
uv pip install --upgrade nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12

SHERPA_LIB="$REPO_ROOT/.venv/lib/python3.12/site-packages/sherpa_onnx/lib"
# Link exact CUDA sonames from site-packages/nvidia into $SHERPA_LIB, then patch RUNPATH:
ln -sfn ../../nvidia/cublas/lib/libcublasLt.so.12 "$SHERPA_LIB/libcublasLt.so.12"
ln -sfn ../../nvidia/cublas/lib/libcublas.so.12 "$SHERPA_LIB/libcublas.so.12"
ln -sfn ../../nvidia/cuda_runtime/lib/libcudart.so.12 "$SHERPA_LIB/libcudart.so.12"
ln -sfn ../../nvidia/cufft/lib/libcufft.so.11 "$SHERPA_LIB/libcufft.so.11"
ln -sfn ../../nvidia/curand/lib/libcurand.so.10 "$SHERPA_LIB/libcurand.so.10"
ln -sfn ../../nvidia/cudnn/lib/libcudnn.so.9 "$SHERPA_LIB/libcudnn.so.9"
patchelf --set-rpath '$ORIGIN' "$SHERPA_LIB/libonnxruntime_providers_cuda.so"
patchelf --set-rpath '$ORIGIN' "$SHERPA_LIB/libonnxruntime_providers_shared.so"
patchelf --set-rpath '$ORIGIN' "$SHERPA_LIB/libonnxruntime.so"
```

#### Characteristics

- Higher token-start latency than NeMo
- Can emit text in bursts
- Efficient CPU option when running on CPU

---

### Moonshine Backend

**Status**: ⚠️ Functional, lower quality than NeMo for many workloads  
**Backend key**: `asr_backend = "moonshine"`  
**Module**: `shuvoice/asr_moonshine.py`  
**Device**: CPU (ONNX runtime)

#### Config

```toml
[asr]
asr_backend = "moonshine"
moonshine_model_name = "moonshine/tiny"
moonshine_model_precision = "float"
moonshine_chunk_ms = 100
moonshine_max_window_sec = 5.0
moonshine_max_tokens = 64
moonshine_provider = "cpu"
moonshine_onnx_threads = 0
```

#### Performance expectations

| Config | Per-phrase avg | Best for |
|---|---|---|
| `moonshine/tiny` + CPU | ~3.4s | Short utterances (<5s) on CPU-only systems |
| `moonshine/tiny` + CUDA | ~0.5s (est.) | Interactive use with GPU |
| `moonshine/base` + CPU | ~7.9s | Not recommended for interactive use |

> Moonshine re-encodes the full audio buffer on every inference call.
> Keep `moonshine_max_window_sec` ≤ 5.0 to limit worst-case latency.
> The `moonshine/tiny` model is ~2.3× faster than `base` with similar
> accuracy for short phrases.

#### Config keys

| Key | Default | Notes |
|---|---|---|
| `moonshine_model_name` | `moonshine/tiny` | `moonshine/tiny` (fast) or `moonshine/base` (slower, slightly more accurate) |
| `moonshine_model_dir` | *none* | Optional local model path |
| `moonshine_model_precision` | `float` | ONNX precision |
| `moonshine_chunk_ms` | `100` | Chunk duration |
| `moonshine_max_window_sec` | `5.0` | Max audio window before reset |
| `moonshine_max_tokens` | `64` | Max generated tokens per window |
| `moonshine_provider` | `cpu` | `cpu` or `cuda` (requires onnxruntime with CUDAExecutionProvider) |
| `moonshine_onnx_threads` | `0` | ONNX intra-op threads; 0 = auto |

#### Dependencies

```bash
uv sync --extra asr-moonshine
# or: uv add useful-moonshine-onnx
```

---

## TTS Backends

### ElevenLabs Backend

**Status**: ✅ Production-ready (streaming path)  
**Backend key**: `tts_backend = "elevenlabs"`  
**Modules**: `shuvoice/tts_elevenlabs.py`, `shuvoice/tts_player.py`, `shuvoice/tts_overlay.py`

#### Config

```toml
[tts]
tts_enabled = true
tts_backend = "elevenlabs"
tts_default_voice_id = "zNsotODqUhvbJ5wMG7Ei"
tts_model_id = "eleven_flash_v2_5"
tts_api_key_env = "ELEVENLABS_API_KEY"
tts_output_format = "pcm_24000"
tts_max_chars = 5000
tts_request_timeout_sec = 30.0
tts_playback_speed = 1.0
```

#### Notes

- API key value is **env-only** (named by `tts_api_key_env`), never stored in config.
- `tts_speak` captures selected text using primary selection first, clipboard fallback second.
- `tts_playback_speed` controls the default synthesis speed (0.5×–2.0×).
- Runtime speed changes restart the current utterance from the beginning.
- Overlay namespace: `tts-overlay` (interactive controls, keyboard mode on-demand).

### OpenAI Backend

**Status**: ✅ Production-ready (raw PCM path)  
**Backend key**: `tts_backend = "openai"`  
**Modules**: `shuvoice/tts_openai.py`, `shuvoice/tts_player.py`, `shuvoice/tts_overlay.py`

#### Config

```toml
[tts]
tts_enabled = true
tts_backend = "openai"
tts_default_voice_id = "onyx"
tts_model_id = "gpt-4o-mini-tts"
tts_api_key_env = "OPENAI_API_KEY"
tts_output_format = "pcm_24000"
tts_max_chars = 5000
tts_request_timeout_sec = 30.0
tts_playback_speed = 1.0
```

#### Dependencies

```bash
uv sync --extra tts-openai
```

#### Notes

- API key value is **env-only** (named by `tts_api_key_env`), never stored in config.
- OpenAI defaults are auto-applied when `tts_backend = "openai"` and the stock ElevenLabs defaults are still present.
- Current ShuVoice playback path expects raw PCM output, so use `tts_output_format = "pcm_24000"`.
- OpenAI speed uses the provider-native `speed` request field (no player-side PCM resampling).

### Local (Piper)

**Status**: ⚠️ Experimental  
**Backend key**: `tts_backend = "local"`  
**Module**: `shuvoice/tts_local.py`

#### Config

```toml
[tts]
tts_backend = "local"
tts_default_voice_id = "default"                      # first discovered local .onnx model
# If you set tts_local_voice, Config normalizes tts_default_voice_id to that value.
tts_playback_speed = 1.0
# Faster ShuVoice speeds map to lower Piper --length-scale values.
tts_local_model_path = "~/.local/share/shuvoice/models/piper" # wizard/setup managed directory
# Manual mode also accepts a single .onnx file path or a directory of .onnx voices.
tts_local_voice = "en_US-amy-medium"                  # optional explicit voice/model stem
tts_local_device = 3                                   # optional output device hint
```

Piper `.onnx.json` sidecar files are used to detect the correct playback sample rate.
If no sidecar is present, ShuVoice falls back to `22050 Hz` for compatibility.

#### Dependencies

```bash
uv sync --extra tts-local
# runtime binary: `piper` or `piper-tts` in PATH
# Arch AUR package / setup automation target: piper-tts
```

#### Automated setup

ShuVoice now supports first-class Local Piper automation in both entrypoints:

```bash
# Wizard path
shuvoice wizard
# choose: Local Piper -> Automatic setup -> curated voice

# CLI path (when config has tts_backend = "local")
shuvoice setup --install-missing --tts-local-voice en_US-amy-medium --non-interactive
```

Automation uses the managed model directory:

```text
~/.local/share/shuvoice/models/piper/
```

Curated downloads are stored as:

```text
~/.local/share/shuvoice/models/piper/<voice-stem>.onnx
~/.local/share/shuvoice/models/piper/<voice-stem>.onnx.json
```

### MeloTTS

**Status**: ⚠️ Experimental  
**Backend key**: `tts_backend = "melotts"`  
**Modules**: `shuvoice/tts_melotts.py` (backend), `shuvoice/melo_helper.py` (subprocess helper)

#### Config

```toml
[tts]
tts_backend = "melotts"
tts_default_voice_id = "EN-US"
tts_model_id = "melotts"
tts_playback_speed = 1.0
tts_melotts_device = "auto"                            # auto | cpu | cuda
# tts_melotts_venv_path = "~/.local/share/shuvoice/melotts-venv"  # default; override if needed
```

#### Config keys

| Key | Default | Notes |
|---|---|---|
| `tts_melotts_device` | `auto` | `auto`, `cpu`, or `cuda`. Controls device used by the MeloTTS helper subprocess. |
| `tts_melotts_venv_path` | *none* (auto: `~/.local/share/shuvoice/melotts-venv/`) | Path to the isolated MeloTTS Python venv. If unset, uses the default managed location. |

#### Available voices

| Voice ID | Name | Model |
|---|---|---|
| `EN-US` | American English | EN_V2 |
| `EN-BR` | British English | EN_V2 |
| `EN-INDIA` | Indian English | EN_V2 |
| `EN-AU` | Australian English | EN_V2 |
| `EN-Newest` | Newest English | EN_NEWEST |

#### Dependencies (isolated venv)

MeloTTS runs in a **separate Python 3.12 venv** to avoid dependency conflicts.
All MeloTTS-specific imports live only in the helper script (`melo_helper.py`).

```bash
# Automated setup (recommended)
shuvoice setup --install-missing   # when tts_backend = "melotts"

# Manual setup
uv python install 3.12
uv venv --python 3.12 ~/.local/share/shuvoice/melotts-venv
~/.local/share/shuvoice/melotts-venv/bin/python -m pip install melotts
~/.local/share/shuvoice/melotts-venv/bin/python -m unidic download
```

#### Characteristics

- MIT-licensed local TTS (MyShell.ai / MIT)
- CPU real-time inference; optional CUDA acceleration
- 44100 Hz PCM int16 mono output
- 5 English voices across two models (EN_V2 and EN_NEWEST)
- Speed control via synthesis parameter (0.5×–2.0×)
- Non-streaming: entire utterance synthesized before playback
- ~9 GB venv footprint (mainly PyTorch); models ~200 MB each

### Kokoro Backend

**Status**: ✅ Production-ready (streaming PCM path)  
**Backend key**: `tts_backend = "kokoro"`  
**Module**: `shuvoice/tts_kokoro.py`

#### Config

```toml
[tts]
tts_enabled = true
tts_backend = "kokoro"
tts_default_voice_id = "af_heart"
tts_model_id = "kokoro"
tts_output_format = "pcm_24000"
tts_kokoro_base_url = "http://localhost:8880/v1"
tts_playback_speed = 1.0
```

#### Config keys

| Key | Default | Notes |
|---|---|---|
| `tts_kokoro_base_url` | `http://localhost:8880/v1` | Base URL for the Kokoro OpenAI-compatible API endpoint. |

#### Notes

- Local self-hosted TTS with an OpenAI-compatible API (no API key required).
- Uses `{base_url}/audio/speech` for synthesis and `{base_url}/audio/voices` for voice listing.
- Voice list is cached for 300 seconds to reduce requests to the local service.
- Speed control via provider-native `speed` request field (0.5×–2.0×).
- Supports both PCM and MP3 output formats.
- A dummy `Authorization: Bearer sk-local` header is sent (Kokoro accepts but ignores it).
- Wizard support is first-class: the TTS step exposes Kokoro backend selection, editable base URL, live voice-list fetching from `/audio/voices` with manual voice-ID fallback, and a `Speak sample` preview button for playback verification before finishing.
- `shuvoice setup` does not install Kokoro, but now prints the configured base URL and relies on `shuvoice preflight` to verify endpoint connectivity via the voice-list API.

---

## Model Locations

| Backend | Model | Location |
|---|---|---|
| NeMo | `nvidia/nemotron-speech-streaming-en-0.6b` | Hugging Face cache (`~/.cache/huggingface/...`) |
| Sherpa | `sherpa_model_name` (default `sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06`) | `~/.local/share/shuvoice/models/sherpa/<sherpa_model_name>/` or custom `sherpa_model_dir` |
| Moonshine | `UsefulSensors/moonshine` | Hugging Face cache (`~/.cache/huggingface/...`) |
| ElevenLabs TTS | `tts_default_voice_id` + `tts_model_id` | Remote API (`api.elevenlabs.io`); key in env (`tts_api_key_env`) |
| OpenAI TTS | `tts_default_voice_id` + `tts_model_id` | Remote API (`api.openai.com/v1/audio/speech`); key in env (`tts_api_key_env`) |
| Local TTS | `tts_local_model_path` / `tts_local_voice` | Local filesystem path (Piper `.onnx` model file(s)); managed automation target: `~/.local/share/shuvoice/models/piper/` |
| MeloTTS | `tts_default_voice_id` + `tts_model_id` | Isolated venv (`~/.local/share/shuvoice/melotts-venv/`); models cached in HuggingFace cache via helper subprocess |
| Kokoro | `tts_default_voice_id` + `tts_model_id` | Local API (`tts_kokoro_base_url`, default `http://localhost:8880/v1`); no API key required |

---

## Build Artifacts

```
build/
├── asr-models/
├── cuda12-compat/
└── sherpa-onnx/
```

---

## System Prerequisites

### Arch Linux packages (required)

```bash
sudo pacman -S \
  gtk4 gtk4-layer-shell python-gobject \
  python-pytorch-cuda \
  wtype wl-clipboard \
  portaudio pipewire pipewire-audio pipewire-alsa \
  cuda cudnn \
  patchelf
```

### Python virtual environment

```bash
uv sync --dev --extra asr-nemo --extra asr-sherpa --extra asr-moonshine
```

### Common dev tools

| Tool | Purpose | Install |
|---|---|---|
| `patchelf` | Patch RUNPATH for CUDA provider libs | `pacman -S patchelf` |
| `xdotool` | XWayland/X11 text injection fallback | `pacman -S xdotool` |
| `gh` | GitHub CLI | `pacman -S github-cli` |
| `uv` | Python package manager | [astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| `ruff` | Lint/format | managed by uv (`uv sync --dev`) |
| `pytest` | Tests | managed by uv (`uv sync --dev`) |

---

## Complete Fresh Install (Clean Slate Test Workflow)

Use this when you want a reproducible “from-scratch” test state (fresh venv,
fresh models, clean service restart).

> ⚠️ **Destructive**: this removes local model caches and recreates `.venv/`.

### 1) Stop services (ShuVoice + optional Waybar helper)

```bash
# Core ShuVoice service
systemctl --user stop shuvoice.service

# Optional: if you run a dedicated shuvoice-waybar user service, stop it too
systemctl --user stop shuvoice-waybar.service 2>/dev/null || true
```

If you want a fully clean slate for unit files too, disable/remove it:

```bash
systemctl --user disable --now shuvoice-waybar.service 2>/dev/null || true
rm -f ~/.config/systemd/user/shuvoice-waybar.service
systemctl --user daemon-reload
```

If the ShuVoice icon still appears in Waybar, it is usually configured as a
Waybar module (not a systemd service). Remove/disable the module and restart
Waybar:

```bash
# Find ShuVoice module references
rg -n "custom/shuvoice|shuvoice-waybar\.sh|shuvoice-waybar" ~/.config/waybar -S

# Edit ~/.config/waybar/config.jsonc:
#   1) remove "custom/shuvoice" from modules-left/center/right
#   2) remove the entire "custom/shuvoice": { ... } block

# Restart Waybar process
pkill -x waybar || true
nohup waybar >/tmp/waybar-restart.log 2>&1 &
```

### 2) Remove local model caches (ShuVoice + relevant HF repos)

```bash
rm -rf ~/.local/share/shuvoice/models
rm -rf ~/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b*
rm -rf ~/.cache/huggingface/hub/models--UsefulSensors--moonshine*
rm -rf ~/.cache/huggingface/hub/models--moonshine*
rm -rf ~/.cache/huggingface/hub/.locks/models--nvidia--nemotron-speech-streaming-en-0.6b*
rm -rf ~/.cache/huggingface/hub/.locks/models--UsefulSensors--moonshine*
```

### 3) Recreate venv + reinstall deps

Use Python 3.12 for best compatibility with current ASR wheels.

```bash
cd /path/to/shuvoice
rm -rf .venv
uv sync --python 3.12 --dev --extra asr-nemo --extra asr-sherpa --extra asr-moonshine
```

### 4) Sherpa runtime compatibility check/fix (if needed)

If `sherpa_onnx` import fails with errors like
`libonnxruntime.so: version 'VERS_1.23.2' not found`, copy a compatible
`libonnxruntime.so` into the venv Sherpa lib dir:

```bash
cp /usr/lib/python3.14/site-packages/sherpa_onnx/lib/libonnxruntime.so \
  .venv/lib/python3.12/site-packages/sherpa_onnx/lib/
```

Then verify:

```bash
.venv/bin/python -c "import sherpa_onnx; print('sherpa_onnx OK')"
```

### 5) Trigger model downloads via ShuVoice (end-user path)

This is the recommended QA flow for validating real user experience.
Do **not** pre-download models manually.

- **Wizard path (best for UX testing):**

```bash
uv run shuvoice wizard
```

Finish the wizard and keep model download enabled in the finish screen to
observe progress/cancel behavior.

- **Service path (lazy runtime download):**

```bash
systemctl --user start shuvoice.service
journalctl --user -u shuvoice.service -f
```

ShuVoice will download backend models lazily on first load when needed.

### 6) Restart + verify service

```bash
systemctl --user daemon-reload
systemctl --user restart shuvoice.service
systemctl --user status shuvoice.service --no-pager
uv run shuvoice preflight
journalctl --user -u shuvoice.service -n 80 --no-pager
```

### 7) Optional (CI/dev only): pre-warm model caches programmatically

Use this only when testing non-interactive setup speed, **not** end-user UX.

```bash
. .venv/bin/activate
python - <<'PY'
from pathlib import Path
import shutil

from shuvoice.asr import get_backend_class
from shuvoice.asr_moonshine import MoonshineBackend
from shuvoice.config import Config
from shuvoice.wizard_state import DEFAULT_SHERPA_MODEL_NAME, PARAKEET_TDT_V3_INT8_MODEL_NAME

model_root = Path.home() / '.local' / 'share' / 'shuvoice' / 'models'
model_root.mkdir(parents=True, exist_ok=True)

sherpa_cls = get_backend_class('sherpa')
for name in [DEFAULT_SHERPA_MODEL_NAME, PARAKEET_TDT_V3_INT8_MODEL_NAME]:
    target = model_root / 'sherpa' / name
    if target.exists():
        shutil.rmtree(target)
    sherpa_cls.download_model(model_name=name, model_dir=str(target))

get_backend_class('nemo').download_model('nvidia/nemotron-speech-streaming-en-0.6b')

for moon_model in ['moonshine/tiny', 'moonshine/base']:
    cfg = Config(asr_backend='moonshine', moonshine_model_name=moon_model, moonshine_provider='cpu')
    MoonshineBackend(cfg).load()
PY
```

### 8) Optional: reset wizard onboarding state too

Use this only when testing first-run UX:

```bash
rm -f ~/.config/shuvoice/config.toml
rm -f ~/.local/share/shuvoice/.wizard-done
uv run shuvoice wizard
```

## Known Issues

| Issue | Description | Status |
|---|---|---|
| [#7](https://github.com/shuv1337/shuvoice/issues/7) | Sherpa may drop trailing words on early key release | Open |
| [#12](https://github.com/shuv1337/shuvoice/issues/12) | Moonshine repetition guard misses some token/long-clause loops | Fixed |
| [#13](https://github.com/shuv1337/shuvoice/issues/13) | Moonshine throughput slower than NeMo/Sherpa | Mitigated (safer defaults, ONNX tuning, GPU provider) |
| — | `sherpa-onnx` source AUR builds may fail on GCC 15 due format-security warning flag interaction | Mitigation available (`python-sherpa-onnx-bin`, upstream patch staged) |
| — | On Linux/x86_64 Python 3.12 repo/venv workflows, plain PyPI `sherpa-onnx` wheels (observed: `1.12.38`, `1.12.39`) can fail at import time due ONNX runtime ABI mismatches, and the `1.12.25+cuda12.cudnn9` CUDA wheel omits a usable ONNX Runtime core in this env. Current mitigation pins `asr-sherpa` to the `1.12.25+cuda12.cudnn9` direct wheel **plus** `onnxruntime-gpu==1.23.2`, and `shuvoice/sherpa_cuda.py` now auto-copies the ORT GPU core/provider libs into `sherpa_onnx/lib`, patches RUNPATH, and creates the `libcublaslt.so.12` alias needed by ORT. This self-heals after `uv sync`, `shuvoice setup`, and service startup. | Mitigated in repo config + runtime repair |
| — | Upstream Sherpa source builds may silently fall back to CPU if they pick preinstalled CPU ONNX Runtime or if the upstream ONNX Runtime GPU download URL is unavailable (observed: hardcoded `hf-mirror.com` path returning 404 in v1.12.39 manual build). | Ongoing / upstream fragility |
| — | `glib2` 2.88 split `GLib.unix_signal_add` into `GLibUnix.signal_add`; older venv `PyGObject` builds (observed: `3.54.x`) may crash in GTK activation with `AttributeError: 'gi.repository.GLib' object has no attribute 'unix_signal_add'`. `shuvoice/app.py` now uses a compatibility shim that prefers `GLibUnix.signal_add` and falls back to the legacy name. | Mitigated in app |
| — | Parakeet streaming is behind explicit safety gate (`sherpa_enable_parakeet_streaming = true`) and requires online-compatible encoder metadata (`window_size`); incompatible models are blocked pre-start with actionable errors | By design |
| — | Sherpa CUDA can fail at decode time (not load time) when the GPU is full elsewhere (symptoms: `CUBLAS_STATUS_ALLOC_FAILED` / `CUDNN_STATUS_INTERNAL_ERROR`). Overlay used to stay blank and nothing was pasted. `shuvoice/app.py::_handle_asr_runtime_error` + `shuvoice/asr_sherpa.py::SherpaBackend.try_fallback_to_cpu` now detect CUDA-OOM-family errors and auto-swap the recognizer to CPU for the rest of the session, showing a transient overlay toast. All other ASR errors also flash a user-visible toast with the failure count (`N/10`). | Mitigated in app |
| — | A stuck push-to-talk (or any runaway recording) would cause the Sherpa offline_instant decoder to allocate activation buffers proportional to audio length — a 15-minute capture asked ORT for ~28 GiB on a 16 GB GPU. The error stringified as `"Failed to allocate memory for requested buffer"` / `"bfc_arena.cc"` wrapped by a `"MemcpyFromHost node"` error, which the original CUDA-OOM marker list did not match, so the auto CPU fallback never triggered. `_CUDA_OOM_ERROR_MARKERS` (`shuvoice/asr_sherpa.py`) now also includes allocation-specific ORT markers (`failed to allocate memory`, `bfc_arena`), and `process_utterance()` enforces `sherpa_offline_max_utterance_sec` (default `60.0s`) by truncating to the trailing window. | Mitigated in app |

---

## Maintaining This File

### ASR runtime-error UX (transient overlay + CUDA fallback)

ASR errors surface to the user via the STT overlay, not just logs.

- Per-call failures (`_process_chunk_safe`, `_process_utterance_safe`) call
  `_flash_overlay_error("⚠ ASR error (N/10) — see logs")`, which pins the
  overlay in the `error` state for ~5s and then auto-hides so subsequent
  utterances render normally.
- CUDA-OOM-family errors (`CUBLAS_STATUS_ALLOC_FAILED`,
  `CUDNN_STATUS_INTERNAL_ERROR`, `CUDA error: out of memory`) are classified
  by `shuvoice/asr_sherpa.py::looks_like_cuda_oom_error` and trigger
  `SherpaBackend.try_fallback_to_cpu()` — a one-shot session-wide swap that
  flips `sherpa_provider` to CPU and reloads the recognizer. The overlay
  shows `⚠ GPU busy — switched ASR to CPU for this session` and the failure
  does NOT advance the circuit breaker.
- The circuit breaker remains intact for persistent non-OOM failures
  (10 in a row disables ASR and shows `⚠ ASR error — will retry in 30s`).
- Backends without `try_fallback_to_cpu()` (NeMo, Moonshine) are untouched
  by this path.

### Optimal Sherpa configuration for memory-constrained hosts

When the host's GPU is heavily contended (e.g. running an llama-server, image
generation, or other ML workloads alongside ShuVoice), prefer **CPU** for the
Sherpa Parakeet path even if a GPU is available:

```toml
[asr]
asr_backend = "sherpa"
sherpa_provider = "cpu"
sherpa_num_threads = 4               # 2-4 is plenty for offline_instant; PTT bursts only
sherpa_model_name = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
instant_mode = true
sherpa_decode_mode = "offline_instant"
sherpa_offline_max_utterance_sec = 60.0
```

Why CPU is the better default in this profile:

- The `parakeet-tdt-0.6b-v3-int8` model is published as int8 specifically for
  CPU; RTF on a modern desktop CPU is well under realtime.
- ShuVoice is idle 99% of the time; PTT release is a short CPU burst, not a
  steady load — it shares the box gracefully with other workloads.
- GPU contention with other ML services (llama-server, etc.) routinely leads
  to ORT BFCArena allocation failures on the GPU EP that don't happen on CPU.
- CPU eliminates the CUDA dependency-management surface (compat libs, RUNPATH
  patching, ORT GPU/CUDA version skew) entirely for this code path.

GPU is still useful for the `nvidia/nemotron-speech-streaming-en-0.6b` NeMo
backend, where streaming throughput matters more.

### When to update AGENTS.md

Update when you:

1. Change backend config keys/defaults (`shuvoice/config.py`) — including `[tts]` keys.
2. Upgrade runtime dependencies (torch, nemo, sherpa-onnx, onnxruntime, CUDA, Piper/TTS tools).
3. Move/add model files or model download defaults (ASR or local TTS).
4. Change Sherpa GPU build/rebuild steps.
5. Add a backend (ASR or TTS).
6. Resolve/discover major issues.
7. Change required system packages.
8. Change Hyprland keybind defaults or control command conventions (`tts_*`, `start/stop`).

### How to update

- Keep heading structure stable.
- Use tables for structured config/version data.
- Include reproducible commands.
- Version-pin where breakage is likely.
- Document failure modes + diagnosis steps for fragile workarounds.

### Verification checklist after updates

```bash
# Config/dataclass sanity
rg -n "^\s+\w+:" shuvoice/config.py

# Model path checks (if using local build artifacts)
ls build/asr-models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06/tokens.txt

# Service startup sanity
systemctl --user restart shuvoice.service
journalctl --user -u shuvoice.service -n 50 --no-pager
```
