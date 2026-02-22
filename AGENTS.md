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
- [Model Locations](#model-locations)
- [Build Artifacts](#build-artifacts)
- [System Prerequisites](#system-prerequisites)
- [Known Issues](#known-issues)
- [Maintaining This File](#maintaining-this-file)

---

## Project Overview

ShuVoice is a streaming speech-to-text overlay for Hyprland/Wayland with
pluggable ASR backends. The user holds push-to-talk, speaks, and transcribed
text is typed into the focused window via clipboard injection.

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
| Package manager | `pacman` / distro equivalent |

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
Wants=graphical-session.target

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
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
```

---

## Runtime Configuration

**Config file**: `~/.config/shuvoice/config.toml`

Config sections map to `shuvoice/config.py::Config`:
`[audio]`, `[asr]`, `[overlay]`, `[hotkey]`, `[typing]`, `[streaming]`, `[feedback]`.
Nested table: `[typing.text_replacements]` for custom phrase corrections.

**Example config**: `examples/config.toml`.

### Switching backends

Set `[asr].asr_backend` and restart service/application.
Only keys for the active backend need to be present.

### Audio gain tuning (app-side)

Applied only for backends that do **not** request raw audio.

| Key | Default | Notes |
|---|---:|---|
| `auto_gain_target_peak` | `0.15` | Target RMS peak for utterance gain |
| `auto_gain_max` | `10.0` | Upper cap for utterance gain |
| `auto_gain_settle_chunks` | `2` | Speech chunks required before gain updates |

### Typing text replacements

Use `[typing.text_replacements]` to correct common ASR mistakes with exact
replacement text.  Matches are case-insensitive and applied to whole
words/phrases only (longest first).  Empty values delete the matched word.

```toml
[typing.text_replacements]
"shove voice" = "ShuVoice"
"speech to text" = "speech-to-text"
"hyper land" = "Hyprland"
"um" = ""
```

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
pip install -e ".[asr-nemo]"
# or: pip install torch nemo-toolkit[asr]
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
sherpa_model_dir = "/path/to/sherpa-model-dir"
sherpa_provider = "cpu"
sherpa_num_threads = 2
sherpa_chunk_ms = 100
```

#### Config (GPU)

```toml
[asr]
asr_backend = "sherpa"
sherpa_model_dir = "/path/to/sherpa-model-dir"
sherpa_provider = "cuda"
sherpa_num_threads = 2
sherpa_chunk_ms = 100
```

#### Config keys

| Key | Default | Notes |
|---|---:|---|
| `sherpa_model_dir` | *none* | If unset, ShuVoice auto-downloads default model to `~/.local/share/shuvoice/models/sherpa/` |
| `sherpa_provider` | `cpu` | `cpu` or `cuda` |
| `sherpa_num_threads` | `2` | CPU threads |
| `sherpa_chunk_ms` | `100` | Chunk duration |

#### Model directory structure

```
tokens.txt
encoder.onnx  (or encoder*.onnx)
decoder.onnx  (or decoder*.onnx)
joiner.onnx   (or joiner*.onnx)
```

#### Dependencies

```bash
pip install -e ".[asr-sherpa]"
# or: pip install sherpa-onnx
```

#### Sherpa GPU (CUDA) support

> ⚠️ Prebuilt wheels often target CUDA 12.x. If your system CUDA/toolkit stack
> does not provide compatible shared libs, GPU mode may fail without compat libs.

Typical rebuild flow:

```bash
cd $REPO_ROOT/build/sherpa-onnx
git checkout v<VERSION>
export SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=89"
$REPO_ROOT/.venv/bin/python setup.py bdist_wheel
pip install dist/sherpa_onnx-*.whl --force-reinstall --no-deps

SHERPA_LIB="$REPO_ROOT/.venv/lib/python3.12/site-packages/sherpa_onnx/lib"
# Copy required CUDA compat libs into $SHERPA_LIB, then patch RUNPATH:
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
moonshine_max_tokens = 128
```

#### Config keys

| Key | Default | Notes |
|---|---|---|
| `moonshine_model_name` | `moonshine/base` | `moonshine/tiny` or `moonshine/base` |
| `moonshine_model_dir` | *none* | Optional local model path |
| `moonshine_model_precision` | `float` | ONNX precision |
| `moonshine_chunk_ms` | `100` | Chunk duration |
| `moonshine_max_window_sec` | `5.0` | Max audio window before reset |
| `moonshine_max_tokens` | `128` | Max generated tokens per window |

#### Dependencies

```bash
pip install -e ".[asr-moonshine]"
# or: pip install useful-moonshine-onnx
```

---

## Model Locations

| Backend | Model | Location |
|---|---|---|
| NeMo | `nvidia/nemotron-speech-streaming-en-0.6b` | Hugging Face cache (`~/.cache/huggingface/...`) |
| Sherpa | `sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` | `build/asr-models/...` or auto-download cache |
| Moonshine | `UsefulSensors/moonshine` | Hugging Face cache (`~/.cache/huggingface/...`) |

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
uv venv .venv --python 3.12
source .venv/bin/activate
pip install -e ".[asr-nemo,asr-sherpa,asr-moonshine,dev]"
```

### Common dev tools

| Tool | Purpose | Install |
|---|---|---|
| `patchelf` | Patch RUNPATH for CUDA provider libs | `pacman -S patchelf` |
| `gh` | GitHub CLI | `pacman -S github-cli` |
| `uv` | Python package manager | `pip install uv` |
| `ruff` | Lint/format | `pip install ruff` |
| `pytest` | Tests | `pip install pytest` |

---

## Known Issues

| Issue | Description | Status |
|---|---|---|
| [#7](https://github.com/shuv1337/shuvoice/issues/7) | Sherpa may drop trailing words on early key release | Open |
| [#12](https://github.com/shuv1337/shuvoice/issues/12) | Moonshine repetition guard misses some token/long-clause loops | Open |
| [#13](https://github.com/shuv1337/shuvoice/issues/13) | Moonshine throughput slower than NeMo/Sherpa | Open |
| — | Prebuilt Sherpa CUDA wheels may be incompatible with newer CUDA stacks | Ongoing |

---

## Maintaining This File

### When to update AGENTS.md

Update when you:

1. Change backend config keys/defaults (`shuvoice/config.py`).
2. Upgrade runtime dependencies (torch, nemo, sherpa-onnx, onnxruntime, CUDA).
3. Move/add model files or model download defaults.
4. Change Sherpa GPU build/rebuild steps.
5. Add a backend.
6. Resolve/discover major issues.
7. Change required system packages.

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
