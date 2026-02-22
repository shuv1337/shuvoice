# AGENTS.md — ShuVoice Developer & Agent Reference

> **Purpose**: This file is the single source of truth for AI coding agents
> working on this codebase.  It documents backend configuration, model
> locations, build artifacts, system prerequisites, and known gotchas.
>
> **Agents: read this file first** before making changes to ASR backends,
> service configuration, or dependency management.

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

ShuVoice is a streaming speech-to-text overlay for Hyprland (Wayland) with
pluggable ASR backends.  The user holds a push-to-talk key, speaks, and
the transcribed text is typed into the focused window via clipboard injection.

**Repo**: `git@github.com:shuv1337/shuvoice.git`
**License**: MIT

---

## Environment

| Component        | Value                                                                 |
|------------------|-----------------------------------------------------------------------|
| Host OS          | Arch Linux (rolling), kernel 6.18.x                                   |
| Python (venv)    | 3.12.12 (`cpython-3.12.12-linux-x86_64-gnu` via `uv`)                |
| Python (system)  | 3.14.x                                                               |
| Virtual env      | `.venv312/` (Python 3.12 — **not** the system 3.14)                   |
| GPU              | NVIDIA GeForce RTX 5080 (SM 89, 16 GB VRAM)                          |
| CUDA toolkit     | 13.1 (system `/opt/cuda/`)                                           |
| cuDNN            | 9.19.0 (system `/usr/lib/`)                                          |
| NVIDIA driver    | 590.48.01                                                            |
| Package manager  | `pacman` + `yay` (AUR)                                               |
| Display server   | Hyprland (Wayland compositor)                                         |

### Important version notes

- **PyTorch** is the Arch `python-pytorch-cuda` system package (2.10.0,
  compiled against CUDA 12.8).  It is symlinked/available inside `.venv312`.
- The system CUDA is **13.1** but PyTorch and all pre-built wheels target
  **CUDA 12.x**.  This version gap affects sherpa-onnx GPU support
  (see [Sherpa GPU](#sherpa-gpu-cuda-support)).

---

## Service Management

ShuVoice runs as a **user systemd service**.

```bash
# Service unit location (user override, not the packaged default)
~/.config/systemd/user/shuvoice.service

# Common commands
systemctl --user start shuvoice.service
systemctl --user stop shuvoice.service
systemctl --user restart shuvoice.service
systemctl --user status shuvoice.service
journalctl --user -u shuvoice.service -f          # live logs
journalctl --user -u shuvoice.service --no-pager   # full log dump
```

### Service unit (current)

```ini
[Unit]
Description=ShuVoice speech-to-text overlay
After=graphical-session.target
Wants=graphical-session.target

[Service]
Type=simple
WorkingDirectory=/home/shuv/repos/shuvoice
ExecStart=/home/shuv/repos/shuvoice/.venv312/bin/shuvoice -v
Restart=on-failure
RestartSec=2
Environment=PYTHONUNBUFFERED=1
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/bin

[Install]
WantedBy=default.target
```

The `-v` flag enables DEBUG logging.  Remove it for production.

---

## Runtime Configuration

**Config file**: `~/.config/shuvoice/config.toml`

The config uses TOML with section headers `[audio]`, `[asr]`, `[overlay]`,
`[hotkey]`, `[typing]`, `[streaming]`, `[feedback]`.  All keys map to fields
in `shuvoice/config.py::Config`.

**Example config**: `examples/config.toml` (full reference with comments).

### Switching backends

To switch backends, edit the `[asr]` section in
`~/.config/shuvoice/config.toml` and restart the service.
Only the keys relevant to the active backend need to be present.

### Audio gain tuning (app-side)

These keys live in `[audio]` and control ShuVoice's per-utterance auto-gain.
They are applied only when the active backend does **not** request raw audio.

| Key                       | Default | Notes |
|---------------------------|---------|-------|
| `auto_gain_target_peak`   | `0.15`  | Target RMS peak used to compute utterance gain |
| `auto_gain_max`           | `10.0`  | Upper cap for utterance gain multiplier |
| `auto_gain_settle_chunks` | `2`     | Speech-level chunks required before gain updates |

---

## ASR Backends

### Gain / audio preprocessing behavior

| Backend | `wants_raw_audio` | App per-chunk auto-gain | Notes |
|---------|-------------------|-------------------------|-------|
| NeMo | `true` | Bypassed | NeMo preprocessor handles feature normalization |
| Moonshine | `true` | Bypassed | Moonshine applies its own buffer-level normalization |
| Sherpa | `false` | Enabled | Benefits from app-side gain on quieter inputs |

### NeMo Backend

**Status**: ✅ Production-ready, best quality
**Backend key**: `asr_backend = "nemo"`
**Module**: `shuvoice/asr_nemo.py`
**Device**: CUDA (GPU) — required for practical use

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

| Key                      | Default                                         | Notes |
|--------------------------|-------------------------------------------------|-------|
| `model_name`             | `nvidia/nemotron-speech-streaming-en-0.6b`      | HuggingFace model ID |
| `right_context`          | `13`                                            | 0–13; higher = better accuracy, more latency |
| `device`                 | `cuda`                                          | `cuda` or `cpu` |
| `use_cuda_graph_decoder` | `false`                                         | Keep false unless verified stable |

NeMo runs with `wants_raw_audio = true`, so app-side per-chunk auto-gain is
bypassed automatically.

#### Dependencies

```bash
pip install -e ".[asr-nemo]"
# or: pip install torch nemo-toolkit[asr]
```

On Arch, PyTorch comes from `python-pytorch-cuda` (system package).

#### Characteristics

- Low latency to first token (~200–400 ms)
- Smooth streaming — tokens emitted incrementally
- ~2.5 GB model download on first run (cached in HuggingFace hub)
- ~5 GB peak memory (GPU)
- Best overall accuracy of the three backends

---

### Sherpa ONNX Backend

**Status**: ⚠️ Functional, GPU support requires source build
**Backend key**: `asr_backend = "sherpa"`
**Module**: `shuvoice/asr_sherpa.py`
**Device**: CPU (default) or CUDA (requires custom wheel)

#### Config (CPU)

```toml
[asr]
asr_backend = "sherpa"
sherpa_model_dir = "/home/shuv/repos/shuvoice/build/asr-models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
sherpa_provider = "cpu"
sherpa_num_threads = 2
sherpa_chunk_ms = 100
```

#### Config (GPU)

```toml
[asr]
asr_backend = "sherpa"
sherpa_model_dir = "/home/shuv/repos/shuvoice/build/asr-models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
sherpa_provider = "cuda"
sherpa_num_threads = 2
sherpa_chunk_ms = 100
```

#### Config keys

| Key                 | Default | Notes |
|---------------------|---------|-------|
| `sherpa_model_dir`  | *none*  | **Required.** Path to streaming transducer model directory |
| `sherpa_provider`   | `cpu`   | `cpu` or `cuda` |
| `sherpa_num_threads`| `2`     | CPU threads for inference |
| `sherpa_chunk_ms`   | `100`   | Chunk duration in milliseconds |

#### Model directory structure

The model dir must contain:
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

> ⚠️ **Pre-built sherpa-onnx wheels (both PyPI CPU and k2-fsa CUDA) link
> against CUDA 12.x libraries.  This system runs CUDA 13.1, which does not
> provide `.so.12` symlinks.  GPU support requires a source-built wheel with
> CUDA 12 compat libraries.**

**Currently installed**: `sherpa-onnx 1.12.25+cuda` (source-built wheel)

The GPU setup involves three pieces:

1. **Source-built wheel** at `build/sherpa-onnx/dist/sherpa_onnx-1.12.25+cuda-cp312-cp312-linux_x86_64.whl`
   - Built with: `SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=89"`
   - The build downloads a pre-compiled `onnxruntime-gpu 1.23.2` which is
     linked against CUDA 12.

2. **CUDA 12 runtime compat libraries** copied into
   `.venv312/lib/python3.12/site-packages/sherpa_onnx/lib/`:
   - `libcublasLt.so.12` (from NVIDIA CUDA 12.9 redistributable)
   - `libcublas.so.12` (from NVIDIA CUDA 12.9 redistributable)
   - `libcudart.so.12` (from NVIDIA CUDA 12.9 redistributable)
   - `libcufft.so.11` (from NVIDIA CUDA 12.9 redistributable)
   - Source tarballs in `build/cuda12-compat/`

3. **`patchelf`** applied to set `RUNPATH=$ORIGIN` on the onnxruntime
   provider `.so` files so they find the compat libs in the same directory.

**To rebuild** (if sherpa-onnx is upgraded or the wheel is lost):
```bash
cd build/sherpa-onnx
git checkout v<VERSION>
export SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=89"
/home/shuv/repos/shuvoice/.venv312/bin/python setup.py bdist_wheel
pip install dist/sherpa_onnx-*.whl --force-reinstall --no-deps

# Copy CUDA 12 compat libs
SHERPA_LIB=".venv312/lib/python3.12/site-packages/sherpa_onnx/lib"
cp build/cuda12-compat/libcublas-*/lib/libcublasLt.so.12 "$SHERPA_LIB/"
cp build/cuda12-compat/libcublas-*/lib/libcublas.so.12   "$SHERPA_LIB/"
cp build/cuda12-compat/cuda_cudart-*/lib/libcudart.so.12 "$SHERPA_LIB/"
cp build/cuda12-compat/libcufft-*/lib/libcufft.so.11     "$SHERPA_LIB/"

# Patch RUNPATH
patchelf --set-rpath '$ORIGIN' "$SHERPA_LIB/libonnxruntime_providers_cuda.so"
patchelf --set-rpath '$ORIGIN' "$SHERPA_LIB/libonnxruntime_providers_shared.so"
patchelf --set-rpath '$ORIGIN' "$SHERPA_LIB/libonnxruntime.so"
```

**To verify GPU is active** (no fallback warning):
```bash
systemctl --user restart shuvoice.service
journalctl --user -u shuvoice.service -n 20 --no-pager | grep -E "Fallback|Ready|ERROR"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv | grep shuvoice
```

#### Characteristics

- High latency to first token (~1.5 s) — model buffers heavily
- Burst emission — tokens appear in large jumps, not smoothly
- Lightweight CPU footprint when running on CPU
- ~330 MiB GPU memory when running on CUDA
- Known issue: trailing words dropped on early key release
  (see [#7](https://github.com/shuv1337/shuvoice/issues/7),
  `ISSUE-sherpa-tail-flush.md`)

---

### Moonshine Backend

**Status**: ⚠️ Functional, lower accuracy than NeMo
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

| Key                        | Default            | Notes |
|----------------------------|--------------------|-------|
| `moonshine_model_name`     | `moonshine/base`   | `moonshine/tiny` or `moonshine/base` |
| `moonshine_model_dir`      | *none*             | Optional local model path |
| `moonshine_model_precision`| `float`            | ONNX precision |
| `moonshine_chunk_ms`       | `100`              | Chunk duration |
| `moonshine_max_window_sec` | `5.0`              | Max audio window before reset |
| `moonshine_max_tokens`     | `128`              | Max generated tokens per window |

#### Dependencies

```bash
pip install -e ".[asr-moonshine]"
# or: pip install useful-moonshine-onnx
```

Models are downloaded automatically from HuggingFace on first use.

#### Characteristics

- CPU-only backend (ONNX runtime)
- Fast startup, but significantly slower than NeMo (CUDA) and Sherpa
- Best suited for short utterances (typically <5 seconds)
- Lower accuracy than NeMo, especially for technical terms
- Good fallback when GPU ASR is unavailable

---

## Model Locations

| Backend    | Model                                              | Location |
|------------|-----------------------------------------------------|----------|
| NeMo       | `nvidia/nemotron-speech-streaming-en-0.6b`          | `~/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/` (2.5 GB) |
| Sherpa     | `sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` | `build/asr-models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06/` |
| Moonshine  | `UsefulSensors/moonshine` (tiny + base)             | `~/.cache/huggingface/hub/models--UsefulSensors--moonshine/` |

---

## Build Artifacts

```
build/
├── asr-models/
│   ├── sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06/   # Sherpa model
│   └── sherpa-kroko.tar.bz2                                   # Model archive
├── cuda12-compat/                                              # CUDA 12 runtime libs
│   ├── cublas.tar.xz
│   ├── cudart.tar.xz
│   ├── cufft.tar.xz
│   └── *-archive/lib/                                         # Extracted .so files
└── sherpa-onnx/                                                # Source build
    ├── dist/sherpa_onnx-1.12.25+cuda-*.whl                    # Built GPU wheel
    └── build/                                                  # CMake build tree
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
# The venv uses uv-managed Python 3.12, NOT the system Python 3.14
# To recreate:
uv venv .venv312 --python 3.12
source .venv312/bin/activate
pip install -e ".[asr-nemo,asr-sherpa,asr-moonshine,dev]"
```

### Tools used by the build/test workflow

| Tool        | Purpose                                     | Install |
|-------------|---------------------------------------------|---------|
| `patchelf`  | Patch RUNPATH on sherpa CUDA .so files       | `pacman -S patchelf` |
| `gh`        | GitHub CLI for issues/PRs                    | `pacman -S github-cli` |
| `uv`        | Fast Python package manager                  | `pip install uv` or `cargo install uv` |
| `ruff`      | Linter/formatter                             | `pip install ruff` |
| `pytest`    | Test runner                                  | `pip install pytest` |

---

## Known Issues

| Issue | Description | Status |
|-------|-------------|--------|
| [#7](https://github.com/shuv1337/shuvoice/issues/7) | Sherpa drops trailing words on early key release | Open — tail flush partially working |
| [#12](https://github.com/shuv1337/shuvoice/issues/12) | Moonshine repetition guard misses token/long-clause loops | Fixes implemented locally in `PLAN-12-13-moonshine-fixes.md`; benchmark validation pending |
| [#13](https://github.com/shuv1337/shuvoice/issues/13) | Moonshine base/tiny throughput is significantly slower than NeMo/Sherpa | Default tuning + throttle changes implemented; benchmark validation pending |
| — | Sherpa GPU requires source-built wheel + CUDA 12 compat libs | Documented above; `PLAN-sherpa-gpu-enable.md` |
| — | Pre-built sherpa-onnx CUDA wheels incompatible with CUDA 13.1 | Permanent until upstream adds CUDA 13 support |

---

## Maintaining This File

### When to update AGENTS.md

Update this file whenever you:

1. **Change a backend config** — add/remove/rename config keys, change
   defaults, or alter the `Config` dataclass in `shuvoice/config.py`.
2. **Upgrade a dependency** — update version numbers in the tables
   (sherpa-onnx, torch, nemo, onnxruntime, CUDA).
3. **Move or add model files** — update the [Model Locations](#model-locations)
   table.
4. **Change the build process** — especially the sherpa GPU build steps.
   If you rebuild sherpa-onnx against a new version, update the version,
   CMake flags, and compat lib paths.
5. **Add a new backend** — add a full section following the pattern of the
   existing three backends.
6. **Resolve or discover issues** — update the [Known Issues](#known-issues)
   table.
7. **Change system packages** — if new pacman packages are needed, add them.

### How to update

- Keep the same structure and heading hierarchy.
- Use tables for structured data (config keys, versions, paths).
- Include the exact commands needed to reproduce a setup step.
- Version-pin where it matters (wheel filenames, CUDA versions).
- If a workaround is fragile (e.g., CUDA compat libs), document the failure
  mode and how to diagnose it.

### Verification checklist after updates

```bash
# Confirm config keys match the dataclass
grep -c "^\s\+\w\+:" shuvoice/config.py   # count Config fields
grep -c '`' AGENTS.md | head -1            # sanity check formatting

# Confirm model paths exist
ls build/asr-models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06/tokens.txt
ls ~/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/
ls ~/.cache/huggingface/hub/models--UsefulSensors--moonshine/

# Confirm service starts with each backend
# (edit ~/.config/shuvoice/config.toml, restart, check logs)
```
