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

| Component       | Value                                                             |
| --------------- | ----------------------------------------------------------------- |
| OS              | Linux (Wayland/Hyprland target)                                   |
| Python (venv)   | 3.12+ (recommended)                                               |
| Python (system) | 3.12+                                                             |
| Virtual env     | `.venv/`                                                          |
| GPU             | Optional (recommended for NeMo / Sherpa CUDA)                     |
| CUDA            | CUDA 12.x-compatible runtime required by many prebuilt GPU wheels |
| Package manager | `uv` (Python), `pacman` (system)                                  |

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
RestartPreventExitStatus=78
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
```

---

## Runtime Configuration

**Config file**: `~/.config/shuvoice/config.toml`

Top-level schema marker: `config_version = 1` (legacy unversioned files are treated as v0 and auto-migrated).

Config sections map to `shuvoice/config.py::Config`:
`[audio]`, `[asr]`, `[overlay]`, `[control]`, `[typing]`, `[streaming]`, `[feedback]`.
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

### Audio gain tuning (app-side)

Applied only for backends that do **not** request raw audio.

| Key                       | Default | Notes                                      |
| ------------------------- | ------: | ------------------------------------------ |
| `auto_gain_target_peak`   |  `0.15` | Target RMS peak for utterance gain         |
| `auto_gain_max`           |  `10.0` | Upper cap for utterance gain               |
| `auto_gain_settle_chunks` |     `2` | Speech chunks required before gain updates |

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

| Key                                |         Default | Notes                                                                                                                                                                                                                           |
| ---------------------------------- | --------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `typing_final_injection_mode`      |          `auto` | `auto`, `clipboard`, `direct`. In `auto`, ShuVoice detects known clipboard watchers (`wl-paste --watch`, `wl-clip-persist`, `elephant`) and switches to direct `wtype` final typing to avoid clipboard-history pollution/races. |
| `typing_clipboard_settle_delay_ms` |            `40` | Delay between `wl-copy` and simulated `Ctrl+V` in clipboard mode to reduce paste timing races.                                                                                                                                  |
| `use_clipboard_for_final`          | `true` (legacy) | Soft-deprecated compatibility flag. If `typing_final_injection_mode` is absent, this maps to `clipboard` (`true`) or `direct` (`false`).                                                                                        |
| `preserve_clipboard`               |         `false` | Capture/restore clipboard around final commit in clipboard mode; direct mode does not touch clipboard.                                                                                                                          |

---

## ASR Backends

### Gain / preprocessing behavior

| Backend   | `wants_raw_audio` | App auto-gain | Notes                            |
| --------- | ----------------- | ------------- | -------------------------------- |
| NeMo      | `true`            | Bypassed      | Backend handles normalization    |
| Moonshine | `true`            | Bypassed      | Backend handles normalization    |
| Sherpa    | `false`           | Enabled       | App-side gain helps quiet inputs |

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

| Key                      | Default                                    | Notes                                 |
| ------------------------ | ------------------------------------------ | ------------------------------------- |
| `model_name`             | `nvidia/nemotron-speech-streaming-en-0.6b` | Hugging Face model ID                 |
| `right_context`          | `13`                                       | 0–13; higher accuracy, higher latency |
| `device`                 | `cuda`                                     | `cuda` or `cpu`                       |
| `use_cuda_graph_decoder` | `false`                                    | Keep false unless validated           |

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

| Key                                |                                               Default | Notes                                                                                                                                      |
| ---------------------------------- | ----------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `sherpa_model_name`                | `sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` | Archive/model name used for auto-download when `sherpa_model_dir` is unset                                                                 |
| `sherpa_model_dir`                 |                                                _none_ | If unset, ShuVoice auto-downloads `sherpa_model_name` to `~/.local/share/shuvoice/models/sherpa/<sherpa_model_name>/`                      |
| `sherpa_decode_mode`               |                                                `auto` | `auto`, `streaming`, or `offline_instant`. `auto` resolves to `offline_instant` for Parakeet + `instant_mode=true`, otherwise `streaming`. |
| `sherpa_enable_parakeet_streaming` |                                               `false` | Safety gate for Parakeet streaming path. Must be `true` to allow Parakeet with `sherpa_decode_mode = "streaming"`.                         |
| `sherpa_provider`                  |                                                 `cpu` | `cpu` or `cuda`                                                                                                                            |
| `sherpa_num_threads`               |                                                   `2` | CPU threads                                                                                                                                |
| `sherpa_chunk_ms`                  |                                                 `100` | Streaming chunk duration (ignored in `offline_instant` mode)                                                                               |

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

#### Sherpa GPU (CUDA) support

> ⚠️ Prebuilt wheels often target CUDA 12.x. If your system CUDA/toolkit stack
> does not provide compatible shared libs, GPU mode may fail without compat libs.

Typical rebuild flow:

```bash
cd $REPO_ROOT/build/sherpa-onnx
git checkout v<VERSION>
export SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=89"
$REPO_ROOT/.venv/bin/python setup.py bdist_wheel
uv pip install dist/sherpa_onnx-*.whl --force-reinstall --no-deps

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
moonshine_max_tokens = 64
moonshine_provider = "cpu"
moonshine_onnx_threads = 0
```

#### Performance expectations

| Config                  | Per-phrase avg | Best for                                   |
| ----------------------- | -------------- | ------------------------------------------ |
| `moonshine/tiny` + CPU  | ~3.4s          | Short utterances (<5s) on CPU-only systems |
| `moonshine/tiny` + CUDA | ~0.5s (est.)   | Interactive use with GPU                   |
| `moonshine/base` + CPU  | ~7.9s          | Not recommended for interactive use        |

> Moonshine re-encodes the full audio buffer on every inference call.
> Keep `moonshine_max_window_sec` ≤ 5.0 to limit worst-case latency.
> The `moonshine/tiny` model is ~2.3× faster than `base` with similar
> accuracy for short phrases.

#### Config keys

| Key                         | Default          | Notes                                                                        |
| --------------------------- | ---------------- | ---------------------------------------------------------------------------- |
| `moonshine_model_name`      | `moonshine/tiny` | `moonshine/tiny` (fast) or `moonshine/base` (slower, slightly more accurate) |
| `moonshine_model_dir`       | _none_           | Optional local model path                                                    |
| `moonshine_model_precision` | `float`          | ONNX precision                                                               |
| `moonshine_chunk_ms`        | `100`            | Chunk duration                                                               |
| `moonshine_max_window_sec`  | `5.0`            | Max audio window before reset                                                |
| `moonshine_max_tokens`      | `64`             | Max generated tokens per window                                              |
| `moonshine_provider`        | `cpu`            | `cpu` or `cuda` (requires onnxruntime with CUDAExecutionProvider)            |
| `moonshine_onnx_threads`    | `0`              | ONNX intra-op threads; 0 = auto                                              |

#### Dependencies

```bash
uv sync --extra asr-moonshine
# or: uv add useful-moonshine-onnx
```

---

## Model Locations

| Backend   | Model                                                                               | Location                                                                                  |
| --------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| NeMo      | `nvidia/nemotron-speech-streaming-en-0.6b`                                          | Hugging Face cache (`~/.cache/huggingface/...`)                                           |
| Sherpa    | `sherpa_model_name` (default `sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06`) | `~/.local/share/shuvoice/models/sherpa/<sherpa_model_name>/` or custom `sherpa_model_dir` |
| Moonshine | `UsefulSensors/moonshine`                                                           | Hugging Face cache (`~/.cache/huggingface/...`)                                           |

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

| Tool       | Purpose                              | Install                                                                 |
| ---------- | ------------------------------------ | ----------------------------------------------------------------------- |
| `patchelf` | Patch RUNPATH for CUDA provider libs | `pacman -S patchelf`                                                    |
| `gh`       | GitHub CLI                           | `pacman -S github-cli`                                                  |
| `uv`       | Python package manager               | [astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| `ruff`     | Lint/format                          | managed by uv (`uv sync --dev`)                                         |
| `pytest`   | Tests                                | managed by uv (`uv sync --dev`)                                         |

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

| Issue                                                 | Description                                                                                                                                                                                                                     | Status                                                                 |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [#7](https://github.com/shuv1337/shuvoice/issues/7)   | Sherpa may drop trailing words on early key release                                                                                                                                                                             | Open                                                                   |
| [#12](https://github.com/shuv1337/shuvoice/issues/12) | Moonshine repetition guard misses some token/long-clause loops                                                                                                                                                                  | Fixed                                                                  |
| [#13](https://github.com/shuv1337/shuvoice/issues/13) | Moonshine throughput slower than NeMo/Sherpa                                                                                                                                                                                    | Mitigated (safer defaults, ONNX tuning, GPU provider)                  |
| —                                                     | `sherpa-onnx` source AUR builds may fail on GCC 15 due format-security warning flag interaction                                                                                                                                 | Mitigation available (`python-sherpa-onnx-bin`, upstream patch staged) |
| —                                                     | Prebuilt Sherpa CUDA wheels may be incompatible with newer CUDA stacks                                                                                                                                                          | Ongoing                                                                |
| —                                                     | Parakeet streaming is behind explicit safety gate (`sherpa_enable_parakeet_streaming = true`) and requires online-compatible encoder metadata (`window_size`); incompatible models are blocked pre-start with actionable errors | By design                                                              |

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
