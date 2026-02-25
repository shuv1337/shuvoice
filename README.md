# ShuVoice

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/branding/shuvoice-variant-dark-lockup.png">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/branding/shuvoice-variant-light-lockup.png">
    <img src="./docs/assets/branding/shuvoice-variant-dark-lockup.png" alt="ShuVoice logo" width="760">
  </picture>
</p>

Streaming speech-to-text overlay for Hyprland with pluggable ASR backends.

[![CI](https://github.com/shuv1337/shuvoice/actions/workflows/ci.yml/badge.svg)](https://github.com/shuv1337/shuvoice/actions/workflows/ci.yml)

<p align="center">
  <img src="./docs/assets/screenshots/splash-overlay.png" alt="ShuVoice splash overlay on Hyprland" width="760">
</p>

## Status

Core pipeline + production hardening are implemented:

- PipeWire capture (`sounddevice`)
- Pluggable ASR backend layer (`nemo`, `sherpa`, `moonshine`) with Sherpa streaming + offline-instant modes
- GTK4 layer-shell overlay
- Hyprland IPC controls via local Unix socket
- `wtype` / clipboard text injection with retry + fallback

## Current backend models & providers

| Backend (`asr_backend`) | Current model(s) | Provider setting | Supported providers |
|---|---|---|---|
| `nemo` | `nvidia/nemotron-speech-streaming-en-0.6b` | `device` | `cuda` (default), `cpu` |
| `sherpa` | `sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` (streaming default), `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` (offline instant) | `sherpa_provider` | `cpu` (default), `cuda` |
| `moonshine` | `moonshine/tiny` (default, also `moonshine/base`) | `moonshine_provider` | `cpu` (default), `cuda` |

Model locations in this repo/runtime:

- NeMo model ID: `nvidia/nemotron-speech-streaming-en-0.6b` (downloaded to Hugging Face cache)
- Sherpa model dir: auto-download selected `sherpa_model_name` to `~/.local/share/shuvoice/models/sherpa/<sherpa_model_name>/` (or custom `sherpa_model_dir`)
- Moonshine models: Hugging Face `UsefulSensors/moonshine` (`base`/`tiny`)

> Note: Sherpa CUDA requires a source-built `sherpa-onnx` GPU wheel plus CUDA 12 compatibility libs on this host stack.
>
> Note: Parakeet TDT (`sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`) is supported via Sherpa offline instant mode (`instant_mode=true` + `sherpa_decode_mode="offline_instant"`, or `sherpa_decode_mode="auto"` with instant mode on). Parakeet + streaming mode remains blocked by startup guards.

## Backend accuracy/performance snapshot (manual regression suite)

Results below were measured on 2026-02-22 using `scripts/tts_roundtrip.py` with the
same 10-utterance phrase set used by `tests/integration/test_roundtrip_regression.py`
(2 phrases × 5 repeats, `--flush-chunks 5`).

| Model/profile | Median similarity | Mean similarity | Empty ratio | Wall time (10 utt) | RTF (wall/audio, lower is faster) |
|---|---:|---:|---:|---:|---:|
| NeMo `nvidia/nemotron-speech-streaming-en-0.6b` (`device=cuda`) | 0.776 | 0.775 | 0.000 | 8.33s | 0.26 |
| Sherpa `...kroko-2025-08-06` (`provider=cuda`) | 0.720 | 0.720 | 0.000 | 2.52s | 0.08 |
| Sherpa `...kroko-2025-08-06` (`provider=cpu`) | 0.720 | 0.720 | 0.000 | 1.62s | 0.05 |
| Moonshine `moonshine/base` | 0.625 | 0.625 | 0.000 | 21.93s | 0.68 |
| Moonshine `moonshine/tiny` | 0.795 | 0.795 | 0.000 | 10.03s | 0.31 |

Notes:
- This is a **regression stress fixture**, not a universal quality ranking.
- Numbers include model load + full roundtrip harness runtime.
- Moonshine throughput improved via deferred chunk-buffer coalescing in `MoonshineBackend.process_chunk()`.
- For day-to-day dictation quality, run your own workload-specific benchmark before choosing defaults.

## Requirements

### Python

- Python `>=3.10`

### System packages (Arch)

```bash
sudo pacman -S \
  gtk4 gtk4-layer-shell python-gobject \
  portaudio pipewire pipewire-audio pipewire-alsa \
  wtype wl-clipboard espeak-ng
```

### Installation (AUR)

ShuVoice is available on the AUR as [`shuvoice-git`](https://aur.archlinux.org/packages/shuvoice-git):

```bash
# Using yay
yay -S shuvoice-git

# Using paru
paru -S shuvoice-git
```

After installation, enable the user service:

```bash
systemctl --user enable --now shuvoice.service
```

`shuvoice-git` includes Sherpa runtime support via dependency
`python-sherpa-onnx`.

If your AUR helper asks which provider to use, pick
`python-sherpa-onnx-bin` (recommended fast path):

```bash
yay -S --needed python-sherpa-onnx-bin shuvoice-git
```

NeMo and Moonshine remain optional.

### Python packages

```bash
# Install base + dev tooling (creates venv automatically):
uv sync --dev

# ASR backend extras:
uv sync --extra asr-nemo
uv sync --extra asr-sherpa
uv sync --extra asr-moonshine

# NeMo convenience alias:
uv sync --extra asr
```

For Python 3.14 + NeMo, prefer the repo override file to avoid
`kaldialign` source-build issues:

```bash
uv sync --extra asr-nemo --override packaging/constraints/py314-overrides.txt
```

If NeMo wheels are unavailable for your environment:

```bash
uv pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
```

## Permissions

ShuVoice uses IPC control commands (no `/dev/input` access required).
Trigger start/stop via Hyprland `bind` / `bindr` commands.

## Preflight

Run before first launch:

```bash
python -m shuvoice preflight
# legacy (still supported): python -m shuvoice --preflight
```

Checks include:

- Python version compatibility
- importability of key Python modules
- audio input device validity
- ASR backend dependencies for selected `asr_backend`
- Sherpa decode diagnostics (resolved `sherpa_decode_mode`, provider requested→effective, Parakeet runnability)
- required binaries (`wtype`, `wl-copy`, `wl-paste`)
- `libgtk4-layer-shell.so`
- output mode validity

## Setup (recommended)

Run the setup workflow once after install to verify backend deps, ensure model
artifacts are ready, and run preflight checks:

```bash
shuvoice setup
```

Useful setup flags:

```bash
# Try to install missing backend deps automatically (when supported)
shuvoice setup --install-missing

# Skip model download and preflight (quick dependency check only)
shuvoice setup --skip-model-download --skip-preflight
```

For Sherpa, `setup` now reports resolved decode mode, provider requested→effective,
and whether a selected Parakeet model is runnable with the current config/runtime.

## Run

```bash
python -m shuvoice
```

Useful commands:

```bash
python -m shuvoice --help
python -m shuvoice setup
python -m shuvoice run --asr-backend nemo --right-context 13
python -m shuvoice run --asr-backend sherpa --sherpa-model-dir /path/to/model
# Parakeet offline instant example (set in config.toml):
# [asr]
# asr_backend = "sherpa"
# sherpa_model_name = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
# instant_mode = true
# sherpa_decode_mode = "offline_instant"
python -m shuvoice run --asr-backend moonshine --moonshine-model-name moonshine/tiny
python -m shuvoice run --asr-backend moonshine --moonshine-provider cuda
python -m shuvoice run --output-mode streaming_partial
python -m shuvoice audio list-devices
python -m shuvoice run --audio-device 2 --input-gain 1.5
python -m shuvoice model download
python -m shuvoice config effective
python -m shuvoice config validate
```

Legacy top-level flags (`--download-model`, `--list-audio-devices`, `--preflight`, `--wizard`, `--control`) remain available for one compatibility cycle and emit deprecation warnings.

`model download` supports NeMo and Sherpa. Moonshine downloads lazily on first load via useful-moonshine-onnx.

## systemd user service

Repo-managed unit template:

- `packaging/systemd/user/shuvoice.service`

Install and start (from this repo):

```bash
mkdir -p ~/.config/systemd/user
cp packaging/systemd/user/shuvoice.service ~/.config/systemd/user/shuvoice.service

# Optional for repo/venv workflows (default template uses /usr/bin/shuvoice)
systemctl --user edit shuvoice.service
# [Service]
# ExecStart=
# ExecStart=%h/.venv/bin/shuvoice

systemctl --user daemon-reload
systemctl --user import-environment WAYLAND_DISPLAY DISPLAY XDG_RUNTIME_DIR HYPRLAND_INSTANCE_SIGNATURE DBUS_SESSION_BUS_ADDRESS XDG_CURRENT_DESKTOP XDG_SESSION_TYPE
systemctl --user enable --now shuvoice.service
systemctl --user status shuvoice.service
```

Packaged unit files include `RestartPreventExitStatus=78` so missing backend
dependencies do not cause endless restart loops.

## Control socket commands (IPC backend)

When ShuVoice is running, send control commands from another terminal:

```bash
python -m shuvoice control start
python -m shuvoice control stop
python -m shuvoice control toggle
python -m shuvoice control status
python -m shuvoice control metrics

# legacy form (still supported):
python -m shuvoice --control status
```

Notes:
- `start`/`stop` is recommended for push-to-talk flows (`bind` + `bindr`).
- `status` may report `processing` briefly after `stop`/`toggle` while final text is being flushed/typed.
- CLI waits up to 2s after `stop` (or a stop-side `toggle`) for processing to finish; adjust with `--control-wait-sec`.

Hyprland example:

```ini
bind = , Insert, exec, shuvoice --control start
bindr = , Insert, exec, shuvoice --control stop
```

## Waybar module (tray-style status icon)

<p align="center">
  <img src="./docs/assets/screenshots/waybar-tooltip.png" alt="Waybar tooltip showing backend, model, device, and keybind" width="420">
</p>

ShuVoice ships a Waybar helper command (`shuvoice-waybar`) that outputs JSON
for a `custom/*` module and can handle click actions.

Quick checks:

```bash
shuvoice-waybar status
shuvoice-waybar menu
shuvoice-waybar launch-wizard
```

Waybar config example:

```jsonc
"custom/shuvoice": {
  "return-type": "json",
  "exec": "shuvoice-waybar status",
  "interval": 1,
  "on-click": "shuvoice-waybar toggle-record",
  "on-click-middle": "shuvoice-waybar service-toggle",
  "on-click-right": "shuvoice-waybar menu",
  "tooltip": true
}
```

Right-click menu uses one of: `omarchy-launch-walker`, `walker`, `wofi`, `rofi`, `bemenu`, or `dmenu`.

If Waybar cannot find the command from your shell PATH, point to the full
venv path (for example `$HOME/.venv/bin/shuvoice-waybar`) or use the wrapper
script in this repo: `scripts/shuvoice-waybar.sh`.

Wrapper quick check:

```bash
./scripts/shuvoice-waybar.sh status
```

Optional: install a PATH symlink (default: `~/.local/bin/shuvoice-waybar`):

```bash
./scripts/install-waybar-wrapper.sh
```

Optional: remove that symlink later:

```bash
./scripts/uninstall-waybar-wrapper.sh
```

If your user unit has a different name, set `SHUVOICE_SERVICE` in Waybar's
exec/on-click commands.

State classes exported by the module:
`recording`, `processing`, `idle`, `starting`, `stopped`, `error`.

Example CSS:

```css
#custom-shuvoice.recording { color: #f38ba8; }
#custom-shuvoice.processing { color: #fab387; }
#custom-shuvoice.idle { color: #a6e3a1; }
#custom-shuvoice.starting { color: #f9e2af; }
#custom-shuvoice.stopped { color: #7f849c; }
#custom-shuvoice.error { color: #f38ba8; }
```

Hyprland blur/transparency for the overlay layer surface:

```ini
# ShuVoice uses layer-shell namespace: stt-overlay
layerrule = blur, stt-overlay
layerrule = ignorealpha 0.20, stt-overlay
layerrule = xray 1, stt-overlay
```

Then tune overlay styling in `~/.config/shuvoice/config.toml`:

```toml
[overlay]
font_size = 24
font_family = "JetBrains Mono" # optional
bg_opacity = 0.55
```

## Configuration

Relaunch setup wizard at any time:

```bash
shuvoice wizard
# legacy: shuvoice --wizard
```

The wizard can optionally auto-add Hyprland push-to-talk binds when the
selected key is not already used (default presets include **Insert** and
**Right Control**).

When Sherpa is selected, the wizard also lets you choose the model variant
(default Zipformer or Parakeet TDT v3 int8), shows download progress in the
finish screen (with a cancel button), and attempts to auto-download the
selected model at finish.

Wizard screens:

<p align="center">
  <img src="./docs/assets/screenshots/wizard-welcome.png" alt="ShuVoice setup wizard welcome screen" width="760">
</p>

<p align="center">
  <img src="./docs/assets/screenshots/wizard-asr-selection.png" alt="ShuVoice setup wizard ASR backend selection screen" width="760">
</p>

<p align="center">
  <img src="./docs/assets/screenshots/wizard-keybind-selection.png" alt="ShuVoice setup wizard keybind selection screen" width="760">
</p>

(From Waybar, map an action to `shuvoice-waybar launch-wizard`.)

Config file path:

- `~/.config/shuvoice/config.toml`
- include top-level `config_version = 1` (added automatically on migration/write)

Example config:

- `examples/config.toml` (full reference)
- `examples/config-nemo-cuda.toml`
- `examples/config-nemo-cpu.toml`
- `examples/config-sherpa-cuda.toml`
- `examples/config-sherpa-cpu.toml`
- `examples/config-sherpa-parakeet-offline.toml`
- `examples/config-moonshine-cpu.toml`
- `examples/waybar-custom-shuvoice.jsonc` (Waybar custom module snippet)
- `examples/waybar-custom-shuvoice-wrapper.jsonc` (wrapper-script variant)
- `examples/waybar-shuvoice.css` (Waybar state color classes)

Backend selection is controlled by `asr_backend`:

- `asr_backend = "sherpa"` (default): uses `sherpa_*` settings; if `sherpa_model_dir` is unset, ShuVoice auto-downloads `sherpa_model_name` (default: `sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06`)
- `asr_backend = "nemo"`: uses `model_name`, `right_context`, `device`
- `asr_backend = "moonshine"`: uses `moonshine_*` settings (16k sample rate expected)

Sherpa decode mode controls whether ShuVoice uses streaming chunk decode or
one-shot offline utterance decode:

```toml
[asr]
asr_backend = "sherpa"
sherpa_decode_mode = "auto"  # auto | streaming | offline_instant
```

Resolution rules:
- `streaming`: always use the existing streaming chunk path.
- `offline_instant`: accumulate audio while PTT is held, decode once on release,
  and commit only the final transcript.
- `auto`: resolves to `offline_instant` for Parakeet model names when
  `instant_mode = true`; otherwise resolves to `streaming`.

Parakeet TDT v3 requires offline instant mode:

```toml
[asr]
asr_backend = "sherpa"
sherpa_model_name = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
instant_mode = true
sherpa_decode_mode = "offline_instant"
```

If Parakeet is selected with streaming decode mode, startup guards block launch
with an actionable error and service exit code 78 (to avoid restart loops).

Optional: set `instant_mode = true` under `[asr]` for low-latency tuning.
This applies backend-specific behavior:

- NeMo: forces `right_context = 0`
- Sherpa streaming mode: caps `sherpa_chunk_ms` at `80`
- Sherpa offline mode: enables one-shot release-to-final decode behavior
- Moonshine: forces `moonshine_model_name = "moonshine/tiny"`, caps `moonshine_max_window_sec` to `3.0`, and caps `moonshine_max_tokens` to `48`

`right_context` applies to NeMo only.

ShuVoice includes built-in brand corrections for common ASR variants of
`ShuVoice` and `Hyprland` (for example: `shove voice`, `shu voice`,
`show voice`, `hyper land`, `hyperland`, `high per land`).

You can add/override corrections with `[typing.text_replacements]`:

```toml
[typing.text_replacements]
# Add your own replacements (or override a built-in one)
"speech to text" = "speech-to-text"
"um" = ""
```

Matches are case-insensitive and only apply to whole words/phrases (longest
source phrases first). Empty values delete the matched word/phrase.

Final text injection behavior is controlled under `[typing]`:

- `typing_final_injection_mode = "auto"` (default): detect known clipboard watchers (`wl-paste --watch`, `wl-clip-persist`, `elephant`) and prefer direct final typing.
- `typing_final_injection_mode = "clipboard"`: always use clipboard paste (`wl-copy` + `Ctrl+V`).
- `typing_final_injection_mode = "direct"`: always use direct `wtype` typing.
- `typing_clipboard_settle_delay_ms = 40`: delay between `wl-copy` and paste in clipboard mode (helps compositor/clipboard-manager timing races).
- `use_clipboard_for_final` is soft-deprecated; if `typing_final_injection_mode` is omitted, legacy `true` maps to `clipboard` and `false` maps to `direct`.

If you hit RNNT CUDA-graph decoder issues on your driver/toolkit combo,
keep this setting disabled (default):

```toml
use_cuda_graph_decoder = false
```

## Smoke test script

```bash
./scripts/smoke-test.sh
```

## Development

Install development tooling and run local quality checks:

```bash
uv sync --dev
uv run ruff check shuvoice tests
uv run ruff format --check shuvoice tests
uv run pytest -m "not gui" -v
```

IPC end-to-end smoke tests (CLI -> control socket):

```bash
pytest -m e2e -k ipc_smoke -v
```

Manual phrase regression (opt-in integration test harness):

```bash
# Runs deterministic TTS->STT regression checks for the two manual phrases
# (quick brown fox + moonshine sentence), repeated multiple times.
SHUVOICE_RUN_ROUNDTRIP=1 \
SHUVOICE_ROUNDTRIP_BACKEND=nemo \
SHUVOICE_ROUNDTRIP_DEVICE=cuda \
pytest -m integration -k roundtrip_regression -v
```

Sherpa GPU low-noise regression suite (from field notes):

```bash
SHUVOICE_RUN_SHERPA_LOW_NOISE=1 \
SHUVOICE_SHERPA_PROVIDER=cuda \
pytest -m integration -k sherpa_gpu_low_noise_phrase_regression -v
```

To remove local build/test artifacts generated during development:

```bash
rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage* coverage.xml
```

## Long-phrase round-trip harness (TTS -> STT)

Use this to reproduce truncation/cut-out behavior with deterministic inputs.

```bash
# Uses built-in defaults (writes WAV + CSV under build/tts-roundtrip)
python scripts/tts_roundtrip.py --asr-backend nemo --device cuda

# Moonshine backend
python scripts/tts_roundtrip.py --asr-backend moonshine --moonshine-model-name moonshine/tiny

# Use fixed phrase fixtures
python scripts/tts_roundtrip.py \
  --phrases-file examples/tts_roundtrip_phrases.txt \
  --asr-backend nemo \
  --device cuda
```

The script:
- generates WAV files via `espeak-ng`
- streams each file through ShuVoice ASR chunking logic
- prints reference vs hypothesis similarity
- writes `build/tts-roundtrip/roundtrip.csv`

## Open source project docs

- Contribution guidelines: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Brand assets: `docs/BRANDING.md`

## License

ShuVoice is released under the MIT License. See `LICENSE`.

## Troubleshooting

- `No module named 'torch'` or `No module named 'nemo'`
  - Install NeMo ASR deps (`uv sync --extra asr-nemo` or `--extra asr`) or Arch CUDA torch package.
- `No module named 'sherpa_onnx'`
  - AUR: `yay -S --needed python-sherpa-onnx-bin` (or another provider for `python-sherpa-onnx`).
  - venv: `uv sync --extra asr-sherpa`.
- `No module named 'moonshine_onnx'`
  - Install Moonshine deps (`uv sync --extra asr-moonshine`).
- `sherpa_model_dir` exists but is missing `encoder/decoder/joiner` artifacts
  - Point `sherpa_model_dir` to a valid transducer model directory containing
    `tokens.txt` and ONNX files for encoder/decoder/joiner.
  - If `sherpa_model_dir` is unset, ShuVoice will auto-download `sherpa_model_name`.
- `Parakeet requires offline instant mode` startup error
  - Use `sherpa_decode_mode = "offline_instant"`, or set `instant_mode = true` with
    `sherpa_decode_mode = "auto"`.
  - Do not run Parakeet with `sherpa_decode_mode = "streaming"`; startup guard blocks this.
- `sherpa_provider='cuda' requested, but runtime does not expose CUDAExecutionProvider`
  - Install a CUDA-enabled sherpa-onnx runtime, or run with `sherpa_provider = "cpu"`.
  - `setup`/`preflight` now report requested vs effective provider and decode mode.
- `moonshine_model_dir` missing `encoder_model.onnx` / `decoder_model_merged.onnx`
  - Point `moonshine_model_dir` to a valid local Moonshine ONNX export, or unset it
    and let useful-moonshine-onnx fetch weights from Hugging Face.
- `No module named 'gi'`
  - Install GTK Python bindings (`uv sync` now includes `PyGObject`).
  - If build fails, install system deps: `sudo pacman -S python-gobject gtk4 gtk4-layer-shell`.
- `Failed to build kaldialign` when installing NeMo extras on Python 3.14
  - Use: `uv sync --extra asr-nemo --override packaging/constraints/py314-overrides.txt`.
  - Or use a Python 3.13 virtualenv for ASR installs.
- `espeak-ng not found` when running `scripts/tts_roundtrip.py`
  - Install with: `sudo pacman -S espeak-ng`.
- `Control socket not found ...`
  - Start ShuVoice first (`python -m shuvoice` or `python -m shuvoice run`) before sending `control` commands.
- `libgtk4-layer-shell.so not found`
  - `sudo pacman -S gtk4-layer-shell`
- `wtype not found in PATH`
  - `sudo pacman -S wtype`
- Clipboard history is polluted with transcription fragments / text not pasting (Omarchy/Walker/Elephant)
  - Set `typing_final_injection_mode = "auto"` in your config (the new default). This detects clipboard managers and uses direct typing to prevent conflicts.
- Recognition quality is poor / start-stop triggers repeatedly
  - Increase ASR context for accuracy (eg. `right_context=13`, with higher latency).
  - Select the correct mic (`python -m shuvoice audio list-devices`, then set `audio_device`). Prefer device *name* over numeric index, because indices can change between runs.
  - Increase `input_gain` moderately (eg. `1.3` to `1.8`) if your mic is too quiet.
  - If silent presses still produce phantom text (eg. "thank you"), raise `silence_rms_threshold` slightly (eg. `0.010` to `0.015`) and/or increase `silence_rms_multiplier` (eg. `2.0`) in config.
- Long phrases plateau or cut out mid-sentence
  - Keep `streaming_stall_guard = true` (default) to inject a tiny silent flush when transcript stalls despite speech energy.
  - Tune `streaming_stall_chunks` (try `3` to `6`) and `streaming_stall_rms_ratio` (try `0.6` to `0.9`) in config.
  - Run `python scripts/tts_roundtrip.py --phrases-file examples/tts_roundtrip_phrases.txt --asr-backend nemo --device cuda` to compare before/after behavior.
