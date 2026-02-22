# ShuVoice

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/branding/shuvoice-variant-dark-lockup.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/branding/shuvoice-variant-light-lockup.png">
    <img src="docs/assets/branding/shuvoice-variant-dark-lockup.png" alt="ShuVoice logo" width="760">
  </picture>
</p>

Streaming speech-to-text overlay for Hyprland with pluggable ASR backends.

[![CI](https://github.com/shuv1337/shuvoice/actions/workflows/ci.yml/badge.svg)](https://github.com/shuv1337/shuvoice/actions/workflows/ci.yml)

## Status

Core pipeline + production hardening are implemented:

- PipeWire capture (`sounddevice`)
- Pluggable streaming ASR backend layer (`nemo`, `sherpa`, `moonshine`)
- GTK4 layer-shell overlay
- evdev hotkey (tap/hold)
- Hyprland IPC fallback controls via local Unix socket
- `wtype` / clipboard text injection with retry + fallback

## Current backend models & providers

| Backend (`asr_backend`) | Current model(s) | Provider setting | Supported providers |
|---|---|---|---|
| `nemo` | `nvidia/nemotron-speech-streaming-en-0.6b` | `device` | `cuda` (default), `cpu` |
| `sherpa` | `sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` (auto-downloaded by default) | `sherpa_provider` | `cpu` (default), `cuda` |
| `moonshine` | `moonshine/base` (also `moonshine/tiny`) | N/A (CPU runtime) | `cpu` |

Model locations in this repo/runtime:

- NeMo model ID: `nvidia/nemotron-speech-streaming-en-0.6b` (downloaded to Hugging Face cache)
- Sherpa model dir: auto-download default `~/.local/share/shuvoice/models/sherpa/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06/` (or custom `sherpa_model_dir`)
- Moonshine models: Hugging Face `UsefulSensors/moonshine` (`base`/`tiny`)

> Note: Sherpa CUDA requires a source-built `sherpa-onnx` GPU wheel plus CUDA 12 compatibility libs on this host stack.

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

### Python packages

```bash
pip install -e .
# ASR runtime (NeMo compatibility alias):
pip install -e .[asr]
# Explicit backend extras:
pip install -e .[asr-nemo]
pip install -e .[asr-sherpa]
pip install -e .[asr-moonshine]
# test tooling:
pip install -e .[dev]
```

Using uv is equivalent:

```bash
uv pip install -e .
uv pip install -e .[asr-nemo]
# or
uv pip install -e .[asr-sherpa]
# or
uv pip install -e .[asr-moonshine]
```

For Python 3.14 + uv with NeMo, prefer the repo override file to avoid
`kaldialign` source-build issues:

```bash
uv pip install -e .[asr-nemo] --overrides packaging/constraints/py314-overrides.txt
```

If NeMo wheels are unavailable for your environment:

```bash
pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
```

## Permissions

### evdev backend

evdev hotkey capture requires access to `/dev/input/event*`.

```bash
sudo usermod -aG input "$USER"
# log out/in after group change
```

### IPC backend (no input group)

Use `hotkey_backend = "ipc"` and trigger start/stop via Hyprland `bind` / `bindr`.

## Preflight

Run before first launch:

```bash
python -m shuvoice --preflight
```

Checks include:

- Python version compatibility
- importability of key Python modules
- audio input device validity
- ASR backend dependencies for selected `asr_backend`
- required binaries (`wtype`, `wl-copy`, `wl-paste`)
- `libgtk4-layer-shell.so`
- configured hotkey backend / hotkey / output mode

## Run

```bash
python -m shuvoice
```

Useful flags:

```bash
python -m shuvoice --help
python -m shuvoice --download-model
python -m shuvoice --asr-backend nemo --right-context 13
python -m shuvoice --asr-backend sherpa --sherpa-model-dir /path/to/model
python -m shuvoice --asr-backend moonshine --moonshine-model-name moonshine/base
python -m shuvoice --hotkey KEY_F9
python -m shuvoice --output-mode streaming_partial
python -m shuvoice --hotkey-backend ipc
python -m shuvoice --list-audio-devices
python -m shuvoice --audio-device 2 --input-gain 1.5
```

`--download-model` supports NeMo and Sherpa. Moonshine downloads lazily on first load via useful-moonshine-onnx.

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

## Control socket commands (IPC backend)

When ShuVoice is running, send control commands from another terminal:

```bash
python -m shuvoice --control start
python -m shuvoice --control stop
python -m shuvoice --control toggle
python -m shuvoice --control status
```

Notes:
- `start`/`stop` is recommended for push-to-talk flows (`bind` + `bindr`).
- `status` may report `processing` briefly after `stop`/`toggle` while final text is being flushed/typed.
- CLI waits up to 2s after `stop` (or a stop-side `toggle`) for processing to finish; adjust with `--control-wait-sec`.

Hyprland example:

```ini
bind = , F9, exec, shuvoice --control start
bindr = , F9, exec, shuvoice --control stop
```

Hyprland blur/transparency for the overlay layer surface:

```ini
# ShuVoice uses layer-shell namespace: stt-overlay
layerrule = blur, stt-overlay
layerrule = ignorealpha 0.20, stt-overlay
layerrule = xray 1, stt-overlay
```

Then tune overlay alpha in `~/.config/shuvoice/config.toml`:

```toml
[overlay]
bg_opacity = 0.55
```

## Configuration

Config file path:

- `~/.config/shuvoice/config.toml`

Example config:

- `examples/config.toml` (full reference)
- `examples/config-nemo-cuda.toml`
- `examples/config-nemo-cpu.toml`
- `examples/config-sherpa-cuda.toml`
- `examples/config-sherpa-cpu.toml`
- `examples/config-moonshine-cpu.toml`

Backend selection is controlled by `asr_backend`:

- `asr_backend = "nemo"` (default): uses `model_name`, `right_context`, `device`
- `asr_backend = "sherpa"`: uses `sherpa_*` settings; if `sherpa_model_dir` is unset, ShuVoice auto-downloads the default streaming model
- `asr_backend = "moonshine"`: uses `moonshine_*` settings (16k sample rate expected)

`right_context` applies to NeMo only.

Final text corrections can be configured with `[typing.text_replacements]`:

```toml
[typing.text_replacements]
"shove voice" = "ShuVoice"
"hyper land" = "Hyprland"
"um" = ""
```

Matches are case-insensitive and only apply to whole words/phrases (longest
source phrases first). Empty values delete the matched word/phrase.

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
pip install -e .[dev]
ruff check shuvoice tests
ruff format --check shuvoice tests
pytest -m "not gui" -v
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
python scripts/tts_roundtrip.py --asr-backend moonshine --moonshine-model-name moonshine/base

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
  - Install NeMo ASR deps (`pip install -e .[asr-nemo]` or `.[asr]`) or Arch CUDA torch package.
- `No module named 'sherpa_onnx'`
  - Install Sherpa deps (`pip install -e .[asr-sherpa]`).
- `No module named 'moonshine_onnx'`
  - Install Moonshine deps (`pip install -e .[asr-moonshine]`).
- `sherpa_model_dir` exists but is missing `encoder/decoder/joiner` artifacts
  - Point `sherpa_model_dir` to a streaming transducer model directory containing
    `tokens.txt` and ONNX files for encoder/decoder/joiner.
  - If `sherpa_model_dir` is unset, ShuVoice will auto-download the default model.
- `moonshine_model_dir` missing `encoder_model.onnx` / `decoder_model_merged.onnx`
  - Point `moonshine_model_dir` to a valid local Moonshine ONNX export, or unset it
    and let useful-moonshine-onnx fetch weights from Hugging Face.
- `No module named 'gi'`
  - Install GTK Python bindings (`pip/uv install -e .` now includes `PyGObject`).
  - If build fails, install system deps: `sudo pacman -S python-gobject gtk4 gtk4-layer-shell`.
- `Failed to build kaldialign` when installing NeMo extras on Python 3.14
  - Use: `uv pip install -e .[asr-nemo] --overrides packaging/constraints/py314-overrides.txt`.
  - Or use a Python 3.13 virtualenv for ASR installs.
- `espeak-ng not found` when running `scripts/tts_roundtrip.py`
  - Install with: `sudo pacman -S espeak-ng`.
- `No keyboard device found ... input group`
  - Add user to `input` group and re-login, or use `hotkey_backend = "ipc"`.
- `Control socket not found ...`
  - Start ShuVoice first (`python -m shuvoice`) before sending `--control` commands.
- `libgtk4-layer-shell.so not found`
  - `sudo pacman -S gtk4-layer-shell`
- `wtype not found in PATH`
  - `sudo pacman -S wtype`
- Recognition quality is poor / start-stop triggers repeatedly
  - Set a single keyboard device in config (`hotkey_device=/dev/input/eventX`) or keep `hotkey_listen_all_devices=false`.
  - Increase ASR context for accuracy (eg. `right_context=13`, with higher latency).
  - Select the correct mic (`python -m shuvoice --list-audio-devices`, then set `audio_device`). Prefer device *name* over numeric index, because indices can change between runs.
  - Increase `input_gain` moderately (eg. `1.3` to `1.8`) if your mic is too quiet.
  - If silent presses still produce phantom text (eg. "thank you"), raise `silence_rms_threshold` slightly (eg. `0.010` to `0.015`) and/or increase `silence_rms_multiplier` (eg. `2.0`) in config.
- Long phrases plateau or cut out mid-sentence
  - Keep `streaming_stall_guard = true` (default) to inject a tiny silent flush when transcript stalls despite speech energy.
  - Tune `streaming_stall_chunks` (try `3` to `6`) and `streaming_stall_rms_ratio` (try `0.6` to `0.9`) in config.
  - Run `python scripts/tts_roundtrip.py --phrases-file examples/tts_roundtrip_phrases.txt --asr-backend nemo --device cuda` to compare before/after behavior.
