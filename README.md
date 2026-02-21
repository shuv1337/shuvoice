# ShuVoice

Streaming speech-to-text overlay for Hyprland with pluggable ASR backends.

## Status

Core pipeline + production hardening are implemented:

- PipeWire capture (`sounddevice`)
- Pluggable streaming ASR backend layer (`nemo`, `sherpa`, `moonshine`)
- GTK4 layer-shell overlay
- evdev hotkey (tap/hold)
- Hyprland IPC fallback controls via local Unix socket
- `wtype` / clipboard text injection with retry + fallback

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

`--download-model` currently supports NeMo only. Sherpa and Moonshine print setup guidance.

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
# ExecStart=%h/repos/shuvoice/.venv312/bin/shuvoice

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

Hyprland example:

```ini
bind = , F9, exec, shuvoice --control start
bindr = , F9, exec, shuvoice --control stop
```

## Configuration

Config file path:

- `~/.config/shuvoice/config.toml`

Example config:

- `examples/config.toml`

Backend selection is controlled by `asr_backend`:

- `asr_backend = "nemo"` (default): uses `model_name`, `right_context`, `device`
- `asr_backend = "sherpa"`: requires `sherpa_model_dir` and uses `sherpa_*` settings
- `asr_backend = "moonshine"`: uses `moonshine_*` settings (16k sample rate expected)

`right_context` applies to NeMo only.

If you hit RNNT CUDA-graph decoder issues on your driver/toolkit combo,
keep this setting disabled (default):

```toml
use_cuda_graph_decoder = false
```

## Smoke test script

```bash
./scripts/smoke-test.sh
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

## Troubleshooting

- `No module named 'torch'` or `No module named 'nemo'`
  - Install NeMo ASR deps (`pip install -e .[asr-nemo]` or `.[asr]`) or Arch CUDA torch package.
- `No module named 'sherpa_onnx'`
  - Install Sherpa deps (`pip install -e .[asr-sherpa]`).
- `No module named 'moonshine_onnx'`
  - Install Moonshine deps (`pip install -e .[asr-moonshine]`).
- `sherpa_model_dir is required` or missing `encoder/decoder/joiner` artifacts
  - Point `sherpa_model_dir` to a streaming transducer model directory containing
    `tokens.txt` and ONNX files for encoder/decoder/joiner.
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
