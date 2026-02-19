# ShuVoice

Streaming speech-to-text overlay for Hyprland using NVIDIA Nemotron Speech Streaming.

## Status

Core pipeline + production hardening are implemented:

- PipeWire capture (`sounddevice`)
- Nemotron streaming ASR (`nemo-toolkit`)
- GTK4 layer-shell overlay
- evdev hotkey (tap/hold)
- Hyprland IPC fallback controls via local Unix socket
- `wtype` / clipboard text injection with retry + fallback

## Requirements

### Python

- Python `>=3.10,<3.13`

### System packages (Arch)

```bash
sudo pacman -S \
  gtk4 gtk4-layer-shell python-gobject \
  portaudio pipewire pipewire-audio pipewire-alsa \
  wtype wl-clipboard
```

### Python packages

```bash
pip install -e .
# ASR runtime:
pip install -e .[asr]
# test tooling:
pip install -e .[dev]
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
- ASR stack (`torch`, `nemo`)
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
python -m shuvoice --hotkey KEY_F9
python -m shuvoice --output-mode streaming_partial
python -m shuvoice --hotkey-backend ipc
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

## Smoke test script

```bash
./scripts/smoke-test.sh
```

## Troubleshooting

- `No module named 'torch'` or `No module named 'nemo'`
  - Install ASR deps (`pip install -e .[asr]`) or Arch CUDA torch package.
- `No keyboard device found ... input group`
  - Add user to `input` group and re-login, or use `hotkey_backend = "ipc"`.
- `Control socket not found ...`
  - Start ShuVoice first (`python -m shuvoice`) before sending `--control` commands.
- `libgtk4-layer-shell.so not found`
  - `sudo pacman -S gtk4-layer-shell`
- `wtype not found in PATH`
  - `sudo pacman -S wtype`
