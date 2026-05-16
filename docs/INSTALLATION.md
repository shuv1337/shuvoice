# Installation

ShuVoice targets Linux Wayland desktops, with Hyprland as the primary tested
environment. The recommended install path on Arch Linux is the AUR package.

## AUR Package

```bash
yay -S shuvoice-git
```

The package installs the CLI, user service unit, Waybar helper, and Sherpa-ONNX
runtime support. If your AUR helper asks which Sherpa package to use, prefer the
prebuilt package:

```bash
yay -S --needed python-sherpa-onnx-bin shuvoice-git
```

After install:

```bash
shuvoice wizard
systemctl --user enable --now shuvoice.service
```

## Source Install

Prerequisites:

| Component | Requirement |
|---|---|
| OS | Linux with Wayland |
| Python | 3.10 or newer |
| Package manager | uv |
| Desktop | Hyprland recommended |
| GPU | Optional, useful for NeMo and CUDA Sherpa |

Arch dependencies:

```bash
sudo pacman -S \
  gtk4 gtk4-layer-shell python-gobject \
  portaudio pipewire pipewire-audio pipewire-alsa \
  wtype wl-clipboard espeak-ng
```

Clone and install:

```bash
git clone https://github.com/shuv1337/shuvoice.git
cd shuvoice
uv sync
```

Install at least one ASR backend:

```bash
uv sync --extra asr-sherpa       # fast default
uv sync --extra asr-nemo         # high accuracy, CUDA recommended
uv sync --extra asr-moonshine    # lightweight
```

For Python 3.14 plus NeMo, use the override file:

```bash
uv sync --extra asr-nemo --override packaging/constraints/py314-overrides.txt
```

Optional TTS extras:

```bash
uv sync --extra tts-elevenlabs
uv sync --extra tts-openai
uv sync --extra tts-local
```

MeloTTS runs in a separate managed venv and is installed by `shuvoice setup
--install-missing` when selected.

## First Setup

Interactive path:

```bash
uv run shuvoice wizard
```

Non-interactive checks and setup:

```bash
uv run shuvoice setup
uv run shuvoice setup --install-missing
uv run shuvoice setup --skip-model-download --skip-preflight
uv run shuvoice preflight
```

Local Piper example:

```bash
uv run shuvoice setup --install-missing \
  --tts-local-voice en_US-amy-medium \
  --non-interactive
```

## Service Setup

The packaged unit expects `/usr/bin/shuvoice`. For source installs, copy the
unit and override `ExecStart`:

```bash
mkdir -p ~/.config/systemd/user
cp packaging/systemd/user/shuvoice.service ~/.config/systemd/user/
systemctl --user edit shuvoice.service
```

Override:

```ini
[Service]
ExecStart=
ExecStart=%h/repos/shuvoice/.venv/bin/shuvoice
```

Reload and start:

```bash
systemctl --user daemon-reload
systemctl --user import-environment WAYLAND_DISPLAY DISPLAY XDG_RUNTIME_DIR HYPRLAND_INSTANCE_SIGNATURE DBUS_SESSION_BUS_ADDRESS XDG_CURRENT_DESKTOP XDG_SESSION_TYPE
systemctl --user enable --now shuvoice.service
```

Foreground launch:

```bash
shuvoice
shuvoice run
```
