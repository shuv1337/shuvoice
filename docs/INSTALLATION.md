# Installation

ShuVoice targets Linux Wayland desktops, with Hyprland as the primary tested
environment. The recommended install path on Arch Linux is the AUR package.

The shipped application is the **Rust** CLI (`shuvoice` / `shuvoice-waybar`).
Optional NeMo, Moonshine, and MeloTTS runtimes are separate worker trees — not
part of the core binary’s link line.

## AUR Package

```bash
yay -S shuvoice-git
```

The package builds `shuvoice-cli` with default **`desktop`** features and
installs:

- `/usr/bin/shuvoice`
- `/usr/bin/shuvoice-waybar`
- user unit `shuvoice.service` (`RestartPreventExitStatus=78`)
- `/usr/lib/shuvoice/workers/` — bundled optional model-worker tree
  (`shuvoice_worker_proto`, `nemo_asr`, `moonshine_asr`, `melotts`; source only)

After install:

```bash
shuvoice wizard
systemctl --user enable --now shuvoice.service
```

Native Sherpa is linked **statically** into the binary (CPU). You do **not**
need `python-sherpa-onnx`, wheel installs, or RUNPATH/CUDA compatibility
patches for the default ASR path.

Optional worker backends (NeMo / Moonshine / MeloTTS) use the bundled workers
tree at `/usr/lib/shuvoice/workers` (or `SHUVOICE_WORKERS_DIR`) plus isolated
venvs created by setup — see [Optional model workers](#optional-model-workers)
and [Troubleshooting](TROUBLESHOOTING.md).

## Source Install

### Prerequisites

| Component | Requirement |
|---|---|
| OS | Linux with Wayland |
| Rust | Workspace `rust-version` (currently **1.92+**), edition 2024 |
| Desktop | Hyprland recommended |
| GPU | Not required for default Sherpa CPU; optional only for worker engines that use it |

Arch packages (desktop overlay + capture + injection). Audio uses **CPAL**
with **PipeWire / ALSA** (not PortAudio, not espeak):

```bash
sudo pacman -S \
  gtk4 gtk4-layer-shell \
  alsa-lib pipewire pipewire-audio pipewire-alsa \
  wtype wl-clipboard
```

Optional helpers:

```bash
sudo pacman -S xdotool          # XWayland injection fallback
# piper / piper-tts             # Local Piper TTS binary (AUR: piper-tts)
# python                        # only if you run optional NeMo/Moonshine/Melo workers
```

### Build

```bash
git clone https://github.com/shuv1337/shuvoice.git
cd shuvoice

# Default features = desktop
#   audio + asr-sherpa + asr-openai + ui + tts + tts-worker
cargo build --release -p shuvoice-cli
```

Binaries:

```text
target/release/shuvoice
target/release/shuvoice-waybar
```

Feature reference (`crates/shuvoice-cli/Cargo.toml`):

| Feature | Role |
|---|---|
| `desktop` (default) | Full packaged set |
| `audio` | CPAL capture |
| `asr-sherpa` | Native static Sherpa |
| `asr-openai` | Native OpenAI Realtime ASR |
| `ui` | GTK4 + layer-shell overlays |
| `tts` | CPAL TTS playback + feedback tones |
| `tts-worker` | MeloTTS via worker-proto only |

Minimal builds compile the CLI surface, but `shuvoice run` **fails closed with
exit 78** when the selected backend or UI surface is not compiled in.

```bash
# Example: headless check surface
cargo build -p shuvoice-cli --no-default-features
```

### First setup

```bash
./target/release/shuvoice wizard
./target/release/shuvoice setup
./target/release/shuvoice setup --install-missing
./target/release/shuvoice setup --skip-model-download --skip-preflight
./target/release/shuvoice preflight
```

Local Piper example:

```bash
./target/release/shuvoice setup --install-missing \
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
ExecStart=%h/repos/shuvoice/target/release/shuvoice
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

Dependency / composition failures exit **78**. The unit sets
`RestartPreventExitStatus=78` so missing features, missing layer-shell, invalid
worker roots, or unsupported Sherpa CUDA config do not restart-loop.

## Paths (XDG)

| Path | Purpose |
|---|---|
| `~/.config/shuvoice/config.toml` | Primary config (`config_version = 1`) |
| `~/.config/shuvoice/local.dev` | Local env (`KEY=value` / `export KEY=value`); process env wins |
| `$XDG_RUNTIME_DIR/shuvoice/control.sock` | Default control socket |
| `~/.local/share/shuvoice/models/sherpa/<name>/` | Auto-downloaded Sherpa models |
| `~/.local/share/shuvoice/models/piper/` | Managed Piper voices |
| `~/.local/share/shuvoice/melotts-venv/` | Isolated MeloTTS worker venv |
| `~/.local/share/shuvoice/workers-nemo-venv/` | Isolated NeMo worker venv |
| `~/.local/share/shuvoice/workers-moonshine-venv/` | Isolated Moonshine worker venv |
| `~/.local/share/shuvoice/.wizard-done` | Wizard completion marker |

Respects `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, and `XDG_RUNTIME_DIR`.

## Optional model workers

Workers are **not** linked into the Rust binary. Production discovery order for
the workers tree:

1. `SHUVOICE_WORKERS_DIR` — absolute UTF-8 path to a valid workers tree
2. `/usr/lib/shuvoice/workers`
3. `/usr/libexec/shuvoice/workers`
4. Dev workspace `workers/` — **debug builds by default**; release only with
   `SHUVOICE_ALLOW_DEV_WORKERS=1` (or `true` / `yes` / `on`)

A valid tree must include `shuvoice_worker_proto/` and the backend module
(`nemo_asr`, `moonshine_asr`, or `melotts`) with `__init__.py` + `__main__.py`.

```bash
# Dev: point at the repo workers tree explicitly
export SHUVOICE_WORKERS_DIR=/path/to/shuvoice/workers

shuvoice setup --install-missing   # creates isolated worker venvs when needed
shuvoice preflight
```

See `workers/README.md` for the framed protocol, fake engines, and unittest
entrypoints. Architecture: `docs/adr/0002-versioned-model-workers.md`.

## Native Sherpa notes

- Provider: **CPU only** on the static in-process binding.
- `sherpa_provider = "cuda"` is **unsupported** and fails closed (setup,
  preflight, and load) with guidance to set `cpu`.
- There is **no** automatic GPU→CPU lie in setup/preflight, and **no** legacy
  Python wheel / `patchelf` RUNPATH repair path for Sherpa.

Recommended default profile (wizard stable instant):

```toml
[asr]
asr_backend = "sherpa"
sherpa_provider = "cpu"
sherpa_model_name = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
instant_mode = true
sherpa_decode_mode = "offline_instant"
```
