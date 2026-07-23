# Environment

Environment variables, external dependencies, and setup notes for agents.

**What belongs here:** required env vars, API key conventions, worker/venv
paths, platform notes.
**What does not belong here:** long-term service port registries (see
`.factory/services.yaml` if present).

---

## Application (Rust)

| Item | Notes |
|---|---|
| Toolchain | Workspace `rust-version` (currently 1.92+), edition 2024 |
| Default features | `desktop` on `shuvoice-cli` |
| Config | `$XDG_CONFIG_HOME/shuvoice/config.toml` |
| Local secrets | `$XDG_CONFIG_HOME/shuvoice/local.dev` (`KEY=value` / `export KEY=value`) |
| Control socket | `$XDG_RUNTIME_DIR/shuvoice/control.sock` |
| Logging | `RUST_LOG` (unit default `info`) |

Common API key env names (values never in `config.toml`):

| Env | Used by |
|---|---|
| `OPENAI_API_KEY` | OpenAI Realtime ASR, OpenAI TTS (default env names) |
| `ELEVENLABS_API_KEY` | ElevenLabs TTS |

Process environment overrides `local.dev`.

---

## Worker discovery env

| Variable | Effect |
|---|---|
| `SHUVOICE_WORKERS_DIR` | Absolute UTF-8 path to a workers tree (highest priority when valid) |
| `SHUVOICE_ALLOW_DEV_WORKERS` | `1`/`true`/`yes`/`on` — allow repo `workers/` probe in **release** builds (debug allows by default) |
| `SHUVOICE_WORKER_FAKE` | Force fake worker engines |
| `SHUVOICE_MELOTTS_VENV` | MeloTTS venv path for worker-side checks |
| `SHUVOICE_MELOTTS_DEVICE` | `auto` / `cpu` / `cuda` |

Packaged roots: `/usr/lib/shuvoice/workers`, `/usr/libexec/shuvoice/workers`.

---

## Isolated worker venvs

Under `$XDG_DATA_HOME/shuvoice/` (default `~/.local/share/shuvoice/`):

| Path | Backend |
|---|---|
| `workers-nemo-venv/` | NeMo ASR worker |
| `workers-moonshine-venv/` | Moonshine ASR worker |
| `melotts-venv/` | MeloTTS worker (override: `tts_melotts_venv_path`) |

MeloTTS notes:

- Prefer Python **3.12** in the isolated venv
- Large footprint (PyTorch-dominated)
- Models typically land in the Hugging Face cache via the worker

Setup automation: `shuvoice setup --install-missing` when the active config
selects these backends.

---

## Model / data paths

| Path | Role |
|---|---|
| `…/models/sherpa/<sherpa_model_name>/` | Native Sherpa auto-download target |
| `…/models/piper/` | Managed Piper `.onnx` + sidecar JSON |
| `…/.wizard-done` | Wizard completion marker |

Native Sherpa is **static CPU**. Do not install CUDA wheels or run RUNPATH
repair for the app binary’s Sherpa path. `sherpa_provider=cuda` fails closed.

---

## System dependencies (desktop)

GTK4, gtk4-layer-shell, ALSA/PipeWire (CPAL capture/playback), `wtype`, `wl-clipboard`.
Optional: `xdotool` (XWayland), `piper`/`piper-tts` (local TTS).

---

## Test commands (no service mutation)

```bash
cargo fmt -- --check
cargo check -p shuvoice-cli
cargo clippy -p shuvoice-cli -- -D warnings
cargo test -p shuvoice-cli

cd workers && python -m unittest discover -s tests -v
```
