# AGENTS.md — ShuVoice Developer & Agent Reference

> **Purpose**: Concise source of truth for agents working in this repository.
> Application code is **Rust**. Optional NeMo / Moonshine / MeloTTS engines are
> **isolated Python workers**, never imported into the app process.
>
> Read this before changing crates, features, config contracts, workers, or the
> user service.

---

## Product shape

ShuVoice is a streaming speech-to-text overlay for Hyprland/Wayland with
pluggable ASR/TTS. Push-to-talk → transcript → type/paste into the focused
window. TTS speaks selection/clipboard. STT and TTS are mutually exclusive.

| Layer | Language | Location |
|---|---|---|
| App shell, runtime, UI, control, setup, wizard, Waybar | **Rust** | `crates/*`, bins `shuvoice` / `shuvoice-waybar` |
| Domain config + pure policy | **Rust** | `crates/shuvoice-core` |
| Optional heavy ML engines | **Python workers** | `workers/` via `shuvoice-worker-proto` |

Default desktop Cargo feature: **`desktop`** on `shuvoice-cli`
(`audio` + `asr-sherpa` + `asr-openai` + `ui` + `tts` + `tts-worker`).

ADRs: `docs/adr/0001-rust-modular-monolith.md`,
`docs/adr/0002-versioned-model-workers.md`.
Map: `docs/architecture/RUST_REWRITE.md`.

---

## Crate ownership

| Crate | Responsibility |
|---|---|
| `shuvoice-core` | Config schema/defaults/migrations, domain types, transcript/TTS speed policy, metrics, pure state machines, XDG paths, exit **78** constant |
| `shuvoice-worker-proto` | Versioned framed stdio protocol (Rust side) |
| `shuvoice-control` | Secure Unix socket server/client; stable command vocabulary |
| `shuvoice-io` | Audio (feature), selection, text injection, Hyprland/process adapters, `local.dev` loader |
| `shuvoice-asr` | ASR trait; native **Sherpa** (`sherpa` feature) + **OpenAI Realtime** (`openai`); worker client for NeMo/Moonshine |
| `shuvoice-tts` | TTS trait + providers; CPAL output feature; MeloTTS **worker-proto only** (`worker-proto` feature) |
| `shuvoice-ui` | Headless view models; optional GTK4/layer-shell (`gtk` feature) |
| `shuvoice-app` | Session owner, STT/TTS lifecycle, orchestration |
| `shuvoice-cli` | Composition root, CLI, setup/preflight/wizard glue, Waybar bin, feature gates |

Workspace: root `Cargo.toml`. Default member: `shuvoice-cli`.
`rust-version` and edition come from workspace package metadata.

### Feature surface (`shuvoice-cli`)

| Feature | Role |
|---|---|
| `desktop` (default) | Full packaged set |
| `audio` | CPAL capture |
| `asr-sherpa` | Native static Sherpa |
| `asr-openai` | Native OpenAI Realtime ASR |
| `ui` | GTK host + overlays |
| `tts` | CPAL player + feedback tones |
| `tts-worker` | MeloTTS via worker-proto (requires `tts`) |

`--no-default-features` builds still compile the CLI; `run` / setup composition
**fails closed with exit 78** when the selected backend or UI surface is missing.

---

## Compatibility contracts (do not break casually)

1. **Config**: `config_version = 1`, v0 migration, section/key names in
   `shuvoice-core` `config_section_fields()`, XDG paths, atomic write/backup.
2. **Control socket**: line protocol `OK …` / `ERROR …`; commands in
   `CONTROL_COMMANDS` (`start`, `stop`, `toggle`, `status`, `ping`, `metrics`,
   `debug_status`, `tts_*`).
3. **Exit 78**: `DEPENDENCY_EXIT_CODE` / `RestartPreventExitStatus=78` on the
   user unit — missing features, layer-shell, worker root, Sherpa CUDA, etc.
4. **Wizard defaults**: Parakeet CPU offline-instant + Kokoro 1.25× (see
   `shuvoice_core::config::defaults::wizard`).
5. **Injection policy**: auto/clipboard/direct, watcher + XWayland behavior,
   text replacements, case policy.
6. **STT/TTS exclusion**, overlay namespaces (`stt-overlay`, `tts-overlay`),
   Waybar JSON helper.
7. **Worker protocol v1** framing shared with `workers/shuvoice_worker_proto`.

User-facing config keys live in Rust core — do not invent keys in docs or
callers. Prefer `shuvoice config effective` and `examples/config.toml`.

---

## Runtime paths

| Path | Role |
|---|---|
| `$XDG_CONFIG_HOME/shuvoice/config.toml` | Config |
| `$XDG_CONFIG_HOME/shuvoice/local.dev` | Local secrets/env |
| `$XDG_RUNTIME_DIR/shuvoice/control.sock` | Control socket default |
| `$XDG_DATA_HOME/shuvoice/models/sherpa/<name>/` | Sherpa models |
| `$XDG_DATA_HOME/shuvoice/models/piper/` | Managed Piper voices |
| `$XDG_DATA_HOME/shuvoice/melotts-venv/` | MeloTTS worker venv |
| `$XDG_DATA_HOME/shuvoice/workers-nemo-venv/` | NeMo worker venv |
| `$XDG_DATA_HOME/shuvoice/workers-moonshine-venv/` | Moonshine worker venv |
| `$XDG_DATA_HOME/shuvoice/.wizard-done` | Wizard marker |

Defaults without XDG overrides: `~/.config`, `~/.local/share`.

---

## ASR / TTS implementation map

| Backend | Where it runs |
|---|---|
| Sherpa | In-process **static** `sherpa-onnx` (CPU). `sherpa_provider=cuda` **fails closed**. No Python wheel / RUNPATH repair. |
| OpenAI Realtime ASR | In-process native client |
| NeMo | Worker: `python -m nemo_asr` |
| Moonshine | Worker: `python -m moonshine_asr` |
| ElevenLabs / OpenAI / Kokoro TTS | Native HTTP in `shuvoice-tts` |
| Local Piper | External `piper`/`piper-tts` binary |
| MeloTTS | Worker-proto only: `python -m melotts` (feature `tts-worker`) |

### Worker discovery

Priority:

1. `SHUVOICE_WORKERS_DIR` (absolute UTF-8 workers tree)
2. `/usr/lib/shuvoice/workers`
3. `/usr/libexec/shuvoice/workers`
4. Dev `workers/` — debug builds default on; release needs
   `SHUVOICE_ALLOW_DEV_WORKERS=1`/`true`/`yes`/`on`

Tree must include `shuvoice_worker_proto/` + backend package
(`__init__.py`, `__main__.py`). See `workers/README.md` and
`crates/shuvoice-cli/src/compose/worker_runtime.rs`.

---

## Service safety

Unit: `packaging/systemd/user/shuvoice.service`

```text
ExecStart=/usr/bin/shuvoice
Restart=on-failure
RestartPreventExitStatus=78
Environment=RUST_LOG=info
```

Source installs override `ExecStart` to `target/release/shuvoice`.
After successful wizard completion the CLI may start/restart the user service
when `systemctl --user` is available.

Never treat exit 78 as a flake — fix composition (features, config, workers,
layer-shell), then restart once.

---

## Development commands

```bash
# Application
cargo fmt
cargo check -p shuvoice-cli                 # desktop default
cargo clippy -p shuvoice-cli -- -D warnings
cargo test -p shuvoice-cli

# Feature-off surface
cargo check -p shuvoice-cli --no-default-features
cargo test  -p shuvoice-cli --no-default-features

# Single crate
cargo test -p shuvoice-core
cargo test -p shuvoice-worker-proto

# Optional workers (stdlib unittest; no ML deps for protocol tests)
cd workers && python -m unittest discover -s tests -v
```

Do not run network installs, user services, or cache wipes unless the user
explicitly asks.

---

## Config change checklist

When adding/changing user-facing keys:

1. `crates/shuvoice-core/src/config/` (`model`, `defaults`, section fields, validation)
2. `examples/config.toml` (+ profile examples if needed)
3. `docs/CONFIGURATION.md`, this file, and any wizard defaults in core/UI
4. Tests in `shuvoice-core` / CLI contract tests

---

## What not to do

- Do not import `workers/**` or Python ML stacks into the Rust app.
- Do not reintroduce Sherpa CUDA wheel/RUNPATH “repair” docs or code paths for
  the native static binding — CUDA is fail-closed; use CPU.
- Do not document MeloTTS legacy helper scripts as the production path —
  worker-proto only.
- Do not weaken control-command allowlisting, injection safety, or exit 78.
- Do not edit user untracked files, secrets, or live `~/.config/shuvoice`
  unless asked.

---

## Docs map

| Doc | Audience |
|---|---|
| `README.md` | Users + quick start |
| `docs/INSTALLATION.md` | Install / build / service |
| `docs/CONFIGURATION.md` | Config keys and profiles |
| `docs/TROUBLESHOOTING.md` | Exit 78, audio, layer-shell, workers |
| `docs/WAYBAR.md` | Waybar module |
| `CONTRIBUTING.md` | Contributor workflow |
| `workers/README.md` | Worker protocol + layout |
| `docs/adr/*` | Accepted architecture decisions |

---

## Maintaining this file

Update when crate boundaries, features, config keys, worker discovery, service
exit behavior, or default wizard profiles change. Keep it short; link to ADRs
and code for deep detail.
