# Architecture

Architectural decisions, patterns, and conventions for ShuVoice backend work
in the **Rust workspace** and optional **Python workers**.

**What belongs here:** crate boundaries, ASR/TTS contracts, worker-proto rules,
config conventions, feature gates.

**Source of truth:** `AGENTS.md`, `docs/adr/0001-rust-modular-monolith.md`,
`docs/adr/0002-versioned-model-workers.md`, `docs/architecture/RUST_REWRITE.md`.

---

## Runtime shape

```text
Hyprland binding / CLI / Waybar
              │
              ▼
      secure control socket  (shuvoice-control)
              │
              ▼
        session owner        (shuvoice-app)
       ┌──────┼────────┐
       ▼      ▼        ▼
  audio q   ASR      TTS player
  (io)   (asr/native  (tts + optional
          or worker)    worker)
       └──────┬──────────┘
              ▼
        typed session events
       ┌──────┼──────────┐
       ▼      ▼          ▼
   GTK view  metrics   injector
   (ui)     (core)     (io)
```

Composition root: `shuvoice-cli` (`compose/`), default feature set **`desktop`**.

Ownership rules:

- Session owner serializes recording/processing/circuit/mutual-exclusion state.
- Audio callback only enqueues; never blocks on model locks.
- One ASR backend owner performs model calls.
- GTK objects stay on the GLib main thread.
- Injection is serialized and must not log transcript contents on failure.
- Config is immutable after validation except explicit runtime selections
  (TTS voice/speed, etc.).

---

## Crate map

| Crate | Role |
|---|---|
| `shuvoice-core` | Config, types, pure policy |
| `shuvoice-worker-proto` | Framed worker protocol (Rust) |
| `shuvoice-control` | Unix control IPC |
| `shuvoice-io` | Audio, selection, injection |
| `shuvoice-asr` | Native Sherpa/OpenAI + worker ASR client |
| `shuvoice-tts` | Providers + player; Melo worker-proto |
| `shuvoice-ui` | Headless models + GTK overlays |
| `shuvoice-app` | Session runtime |
| `shuvoice-cli` | Bins, setup, composition, features |

The former in-process Python application has been removed. Optional engine
adapters live under `workers/` only.

---

## ASR contract

- Trait / factory: `crates/shuvoice-asr`
- Native: Sherpa (`feature = "sherpa"`, **static CPU** — `cuda` fails closed),
  OpenAI Realtime (`feature = "openai"`)
- Workers: NeMo + Moonshine via `WorkerAsrBackend` + full spawn
  (`PYTHONPATH`, `current_dir`) from `compose/worker_runtime.rs`
- Never use lossy factory attach for production worker spawn
- Capabilities include `wants_raw_audio` (drives app auto-gain bypass)

---

## TTS contract

Implementations live in `crates/shuvoice-tts` (`TtsBackend` trait):

| Method | Purpose |
|---|---|
| `sample_rate_hz()` | Nominal PCM rate |
| `synthesize_stream(request, cancel)` | PCM s16le mono stream |
| `list_voices()` | UI voice list |
| `dependency_errors()` | Missing runtime deps |
| `capabilities()` | Speed flags / ranges |

Audio format:

- Raw PCM int16 mono little-endian unless the backend decodes MP3 first and
  reports the decoded rate
- Typical rates: 24000 (cloud/Kokoro), ~22050 (Piper sidecar/default), 44100 (MeloTTS)

### Registration

- CLI composition: `shuvoice-cli` `compose/tts_adapter.rs` maps
  `shuvoice_core::TtsBackendKind` → `shuvoice_tts` settings/factory
- Config fields: `shuvoice-core` `Config` + `config_section_fields()["tts"]`
- Wizard: `shuvoice-ui` wizard view models + `shuvoice_core::config::defaults::wizard`

### MeloTTS — worker-proto only

- Requires `shuvoice-cli` feature `tts-worker` / `shuvoice-tts` feature
  `worker-proto`
- Spawn: workers tree `melotts` module + isolated venv interpreter
- `melotts_helper_script` must remain unused in production settings
- Fail closed if worker-proto feature or workers root is missing

### Piper

- External binary (`piper` / `piper-tts`), not a Python import
- Managed models: `$XDG_DATA_HOME/shuvoice/models/piper/`

---

## Worker protocol

Wire format v1 (identical Rust/Python):

```text
u32 BE length | u8 kind | payload
```

Kinds: JSON control (`1`), f32 PCM (`2`), i16 PCM (`3`), opaque (`4`).
Length rejected unless `1 ..= 16 MiB` before allocation.

Handshake: `hello` → `hello_ok` + manifest (or `hello_err`).

Reference workers: `workers/{nemo_asr,moonshine_asr,melotts,shuvoice_worker_proto}`.
Tests: `cd workers && python -m unittest discover -s tests -v`.

Discovery env:

| Variable | Role |
|---|---|
| `SHUVOICE_WORKERS_DIR` | Absolute workers tree override |
| `SHUVOICE_ALLOW_DEV_WORKERS` | Release opt-in for repo `workers/` |
| `SHUVOICE_WORKER_FAKE` | Fake engines in workers |
| `SHUVOICE_MELOTTS_VENV` / `SHUVOICE_MELOTTS_DEVICE` | Melo worker env |

---

## Config pattern

1. Add field to `Config` in `crates/shuvoice-core/src/config/model.rs`
2. Register in `config_section_fields()` (`defaults.rs`)
3. Validate in normalize/`validate` paths
4. Add defaults / wizard constants when user-visible
5. Update `examples/config.toml`, `docs/CONFIGURATION.md`, `AGENTS.md`
6. Cover with core + CLI contract tests

---

## Feature-off / exit 78

Composition validates compiled features + layer-shell + worker discovery.
Failures return dependency exit **78** (systemd `RestartPreventExitStatus`).
Messages must stay actionable and avoid leaking secret path bytes where the
code already redacts.

---

## Rejected patterns

- Embedding CPython / PyO3 for NeMo or MeloTTS in the app process
- Per-backend one-off subprocess protocols (use worker-proto)
- Documenting or implementing Sherpa CUDA wheel + RUNPATH repair for the
  static native binding
- Legacy Melo helper scripts as the production path
