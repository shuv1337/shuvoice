# Contributing to ShuVoice

Thanks for your interest in contributing to ShuVoice!

Need logo assets for docs or release notes? See `docs/BRANDING.md`.

## Project shape

ShuVoice is a **Rust modular monolith** (`crates/*`) with optional **Python model
workers** under `workers/`. The application shell, runtime, UI, control plane,
setup, and packaging binaries are Rust. NeMo, Moonshine, and MeloTTS remain
isolated worker processes that speak `shuvoice-worker-proto` — never imported
into the app.

Read first:

- `AGENTS.md` — crate ownership, contracts, tests, service safety
- `docs/adr/0001-rust-modular-monolith.md`
- `docs/adr/0002-versioned-model-workers.md`
- `docs/architecture/RUST_REWRITE.md`
- `workers/README.md` — worker wire format and layout

## Development setup

### Application (Rust)

Requirements:

- Rust toolchain matching `rust-version` in the workspace `Cargo.toml` (currently **1.92+**)
- System libs for the `desktop` feature: GTK4, gtk4-layer-shell, ALSA/PipeWire (CPAL), `wtype`, `wl-clipboard`

```bash
git clone https://github.com/shuv1337/shuvoice.git
cd shuvoice

# Default features = desktop (audio + native Sherpa/OpenAI + UI + TTS + tts-worker)
cargo build -p shuvoice-cli
```

Binaries: `target/debug/shuvoice`, `target/debug/shuvoice-waybar`.

### Optional workers (Python)

Only needed when developing or running NeMo / Moonshine / MeloTTS workers.
The Rust build does **not** depend on this tree.

```bash
# Protocol + golden-frame tests (no ML packages)
cd workers
python -m unittest discover -s tests -v

# Fake engines (no NeMo/Melo/Moonshine installed)
python -m nemo_asr --fake
python -m melotts --fake
python -m moonshine_asr --fake
```

Isolated venvs (created by `shuvoice setup --install-missing` when selected)
live under `$XDG_DATA_HOME/shuvoice/` — see `AGENTS.md` and `workers/README.md`.

## Recommended local checks

Run these before opening a pull request:

```bash
cargo fmt --all -- --check
cargo check -p shuvoice-cli
cargo clippy -p shuvoice-cli -- -D warnings
cargo test -p shuvoice-cli

# When touching worker protocol or workers/**
cd workers && python -m unittest discover -s tests -v
```

Useful feature matrix (mirrors CI intent):

```bash
cargo check -p shuvoice-cli --no-default-features
cargo test  -p shuvoice-cli --no-default-features
cargo check -p shuvoice-cli                 # desktop default
cargo test  -p shuvoice-cli
```

Headless crates (`shuvoice-core`, `shuvoice-control`, `shuvoice-worker-proto`,
etc.) can be tested individually with `cargo test -p <crate>`.

## Commit and PR expectations

- Keep commits focused and descriptive.
- Include tests for behavior changes when practical (Rust unit/integration tests;
  worker stdlib unittest for protocol changes).
- Update documentation for any user-facing change.
- If config keys/defaults change, update **all** of:
  - `crates/shuvoice-core/src/config/` (model + defaults + section fields)
  - `examples/config.toml` (and relevant profile examples)
  - `docs/CONFIGURATION.md` and `AGENTS.md`
- Do not commit generated artifacts (`target/`, `build/`, `dist/`, coverage, caches).
- Do not add Python imports or build-deps from `workers/` into the Rust app.

## Reporting issues

Please open an issue with:

- steps to reproduce
- expected behavior
- actual behavior
- logs and environment details (OS, Rust/`shuvoice --help` version if available,
  `asr_backend` / `tts_backend`, feature set if self-built)

## Code of Conduct

By participating in this project, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
