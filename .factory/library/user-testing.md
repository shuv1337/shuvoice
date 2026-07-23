# User Testing

Testing surface, resource cost classification, and validation approach.

**What belongs here:** How to test the application, what surfaces exist, concurrency limits.

---

## Validation Surface

ShuVoice is a desktop Wayland/GTK4 application with a **Rust** shell and optional
**Python model workers**. Primary automated surfaces:

1. **Cargo tests (primary)** — domain, control, composition, CLI contracts.
   ```bash
   cargo test -p shuvoice-cli
   cargo test -p shuvoice-core
   cargo test -p shuvoice-worker-proto
   # feature-off surface
   cargo test -p shuvoice-cli --no-default-features
   ```
2. **Cargo check / clippy / fmt** — compile and lint gates.
   ```bash
   cargo fmt --all -- --check
   cargo check -p shuvoice-cli
   cargo clippy -p shuvoice-cli -- -D warnings
   ```
3. **Optional worker protocol tests** — stdlib unittest, no ML packages required.
   ```bash
   cd workers && python -m unittest discover -s tests -v
   ```
4. **CLI smoke (manual / harness)** — `shuvoice preflight`, `shuvoice config effective`,
   `shuvoice control status` against a running instance when the user allows service use.

Do **not** drive current work through legacy `uv run pytest tests/` or
in-process `shuvoice/**` Python app modules. Historical Factory validation
artifacts under `.factory/validation/` may remain for archive; **do not** treat
them as the live test entrypoint.

### Surfaces NOT testable in automation

- GTK4 overlay UI (requires Wayland + layer-shell)
- Audio capture/playback (requires PipeWire/ALSA + hardware)
- Push-to-talk interaction (requires keyboard + Hyprland)
- Real NeMo / MeloTTS / Moonshine model inference (optional workers; heavy deps)

## Validation Concurrency

- Machine: multi-core desktop; cargo and worker unittest are the light path
- Max concurrent validators: **4** (prefer separate crates / worker test files)
- No browser or heavy GUI testing in the default agent path

## Optional workers / MeloTTS

Production MeloTTS is **worker-proto only** (`python -m melotts` under a workers
tree). Protocol and framing tests live in `workers/tests/` with fake engines.
Host discovery/spawn coverage lives in `shuvoice-cli` / `shuvoice-tts` cargo
tests. Do not require a real Melo venv for protocol unit tests.

## Flow Validator Guidance: cargo + workers unittest

**Surface**: Rust crates + `workers/tests`

**Tools**: `cargo test` / `cargo clippy` / `cargo fmt`; `python -m unittest`

**Isolation**: Separate crates or worker test modules; avoid mutating live
`~/.config/shuvoice` or user systemd units unless the user explicitly asked.

**Boundaries**:

- Do NOT modify source unless implementing an assigned change
- Do NOT install ML packages or wipe caches unless asked
- Prefer headless cargo tests and workers unittest
- Capture command exit codes and failing test names as evidence

**Concurrency**: Safe to run distinct crate test packages in parallel when
resources allow.

**Evidence**: Full command lines + pass/fail summary mapped to assertions.
