# PLAN: ShuVoice production hardening and full implementation

## Objective

Bring the current ShuVoice prototype to a reliable, installable, and operable state on Hyprland by closing gaps identified in the code/plan review:

- bootstrap/runtime dependency failures
- Python compatibility mismatch
- ASR threading safety and audio queue behavior under load
- hotkey robustness and fallback strategy
- packaging/docs/testing required for repeatable deployment

## Current baseline

Implemented and working at prototype level:

- streaming ASR pipeline glue (`shuvoice/asr.py`, `shuvoice/app.py`)
- 100ms capture → 1120ms accumulation (`shuvoice/audio.py`, `shuvoice/app.py`)
- GTK4 layer-shell overlay (`shuvoice/overlay.py`)
- evdev tap/hold hotkey state machine (`shuvoice/hotkey.py`)
- final text injection via `wtype` + `wl-copy` (`shuvoice/typer.py`)

Missing or incomplete:

- robust dependency/bootstrap path (`torch`, `nemo`, system binaries)
- full Python support alignment with NeMo constraints
- synchronization around ASR state resets
- queue overflow policy tuned for low-latency streaming
- Hyprland IPC fallback path from plan not yet implemented
- docs/tests/packaging artifacts

---

## Milestone 1 — Runtime bootstrap + compatibility (critical)

### Tasks

- [x] Align Python support in packaging to NeMo constraints (3.10–3.12)
  - File: `pyproject.toml`
- [x] Ensure Python 3.10 compatibility for config loading (`tomllib` fallback)
  - File: `shuvoice/config.py`
- [x] Add explicit ASR dependency checks with actionable errors instead of traceback explosions
  - File: `shuvoice/asr.py`
- [x] Add CLI preflight mode to verify runtime requirements before full startup
  - File: `shuvoice/__main__.py`
  - Checks: Python version, key imports, `wtype`, `wl-copy`, `libgtk4-layer-shell.so`, hotkey validity

### Validation

```bash
python -m shuvoice --help
python -m shuvoice --preflight
python -m compileall shuvoice
```

---

## Milestone 2 — Streaming stability + thread safety (critical)

### Tasks

- [x] Add ASR state synchronization between hotkey thread (`reset`) and ASR worker (`process_chunk`)
  - File: `shuvoice/app.py`
- [x] Update audio overflow policy to drop oldest queued chunks, preserving freshest audio
  - File: `shuvoice/audio.py`
- [x] Fix overlay click-through signal hookup order (connect before present)
  - File: `shuvoice/overlay.py`
- [x] Add observable logging for drop/backpressure conditions
  - File: `shuvoice/audio.py`

### Validation

```bash
python -m compileall shuvoice
# Manual: hold hotkey while CPU/GPU loaded, verify no crashes and low-latency behavior
```

---

## Milestone 3 — Hotkey robustness + Hyprland fallback (important)

### Tasks

- [x] Validate configured hotkey name early with clear error messaging
  - File: `shuvoice/hotkey.py`
- [x] Support multi-keyboard listening and optional explicit device path
  - Files: `shuvoice/hotkey.py`, `shuvoice/config.py`, `shuvoice/app.py`
- [x] Implement Hyprland IPC fallback control path (no `input` group requirement)
  - Files: `shuvoice/__main__.py`, `shuvoice/app.py`, new helper module(s)
  - Add start/stop/toggle control surface suitable for `bind`/`bindr`

### Validation

```bash
# Manual: multiple keyboards attached
# Manual: evdev path works
# Manual: Hyprland bind/bindr fallback works without input-group access
```

---

## Milestone 4 — Typing UX and safety (important)

### Tasks

- [x] Add configurable output mode: `final_only` (default) vs `streaming_partial`
  - Files: `shuvoice/config.py`, `shuvoice/app.py`
- [x] Optional clipboard preservation/restore around final paste
  - File: `shuvoice/typer.py`
- [x] Add typing failure fallback strategy (retry / direct wtype)
  - File: `shuvoice/typer.py`

### Validation

```bash
# Manual: partial updates replace correctly in focused app
# Manual: final paste works for long/unicode text
# Manual: clipboard preservation option behavior
```

---

## Milestone 5 — Packaging, docs, and testability (important)

### Tasks

- [x] Add user-facing `README.md` with install/run/troubleshooting
  - File: `README.md`
- [x] Add `config.toml` example with all supported keys
  - File: `examples/config.toml` (or similar)
- [x] Add Arch `PKGBUILD` skeleton matching runtime assumptions
  - File: `packaging/PKGBUILD` (or project root PKGBUILD)
- [x] Add smoke test checklist/script for end-to-end validation on Hyprland
  - File: `scripts/smoke-test.sh` + docs
- [x] Add unit tests for:
  - hotkey state machine transitions
  - config loading/flattening and invalid key behavior
  - Files: `tests/test_hotkey.py`, `tests/test_config.py`

### Validation

```bash
pytest
python -m shuvoice --preflight
```

---

## External references used by this plan

- NVIDIA NeMo: https://github.com/NVIDIA/NeMo
- Nemotron model card: https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- wtype: https://github.com/atx/wtype
- gtk4-layer-shell: https://github.com/wmww/gtk4-layer-shell

---

## Execution order

1. Milestone 1 (bootstrap)  
2. Milestone 2 (runtime stability)  
3. Milestone 3 (hotkey robustness + fallback)  
4. Milestone 4 (typing UX)  
5. Milestone 5 (packaging/docs/tests)

This ordering ensures the app can be started/debugged reliably before adding new feature surface area.
