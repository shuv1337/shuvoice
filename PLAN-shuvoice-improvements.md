# PLAN: ShuVoice Improvements — Revised (Post-Review)

## Objective

Address the project findings with a safe implementation sequence:

1. Fix correctness bugs (queue overflow dead path, Python version drift, recording-start race)
2. Improve architecture and maintainability of ASR processing
3. Expand reliable, headless-friendly test coverage
4. Add runtime safety nets and better observability
5. Improve performance and UX
6. Establish packaging/lint/CI hygiene

Target: increase confidence and maintainability while avoiding regressions in the live recording/transcription path.

---

## Review-driven constraints (must follow)

1. **Headless CI compatibility:** default unit tests must not require GTK/GI/layer-shell runtime.
2. **Pure logic stays in pure modules:** testable core logic cannot live only in `app.py`/`overlay.py`.
3. **Characterization before refactor:** add behavior-locking tests before large `_asr_loop` decomposition.
4. **API consistency:** keep `consume_native_chunk()` return type stable across milestones.
5. **Control socket validation must be explicit:** define accepted path rules, not vague “reasonable location”.
6. **ASR failure states must be explicit in status API:** disabled/dead thread states must be surfaced.

---

## Current baseline

- ~2.5k LOC across 10 source files
- 13 tests passing (`pytest -q`)
- Existing tests are currently headless-safe (`audio/config/control/hotkey`)
- Running successfully on Python 3.14.2 while metadata/preflight still enforce `<3.13`

---

## Milestone 0 — Testability foundation (critical prerequisite)

### 0.1 Extract pure modules to avoid GTK-import coupling

**Problem:** `shuvoice/app.py` imports GI/GTK at module load, so unit-testing internals from that module is brittle in CI.

- [x] Create `shuvoice/transcript.py` with pure `prefer_transcript(previous, candidate)`
- [x] Create `shuvoice/utterance_state.py` with pure `_UtteranceState` dataclass (private by convention, testable module)
- [x] Keep `ShuVoiceApp` as orchestration; import pure helpers from these modules

### 0.2 Define test tiers and markers

- [x] Add pytest markers:
  - `unit` (default)
  - `gui` (requires GTK/GI runtime)
- [x] In CI default test command, run only non-GUI tests:
  - `pytest -m "not gui" -v`
- [x] Keep any GTK overlay smoke tests under `@pytest.mark.gui`

### 0.3 Characterization-first guardrails

- [x] Add/update tests for transcript merge behavior and typer behavior before refactoring `_asr_loop`
- [x] Ensure each planned refactor has a failing test opportunity if behavior changes

**Validation:**
```bash
pytest -m "not gui" -v
```

---

## Milestone 1 — Correctness bugs (critical)

### 1.1 Activate audio queue overflow handling with bounded queue

**Files:** `shuvoice/audio.py`, `shuvoice/config.py`, `examples/config.toml`, `shuvoice/app.py`, `tests/test_audio.py`

- [x] Add module constant in `audio.py`:
  - `DEFAULT_AUDIO_QUEUE_MAX_SIZE = 200`  
  (20s at 100ms/chunk; safer latency profile than 500 while still generous)
- [x] Change queue init from unbounded to finite:
  - `queue.Queue(maxsize=audio_queue_max_size)`
- [x] Add config field:
  - `audio_queue_max_size: int = 200`
- [x] Validate config value (e.g. minimum 1)
- [x] Wire config -> `AudioCapture(...)` from `ShuVoiceApp.__init__`
- [x] Keep existing drop-oldest overflow behavior and make it testable
- [x] Add tests:
  - overflow path triggers when full
  - oldest dropped, newest retained
  - `_dropped_chunks` increments
  - `clear()` empties queue

### 1.2 Fix Python version constraint drift

**Files:** `pyproject.toml`, `shuvoice/__main__.py`, `README.md`

- [x] Set `requires-python = ">=3.10"`
- [x] Update preflight `check_python()` to require only `>=3.10`
- [x] Update README Python requirement line to `>=3.10`

### 1.3 Fix race window in `_on_recording_start`

**File:** `shuvoice/app.py`

- [x] Move initial `self.audio.clear()` inside `_asr_lock`
- [x] After `self.asr.reset()`, perform second `self.audio.clear()` drain
- [x] Keep behavior idempotent when already recording

### 1.4 Project root cleanup (single source of truth)

- [x] Delete stale `test_script.py` (tracked once, here only)

**Validation:**
```bash
pytest tests/test_audio.py -v
python -m shuvoice --preflight
python -m compileall shuvoice/
```

---

## Milestone 2 — Coverage expansion before major refactor (important)

### 2.1 Transcript tests (pure module)

**Files:** `shuvoice/transcript.py`, `tests/test_transcript.py`

- [x] Normal growth: `hello -> hello world`
- [x] Regression rejection: longer previous beats shorter candidate
- [x] Empty/whitespace handling
- [x] Stitching overlap behavior
- [x] False-positive guard (short overlap does not stitch)
- [x] Rewrite acceptance (longer contextual rewrite can replace)
- [x] Equal-length divergent determinism
- [x] Increase overlap threshold from 5 to 8 chars (document and test)

### 2.2 Typer tests

**File:** `tests/test_typer.py`

- [x] Mock `subprocess.run`
- [x] Cover `update_partial()`, `commit_final()`, `_run()` retries, `reset()`
- [x] Include clipboard preservation cases
- [x] Include fallback-to-direct-type path

### 2.3 ASR engine mocked tests

**File:** `tests/test_asr.py`

- [x] `dependency_errors()` when deps missing / present
- [x] `_normalize_transcript_item()` matrix
- [x] `process_chunk()` and `reset()` error when model unloaded

### 2.4 Audio tests extension

**File:** `tests/test_audio.py`

- [x] Queue overflow behavior with finite maxsize
- [x] `_select_input_device()` preference for Pulse/PipeWire
- [x] `audio_rms()` utility tests (empty, zeros, known signal)

### 2.5 Utterance state tests (pure module)

**Files:** `shuvoice/utterance_state.py`, `tests/test_utterance_state.py`

- [x] `reset()` clears all fields
- [x] `add_chunk()` increments totals
- [x] `consume_native_chunk(native)` correctness (chunk+remainder+has_more)

### 2.6 Overlay tests split by tier

- [x] Add headless-safe tests for any pure overlay state logic (non-GUI)
- [x] Optional `tests/test_overlay_gui.py` with `@pytest.mark.gui` for direct GI calls

**Validation:**
```bash
pytest -m "not gui" -v
# Optional local GUI smoke:
pytest -m gui -v
```

---

## Milestone 3 — ASR loop decomposition and utility extraction (important)

**Primary file:** `shuvoice/app.py`

### 3.1 Extract shared RMS helper

- [x] Add `audio_rms(audio: np.ndarray) -> float` in `shuvoice/audio.py` (or `shuvoice/utils.py`)
- [x] Replace inline RMS calculations in app loop with helper calls

### 3.2 Use extracted utterance state object

- [x] Import `_UtteranceState` from `shuvoice/utterance_state.py`
- [x] Replace ad-hoc local variable resets with `state.reset(...)`
- [x] Replace manual buffer bookkeeping with `state.add_chunk(...)` / `state.consume_native_chunk(...)`

### 3.3 Decompose `_asr_loop` into focused methods

- [x] `_process_recording_chunks(state)`
- [x] `_handle_recording_stop(state)`
- [x] `_drain_and_buffer(state)`
- [x] `_flush_tail_silence(state)`
- [x] `_commit_utterance(state)`
- [x] `_apply_utterance_gain(audio, gain)`
- [x] `_update_noise_floor(chunk_rms)`
- [x] Keep `_asr_loop` readable state-machine driver (~60–90 lines, clarity over strict size)

### 3.4 Keep `consume_native_chunk` API consistent

- [x] Stable signature across all milestones:
  - `consume_native_chunk(native: int) -> tuple[np.ndarray, bool]`
- [x] `bool` always means `has_more_native_chunks`

**Validation:**
```bash
pytest -m "not gui" -v
```

---

## Milestone 4 — Runtime safety and observability (important)

### 4.1 ASR failure circuit breaker with explicit state semantics

**File:** `shuvoice/app.py`

- [x] Add:
  - `_consecutive_asr_failures: int = 0`
  - `_ASR_MAX_FAILURES = 10`
  - `_asr_disabled: bool = False`
- [x] On successful `process_chunk`: reset failure counter
- [x] On failed `process_chunk`: increment counter
- [x] On threshold reached:
  - log `CRITICAL`
  - set `_asr_disabled=True`
  - clear `_recording`
  - overlay: `⚠ ASR error — restart ShuVoice`
- [x] On next recording start while disabled:
  - attempt one `asr.reset()`
  - if success: clear disabled+counter and continue start
  - if failure: remain disabled and refuse recording start

### 4.2 Thread crash detection for ASR and hotkey workers

- [x] Add health flags:
  - `_asr_thread_alive: bool = True`
  - `_hotkey_thread_alive: bool = True` (when hotkey thread enabled)
- [x] Wrap top-level worker bodies in fail-safe try/except with `CRITICAL` logs
- [x] On crash: set corresponding alive flag false and show overlay error if relevant

### 4.3 Status API must surface failure states

- [x] Expand `_recording_status()` to return:
  - `recording`
  - `idle`
  - `error:asr_disabled`
  - `error:asr_thread_dead`
  - `error:hotkey_thread_dead` (if applicable)

### 4.4 Journald-aware logging format

**File:** `shuvoice/__main__.py`

- [x] Detect systemd journal via `JOURNAL_STREAM`
- [x] Use no-timestamp format under journald, timestamped format otherwise

---

## Milestone 5 — Performance improvements (moderate)

### 5.1 Optimize buffer concatenation without API changes

**File:** `shuvoice/utterance_state.py`

- [x] In `consume_native_chunk(native)`, avoid `np.concatenate()` when one buffer element is already large enough
- [x] Preserve return type:
  ```python
  def consume_native_chunk(self, native: int) -> tuple[np.ndarray, bool]:
      if len(self.buffer) == 1 and len(self.buffer[0]) >= native:
          audio_data = self.buffer[0]
      else:
          audio_data = np.concatenate(self.buffer)
      to_process = audio_data[:native]
      remainder = audio_data[native:]
      self.buffer = [remainder] if len(remainder) > 0 else []
      self.total = len(remainder)
      return to_process, self.total >= native
  ```

### 5.2 Batch backspace commands in typer

**File:** `shuvoice/typer.py`

- [x] Batch backspace keystrokes in `_backspace_partial()` (e.g. chunks of 50)
- [x] Apply same batching strategy in `update_partial()` backspace path
- [x] Keep retry semantics unchanged

---

## Milestone 6 — UX improvements (nice to have)

### 6.1 Audio feedback tones on start/stop

**Files:** `shuvoice/feedback.py`, `shuvoice/config.py`, `shuvoice/app.py`, `examples/config.toml`, `tests/test_feedback.py`

- [x] Add config:
  - `audio_feedback: bool = True`
  - `feedback_start_freq`, `feedback_stop_freq`, `feedback_duration_ms`, `feedback_volume`
- [x] Implement `play_tone(freq, duration_ms, volume)` (non-blocking, exception-safe)
- [x] Trigger tone on recording start/stop
- [x] Add unit test for generated array length + `sd.play` call

### 6.2 Overlay state indication

**Files:** `shuvoice/overlay.py`, `shuvoice/app.py`

- [x] Add states: `listening`, `processing`, `error`
- [x] Add CSS classes and `set_state()` API in overlay
- [x] Wire transitions in app lifecycle events

### 6.3 Basic post-processing capitalization

**Files:** `shuvoice/postprocess.py`, `shuvoice/config.py`, `shuvoice/app.py`, `tests/test_postprocess.py`, `examples/config.toml`

- [x] Add `auto_capitalize: bool = True`
- [x] Implement `capitalize_first(text)`
- [x] Apply only to final committed text (config-gated)

---

## Milestone 7 — Packaging, linting, CI, and path hardening (important)

### 7.1 Hygiene

- [x] Add `.jules/` and `.Jules/` to `.gitignore`
- [x] Add `shuvoice/py.typed` marker
- [x] Ensure package includes marker (setuptools package-data config)

### 7.2 Ruff linting

**File:** `pyproject.toml`

- [x] Add `ruff>=0.9` to dev dependencies
- [x] Add Ruff config (`line-length`, lint select set, formatting settings)
- [x] Run:
  ```bash
  ruff check --fix shuvoice/ tests/
  ruff format shuvoice/ tests/
  ```

### 7.3 CI workflow (headless-safe by default)

**File:** `.github/workflows/ci.yml`

- [x] `lint` job (ruff check + format --check)
- [x] `unit` job matrix (Python 3.10–3.13 minimum):
  - install required system deps for current tests
  - run `pytest -m "not gui" -v`
- [x] Optional `gui-smoke` job (manual trigger or separate workflow):
  - install GTK/GI/layer-shell deps
  - run `pytest -m gui -v`

### 7.4 Control socket path validation (explicit policy)

**File:** `shuvoice/control.py`, tests in `tests/test_control.py`

- [x] `resolve_control_socket_path(path)` rules:
  - if custom path provided, require absolute path
  - reject directory paths
  - require `.sock` suffix
  - resolve parent and enforce allowed roots:
    - `$XDG_RUNTIME_DIR` (preferred)
    - `/tmp` (fallback)
  - create parent directory with `0700`
- [x] Add tests for:
  - directory rejection
  - relative path rejection
  - outside-root rejection
  - valid runtime/tmp paths accepted

---

## Revised execution order

```text
Milestone 0  (testability foundation, CI-safe test tiers)
    ↓
Milestone 1  (critical bug fixes)
    ↓
Milestone 2  (coverage expansion / characterization)
    ↓
Milestone 3  (ASR-loop decomposition)
    ↓
Milestone 4  (runtime safety + status semantics)
    ↓
Milestone 5  (performance)
    ↓
Milestone 6  (UX)
    ↓
Milestone 7  (packaging/lint/CI hardening; some parts can run earlier)
```

---

## End-to-end validation checklist

```bash
# Lint
ruff check shuvoice/ tests/
ruff format --check shuvoice/ tests/

# Unit tests (CI-equivalent)
pytest -m "not gui" -v --tb=short

# Optional GUI tests (local/GUI-capable runner)
pytest -m gui -v --tb=short

# Preflight (Python 3.14 should pass Python-version gate)
python -m shuvoice --preflight

# Compile
python -m compileall shuvoice/

# Manual smoke
./scripts/smoke-test.sh

# Manual focused checks
# - rapid start/stop race
# - long recording overflow behavior
# - forced ASR failures -> circuit breaker + status/error overlay
# - control socket invalid-path rejection
# - overlay state colors + feedback tones
```

---

## Files created/modified summary (revised)

| File | Action | Milestone |
|------|--------|-----------|
| `shuvoice/audio.py` | bounded queue, `audio_rms`, queue config plumbing | M1, M3 |
| `shuvoice/config.py` | `audio_queue_max_size` + UX fields | M1, M6 |
| `shuvoice/app.py` | race fix, decomposition, runtime safety, state/status handling | M1, M3, M4 |
| `shuvoice/transcript.py` | **new** pure transcript merge helper | M0, M2 |
| `shuvoice/utterance_state.py` | **new** pure utterance state dataclass | M0, M3, M5 |
| `shuvoice/typer.py` | backspace batching | M5 |
| `shuvoice/overlay.py` | overlay state API/CSS | M6 |
| `shuvoice/feedback.py` | **new** tone feedback helper | M6 |
| `shuvoice/postprocess.py` | **new** capitalization helper | M6 |
| `shuvoice/control.py` | explicit socket path policy validation | M7 |
| `shuvoice/__main__.py` | Python version check + journald-aware logging | M1, M4 |
| `pyproject.toml` | Python floor only, dev deps, Ruff, pytest markers, package-data | M1, M7 |
| `README.md` | Python requirement update and docs alignment | M1 |
| `.gitignore` | add `.jules/` and `.Jules/` | M7 |
| `shuvoice/py.typed` | **new** marker | M7 |
| `tests/test_transcript.py` | **new** transcript tests | M2 |
| `tests/test_typer.py` | **new** typer tests | M2 |
| `tests/test_asr.py` | **new** ASR mocked tests | M2 |
| `tests/test_audio.py` | extend with overflow/device/rms tests | M1, M2 |
| `tests/test_utterance_state.py` | **new** state tests | M2 |
| `tests/test_overlay_gui.py` | **new optional** GUI smoke tests | M2 |
| `.github/workflows/ci.yml` | **new** lint + unit CI, optional GUI workflow strategy | M7 |
| `examples/config.toml` | add queue + UX config fields | M1, M6 |
| `test_script.py` | **delete** | M1 |

---

## External references

- Python queue maxsize: https://docs.python.org/3/library/queue.html#queue.Queue
- PEP 561 (`py.typed`): https://peps.python.org/pep-0561/
- Ruff docs: https://docs.astral.sh/ruff/
- GitHub Actions setup-python: https://github.com/actions/setup-python
- sounddevice play(): https://python-sounddevice.readthedocs.io/en/latest/#sounddevice.play
- NeMo streaming ASR docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/
