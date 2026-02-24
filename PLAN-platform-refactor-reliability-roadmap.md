# PLAN: ShuVoice Platform Refactor & Reliability Roadmap

## Scope

This plan covers all improvement themes from the architecture review:

1. Config durability + migrations
2. Refactor monolith hotspots (`app`, `wizard`, `wizard_state`, CLI, `waybar`)
3. CLI redesign with subcommands
4. Backend contract hardening
5. Observability + diagnostics
6. Docs drift prevention in CI
7. Targeted performance wins (diff-based partial typing, regex caching)

> This is the **canonical** plan. It supersedes
> `PLAN-refactor-performance-enhancements-roadmap.md` (which covered the same
> themes in an earlier draft). That file should be deleted once work begins.

> This plan is implementation-only guidance. It does **not** include code changes.

---

## Current-State Baseline (evidence)

### Code concentration hotspots

| Module | LOC | Notes |
|---|---:|---|
| `shuvoice/app.py` | 822 | Core orchestration + ASR loop |
| `shuvoice/wizard.py` | 602 | GTK wizard UI + business logic interleaved |
| `shuvoice/wizard_state.py` | 554 | Config writes, keybind edits, markers |
| `shuvoice/waybar.py` | 487 | Waybar integration, keybind detection |
| `shuvoice/__main__.py` | 479 | CLI parsing + all command routing |
| `shuvoice/config.py` | 296 | Dataclass, validation, loading |

### Prior extraction work (already done)

The following modules have already been extracted from `app.py` and are clean,
tested, and should be **preserved as-is** (not re-extracted):

| Module | LOC | Purpose |
|---|---:|---|
| `shuvoice/utterance_state.py` | 50+ | `_UtteranceState` dataclass |
| `shuvoice/streaming_health.py` | 30 | `should_trigger_stall_flush()` heuristic |
| `shuvoice/postprocess.py` | 44 | `apply_text_replacements()`, `capitalize_first()` |
| `shuvoice/transcript.py` | 128 | `prefer_transcript()` logic |
| `shuvoice/feedback.py` | 50 | `play_tone()` audio feedback |
| `shuvoice/overlay_state.py` | 21 | Overlay state enum |
| `shuvoice/audio.py` | 218 | `AudioCapture`, `audio_rms()` |
| `shuvoice/typer.py` | 157 | `StreamingTyper` |

The remaining 822 lines in `app.py` are the genuinely coupled orchestration
layer (state machine transitions, chunk buffering, gain, flush strategies).
Further extraction should target these internal concerns.

### Relevant current behavior

- Config load/validation lives in `shuvoice/config.py` (`Config.load`, `__post_init__`)
- `DEFAULT_TEXT_REPLACEMENTS` (37 lines) and `_default_text_replacements()` are in `config.py`
- Wizard config writes are in `shuvoice/wizard_state.py` (`write_config`, `_upsert_asr_key`)
- CLI command routing currently lives in `shuvoice/__main__.py`
- Control IPC is in `shuvoice/control.py`
- ASR backend contract is in `shuvoice/asr_base.py` — `wants_raw_audio` already exists as a property
- Backend registry is in `shuvoice/asr.py` — note: `ASREngine` is a compatibility export that eagerly resolves NeMo today; replace with a lazy deprecated shim before full removal
- Waybar integration is in `shuvoice/waybar.py`
- CI pipeline currently in `.github/workflows/ci.yml`

### Test baseline

- **209 tests** collected (headless tier green)
- Test counts per key module:
  - `test_config.py`: 18 tests
  - `test_config_validation.py`: 8 tests
  - `test_app_flow.py`: 17 tests
  - `test_wizard.py`: 27 tests
  - `test_wizard_ui.py`: 4 tests
  - `test_asr.py`: 16 tests
  - `test_control.py`: 10 tests
  - `test_main_cli.py`: **1 test** ← critical gap

### Operational risks to address

- Config file durability/recoverability is limited (no explicit schema version or migration framework)
- Large modules increase regression risk and reduce test isolation
- One large CLI parser makes behavior coupling hard to reason about
- Waybar module (487 LOC) blends UI formatting, systemd service management, and Hyprland detection
- Observability is mostly log-line based (limited structured runtime metrics)
- Docs/examples can drift from config/code behavior over time

---

## Goals & Success Criteria

### Primary goals

- Safer, versioned config lifecycle with predictable upgrades and recovery
- Smaller, testable runtime modules with clearer responsibilities
- Subcommand-based CLI UX with backward compatibility path
- Explicit backend capability contract and consistent diagnostics
- Structured runtime metrics and easier field debugging
- Automated checks to keep docs aligned with code
- Targeted performance wins for partial typing and regex processing

### Program-level Definition of Done

- [x] All existing test suites remain green (`pytest`, `ruff`)
- [x] New tests cover migration, CLI routing, metrics, and refactored modules
- [x] No user-visible regressions for current commands (`--control`, `--wizard`, standard run)
- [x] README/examples/AGENTS are updated and CI enforces key drift checks
- [x] Rollback instructions documented per phase

---

## Guiding Constraints

- Keep Python dependency surface minimal unless justified.
- Preserve current package entry points (`shuvoice`, `shuvoice-waybar`) in `pyproject.toml`.
- Keep compatibility with user systemd workflow (`packaging/systemd/user/shuvoice.service`).
- Do incremental PRs with passing tests at each step (no mega-merge).
- Preserve existing extracted modules (`utterance_state.py`, `streaming_health.py`, etc.) — extend, don't replace.

---

## Implementation Order (Milestones)

| Milestone | Focus | Depends On |
|---|---|---|
| M0 | Baseline + test harness prep | none |
| M1 | Config durability + migrations | M0 |
| M2a | CLI parser/subcommand scaffolding | M0 |
| M2b | CLI config subcommands | M1, M2a |
| M3 | Runtime refactor (`app.py`) | M1 |
| M4 | Wizard and wizard_state refactor | M1 |
| M5 | Waybar integration refactor | M1 |
| M6 | Backend contract hardening | M3 |
| M7 | Observability + diagnostics | M3, M6 |
| M8 | Performance wins (typing, regex) | M3, M5 |
| M9 | Docs drift CI checks + doc refresh | M1–M8 |

> **Dependency corrections from review**: M4 (Wizard) depends on M1 only (not
> M3) — wizard and app.py share zero modules. M2 is split: parser restructure
> (M2a) is independent of config versioning; only config subcommands (M2b)
> need M1.

---

## Workstream A — Config Durability + Migrations (Highest Priority)

### Target design

- Introduce explicit config schema versioning.
- Separate concerns: parsing, migration, validation, writing.
- Ensure atomic writes and backup/restore path for wizard and future config-edit commands.
- Add "effective config" inspection command.

> **Note**: `DEFAULT_TEXT_REPLACEMENTS` and `_default_text_replacements()` stay
> in `config.py`. They are only 37 lines and tightly coupled with `__post_init__`
> validation. Splitting to a separate `config_defaults.py` adds import
> complexity for minimal gain.

### Planned file changes

- **Refactor:**
  - `shuvoice/config.py` — add `config_version` field, schema version constant
  - `shuvoice/wizard_state.py` — switch to `config_io` API
  - `shuvoice/__main__.py` — add config subcommand routing (in M2b)
- **New modules:**
  - `shuvoice/config_io.py` — atomic read/write, backup, schema version handling
  - `shuvoice/config_migrations.py` — migration registry and stepwise transforms
- **Tests:**
  - `tests/test_config.py` (update)
  - `tests/test_config_validation.py` (update)
  - `tests/test_config_io.py` (new)
  - `tests/test_config_migrations.py` (new)

### Tasks

- [x] A1. Define config schema metadata
  - [x] Add `config_version = 1` top-level key in config.toml
  - [x] Add `CURRENT_CONFIG_VERSION` constant in `config.py`
  - [x] Document schema policy (when to bump, what is compatible)

- [x] A2. Implement `config_io.py`
  - [x] `load_raw(path) -> dict` — parse TOML with schema version detection
  - [x] `write_atomic(path, data)` — `tempfile` + `fsync` + `rename`
  - [x] `backup_config(path)` — copy to `config.toml.bak-<timestamp>`
  - [x] Recovery path if write fails midway

- [x] A3. Implement migration framework (`config_migrations.py`)
  - [x] Migration registry: `dict[int, Callable[[dict], dict]]`
  - [x] `migrate_to_latest(raw_config)` with stepwise transforms
  - [x] Migration report object (`from_version`, `to_version`, `changed_keys`)

- [x] A4. Integrate wizard writes
  - [x] Replace `_upsert_asr_key()` line-editing in `wizard_state.py` with `config_io` API
  - [x] Preserve existing behavior for `overwrite_existing` toggle
  - [x] Keep keybind auto-edit logic separate from config write logic

- [x] A5. Expand test coverage
  - [x] Missing file → default schema version + defaults
  - [x] Old version file → migrates correctly
  - [x] Unknown keys policy verified (ignored/warned as chosen)
  - [x] Backup/atomic write behavior tested under failure simulation
  - [x] Round-trip: load → migrate → write → reload produces identical config

### Rollback

Revert `config_io.py` and `config_migrations.py`, restore `wizard_state.py`
`_upsert_asr_key()` calls. Config files without `config_version` key continue
to work (treated as version 0/unversioned).

### Validation

```bash
uv run ruff check shuvoice tests
uv run pytest tests/test_config.py tests/test_config_validation.py tests/test_config_io.py tests/test_config_migrations.py -v
uv run pytest -m "not gui" -v
```

### Acceptance criteria

- [x] Config migrations are deterministic and tested
- [x] Wizard config writes use atomic/backup path
- [x] Unversioned config files load without error (backward compatible)

---

## Workstream B — CLI Redesign with Subcommands

### Target design

Move from a single large argument surface to explicit subcommands while preserving current workflows.

Recommended command structure:

- `shuvoice run` (default runtime)
- `shuvoice control start|stop|toggle|status|ping`
- `shuvoice preflight`
- `shuvoice wizard`
- `shuvoice config effective|path|validate` (needs M1/config_io)
- `shuvoice model download`
- `shuvoice audio list-devices`

### Planned file changes

- **Refactor:** `shuvoice/__main__.py` — thin dispatcher only
- **New modules:**
  - `shuvoice/cli/__init__.py`
  - `shuvoice/cli/parser.py` — subcommand parser factory
  - `shuvoice/cli/commands/run.py`
  - `shuvoice/cli/commands/control.py`
  - `shuvoice/cli/commands/preflight.py`
  - `shuvoice/cli/commands/config.py` (M2b — depends on config_io)
  - `shuvoice/cli/commands/model.py`
  - `shuvoice/cli/commands/audio.py`
- **Tests:**
  - `tests/test_main_cli.py` (expand — currently only 1 test)
  - `tests/test_cli_subcommands.py` (new)

### Tasks

#### M2a — Parser scaffolding (depends: M0)

- [x] B1. Introduce `shuvoice/cli/parser.py` with subcommand dispatcher
- [x] B2. Move each command path into dedicated handler module
- [x] B3. Preserve legacy top-level flags for one compatibility cycle
  - [x] Map `--control`, `--preflight`, `--wizard`, `--list-audio-devices`, `--download-model` to new handlers
  - [x] Emit deprecation warnings where appropriate
- [x] B4. Keep runtime default behavior unchanged when no subcommand is provided
- [x] B5. Update help text and examples
- [x] B6. Bring `test_main_cli.py` to ≥10 tests covering all legacy flag paths

#### M2b — Config subcommands (depends: M1, M2a)

- [x] B7. Implement `shuvoice config effective` (merged defaults + user config)
- [x] B8. Implement `shuvoice config path` and `shuvoice config validate`

### Rollback

Revert `shuvoice/cli/` package; restore monolithic `__main__.py`. Legacy
flags never stop working during the compatibility cycle so user binds are safe.

### Validation

```bash
uv run pytest tests/test_main_cli.py tests/test_cli_subcommands.py -v
python -m shuvoice --help
python -m shuvoice control --help
python -m shuvoice config --help
# Legacy compat
python -m shuvoice --preflight
python -m shuvoice --control status
```

### Acceptance criteria

- [x] Existing user commands continue to work (legacy flags → deprecation warning)
- [x] `--help` is shorter and grouped by subcommand
- [x] CLI logic in `__main__.py` reduced to ≤50 lines (dispatcher + logging setup)
- [x] ≥10 CLI tests covering all command paths

---

## Workstream C — Runtime Refactor (`app.py`)

### Target design

Break remaining orchestration concerns into cohesive units with explicit state
machine boundaries. Build on already-extracted modules rather than duplicating.

Proposed new modules:

- `shuvoice/runtime/__init__.py`
- `shuvoice/runtime/state_machine.py` — recording state transitions (start/stop/toggle)
- `shuvoice/runtime/chunk_pipeline.py` — chunk buffering + gain + ASR dispatch
- `shuvoice/runtime/flush_policy.py` — tail flush/noise strategies (extracts `_flush_tail_silence`, `_make_flush_noise`)

Keep `ShuVoiceApp` as orchestration facade. **Preserve** existing extracted
modules:

- `utterance_state.py` — `_UtteranceState` stays where it is
- `streaming_health.py` — `should_trigger_stall_flush()` stays where it is
- `postprocess.py`, `transcript.py`, `feedback.py` — unchanged

### Planned file changes

- **Refactor:** `shuvoice/app.py`
- **New:** `shuvoice/runtime/__init__.py`, `state_machine.py`, `chunk_pipeline.py`, `flush_policy.py`
- **Cleanup:** Replace eager `ASREngine` export in `shuvoice/asr.py` with a lazy compatibility shim + deprecation warning
- **Tests:**
  - `tests/test_app_flow.py` (adapt)
  - `tests/test_runtime_state_machine.py` (new)
  - `tests/test_runtime_chunk_pipeline.py` (new)
  - `tests/test_runtime_flush_policy.py` (new)

### Tasks

- [x] C1. Extract recording state transitions (`_on_recording_start`, `_on_recording_stop`, `_on_recording_toggle`, `_recording_status`) into `runtime/state_machine.py`
- [x] C2. Extract chunk processing pipeline (`_append_recording_chunk`, `_transcribe_native_chunk`, `_process_recording_chunks`, `_drain_and_buffer`, `_apply_utterance_gain`) into `runtime/chunk_pipeline.py`
- [x] C3. Extract tail flush strategy (`_flush_tail_silence`, `_make_flush_noise`, `_flush_streaming_stall`) into `runtime/flush_policy.py`
- [x] C4. Keep thread/lock semantics equivalent; add regression tests
- [x] C5. Reduce direct side effects (overlay/typer updates) via callback interface
- [x] C6. Replace eager `ASREngine` export with lazy compatibility shim and deprecation warning (no eager NeMo import)

### Rollback

Inline extracted modules back into `app.py`. Revert is safe because the
public API (`ShuVoiceApp`) never changes.

### Validation

```bash
uv run pytest tests/test_app_flow.py tests/test_runtime_state_machine.py tests/test_runtime_chunk_pipeline.py tests/test_runtime_flush_policy.py -v
uv run pytest -m "not gui" -v
```

### Acceptance criteria

- [x] `app.py` reduced to ≤300 lines (orchestration facade only)
- [x] No regression in recording lifecycle behavior
- [x] ASR error-recovery behavior remains intact and tested

---

## Workstream D — Wizard + Wizard State Refactor

> **Depends on M1 only** — wizard and app.py share zero modules.

### Target design

Separate UI rendering from pure business logic and config/hyprland side effects.

Proposed split (new `shuvoice/wizard/` package):

- `shuvoice/wizard/__init__.py` — re-exports `WelcomeWizard` for backward compat
- `shuvoice/wizard/ui.py` — GTK page composition
- `shuvoice/wizard/flow.py` — step transitions, state
- `shuvoice/wizard/actions.py` — finish actions (write config, write marker)
- `shuvoice/wizard/hyprland.py` — keybind parse/edit logic (from `wizard_state.py`)

### Migration step

The current `shuvoice/wizard.py` single-file must become `shuvoice/wizard/__init__.py`
(or thin re-export). All consumers import `from shuvoice.wizard import WelcomeWizard`
— the `__init__.py` re-export preserves this. Similarly, `wizard_state.py` helpers
move into the package but retain backward-compatible imports.

### Planned file changes

- **Refactor:** `shuvoice/wizard.py` → `shuvoice/wizard/` package
- **Refactor:** `shuvoice/wizard_state.py` — split into `wizard/actions.py` + `wizard/hyprland.py`
- **Tests:**
  - `tests/test_wizard.py` (update imports)
  - `tests/test_wizard_ui.py` (update imports)
  - `tests/test_wizard_hyprland.py` (new — keybind parse/edit logic)

### Tasks

- [x] D1. Create `shuvoice/wizard/` package with `__init__.py` re-export
- [x] D2. Extract page-building logic from `WelcomeWizard` into `wizard/ui.py`
- [x] D3. Extract finish pipeline (`write_config`, keybind setup, marker write) into `wizard/actions.py`
- [x] D4. Move Hyprland bind parse/edit logic to `wizard/hyprland.py`
- [x] D5. Standardize status/result enums for keybind setup outcomes
- [x] D6. Ensure wizard remains headless-test friendly for non-GUI core logic
- [x] D7. Update `__main__.py` and test imports

### Rollback

Collapse `shuvoice/wizard/` package back to `wizard.py` + `wizard_state.py`.
The `__init__.py` re-export pattern makes this a clean revert.

### Validation

```bash
uv run pytest tests/test_wizard.py tests/test_wizard_ui.py tests/test_wizard_hyprland.py -v
uv run pytest -m gui -v  # optional/dispatch-run in CI as today
```

### Acceptance criteria

- [x] UI code and side-effect code are clearly separated
- [x] Existing wizard UX remains unchanged
- [x] Hyprland config edit paths are covered by dedicated tests
- [x] `from shuvoice.wizard import WelcomeWizard` still works

---

## Workstream E — Waybar Integration Refactor

> **Depends on M1 only** — waybar interacts heavily with config and systemd but not app core.

### Target design

Separate Waybar CLI routing from systemd interactions, Hyprland queries, and UI string formatting to improve test isolation and reduce module size.

Proposed split (new `shuvoice/waybar/` package):

- `shuvoice/waybar/__init__.py` — re-exports `main` for backward compat
- `shuvoice/waybar/cli.py` — command parser and action dispatch
- `shuvoice/waybar/systemd.py` — `systemctl` interactions and service state checks
- `shuvoice/waybar/format.py` — JSON payload and tooltip info line generation
- `shuvoice/waybar/hyprland.py` — `detect_keybind()` (can be shared with wizard later)

### Planned file changes

- **Refactor:** `shuvoice/waybar.py` → `shuvoice/waybar/` package
- **Tests:**
  - `tests/test_waybar.py` (update imports)
  - `tests/test_waybar_format.py` (new — UI strings)
  - `tests/test_waybar_systemd.py` (new — systemctl wrappers)

### Tasks

- [x] E1. Create `shuvoice/waybar/` package with `__init__.py` re-export
- [x] E2. Extract UI string building into `waybar/format.py`
- [x] E3. Extract `_run_systemctl_user` and related helpers to `waybar/systemd.py`
- [x] E4. Move `detect_keybind` to `waybar/hyprland.py`
- [x] E5. Move `main` and arg parsing to `waybar/cli.py`
- [x] E6. Ensure existing `shuvoice-waybar` entry point and command names remain backward compatible

### Rollback

Collapse `shuvoice/waybar/` package back to `waybar.py`. Clean revert.

### Validation

```bash
uv run pytest tests/test_waybar*.py -v
shuvoice-waybar status
```

### Acceptance criteria

- [x] Waybar command outputs identical JSON format as before
- [x] Test coverage improved for systemd error handling and formatting logic
- [x] `waybar.py` refactored away from being a monolith

---

## Workstream F — Backend Contract Hardening

### Target design

Extend the ASR contract to explicit capabilities and standardized diagnostics.

**Important**: `wants_raw_audio` already exists as a property on `ASRBackend`
(`asr_base.py:25`). The new `ASRCapabilities` dataclass should **subsume** it.
During transition, keep the base class property as a passthrough to capabilities.

Proposed interface additions in `asr_base.py`:

```python
@dataclass(frozen=True)
class ASRCapabilities:
    supports_gpu: bool = False
    supports_model_download: bool = False
    wants_raw_audio: bool = False  # migrated from base class property
    expected_chunking: str = "streaming"  # "streaming" | "windowed"
```

Standard dependency diagnostics payload (structured, not only string list).

### Planned file changes

- `shuvoice/asr_base.py` — add `ASRCapabilities`, migrate `wants_raw_audio`
- `shuvoice/asr.py` — keep `ASREngine` compatibility shim (from C6); remove only after deprecation cycle and test migration
- `shuvoice/asr_nemo.py` — implement `capabilities`
- `shuvoice/asr_sherpa.py` — implement `capabilities`
- `shuvoice/asr_moonshine.py` — implement `capabilities`
- `shuvoice/cli/commands/preflight.py` (or `shuvoice/__main__.py` until M2a lands) for preflight output enrichment
- Tests:
  - `tests/test_asr.py` (update)
  - `tests/test_asr_moonshine.py` (update)
  - `tests/test_backend_capabilities.py` (new)

### Tasks

- [x] F1. Define `ASRCapabilities` dataclass in `asr_base.py`
- [x] F2. Add `capabilities` class property to `ASRBackend` base
- [x] F3. Migrate `wants_raw_audio` from base property to `capabilities.wants_raw_audio`
  - [x] Keep deprecated `wants_raw_audio` property as passthrough for one cycle
- [x] F4. Implement `capabilities` for each backend (nemo, sherpa, moonshine)
- [x] F5. Update preflight to use capability-aware checks
- [x] F6. Keep backward compatibility for `dependency_errors()` API during transition
- [x] F7. Add tests to ensure registry exposes capabilities consistently
- [x] F8. Start migrating tests/callers from `ASREngine` to `get_backend_class("nemo")` ahead of final alias removal

### Rollback

Remove `ASRCapabilities` dataclass; restore direct `wants_raw_audio` property.
No behavioral change to callers.

### Validation

```bash
uv run pytest tests/test_asr.py tests/test_backend_capabilities.py -v
python -m shuvoice preflight  # or: python -m shuvoice --preflight (legacy)
```

### Acceptance criteria

- [x] Backend features are inspectable in a uniform way
- [x] Preflight output is clearer and backend-specific
- [x] `wants_raw_audio` backward-compat property still works

---

## Workstream G — Observability + Diagnostics

### Target design

Add structured runtime metrics for troubleshooting and performance visibility.

Proposed metrics domains:

- Recording lifecycle: start/stop counts, average utterance duration
- ASR pipeline: first-token latency, finalization latency, reset/recovery counters
- Audio queue: queue growth/drops, max observed queue depth
- Output pipeline: partial updates count, final commit count, commit failures

### Planned file changes

- **New modules:**
  - `shuvoice/metrics.py`
  - `shuvoice/diagnostics.py`
- **Refactor:**
  - `shuvoice/app.py` (or `shuvoice/runtime/` modules by this point)
  - `shuvoice/control.py` (optional metrics command path)
  - `shuvoice/waybar/format.py` (optional debug stats section)
- **Tests:**
  - `tests/test_metrics.py` (new)
  - `tests/test_control.py` (update for diagnostics endpoint if added)

### Tasks

- [x] G1. Add metrics collector with low overhead (in-memory counters + rolling timings)
- [x] G2. Emit structured periodic summary logs (debug/info gated)
- [x] G3. Add optional CLI/IPC diagnostics endpoint
  - Option A: `shuvoice control metrics` (extends IPC protocol)
  - Option B: `shuvoice diagnostics` subcommand
- [x] G4. Add optional Waybar tooltip extension (only when enabled)
- [x] G5. Add privacy guardrails (no transcript content in metrics)

### Rollback

Remove `metrics.py` and `diagnostics.py`; revert instrumentation points in
app/control. No data persistence to clean up.

### Validation

```bash
uv run pytest tests/test_metrics.py tests/test_security_logging.py -v
# manual smoke
python -m shuvoice control status  # or: shuvoice control status
```

### Acceptance criteria

- [x] Troubleshooting can be done without enabling invasive debug logs
- [x] Security logging tests still pass (no transcript leakage)

---

## Workstream H — Performance Wins

### Target design

Targeted performance improvements identified in the architecture review.
These are low-risk, high-value changes.

### Tasks

- [x] H1. **Diff-based partial typing** in `StreamingTyper.update_partial()`
  - Current: full backspace + retype every update
  - Target: compute common prefix, only retype the changed suffix
  - Module: `shuvoice/typer.py`
  - Test: `tests/test_typer.py` (update with keystroke-count assertions)

- [x] H2. **Cache compiled regex in `apply_text_replacements()`**
  - Current: recompiles patterns on every call
  - Target: compile once at `Config` init, pass pattern cache to postprocess
  - Module: `shuvoice/postprocess.py`, `shuvoice/config.py`
  - Test: `tests/test_postprocess.py` (update)

- [x] H3. **Waybar `detect_keybind()` caching**
  - Current: re-runs `hyprctl binds -j` on each invocation
  - Target: cache with short TTL or on-change invalidation
  - Module: `shuvoice/waybar/hyprland.py`
  - Test: `tests/test_waybar.py` or `tests/test_waybar_hyprland.py` (update)

### Rollback

Each is a self-contained optimization. Revert individual commits.

### Validation

```bash
uv run pytest tests/test_typer.py tests/test_postprocess.py tests/test_waybar.py -v
```

### Acceptance criteria

- [x] Partial typing keystroke count reduced by ≥50% for typical updates
- [x] No regex recompilation in hot path
- [x] No behavioral regressions

---

## Workstream I — Docs Drift Prevention + CI Guardrails

### Target design

Automate consistency checks between code and docs/examples.

### Planned file changes

- `.github/workflows/ci.yml`
- `scripts/validate-doc-config-sync.py` (new)
- `scripts/validate-cli-help-sync.py` (new, optional)
- `README.md`
- `AGENTS.md`
- `examples/config.toml`

### Tasks

- [x] I1. Add script to verify documented config keys exist in `Config` dataclass
- [x] I2. Add script to verify example config sections/keys remain valid
- [x] I3. Add CI job stage to run drift checks
- [x] I4. Add doc update checklist to `CONTRIBUTING.md`
- [x] I5. Optionally generate a canonical config-key table from code metadata

### Rollback

Remove CI job and scripts. No functional impact.

### Validation

```bash
uv run python scripts/validate-doc-config-sync.py
uv run ruff check shuvoice tests scripts
```

### Acceptance criteria

- [x] CI fails on stale config docs/examples
- [x] Contributors get clear actionable drift errors

---

## PR Slicing Plan (Suggested)

| PR | Content | Milestone |
|---|---|---|
| PR-01 | Config versioning scaffold + `config_io` (no behavior change) | M1 |
| PR-02 | Migration framework + tests | M1 |
| PR-03 | Wizard config write path switched to `config_io` | M1 |
| PR-04 | CLI parser/subcommand scaffolding + legacy compat + CLI tests | M2a |
| PR-05 | CLI config subcommands (`config effective/path/validate`) | M2b |
| PR-06 | Runtime extraction: state machine + chunk pipeline + flush policy | M3 |
| PR-07 | Wizard UI/business/hyprland split into package | M4 |
| PR-08 | Waybar integration refactored into package | M5 |
| PR-09 | Backend capabilities contract + `wants_raw_audio` migration | M6 |
| PR-10 | Metrics collector + structured diagnostics + Waybar integration | M7 |
| PR-11 | Performance: diff typing + regex cache + waybar cache | M8 |
| PR-12 | Docs drift scripts + CI job + doc refresh + release notes | M9 |

Each PR must include:
- [x] Tests
- [x] Docs updates (if behavior/user-facing)
- [x] Rollback note in PR description

> Contains 12 PR slices to keep reviews focused and enable incremental shipping.

---

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| CLI compatibility breakage | Users' scripts/binds fail | Keep old flags for one cycle; add tests for legacy paths |
| `ASREngine` compatibility removal too early | Backend/unit tests and external imports break | Use lazy shim + deprecation cycle; migrate tests to `get_backend_class("nemo")` first |
| Runtime race regression during refactor | Recording instability | Extract logic behind tests first, then move code |
| Config migration bug | User config corruption | Atomic writes + automatic backup + migration tests |
| Metrics overhead | Increased latency | Keep lightweight counters; no heavy per-chunk serialization |
| Docs check false positives | CI friction | Provide explicit ignore rules + actionable error messages |
| `wizard.py` → `wizard/` import breakage | Test/consumer failures | `__init__.py` re-export pattern; update all imports in same PR |
| Two competing plan files | Developer confusion | Delete `PLAN-refactor-performance-enhancements-roadmap.md` when work begins |

---

## Rollout & Backout Strategy

### Rollout

- Ship incrementally by PR slices above.
- After M2a (CLI), publish migration note in README with old/new examples.
- After M1 and M7, run a real-user smoke pass on systemd service workflow.

### Backout

- Keep changes isolated by module so individual PRs can be reverted cleanly.
- Preserve old CLI compatibility until after one stable release cycle.
- Config system always writes backups before mutating user files.
- Each workstream section above includes a **Rollback** subsection.

---

## Cleanup Tasks (Housekeeping)

These can be done opportunistically during any milestone:

- [x] Remove/retire `ASREngine` compatibility alias only after deprecation cycle and test migration (C6/F8), ensuring no eager NeMo import path remains
- [x] Delete `PLAN-refactor-performance-enhancements-roadmap.md` once this plan is accepted as canonical

---

## Final Verification Matrix

Run before declaring roadmap complete:

```bash
uv run ruff check shuvoice tests scripts
uv run ruff format --check shuvoice tests scripts
uv run pytest -m "not gui" -v
uv run pytest -m e2e -v
# optional
uv run pytest -m gui -v
python -m shuvoice preflight       # new subcommand
python -m shuvoice --preflight     # legacy compat
python -m shuvoice --help
```

And for service-level sanity:

```bash
systemctl --user restart shuvoice.service
journalctl --user -u shuvoice.service -n 80 --no-pager
```

---

## Deliverables Checklist

- [x] Versioned config + migration engine
- [x] Durable config write path with backups
- [x] Subcommand-based CLI + compatibility layer
- [x] Refactored runtime modules preserving prior extractions
- [x] Wizard refactored into package with UI/logic/hyprland separation
- [x] Waybar refactored into package separating CLI, format, and systemd logic
- [x] Hardened backend capability contract with `wants_raw_audio` migration
- [x] Structured diagnostics + optional Waybar debug stats
- [x] Performance: diff typing, regex cache, waybar cache
- [x] CI-enforced docs/config drift checks
- [x] Updated user/developer documentation
- [x] Dead code cleanup (`ASREngine` eager-import path retired via deprecation, duplicate plan file)
