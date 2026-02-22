# PLAN: Fix Moonshine Repetition Guard & Performance (#12, #13)

## Plan QA Summary (Implementation Readiness)

Pre-implementation review checks:

- ✅ #12/#13 causality corrected (guard correctness vs perf knobs)
- ✅ Guard ordering explicitly handles short/single-token runaway text
- ✅ Files list includes all impacted defaults/docs/tests
- ✅ Benchmark commands are reproducible (explicit Moonshine flags)
- ✅ Test expectations are deterministic (no “or similar” assertions)

**Status: IMPLEMENTED — core tasks complete; benchmark improvement targets partially unmet**

---

## Objective

Address:

- **#12**: Moonshine repetition guard misses token-level loops (hyphen/digits)
  and long repeated clauses.
- **#13**: Moonshine CPU-time/latency is too high vs NeMo/Sherpa.

Important framing:

- #12 is a **quality/safety guard** fix.
- #13 gains come primarily from reducing encoder/decoder work
  (window size, token budget, inference cadence).

---

## Scope

### In scope

- `shuvoice/asr_moonshine.py`: guard robustness, infer throttle, startup warning
- `shuvoice/config.py`: Moonshine defaults
- `tests/test_asr_moonshine.py`: new Moonshine guard tests
- `tests/test_config.py`: update default assertions
- `examples/config.toml`, `examples/config-moonshine-cpu.toml`: sync defaults
- `AGENTS.md`: sync Moonshine defaults/caveats + issue status

### Out of scope

- Non-Moonshine backend behavior changes (except regression validation)
- Moonshine model architecture changes upstream

---

## Phase 0 — Baseline capture (required before code changes)

Run baseline with explicit settings to avoid local config drift:

```bash
/usr/bin/time -f "wall=%e user=%U sys=%S cpu=%P" \
  .venv312/bin/python scripts/tts_roundtrip.py \
  --phrases-file build/benchmark-phrases.txt \
  --output-dir build/benchmark-moonshine-base-baseline \
  --asr-backend moonshine --moonshine-model-name moonshine/base \
  --moonshine-max-window-sec 10 --moonshine-max-tokens 192 \
  --flush-chunks 5

/usr/bin/time -f "wall=%e user=%U sys=%S cpu=%P" \
  .venv312/bin/python scripts/tts_roundtrip.py \
  --phrases-file build/benchmark-phrases.txt \
  --output-dir build/benchmark-moonshine-tiny-baseline \
  --asr-backend moonshine --moonshine-model-name moonshine/tiny \
  --moonshine-max-window-sec 10 --moonshine-max-tokens 192 \
  --flush-chunks 5
```

### Acceptance criteria

- [x] Baseline wall/user/sys metrics recorded for base + tiny
- [x] Baseline CSV outputs saved in both output dirs

Execution note: `/usr/bin/time` was unavailable on this host; bash `time` with
`TIMEFORMAT='wall=%R user=%U sys=%S cpu=%P'` was used instead.

---

## Phase 1 — Repetition guard correctness (#12)

### Task 1.1 — Token-scoped repetition regex (before word checks)

Implementation:

- [x] Add `import re` in `shuvoice/asr_moonshine.py`
- [x] Add class-level token repetition regex (1–10 chars, >=4 repeats)
- [x] Apply regex over non-space spans (tokens), not across arbitrary spaces
- [x] Truncate to one occurrence on match and emit `log.debug(...)`

### Acceptance criteria

- [x] Hyphen token loop truncates deterministically
- [x] Digit token loop truncates deterministically
- [x] No cross-word false positive from regex spanning spaces
- [x] Debug log fires when rule triggers

---

### Task 1.2 — Character cap applies even for short/single-token text

Implementation:

- [x] Add `_MAX_CHARS_PER_SEC = 40.0`
- [x] Compute `max_chars = max(100, int(audio_seconds * cls._MAX_CHARS_PER_SEC) + 20)`
- [x] If `len(text) > max_chars`, truncate safely
  - prefer word boundary when possible
  - preserve non-empty output for single-token text
- [x] Run this cap **before** any `len(words) <= 5` early return
- [x] Add `log.debug(...)` when cap triggers

### Acceptance criteria

- [x] Very long 1-token output is truncated
- [x] Short non-pathological text (<=5 words) remains unchanged
- [x] Character cap never returns empty string unless input is empty
- [x] Debug log fires when cap triggers

---

### Task 1.3 — Expand clause-level n-gram repetition detection

Implementation:

- [x] Expand `plen` from `1..4` to `1..12`
- [x] Increase start scan window cap from `8` to `20`
- [x] Preserve feasibility bound:
      `start_limit = min(len(words) - plen * threshold + 1, 20)`
- [x] Dynamic repetition threshold:
  - `4` for `plen <= 4`
  - `3` for `plen > 4`

### Acceptance criteria

- [x] 11-word clause repeated 3x truncates to one clause
- [x] Existing short n-gram behavior remains effective
- [x] No index/loop bounds errors on small inputs

---

### Task 1.4 — Add deterministic Moonshine guard tests

Implementation:

- [x] Create `tests/test_asr_moonshine.py`
- [x] Add tests:
  - hyphen loop truncation (exact expected output)
  - digit loop truncation with repeatable input (`127127127127` style)
  - 11-word clause repetition truncation
  - char-cap for long single-token text
  - normal text unchanged
  - short non-pathological text unchanged

### Acceptance criteria

- [x] `tests/test_asr_moonshine.py` passes locally
- [x] Assertions are deterministic and exact (no fuzzy “similar” wording)

---

## Phase 2 — Performance knobs and defaults (#13)

### Task 2.1 — Reduce default `moonshine_max_window_sec` to 5.0

Implementation:

- [x] Update `shuvoice/config.py` default to `5.0`
- [x] Update `tests/test_config.py` default assertions
- [x] Update `examples/config.toml`
- [x] Update `examples/config-moonshine-cpu.toml`
- [x] Update `AGENTS.md` Moonshine snippet and config table

### Acceptance criteria

- [x] Code default is `5.0`
- [x] Tests assert `5.0`
- [x] Both example configs and AGENTS show `5.0`

---

### Task 2.2 — Reduce default `moonshine_max_tokens` to 128

Implementation:

- [x] Update `shuvoice/config.py` default to `128`
- [x] Update `tests/test_config.py` default assertions
- [x] Update `examples/config.toml`
- [x] Update `examples/config-moonshine-cpu.toml`
- [x] Update `AGENTS.md` Moonshine snippet and config table

### Acceptance criteria

- [x] Code default is `128`
- [x] Tests assert `128`
- [x] Both example configs and AGENTS show `128`

---

### Task 2.3 — Increase Moonshine inference throttle interval

Implementation:

- [x] Change `_INFER_INTERVAL_S` from `0.30` to `0.50`
- [x] Update nearby comments/docstring to match

### Acceptance criteria

- [x] Constant is `0.50` in `shuvoice/asr_moonshine.py`
- [x] Comments/docs in same file are consistent

---

### Task 2.4 — Add startup CPU-only performance warning

Implementation:

- [x] In `load()`, after successful init, log warning that Moonshine is CPU-only
      and slower than NeMo/Sherpa
- [x] Keep warning actionable (best for short utterances / no-GPU setups)

### Acceptance criteria

- [x] Warning appears once on successful Moonshine backend load
- [x] Wording is clear and non-alarming

---

## Phase 3 — Validation and regression safety

### Task 3.1 — Post-change benchmark run

```bash
/usr/bin/time -f "wall=%e user=%U sys=%S cpu=%P" \
  .venv312/bin/python scripts/tts_roundtrip.py \
  --phrases-file build/benchmark-phrases.txt \
  --output-dir build/benchmark-moonshine-base-fixed \
  --asr-backend moonshine --moonshine-model-name moonshine/base \
  --moonshine-max-window-sec 5 --moonshine-max-tokens 128 \
  --flush-chunks 5

/usr/bin/time -f "wall=%e user=%U sys=%S cpu=%P" \
  .venv312/bin/python scripts/tts_roundtrip.py \
  --phrases-file build/benchmark-phrases.txt \
  --output-dir build/benchmark-moonshine-tiny-fixed \
  --asr-backend moonshine --moonshine-model-name moonshine/tiny \
  --moonshine-max-window-sec 5 --moonshine-max-tokens 128 \
  --flush-chunks 5
```

### Acceptance criteria

- [x] No runaway repeated segment >100 chars in hypotheses
- [ ] Relative wall-time improvement vs baseline:
  - base: **>=35% faster** target
  - tiny: **>=25% faster** target
- [ ] Median similarity is maintained or improved

Observed benchmark results (current implementation):
- base wall: 39.633s → 29.392s (**25.84% faster**, target not met)
- tiny wall: 26.383s → 22.510s (**14.68% faster**, target not met)
- base median similarity: 0.840 → 0.834 (slight decrease)
- tiny median similarity: 0.809 → 0.798 (slight decrease)

---

### Task 3.2 — Unit test suite

```bash
.venv312/bin/python -m pytest tests/test_asr_moonshine.py tests/test_config.py -v
.venv312/bin/python -m pytest tests/ -x --ignore=tests/integration --ignore=tests/e2e -v
```

### Acceptance criteria

- [x] New Moonshine guard tests pass
- [x] Config default tests pass
- [x] No regressions in NeMo/Sherpa unit tests

---

### Task 3.3 — Documentation consistency

Implementation:

- [x] Update Moonshine caveats/perf language in `AGENTS.md`
- [x] Ensure AGENTS Moonshine snippet and config table are consistent
- [x] Update Known Issues entries for #12/#13 status

### Acceptance criteria

- [x] AGENTS has no internal default mismatch for Moonshine
- [x] Known Issues reflects post-change state

---

## Implementation order

```text
Phase 0 (baseline):          0
Phase 1 (correctness):       1.1 → 1.2 → 1.3 → 1.4
Phase 2 (performance):       2.1 → 2.2 → 2.3 → 2.4
Phase 3 (validation/docs):   3.1 → 3.2 → 3.3
```

---

## Suggested commits

1. `fix(moonshine): harden repetition guard for token and clause loops (#12)`
2. `perf(moonshine): reduce defaults and increase inference throttle (#13)`
3. `test(moonshine): add guard regressions and update config default assertions`
4. `docs: sync Moonshine defaults/caveats in AGENTS and examples`

---

## Files touched

- `shuvoice/asr_moonshine.py`
- `shuvoice/config.py`
- `tests/test_asr_moonshine.py` (new)
- `tests/test_config.py`
- `examples/config.toml`
- `examples/config-moonshine-cpu.toml`
- `AGENTS.md`
