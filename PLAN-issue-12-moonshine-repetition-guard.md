# PLAN: Issue #12 — Moonshine repetition guard hardening

> Issue: https://github.com/shuv1337/shuvoice/issues/12  
> Scope: **Planning only** (no code changes in this document)  
> Target file for implementation: `shuvoice/asr_moonshine.py`

## Context

Issue #12 reports three real-world Moonshine failure modes that can produce very long garbage output:

1. Hyphenated/token-local loops (e.g. `hake-hake-hake-...`)
2. Long clause-level loops (repeating ~10+ word spans)
3. Numeric/character-level runs inside one token (e.g. `1270000000...`)

Repro from issue body (via `gh issue view 12`) uses `scripts/tts_roundtrip.py` with Moonshine base/tiny and challenging phrases.

Repository context shows this is still tracked as open (`AGENTS.md:378`).

## Goals

- Close issue #12 with deterministic, test-backed repetition guard behavior for Moonshine.
- Improve detection for:
  - hyphenated/token-local repetition,
  - long clause loops,
  - numeric/character-level runs.
- Minimize false positives (do not aggressively truncate valid text like normal hyphenated words and ordinary numbers).
- Add explicit regression coverage (unit + opt-in integration) tied to the issue’s reproduction cases.

## Non-goals

- No model-quality retraining or Moonshine decoder architecture changes.
- No changes to NeMo/Sherpa behavior.
- No throughput/perf work from #13 beyond ensuring new logic is not pathologically expensive.
- No always-on GPU/integration CI expansion in this issue (keep integration tests opt-in like current pattern).

## Current-state findings (with code references)

### 1) Repetition guard already has multi-stage logic (partial mitigation exists)

`MoonshineBackend._guard_repetition()` currently applies:

- token-local repetition regex,
- character-count cap,
- word-count cap,
- n-gram repetition scan.

References:
- Constants: `shuvoice/asr_moonshine.py:64-73`
- Guard implementation: `shuvoice/asr_moonshine.py:319-416`
- Current long-pattern limit is 12 words (`_MAX_PATTERN_WORDS=12`) and scan start cap is 20 (`_MAX_PATTERN_STARTS=20`) at `shuvoice/asr_moonshine.py:68-69,391-392`.

### 2) Unit tests already cover basic classes, but are narrow and mostly synthetic

Existing tests in `tests/test_asr_moonshine.py` include:
- hyphenated token truncation (`:9-12`)
- digit token repetition (`:15-20`)
- clause-level repetition (`:24-28`)
- long single-token char-cap truncation (`:31-37`)
- non-pathological keep-cases (`:40-49`)

However, gaps remain for:
- clause loops starting later than the first ~20 words,
- long numeric tails like `127000000...` (issue example form),
- explicit false-positive controls (valid long numbers/hyphenated terms).

### 3) Security logging constraints already exist and must remain intact

`tests/test_security_logging.py:68-78` asserts repetition logs avoid secret raw text and only log abstract pattern information.

### 4) Integration harness exists, but issue-12 phrases are not pinned in deterministic regression tests

- Current manual integration phrase set is only two phrases (`tests/integration/test_roundtrip_regression.py:18-20`).
- Moonshine is supported in that harness (`tests/integration/test_roundtrip_regression.py:99-101`), but no dedicated issue-12 phrase matrix.
- `scripts/tts_roundtrip.py` computes similarity only (`scripts/tts_roundtrip.py:365-367`) and does not emit repetition-specific diagnostics.

### 5) CI behavior implies unit tests are the reliable merge gate

GitHub CI currently runs headless unit tests excluding integration/gpu (`.github/workflows/ci.yml:78`), so critical regressions for #12 should have strong unit coverage, with integration as opt-in local validation.

---

## Proposed detection strategy (implementation design)

Use a staged detector pipeline in `_guard_repetition()` with explicit false-positive controls:

### A) Hyphenated/token-local repetition detector

- Keep token-local pass first (before word split), but replace single regex-only truncation with helper logic that:
  - detects repeated subunits with optional delimiters (`-`, `_`) across a bounded unit-size range,
  - preserves prefix context,
  - truncates at first pathological run.
- Add guards to avoid cutting normal hyphenated words (`state-of-the-art`) and short expressive forms (`ha-ha-ha`).

### B) Clause-length loop detector (word-level)

- Move from early-start-only n-gram scan (`_MAX_PATTERN_STARTS`) to a suffix/rolling-window strategy that can detect loops starting later in the utterance.
- Keep bounded complexity (small max pattern words, bounded scan depth), but ensure patterns >4 words and up to long-clause lengths are covered.
- Normalize punctuation/case during comparisons to catch comma/period variants.

### C) Numeric/character-level repetition detector

- Add explicit character-run and numeric-tail handling (e.g., very long repeated digits or repeated short char units inside one token).
- Differentiate numeric heuristics from general text heuristics to reduce false positives:
  - allow ordinary magnitudes (`1000`, `100000`) and normal IDs,
  - cut only on clearly pathological lengths/runs.

### D) Safety + determinism

- Keep/retain dynamic char-count and word-count caps as final safety net.
- Ensure logs remain redacted (no full token/text leakage).
- Keep guard behavior deterministic from `(text, audio_seconds)` input.

---

## Task breakdown (ordered, concrete)

### Phase 0 — Baseline + fixture capture

- [ ] Reproduce issue #12 baseline outputs with existing code using `scripts/tts_roundtrip.py` for:
  - [ ] Moonshine `base` tongue-twister + long-clause phrase
  - [ ] Moonshine `tiny` numeric phrase
- [ ] Save baseline outputs under `build/tts-roundtrip-issue12-baseline/` and capture CSV + terminal transcript snippets for before/after comparison.

### Phase 1 — Guard algorithm hardening in `shuvoice/asr_moonshine.py`

- [ ] Refactor `_guard_repetition()` into small helper methods (token-local, char/digit-run, clause-loop, caps) to make detector behavior independently testable.
- [ ] Replace/augment current token regex handling with delimiter-aware repeated-subunit detection and conservative truncation rules.
- [ ] Replace start-capped n-gram loop logic with a bounded strategy that can detect repeated long clauses appearing beyond early word positions.
- [ ] Add explicit numeric/character repetition rules for pathological digit floods and repeated short char units.
- [ ] Keep runtime bounded and avoid large quadratic scans; document complexity assumptions in code comments.
- [ ] Preserve redacted logging behavior (pattern metadata only).

### Phase 2 — Unit test expansion

- [ ] Extend `tests/test_asr_moonshine.py` with parametrized coverage for:
  - [ ] hyphenated loops from issue phrase variants,
  - [ ] clause loops of length 8–15 words,
  - [ ] clause loops beginning after >20 initial words,
  - [ ] numeric tail flood (`127000000...`) and char-run flood,
  - [ ] false-positive controls (valid hyphenated compounds, normal large numbers, short emphasis repeats).
- [ ] Add regression tests for boundary thresholds (just below/just above trigger).
- [ ] Update/add log-redaction assertions in `tests/test_security_logging.py` for any new detector log paths.

### Phase 3 — Deterministic integration regression coverage

- [ ] Add a Moonshine-specific opt-in integration test (new file: `tests/integration/test_moonshine_repetition_regression.py`) following existing env-gated style.
- [ ] Add an issue-12 phrase fixture file (e.g., `examples/tts_roundtrip_issue12_phrases.txt`) including:
  - [ ] tongue twister,
  - [ ] long clause sentence,
  - [ ] numeric invoice sentence,
  - [ ] control phrases for false-positive checks.
- [ ] In integration assertions, include both similarity and repetition constraints (e.g., max repeated-char run, max repeated n-gram run in hypothesis text).
- [ ] Run same matrix against `moonshine/base` and `moonshine/tiny`.

### Phase 4 — Docs + closeout

- [ ] Update README regression section with issue-12 reproduction/validation commands.
- [ ] Update `AGENTS.md` known issue entry for #12 after validation is complete.
- [ ] Prepare issue close comment template with before/after examples and exact commands used.

---

## Validation checklist

### Fast local gate (required)

- [ ] `uv run pytest tests/test_asr_moonshine.py -v`
- [ ] `uv run pytest tests/test_security_logging.py -v`
- [ ] `uv run pytest -m "not gui and not e2e and not integration and not gpu" -v`

### Issue reproduction commands (explicit)

- [ ] Moonshine base (hyphenated + clause):

```bash
uv run python scripts/tts_roundtrip.py \
  --asr-backend moonshine \
  --moonshine-model-name moonshine/base \
  --phrase "The sixth sick sheik's sixth sheep's sick" \
  --phrase "We still have issues with recording cutting out on long sentences, and we need deterministic regression tests to catch regressions before they ship" \
  --output-dir build/tts-roundtrip-issue12-base
```

- [ ] Moonshine tiny (numeric):

```bash
uv run python scripts/tts_roundtrip.py \
  --asr-backend moonshine \
  --moonshine-model-name moonshine/tiny \
  --phrase "Invoice 4827 totals one hundred and fifty three dollars and twelve cents" \
  --output-dir build/tts-roundtrip-issue12-tiny
```

### Integration regression (opt-in)

- [ ] Base model:

```bash
SHUVOICE_RUN_MOONSHINE_REPETITION=1 \
SHUVOICE_MOONSHINE_MODEL_NAME=moonshine/base \
uv run pytest -m integration -k moonshine_repetition_regression -v
```

- [ ] Tiny model:

```bash
SHUVOICE_RUN_MOONSHINE_REPETITION=1 \
SHUVOICE_MOONSHINE_MODEL_NAME=moonshine/tiny \
uv run pytest -m integration -k moonshine_repetition_regression -v
```

### Acceptance checks

- [ ] Tongue-twister output no longer contains runaway hyphenated loops.
- [ ] Long sentence no longer contains repeating long-clause loops.
- [ ] Numeric hallucination is truncated without damaging normal-number control cases.
- [ ] Existing ASR and transcript tests remain green.

---

## Regression-test matrix

| Case class | Example input | Detector path | Test layer | Expected assertion |
|---|---|---|---|---|
| Hyphenated token loop | `six-six-hake-hake-hake...` | token-local repeated-subunit detector | Unit + Integration | Truncated near first repeated run; no runaway suffix |
| Long clause loop (early) | repeated 10–12 word clause ×3 | clause-loop detector | Unit | Returns first clause only (or first stable occurrence) |
| Long clause loop (late start) | normal prefix (20+ words) then repeated clause | clause-loop detector (late-position coverage) | Unit | Loop is cut even when start index is late |
| Numeric tail flood | `... 1270000000000...` | char/digit-run detector | Unit + Integration (tiny) | Output truncates pathological digit run |
| Repeated short char unit | `ababababab...` in token | token/char detector | Unit | Pathological run truncated |
| Normal hyphenated word | `state-of-the-art` | false-positive guard | Unit | Text unchanged |
| Normal large number | `100000` / `20260223` | numeric false-positive guard | Unit | Text unchanged |
| Control natural phrase | ordinary sentence | no detector triggers | Unit + Integration | Text unchanged aside from normal ASR variance |

---

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Over-aggressive truncation (false positives) | Legit text clipped (especially numbers/hyphenated compounds) | Add explicit control fixtures + boundary tests; separate numeric thresholds from alphabetic ones |
| Under-detection of late clause loops | Issue remains reproducible in long outputs | Remove dependence on early-start scan cap; add late-start loop tests |
| Runtime overhead from heavier detection | Moonshine latency worsens | Keep bounded scan windows and max pattern sizes; micro-benchmark guard path with long synthetic transcripts |
| Logging leaks sensitive text | Security regression | Preserve/extend redaction tests in `tests/test_security_logging.py` |
| Integration nondeterminism | Flaky regression signals | Use deterministic TTS phrase fixtures, repetitions, and tolerant-but-specific repetition metrics |

---

## Rollback strategy

1. **Isolate changes** to repetition-guard helpers in `shuvoice/asr_moonshine.py` so rollback scope is a single file/function path.
2. If regressions appear, **revert to previous guard behavior** via targeted git revert of the issue-12 commit(s).
3. Keep regression fixtures/tests in place (or mark known-failing with issue reference temporarily) so regressions remain visible during rollback.
4. After rollback, run minimum safety checks:

```bash
uv run pytest tests/test_asr_moonshine.py -v
uv run pytest tests/test_security_logging.py -v
uv run pytest -m "not gui and not e2e and not integration and not gpu" -v
```

5. Re-open/tighten thresholds in a follow-up PR using the same issue-12 matrix before reattempting.

---

## Definition of done

- [ ] All new/updated unit tests pass.
- [ ] Moonshine issue-12 integration regression passes for base + tiny (opt-in harness).
- [ ] Before/after outputs demonstrate removal of runaway repetition classes.
- [ ] Documentation and known-issue status updated.
- [ ] Issue #12 can be closed with reproducible evidence.
