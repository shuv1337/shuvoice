# PLAN: Fix Moonshine Repetition Guard & Performance (#12, #13)

## Objective

Address the two open Moonshine issues:

- **#12** — Repetition guard fails on hyphenated patterns, clause-level loops,
  and digit repetition, producing garbage output hundreds of characters long.
- **#13** — Base model takes 63s wall / 22min CPU for 8 phrases; both models
  are an order of magnitude slower than NeMo and Sherpa.

These are tackled together because the repetition guard fix (#12) directly
reduces wasted inference time (#13) — runaway hallucination loops are the
dominant contributor to the base model's extreme CPU usage.

---

## Phase 1: Harden the repetition guard (#12)

The current `_guard_repetition()` in `shuvoice/asr_moonshine.py` (line ~266)
has three blind spots.  Each needs a targeted fix.

### Task 1.1 — Character-level repetition detection

**Problem**: Hyphenated tokens like `"six-six-hake-hake-hake-hake-..."` are a
single word when split by spaces.  Digit runs like `"127000000000..."` are
also a single token.  The word-level n-gram loop never sees them.

**Fix**: Add a regex-based character-level check *before* the word-level
checks.  Detect any substring of 1–10 characters repeating ≥4 consecutive
times and truncate to a single occurrence.

```python
import re
_CHAR_REPETITION_RE = re.compile(r'(.{1,10}?)\1{3,}')
```

**File**: `shuvoice/asr_moonshine.py` — `_guard_repetition()`

- [ ] Add `_CHAR_REPETITION_RE` class constant (compiled regex)
- [ ] Insert character-level scan at the top of `_guard_repetition()`,
      before the word split
- [ ] On match: truncate text to `text[:match.start() + len(match.group(1))]`
- [ ] Add `log.debug()` when triggered

### Task 1.2 — Character-count cap

**Problem**: Even if word count is within bounds, a single "word" can be
hundreds of characters (hyphenated loops, digit spam).  There is no
character-level length cap.

**Fix**: Add `_MAX_CHARS_PER_SEC` constant and enforce it alongside the
existing `_MAX_WORDS_PER_SEC` word cap.

```python
_MAX_CHARS_PER_SEC = 40.0  # generous; typical speech ≈ 15-20 cps
```

**File**: `shuvoice/asr_moonshine.py` — `_guard_repetition()`

- [ ] Add `_MAX_CHARS_PER_SEC = 40.0` class constant
- [ ] After the word-count cap, add a character-count cap:
      `max_chars = max(100, int(audio_seconds * cls._MAX_CHARS_PER_SEC) + 20)`
- [ ] If `len(text) > max_chars`, truncate at the last word boundary:
      `text = text[:max_chars].rsplit(' ', 1)[0]`
- [ ] Add `log.debug()` when triggered

### Task 1.3 — Expand n-gram window for clause-level repetition

**Problem**: The n-gram loop only checks patterns of 1–4 words.  Clause-level
repetition like `"and we still have issues with recording cutting out on long
sentences"` (11 words) is invisible to it.

**Fix**: Increase the max pattern length from 4 to 12 words.  Also expand the
starting-position search window from 8 to cover more of the output.

**File**: `shuvoice/asr_moonshine.py` — `_guard_repetition()`, inner loop

- [ ] Change `range(1, 5)` to `range(1, 13)` for pattern length
- [ ] Change the start-position limit from `8` to `min(len(words), 20)`
- [ ] Reduce threshold from 4 to 3 consecutive repeats for patterns >4 words
      (long clause patterns rarely repeat 4× but 3× is already pathological)

### Task 1.4 — Unit tests for repetition guard

**File**: `tests/test_asr.py` (or new `tests/test_moonshine_guard.py`)

- [ ] Test: hyphenated repetition `"The six-six-hake-hake-hake-hake-..."` →
      truncated to `"The six-six-hake"` or similar
- [ ] Test: digit repetition `"Invoice 4827 totals 12700000000..."` →
      truncated to `"Invoice 4827 totals 127"`
- [ ] Test: clause-level repetition (11-word pattern repeated 3×) → truncated
      to a single occurrence
- [ ] Test: character-count cap fires for long single-token output
- [ ] Test: normal text (no repetition) passes through unchanged
- [ ] Test: short text (≤5 words) passes through unchanged (existing behavior)

---

## Phase 2: Performance improvements (#13)

### Task 2.1 — Reduce default `moonshine_max_window_sec` from 10.0 to 5.0

**Rationale**: The encoder re-processes the full buffer every inference call.
At 10s × 16kHz = 160,000 samples per call.  Halving to 5s cuts encoder work
by 50%.  Most push-to-talk utterances are <5s.

**Files**:
- `shuvoice/config.py` — change default `moonshine_max_window_sec: float = 5.0`
- `examples/config.toml` — update example comment
- `AGENTS.md` — update Moonshine config table

- [ ] Change default in `config.py`
- [ ] Update `examples/config.toml`
- [ ] Update `AGENTS.md` config keys table

### Task 2.2 — Reduce default `moonshine_max_tokens` from 192 to 128

**Rationale**: Autoregressive decoding is O(n) in token count.  192 tokens is
generous for spoken utterances; 128 covers ~30s of speech.  Shorter cap also
limits how long a hallucination loop can run before the guard catches it.

**Files**:
- `shuvoice/config.py` — change default `moonshine_max_tokens: int = 128`
- `examples/config.toml` — update example
- `AGENTS.md` — update Moonshine config table

- [ ] Change default in `config.py`
- [ ] Update `examples/config.toml`
- [ ] Update `AGENTS.md`

### Task 2.3 — Increase inference throttle interval

**Rationale**: `_INFER_INTERVAL_S = 0.30` means ~17 full re-encode calls for
a 5s utterance.  Increasing to 0.50s reduces to ~10 calls with only slightly
chunkier streaming updates.

**File**: `shuvoice/asr_moonshine.py`

- [ ] Change `_INFER_INTERVAL_S` from `0.30` to `0.50`
- [ ] Update class docstring to reflect new value

### Task 2.4 — Add startup warning for Moonshine on CPU

**Rationale**: Users selecting Moonshine should be aware of the performance
characteristics.  Log a warning at `load()` time.

**File**: `shuvoice/asr_moonshine.py` — `load()`

- [ ] After successful model load, emit:
      `log.warning("Moonshine runs on CPU only and is significantly slower "
      "than NeMo (CUDA) or Sherpa. Best suited for short utterances (<5s) "
      "on systems without GPU support.")`

---

## Phase 3: Benchmark & validate

### Task 3.1 — Re-run roundtrip benchmark after changes

Using the same 8-phrase benchmark set from the original evaluation.

```bash
.venv312/bin/python scripts/tts_roundtrip.py \
  --phrases-file build/benchmark-phrases.txt \
  --output-dir build/benchmark-moonshine-base-fixed \
  --asr-backend moonshine --moonshine-model-name moonshine/base \
  --flush-chunks 5

.venv312/bin/python scripts/tts_roundtrip.py \
  --phrases-file build/benchmark-phrases.txt \
  --output-dir build/benchmark-moonshine-tiny-fixed \
  --asr-backend moonshine --moonshine-model-name moonshine/tiny \
  --flush-chunks 5
```

- [ ] Verify no phrase produces >100 characters of repeated pattern
- [ ] Verify wall time for base model drops significantly (target: <30s)
- [ ] Verify wall time for tiny model drops (target: <15s)
- [ ] Compare median similarity before/after (should improve or hold steady,
      since truncating garbage improves the score)

### Task 3.2 — Run full test suite

```bash
.venv312/bin/python -m pytest tests/ -x --ignore=tests/integration --ignore=tests/e2e -v
```

- [ ] All unit tests pass
- [ ] No regressions in NeMo or Sherpa backend tests

### Task 3.3 — Update AGENTS.md

- [ ] Update Moonshine "Characteristics" section with realistic performance
      expectations and the CPU-only caveat
- [ ] Update Known Issues table to reference #12 and #13 status

---

## Implementation order

```
Phase 1 (accuracy):  1.1 → 1.2 → 1.3 → 1.4 (test)
Phase 2 (perf):      2.1 → 2.2 → 2.3 → 2.4
Phase 3 (validate):  3.1 → 3.2 → 3.3
```

Phase 1 and Phase 2 are independent and can be committed separately.
Phase 3 must run after both.

## Expected commits

1. `fix: harden Moonshine repetition guard for char-level and clause-level loops (#12)`
2. `perf: reduce Moonshine defaults and increase inference throttle (#13)`
3. `docs: update AGENTS.md with Moonshine performance caveats`

---

## Files touched

| File | Changes |
|------|---------|
| `shuvoice/asr_moonshine.py` | Repetition guard overhaul, throttle increase, startup warning |
| `shuvoice/config.py` | Default `moonshine_max_window_sec` 10→5, `moonshine_max_tokens` 192→128 |
| `tests/test_asr.py` (or new file) | Repetition guard regression tests |
| `tests/test_app_flow.py` | No changes expected |
| `examples/config.toml` | Updated defaults |
| `AGENTS.md` | Performance caveats, config table updates |
