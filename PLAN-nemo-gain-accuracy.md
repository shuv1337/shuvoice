# PLAN: NeMo Auto-Gain & Tail Accuracy Fix

**Date**: 2026-02-20
**Status**: Revised (implemented; manual benchmark pending)
**Goal**: Improve NeMo transcription accuracy by fixing the auto-gain pipeline and tail flush, targeting ≥75% correct across the benchmark set (currently ~35%).

---

## Plan Review Updates (2026-02-20)

This plan was reviewed against the current codebase before implementation.

### Critical fixes applied to the plan

1. **Add automated verification tasks**
   - The original draft relied heavily on manual log checks.
   - Added explicit test tasks for backend raw-audio routing, config defaults/validation, and utterance state reset behavior.

2. **Clarify NeMo tail-flush follow-up as data-driven**
   - The draft proposed lowering `stable_required` to 3 as a conditional tweak.
   - This can prematurely stop flushing and is now treated as an *optional, benchmark-gated* change rather than a default recommendation.

3. **Explicitly preserve backend split**
   - NeMo/Moonshine should bypass app-side per-chunk gain (`wants_raw_audio=True`).
   - Sherpa keeps app-side gain, but with safer defaults and settle delay.

### Ready/blocked scope

- **Ready now**: code + docs + automated tests for Phase 1 and Phase 2.
- **Requires manual microphone benchmark**: final acceptance decision for Phase 3 tuning and ≥75% target claim.

### Implementation checkpoint

- Completed: Phases 1, 2, 4.1, and 4.3
- Pending: Phase 3 and Phase 4.2 (manual spoken benchmark)
- Automated validation run:
  - `pytest -q tests/test_asr.py tests/test_config.py tests/test_utterance_state.py`
  - `pytest -q`

---

## Problem Summary

Benchmark testing with NeMo + CUDA revealed three systemic failure modes, all traceable to the audio preprocessing pipeline — not the model itself:

1. **Over-amplification** (gains of 20-40×) distorts phonemes, causing word substitution ("pull"→"poll", "Hey"→"Hake", "Invoice"→"Voice") and spurious sentence boundaries ("moonshine. Demo", "config. Changes.")
2. **Gain instability** — gain swings from 40× (noise-level first chunk) to 8-17× (speech arrives) within a single utterance, creating volume discontinuities the model wasn't trained on.
3. **Tail truncation** — final words consistently lost ("milliseconds" dropped 4/4 trials) because the tail flush feeds high-gain amplified noise that produces hallucinations instead of flushing real buffered tokens.

## Evidence (from session logs)

| Utterance | Gain range | Failure |
|-----------|-----------|---------|
| "Hey, can you review the poll?" | 36.7→24.4× | "pull"→"poll" |
| "Hake, can you review..." | 23.9→18.9× | "Hey"→"Hake" |
| "Voice four eight two seven..." | 12.9→10.9× | "Invoice"→"Voice" |
| "Invoice. Totals one hundred." | 39.8→17.5× | spurious periods |
| "...less than 360." | 31.7→28.5× | "350 milliseconds"→"360." |

All gain values above are applied **before** the audio reaches NeMo's preprocessor.

---

## Affected Files

| File | Role |
|------|------|
| `shuvoice/app.py` | Gain calculation, chunk pipeline, tail flush |
| `shuvoice/utterance_state.py` | Per-utterance gain state |
| `shuvoice/asr_base.py` | `wants_raw_audio` property (already exists for Moonshine) |
| `shuvoice/asr_nemo.py` | NeMo backend — may need to opt out of gain |
| `shuvoice/config.py` | New config knobs |
| `examples/config.toml` | Document new config keys |
| `AGENTS.md` | Update NeMo config keys table |

---

## Tasks

### Phase 1: Tame the auto-gain

The core fix. NeMo's neural preprocessor already normalizes mel spectrograms internally — aggressive external gain is counterproductive.

#### 1.1 Add `wants_raw_audio` override for NeMo

- [x] In `shuvoice/asr_nemo.py`, override the `wants_raw_audio` property to return `True`

This uses the existing mechanism (already used by Moonshine) to tell `app.py` to skip per-chunk gain entirely. NeMo's preprocessor (`AudioToMelSpectrogramPreprocessor`) applies its own feature normalization.

**Rationale**: The simplest, lowest-risk change. Sherpa still gets gain (it benefits from it on CPU). NeMo and Moonshine get raw audio. Zero config changes needed.

```python
# shuvoice/asr_nemo.py
@property
def wants_raw_audio(self) -> bool:
    return True
```

#### 1.2 Update `_flush_tail_silence` to skip gain when `wants_raw_audio`

- [x] In `shuvoice/app.py` `_flush_tail_silence()`, check `self.asr.wants_raw_audio` and skip the `_apply_utterance_gain` call on flush audio when True

Current code (around line 320):
```python
flush_audio = self._make_flush_noise(native)
if not self.asr.wants_raw_audio:
    flush_audio = self._apply_utterance_gain(flush_audio, state.utterance_gain)
```
This is already correct — `_flush_tail_silence` already checks `wants_raw_audio`. Verify this path is exercised correctly by reviewing the existing guard. **No code change needed if already guarded** — just verify.

#### 1.3 Update `_flush_streaming_stall` similarly

- [x] Verify `_flush_streaming_stall()` in `app.py` also respects `wants_raw_audio` — currently it feeds silence (`np.zeros`) which bypasses gain, so this should be fine. Confirm and add a comment if no change needed.

#### 1.4 Update `asr_base.py` docstring

- [x] Amend the `wants_raw_audio` docstring in `shuvoice/asr_base.py` to mention NeMo alongside Moonshine as a backend that uses this flag, and clarify that it's not just about batch vs streaming — it's about backends with their own normalization.

### Phase 2: Reduce gain aggressiveness for backends that still use it (Sherpa)

Even for Sherpa, the current 40× max gain is too aggressive. Make it saner as a safety net.

#### 2.1 Lower gain constants

- [x] In `shuvoice/app.py` `_append_recording_chunk()`, change:
  - `target_peak`: `0.3` → `0.15`
  - max gain cap: `40.0` → `10.0`

Current code:
```python
if state.peak_rms > 0.003:
    target_peak = 0.3
    state.utterance_gain = min(target_peak / state.peak_rms, 40.0)
```

New code:
```python
if state.peak_rms > 0.003:
    target_peak = 0.15
    state.utterance_gain = min(target_peak / state.peak_rms, 10.0)
```

#### 2.2 Add gain settle delay

- [x] Don't update `utterance_gain` until at least 2 speech-level chunks have been seen. This prevents the first noisy chunk from setting gain to max.

Add a `speech_chunks_seen` counter to `_UtteranceState`:

```python
# shuvoice/utterance_state.py
speech_chunks_seen: int = 0  # new field
```

Reset it in `reset()`. Increment in `_append_recording_chunk()` when `chunk_rms >= threshold`. Only compute gain when `speech_chunks_seen >= 2`.

#### 2.3 Expose gain constants as config (optional, low priority)

- [x] Add to `Config` in `shuvoice/config.py`:
  - `auto_gain_target_peak: float = 0.15`
  - `auto_gain_max: float = 10.0`
  - `auto_gain_settle_chunks: int = 2`
- [x] Use these in `app.py` instead of hardcoded values
- [x] Add to `examples/config.toml` with comments
- [x] Add to `AGENTS.md` config keys table under a new "Audio gain" section

### Phase 3: Improve tail flush for NeMo's right-context buffer

NeMo with `right_context=13` needs ~1120ms of future audio to finalize its last tokens. The current flush helps but is undermined by gain distortion.

#### 3.1 Increase flush stability threshold for NeMo

- [ ] After Phase 1 (gain removed for NeMo), verify the tail flush works better with raw-level noise. The fix in Phase 1 should resolve most tail truncation since the flush noise will no longer be amplified 20-40×.

Run the "350 milliseconds" test 4 times after Phase 1. If tail truncation is still ≥50% failure rate, proceed to 3.2.

#### 3.2 (Conditional) Tune flush budget only with benchmark evidence

- [ ] If tail truncation persists after Phase 1, prefer **backend-tunable** flush controls over hardcoding NeMo-specific constants directly in `app.py`.
- [ ] First change: increase `max_flush` (e.g. `20 -> 30`) while keeping stability logic conservative.
- [ ] Only change `stable_required` after confirming with repeated trials that convergence is not cut off early.

Candidate API (if needed):
```python
# shuvoice/asr_base.py
@property
def flush_budget(self) -> int:
    """Max tail-flush chunks to feed after key release."""
    return 20

@property
def flush_stable_threshold(self) -> int:
    """Consecutive unchanged steps required before considering flush converged."""
    return 5
```

Then override in `asr_nemo.py` only if benchmark data supports it.

### Phase 4: Documentation & validation

#### 4.1 Update AGENTS.md

- [x] Update the NeMo config keys table to mention that auto-gain is bypassed for NeMo (`wants_raw_audio = True`)
- [x] If Phase 2.3 config keys were added, document them in the appropriate tables
- [x] Add a "Gain / audio preprocessing" section under ASR Backends explaining the per-backend behavior

#### 4.2 Re-run benchmark

- [ ] Re-run the full test set (same sentences, 4 trials each) after Phase 1+2
- [ ] Record results in the same format for comparison
- [ ] Target: ≥75% overall accuracy (up from ~35%)

#### 4.3 Automated regression coverage

- [x] Add/adjust tests to verify backend raw-audio routing (`NemoBackend.wants_raw_audio == True`)
- [x] Add/adjust tests for new auto-gain config defaults + validation
- [x] Add/adjust tests for `_UtteranceState` reset behavior with `speech_chunks_seen`
- [x] Run focused pytest suite for touched modules

Expected improvements per test:
| Test | Current | Expected |
|------|---------|----------|
| Quick brown fox | 3/4 | 4/4 |
| Moonshine demo | 3/4 | 4/4 |
| Sixth sheik (tongue twister) | 0/4 | 1-2/4 (model limitation) |
| Invoice 4827 | 1/4 | 3/4 |
| Pull request | 2/4 | 3-4/4 |
| 350 milliseconds | 0/4 | 2-3/4 |

---

## Implementation Order

```
Phase 1.1  →  Phase 1.2-1.4
   ↓
Phase 2.1-2.2  →  Phase 2.3
   ↓
Phase 4.1 + 4.3 (docs + automated tests)
   ↓
Phase 3.1 (manual benchmark check)  →  Phase 3.2 (conditional tuning)
   ↓
Phase 4.2 (full benchmark + scorecard)
```

**Phase 1 alone should fix 60-70% of the failures.** Phase 2 is a safety net for Sherpa users. Phase 3 is conditional on Phase 1 results.

## Risks

| Risk | Mitigation |
|------|-----------|
| NeMo accuracy worse without any gain (very quiet mic) | Users have `input_gain` config for manual boost; NeMo's preprocessor handles a wide dynamic range |
| Sherpa regresses with lower max gain | 10× is still substantial; Sherpa on CPU with quiet audio may need `input_gain` tuning |
| Tail flush change interacts with stall guard | Stall guard already feeds `np.zeros` (no gain); only tail flush changes |
| `speech_chunks_seen` counter adds state complexity | Minimal — one int field, reset on utterance start |
