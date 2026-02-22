# Issue: Sherpa streaming transducer drops trailing words on early key release

## Summary

When using the Sherpa ONNX ASR backend, releasing the push-to-talk key as soon
as you finish speaking causes the final transcript to lose trailing words.
The user must hold the key ~1–2 seconds **past** the end of speech for the full
sentence to appear.  NeMo does not have this problem.

**Example**: Speaking "The quick brown fox jumped over the lazy dog" and
releasing immediately after "dog":

| Scenario | Final output |
|----------|-------------|
| Hold key through silence ✅ | The quick brown fox jumped over the lazy dog |
| Release right after speaking ❌ | The quick brown fox jumped over |
| Release right after speaking ❌ | The quick brown fox jumped over the la |

## Root cause

Sherpa's streaming zipformer transducer buffers significantly more internal
context than NeMo before emitting tokens.  Observed behaviour:

1. **~1.5 s latency to first token** — the first 15+ chunks (100 ms each)
   return `raw_text=''` even though speech energy is high.
2. **Burst emission** — text appears in large jumps
   (`'' → 'The quick'`, then 10+ unchanged steps, then
   `'The quick' → 'The quick brown fox jumped over'`).
3. **Trailing tokens stay buffered** — the transducer holds 2–4 final words
   internally until enough *future* context (real or silence) pushes them out.

When the key is released, `_recording.clear()` fires immediately,
the real-time processing loop (`_process_recording_chunks`) exits, and
`_handle_recording_stop` takes over.  The post-release path is:

```
_drain_and_buffer        → collect any remaining queued audio
process remaining chunks → run buffered audio through the model
_flush_tail_silence      → feed synthetic silence/noise to flush the transducer
```

The flush exists specifically to push out buffered hypotheses, but it
is not reliably triggering Sherpa to emit the remaining words.

## What we tried

### 1. Increase flush budget (5 → 20 chunks, 500 ms → 2 s)
- **Result**: No effect.  Sherpa returned the same partial text on every
  silence chunk; the loop hit `stable_steps` and exited.

### 2. Feed ambient-level Gaussian noise instead of `np.zeros`
- **Rationale**: Sherpa may treat perfect digital silence differently from
  natural ambient noise.  During held-key tests, the model flushed correctly
  when fed real ambient noise (RMS 0.006–0.01).
- **Result**: No effect at raw noise-floor amplitude.  The noise floor
  (~0.001–0.003 RMS) is far below what the model was seeing during speech
  because all recording audio goes through utterance gain (10–40×).

### 3. Apply utterance gain to flush noise
- **Rationale**: Match the amplitude scale the model has been processing.
  With gain 18× and noise floor 0.003, flush noise RMS ≈ 0.05 — similar to
  the ambient noise that triggers flushing when the user holds the key.
- **Result**: **Partial success.**
  - Works when `noise_floor_rms` is ≥ 0.003 (flush RMS ≈ 0.05 after gain).
  - Fails when `noise_floor_rms` is very low (0.001 in a quiet room → flush
    RMS ≈ 0.02 after gain), which is too quiet to trigger emission.
  - When it works, the flush recovers the full sentence in 1–2 steps.

### 4. Increase `stable_required` (3 → 5)
- **Rationale**: Give more chances for delayed emission.
- **Result**: No effect when the flush noise is too quiet to trigger any
  change in the first place.

## Evidence: successful flush vs. failed flush

### Successful (noise_floor=0.0035, gain=18.1, flush RMS ≈ 0.063)
```
Recording stopped → text was "The quick brown fox jumped"
Tail flush step 1: → "The quick brown fox jumped over the lazy dog"  ✅
Final: The quick brown fox jumped over the lazy dog
```

### Failed (noise_floor=0.0012, gain=19.4, flush RMS ≈ 0.023)
```
Recording stopped → text was "The quick brown fox jumped over"
(no tail flush steps triggered)
Final: The quick brown fox jumped over  ❌
```

## Proposed fixes (not yet implemented)

### A. Minimum flush noise floor (quick fix)
Clamp the flush noise RMS to a minimum value (e.g., 0.005 pre-gain) so that
even in a very quiet room the flush audio is loud enough to trigger emission:

```python
rms = max(self._noise_floor_rms, 0.005)  # minimum flush energy
```

This is the simplest change and directly addresses the failure mode.

### B. Adaptive flush with escalating amplitude
Start with ambient-level noise and progressively increase amplitude on each
flush step if no new text appears.  Cap at some reasonable maximum (e.g., 0.1
pre-gain) to avoid injecting perceptible artifacts.

### C. Feed recorded silence instead of synthetic noise
Capture a short buffer of real ambient audio *before* recording starts (during
the noise-floor estimation phase).  Use this captured ambient audio for the
tail flush instead of synthetic noise.  This would most closely replicate the
"user holds key in silence" condition that reliably flushes the model.

### D. Backend-specific flush method
Add a `flush()` method to the `ASRBackend` ABC that backends can override.
Sherpa could implement a more aggressive flush strategy (e.g., feeding the
model's own silence token or resetting the stream and extracting final text).

## Impact

- **Sherpa backend only** — NeMo emits tokens with much lower latency and
  does not exhibit this buffering behaviour.
- **Moonshine untested** for this specific pattern but likely similar since
  it also uses chunk-based processing.
- The workaround is to hold the key ~1–2 s longer than natural speech, which
  is usable but not ergonomic.

## Related files

- `shuvoice/app.py` — `_flush_tail_silence`, `_handle_recording_stop`,
  `_make_flush_noise`
- `shuvoice/asr_sherpa.py` — `process_chunk`, `reset`
- `PLAN-sherpa-gpu-enable.md` — GPU enablement plan (separate concern)
