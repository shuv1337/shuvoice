# PLAN: Moonshine Throughput Improvements

**Issue**: [#13 — Moonshine: base model too slow for streaming use (63s wall / 22min CPU for 8 phrases)](https://github.com/shuv1337/shuvoice/issues/13)

---

## Context & Observed Problem

The Moonshine ONNX backend is **an order of magnitude slower** than every other
ShuVoice ASR backend for streaming push-to-talk use:

| Backend | Wall time | CPU time | Per-phrase avg |
|---------|-----------|----------|----------------|
| NeMo (CUDA) | 8.5s | 11.3s | ~1.1s |
| Sherpa CUDA | 2.8s | 5.4s | ~0.35s |
| Sherpa CPU | 1.5s | 4.1s | ~0.19s |
| **Moonshine base** | **63s** | **22m 8s** | **~7.9s** |
| **Moonshine tiny** | **27s** | **9m 33s** | **~3.4s** |

**Root cause**: Moonshine is a batch encoder-decoder.  Every inference call
re-encodes the **entire** accumulated audio buffer (up to
`moonshine_max_window_sec` seconds) and then runs an autoregressive decoder for
up to `moonshine_max_tokens` tokens.  As the buffer grows the total work is
**O(n²)** in audio length.  The existing `_INFER_INTERVAL_S = 0.50` throttle
limits call frequency but doesn't change the per-call cost.

**Key code locations**:
- `shuvoice/asr_moonshine.py` — full backend; `process_chunk()` (line ~198),
  `_INFER_INTERVAL_S = 0.50` (line ~57), `_commit_pending_audio()` (line ~233),
  `_normalize_buffer()` (line ~258).
- `shuvoice/config.py` — `moonshine_max_window_sec = 5.0` (default),
  `moonshine_max_tokens = 128`, `moonshine_model_name = "moonshine/base"`.
- `.venv312/.../moonshine_onnx/model.py` — upstream `MoonshineOnnxModel`:
  creates `onnxruntime.InferenceSession` without specifying `providers`,
  so it defaults to `CPUExecutionProvider`.
- `scripts/tts_roundtrip.py` — benchmark harness used in issue data.
- `tests/test_asr_moonshine.py` — unit tests for repetition guard and buffer
  management (no latency/throughput tests today).

---

## Goals

1. **Reduce Moonshine per-phrase latency** to ≤1.5s (tiny) / ≤3s (base) for a
   typical 2–4 second utterance on CPU, making interactive push-to-talk usable
   for short inputs.
2. **Evaluate ONNX GPU execution** (`CUDAExecutionProvider`) as a path to
   near-parity with Sherpa/NeMo latency.
3. **Ship safer defaults** so new users don't hit the pathological 63s case.
4. **Emit a clear startup warning** when the user selects a slow configuration.
5. **Document realistic performance expectations** in `AGENTS.md`.
6. **Establish benchmark pass/fail thresholds** in CI-compatible form.

## Non-Goals

- Rewriting Moonshine as a true streaming encoder (architectural redesign of
  the upstream model).
- Achieving accuracy parity with NeMo (tracked separately in #12).
- Implementing KV-cache optimizations inside the upstream `moonshine_onnx`
  library (we'll evaluate feasibility only; changes are out of scope for this
  issue).
- Replacing Moonshine with a different lightweight backend.

---

## Current-State Findings

### 1. Full-buffer re-encoding is the dominant cost

`process_chunk()` calls `self._model.generate(audio_2d, max_len=max_tokens)`
where `audio_2d` is the **entire committed buffer** up to `_max_window_samples`.
The upstream `MoonshineOnnxModel.generate()` runs the ONNX encoder on the full
waveform every call:

```python
# moonshine_onnx/model.py
last_hidden_state = self.encoder.run(None, encoder_inputs)[0]
```

For a 5-second buffer at 16 kHz = 80,000 samples, the encoder processes all
80k samples on every inference cycle.  At `_INFER_INTERVAL_S = 0.50` that means
~10 full-buffer encoder runs for a 5-second utterance.

### 2. Autoregressive decoder adds linear overhead

After encoding, the decoder runs token-by-token up to `max_tokens=128`.
Each step is a full decoder ONNX session run (the KV-cache is maintained
correctly via `use_cache_branch`, so this is not quadratic — but 128 steps
is still significant on CPU).

### 3. Default model is `moonshine/base` (8 layers, head_dim=52)

`moonshine/base` has 8 decoder layers × 8 KV heads × 52 head_dim.
`moonshine/tiny` has 6 layers × 8 heads × 36 head_dim — roughly 40% less
compute.  The issue data shows tiny is ~2.3× faster than base.

### 4. ONNX GPU providers are available

The installed `onnxruntime` exposes `CUDAExecutionProvider` and
`TensorrtExecutionProvider`, but `moonshine_onnx.MoonshineOnnxModel` hard-codes
no `providers` argument in `InferenceSession(encoder)` / `InferenceSession(decoder)`.
GPU acceleration would require either:
- Monkey-patching the sessions after construction, or
- Constructing the `InferenceSession` objects ourselves with
  `providers=['CUDAExecutionProvider', 'CPUExecutionProvider']`.

### 5. No ONNX session options tuning

The upstream library creates sessions with default `SessionOptions`.  Tuning
`inter_op_num_threads`, `intra_op_num_threads`, and
`execution_mode = ORT_PARALLEL` could improve CPU throughput.

### 6. Window defaults already reduced

`moonshine_max_window_sec` defaults to `5.0` (config.py line ~76).  The issue
body mentions the original benchmark used `10.0`.  The current 5.0 default
already halves worst-case buffer size vs. 10.0.

---

## Hypotheses to Validate

| # | Hypothesis | Measurement |
|---|---|---|
| H1 | Reducing `moonshine_max_window_sec` from 5.0→3.0 cuts per-phrase latency proportionally | Roundtrip benchmark wall time & per-phrase avg |
| H2 | Switching default model from `base` to `tiny` gives ≥2× speedup with acceptable quality | Roundtrip wall time + median similarity |
| H3 | Increasing `_INFER_INTERVAL_S` from 0.50→0.80 reduces total encoder calls without harming final accuracy | Roundtrip wall time + similarity delta |
| H4 | Lowering `moonshine_max_tokens` from 128→64 reduces decoder time for short utterances | Per-phrase wall time on phrases <3s |
| H5 | Tuning ONNX `SessionOptions` (thread count, parallel execution) improves CPU throughput | Roundtrip wall time delta |
| H6 | `CUDAExecutionProvider` on encoder+decoder brings Moonshine within 2× of Sherpa CPU | Roundtrip wall time on GPU system |

### Measurement plan

All benchmarks use the existing roundtrip harness:

```bash
time python scripts/tts_roundtrip.py \
  --asr-backend moonshine \
  --moonshine-model-name moonshine/tiny \
  --moonshine-max-window-sec 3.0 \
  --moonshine-max-tokens 64
```

Capture: wall time (`time`), per-phrase avg, median similarity.
Compare against baseline (current defaults) and against Sherpa CPU.

---

## Task Breakdown

### Track A: Quick Tuning (config defaults & constants)

- [ ] **A1. Benchmark current defaults as baseline**
  - Run `scripts/tts_roundtrip.py` with `moonshine/base` and `moonshine/tiny`
    using current defaults (`max_window_sec=5.0`, `max_tokens=128`,
    `_INFER_INTERVAL_S=0.50`).
  - Record wall time, CPU time, per-phrase avg, and median similarity.
  - Save results to `build/tts-roundtrip/baseline-moonshine-{base,tiny}.csv`.

- [ ] **A2. Sweep `moonshine_max_window_sec` = {2.0, 3.0, 4.0, 5.0}**
  - For each value, run roundtrip with `moonshine/tiny`.
  - Record wall time and similarity.  Identify the knee of the
    latency-vs-accuracy curve.

- [ ] **A3. Sweep `moonshine_max_tokens` = {32, 64, 96, 128}**
  - For each value, run roundtrip with `moonshine/tiny` at best
    `max_window_sec` from A2.
  - Identify if lower token caps help for short phrases without hurting long
    phrase accuracy.

- [ ] **A4. Sweep `_INFER_INTERVAL_S` = {0.50, 0.80, 1.00, 1.50}**
  - Test at best config from A2+A3.
  - Measure total encoder call count (add temporary counter log) and
    wall time.

- [ ] **A5. Update defaults based on A2–A4 findings**
  - In `shuvoice/config.py`, update:
    - `moonshine_model_name` default → `"moonshine/tiny"` (if H2 confirmed)
    - `moonshine_max_window_sec` → best value from A2
    - `moonshine_max_tokens` → best value from A3
  - In `shuvoice/asr_moonshine.py`, update:
    - `_INFER_INTERVAL_S` → best value from A4

### Track B: ONNX Session Tuning (CPU)

- [ ] **B1. Add ONNX `SessionOptions` tuning to `MoonshineBackend.load()`**
  - After `moonshine_onnx.MoonshineOnnxModel(**kwargs)` constructs the model,
    replace `self._model.encoder` and `self._model.decoder` with new
    `onnxruntime.InferenceSession` objects that have tuned `SessionOptions`:
    - `sess_options.inter_op_num_threads = os.cpu_count()` (or configurable)
    - `sess_options.intra_op_num_threads = 2` (avoid over-subscription)
    - `sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL`
  - Read encoder/decoder model paths from the existing model object's session
    metadata or by re-resolving via `moonshine_onnx` helpers.

- [ ] **B2. Benchmark B1 vs. baseline**
  - Run roundtrip and compare wall time.  Accept if ≥10% improvement.

- [ ] **B3. Expose optional config key `moonshine_onnx_threads`**
  - Add `moonshine_onnx_threads: int = 0` to `Config` (0 = auto).
  - Wire into the `SessionOptions` creation from B1.
  - Document in AGENTS.md.

### Track C: ONNX GPU Execution Provider Feasibility

- [ ] **C1. Prototype GPU session creation**
  - In `MoonshineBackend.load()`, when a new config key
    `moonshine_provider` is `"cuda"`, reconstruct the encoder and decoder
    `InferenceSession` with
    `providers=['CUDAExecutionProvider', 'CPUExecutionProvider']`.
  - Use the model file paths resolved from HuggingFace cache or
    `moonshine_model_dir`.

- [ ] **C2. Add `moonshine_provider` config key**
  - Add `moonshine_provider: str = "cpu"` to `Config` dataclass.
  - Add validation (`cpu` or `cuda`).
  - Wire through CLI args in `scripts/tts_roundtrip.py` for benchmarking.

- [ ] **C3. Benchmark GPU vs. CPU**
  - Run roundtrip with `moonshine_provider=cuda` and compare to CPU baseline.
  - Record wall time, GPU memory usage (`nvidia-smi`), and any errors.
  - **Pass criteria**: GPU must be ≤2× Sherpa CPU wall time (≤3.0s for 8
    phrases) with no OOM or CUDA errors.

- [ ] **C4. Handle GPU unavailability gracefully**
  - If `CUDAExecutionProvider` is requested but not available in the
    installed `onnxruntime`, log a warning and fall back to CPU.
  - Add unit test: mock `onnxruntime.get_available_providers()` to exclude
    CUDA; verify fallback + warning.

- [ ] **C5. Document GPU setup in AGENTS.md**
  - Add Moonshine GPU section parallel to Sherpa GPU docs.
  - Note: requires `onnxruntime-gpu` (or system `onnxruntime` with CUDA
    provider compiled in, which is already the case on this system).

### Track D: Startup Warning & Documentation

- [ ] **D1. Emit startup warning for slow configurations**
  - In `MoonshineBackend.load()`, **after** model load succeeds:
    - If `moonshine_model_name` contains `"base"` and provider is `"cpu"`:
      emit `log.warning()` advising the user to switch to `moonshine/tiny`
      or enable CUDA for interactive use.
    - If `moonshine_max_window_sec > 5.0` on CPU: warn about long-utterance
      latency.
  - Note: a startup warning already exists (line ~176 in current code);
    enhance it with specific config advice.

- [ ] **D2. Update AGENTS.md Moonshine section**
  - Add realistic performance expectations table.
  - Update config keys table with new keys (`moonshine_provider`,
    `moonshine_onnx_threads`).
  - Update default values to match A5 changes.
  - Add "Moonshine GPU" subsection if C3 proves feasible.
  - Document recommended configs for CPU-only vs. GPU systems.

- [ ] **D3. Update `examples/config.toml`**
  - Reflect new defaults and add commented-out GPU config example.

### Track E: Benchmark Automation & Regression Guard

- [ ] **E1. Add Moonshine-specific latency threshold to roundtrip regression**
  - In `tests/integration/test_roundtrip_regression.py`, when
    `SHUVOICE_ROUNDTRIP_BACKEND=moonshine`:
    - Add `SHUVOICE_ROUNDTRIP_MAX_WALL_SEC` env var (default: 30 for tiny,
      60 for base).
    - Assert total wall time ≤ threshold.
    - Lower similarity thresholds vs. NeMo: `min_median_similarity_total=0.70`,
      `min_median_similarity_per_phrase=0.65` (Moonshine's known accuracy
      gap).

- [ ] **E2. Create `scripts/moonshine_sweep.sh` convenience wrapper**
  - Shell script that runs the roundtrip harness across parameter
    combinations from A2–A4 and outputs a summary table.
  - Used for development; not required in CI.

---

## Benchmark Methodology

### Harness

All measurements use `scripts/tts_roundtrip.py` with the default 5 phrases
from `examples/tts_roundtrip_phrases.txt` (or the 8-phrase set from the issue).

### Metrics

| Metric | Capture method |
|---|---|
| Wall time | `time` command wrapper |
| Per-phrase avg | Wall time ÷ phrase count |
| CPU time | `time` user+sys |
| Median similarity | CSV output `similarity` column |
| Encoder calls | Temporary `log.debug` counter in `process_chunk()` |
| GPU memory | `nvidia-smi --query-gpu=memory.used --format=csv` |

### Pass/Fail Thresholds (interactive latency target: ≤500ms per update)

| Scenario | Metric | Pass | Fail |
|---|---|---|---|
| Tiny CPU (tuned) | Per-phrase avg | ≤1.5s | >2.0s |
| Tiny CPU (tuned) | Total wall (8 phrases) | ≤15s | >20s |
| Base CPU (tuned) | Per-phrase avg | ≤3.0s | >5.0s |
| Tiny GPU | Per-phrase avg | ≤0.5s | >1.0s |
| Tiny GPU | Total wall (8 phrases) | ≤5s | >8s |
| Any config | Median similarity | ≥0.65 | <0.55 |

> **Note**: The 500ms interactive target applies to *partial update latency*
> (time between consecutive `process_chunk` returns with changed text).
> The per-phrase averages above are *total transcription time* including all
> inference passes.  The throttle interval (`_INFER_INTERVAL_S`) directly
> controls update cadence — at 0.80s the user sees updates at most every
> ~800ms, which is acceptable for a "lightweight fallback" backend.

---

## Validation Checklist

```bash
# 1. Unit tests pass (existing + new)
uv run pytest tests/test_asr_moonshine.py tests/test_asr.py tests/test_config.py -v

# 2. Roundtrip baseline (before changes)
time python scripts/tts_roundtrip.py --asr-backend moonshine \
  --moonshine-model-name moonshine/tiny

# 3. Roundtrip with tuned defaults (after Track A)
time python scripts/tts_roundtrip.py --asr-backend moonshine \
  --moonshine-model-name moonshine/tiny \
  --moonshine-max-window-sec 3.0 \
  --moonshine-max-tokens 64

# 4. GPU benchmark (after Track C)
time python scripts/tts_roundtrip.py --asr-backend moonshine \
  --moonshine-model-name moonshine/tiny
  # (with moonshine_provider=cuda in config or new CLI arg)

# 5. Config validation
rg -n '^\s+moonshine' shuvoice/config.py

# 6. AGENTS.md consistency
grep -c 'moonshine' AGENTS.md  # should increase

# 7. Integration regression (requires SHUVOICE_RUN_ROUNDTRIP=1)
SHUVOICE_RUN_ROUNDTRIP=1 SHUVOICE_ROUNDTRIP_BACKEND=moonshine \
  SHUVOICE_MOONSHINE_MODEL_NAME=moonshine/tiny \
  uv run pytest tests/integration/test_roundtrip_regression.py -v

# 8. Service smoke test
systemctl --user restart shuvoice.service
journalctl --user -u shuvoice.service -n 20 --no-pager | grep -i moonshine

# 9. Lint
uv run ruff check shuvoice/asr_moonshine.py shuvoice/config.py
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Reducing `max_window_sec` truncates long utterances | Words beyond the window are silently dropped | Keep ≥3.0s (covers ~15 words of natural speech); document the trade-off |
| Lowering `max_tokens` causes truncation on long phrases | Output cut short mid-sentence | Set floor at 64; add log warning when token limit is hit |
| ONNX GPU sessions may not work with all model precisions | `CUDAExecutionProvider` may reject `float16` models | Default to `float` precision on GPU; test both |
| Monkey-patching `moonshine_onnx` model sessions is fragile | Upstream library updates may break our session replacement | Pin `useful-moonshine-onnx` version; add defensive try/except with CPU fallback |
| Thread tuning may degrade on systems with few cores | Over-subscribing threads causes contention | Use `os.cpu_count()` with a cap; make configurable via `moonshine_onnx_threads` |
| Changing default model from `base` to `tiny` reduces accuracy | Users expecting `base` quality get `tiny` | Document in AGENTS.md; make the switch opt-in if similarity delta >0.10 |

---

## Rollback Strategy

All changes are **backward-compatible config default changes** and
**additive code** (new config keys, new ONNX session options, new log
warnings).  Rollback paths:

1. **Config defaults**: User overrides any default in
   `~/.config/shuvoice/config.toml`.  Old values remain valid.
2. **ONNX session tuning (Track B)**: Controlled by `moonshine_onnx_threads`.
   Set to `0` (auto) or remove the key to revert to upstream defaults.
3. **GPU provider (Track C)**: Controlled by `moonshine_provider`.  Defaults
   to `"cpu"` — no behavioral change unless explicitly opted in.
4. **Full revert**: `git revert` the implementation commits.  No data
   migration or state changes involved.

---

## Definition of Done

- [ ] **Moonshine tiny (CPU, tuned defaults)** completes 8-phrase roundtrip in
      ≤15s wall time with ≥0.65 median similarity.
- [ ] **Default model** is `moonshine/tiny` (or `base` with documented
      caveats if tiny quality is unacceptable).
- [ ] **Startup warning** is emitted when a slow Moonshine configuration is
      detected (base on CPU, or `max_window_sec > 5`).
- [ ] **ONNX GPU feasibility** is evaluated and documented:
  - If feasible: `moonshine_provider = "cuda"` works end-to-end with graceful
    CPU fallback.
  - If not feasible: documented in AGENTS.md with specific failure reason.
- [ ] **AGENTS.md** updated with:
  - Realistic performance table.
  - New config keys.
  - Recommended config for CPU-only vs. GPU systems.
- [ ] **All existing tests pass** (`uv run pytest` green).
- [ ] **Roundtrip regression test** has Moonshine-specific thresholds.
- [ ] **No regressions** in NeMo or Sherpa backend behavior.
