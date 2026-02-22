# PLAN: Enable GPU-backed Sherpa ONNX in ShuVoice

## Objective

Run ShuVoice with `asr_backend = "sherpa"` and **actual CUDA execution** (not CPU fallback), then keep service configuration and docs reproducible.

---

## Current state (observed)

- ShuVoice service can run Sherpa backend with current model directory.
- Current installed `sherpa-onnx` wheel falls back to CPU even when `sherpa_provider = "cuda"`.
- Runtime warning seen in service logs:
  - `Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON ... Fallback to cpu!`
- Environment uses `.venv312` (Python 3.12) and NVIDIA GPU is present.

---

## Success criteria

- `sherpa_provider = "cuda"` in config.
- No Sherpa fallback warning in service logs.
- Sherpa inference smoke test passes with expected transcript quality.
- During/after inference, GPU usage attributable to ShuVoice/Sherpa is observable.
- Service remains stable after restart.

---

## Prerequisites

- `.venv312` active/usable.
- NVIDIA driver working (`nvidia-smi` succeeds).
- Existing Sherpa model directory:
  - `build/asr-models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06`

---

## Milestone 0 — Baseline capture

- [ ] Record current package versions:
  - `python -m pip show sherpa-onnx sherpa-onnx-core`
- [ ] Capture baseline fallback evidence from logs:
  - `journalctl --user -u shuvoice.service -n 100 --no-pager | rg -n "Fallback to cpu|SHERPA_ONNX_ENABLE_GPU"`
- [ ] Save baseline runtime snapshot in `build/asr-smoke/sherpa-gpu-baseline.txt`.

Validation:
```bash
nvidia-smi
.venv312/bin/python -m shuvoice --preflight --asr-backend sherpa
```

---

## Milestone 1 — Try prebuilt CUDA Sherpa wheel (preferred)

Use official Sherpa CUDA wheels (documented in `https://k2-fsa.github.io/sherpa/onnx/python/install.html`).

- [ ] Enumerate available CUDA wheel versions from:
  - `https://k2-fsa.github.io/sherpa/onnx/cuda.html`
- [ ] Select a wheel compatible with:
  - Python 3.12 (`cp312`)
  - Linux x86_64
  - local CUDA/CUDNN runtime stack
- [ ] Install selected wheel into `.venv312` (replace CPU wheel):
  - Example pattern:
    - `pip install --verbose sherpa-onnx=="<version+cuda...>" --no-index -f https://k2-fsa.github.io/sherpa/onnx/cuda.html`
- [ ] Verify version tag indicates CUDA build (e.g. `+cuda`/`+cuda12.cudnn9`).

Validation:
```bash
.venv312/bin/python - <<'PY'
import sherpa_onnx
print(sherpa_onnx.__version__)
PY
```

---

## Milestone 2 — Backend smoke test with `provider=cuda`

- [ ] Run isolated Sherpa backend decode test in `.venv312` with:
  - `asr_backend='sherpa'`
  - `sherpa_provider='cuda'`
  - known WAV fixture (`tests/audio-sample.wav`)
- [ ] Confirm transcript is produced and comparable to CPU run.
- [ ] Confirm no fallback warning emitted in test output.
- [ ] Capture output to `build/asr-smoke/sherpa-gpu-smoke.txt`.

Validation:
```bash
# Should NOT print fallback warning
.venv312/bin/python <sherpa-cuda-smoke-script>
```

---

## Milestone 3 — Service rollout (Sherpa CUDA)

- [ ] Set user config:
  - `asr_backend = "sherpa"`
  - `sherpa_provider = "cuda"`
- [ ] Restart service:
  - `systemctl --user restart shuvoice.service`
- [ ] Confirm active state and control socket healthy.
- [ ] Trigger short dictation via IPC (`start` / `stop`) and inspect logs.

Validation:
```bash
systemctl --user status shuvoice.service --no-pager
.venv312/bin/python -m shuvoice --control status
journalctl --user -u shuvoice.service -n 120 --no-pager
```

Acceptance checks:
- [ ] No `Fallback to cpu` warning.
- [ ] Dictation output appears normal.

---

## Milestone 4 — GPU proof and performance sanity

- [ ] Capture GPU activity during active inference (process/memory/utilization):
  - `nvidia-smi` snapshots before/during recording
- [ ] Compare rough decode latency vs CPU run on same test WAV.
- [ ] Record findings in `build/asr-smoke/sherpa-gpu-report.md`.

Validation:
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

---

## Milestone 5 — Fallback path (only if prebuilt CUDA wheel fails)

If Milestone 1 fails (no compatible wheel/provider still CPU):

- [ ] Build `sherpa-onnx` from source with GPU enabled:
  - clone `k2-fsa/sherpa-onnx`
  - export `SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON"`
  - run `python setup.py install` (or build wheel and install)
- [ ] Ensure ONNX Runtime GPU dependencies resolve at runtime.
- [ ] Re-run Milestones 2–4.

Validation:
```bash
.venv312/bin/python - <<'PY'
import sherpa_onnx
print(sherpa_onnx.__version__)
PY
```

---

## Milestone 6 — Harden docs and operator workflow

- [ ] Update project docs with a dedicated Sherpa CUDA setup section:
  - required wheel source
  - compatible version examples
  - common fallback warning and fix
- [ ] Add a one-command smoke script for Sherpa CUDA verification.
- [ ] (Optional) Add strict provider check in backend/app startup to fail fast when `cuda` requested but unavailable.

---

## Risks and mitigations

1. **No matching CUDA wheel for local Python/ABI**
   - Mitigation: source build path (Milestone 5).
2. **CUDA/CUDNN mismatch at runtime**
   - Mitigation: pin working wheel version and document exact env.
3. **Silent CPU fallback despite `provider=cuda`**
   - Mitigation: explicit log checks + GPU telemetry checks in acceptance criteria.
4. **Service instability after dependency replacement**
   - Mitigation: test in script first, then restart service; keep rollback command ready.

---

## Rollback plan

- [ ] Reinstall CPU wheel:
  - `pip install sherpa-onnx==1.12.25`
- [ ] Set `sherpa_provider = "cpu"` in config.
- [ ] Restart service and verify healthy operation.

---

## Final sign-off checklist

- [ ] Sherpa uses CUDA without fallback warning.
- [ ] Service stable after restart.
- [ ] Dictation quality acceptable.
- [ ] GPU usage evidence captured.
- [ ] Docs updated with reproducible install/run steps.
