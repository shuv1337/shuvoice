# PLAN: Pluggable ASR Backends (NeMo ➜ Sherpa ONNX first)

## Objective

Migrate ShuVoice from a single NeMo-bound ASR implementation to a **pluggable backend architecture** while keeping the existing app skeleton (audio capture, hotkey/control, overlay, typing, orchestration) intact.

Primary deliverable:
- Ship `sherpa-onnx` as the first non-NeMo backend with true streaming behavior.

Secondary deliverable:
- Preserve NeMo as default and fully backward-compatible path.

---

## Plan Review Summary (2026-02-20)

Overall assessment: **feasible with medium risk**. The core app is already mostly backend-agnostic; main work is adapter boundaries and dependency/CLI wiring.

### Critical issues found in the prior draft (now fixed in this revision)

1. **`ASREngine` compatibility strategy was ambiguous/risky**
   - Prior draft proposed turning `ASREngine` into a factory-like wrapper object.
   - Risk: breaks existing callsites/tests that rely on NeMo-class behavior (`ASREngine()` constructor, static/class methods).
   - **Fix**: keep `ASREngine` symbol as NeMo compatibility alias; add explicit factory APIs (`get_backend_class`, `create_backend`) for new backend selection.

2. **Config/docs mismatch for Sherpa keys**
   - Loader (`Config.load`) flattens sections and only accepts dataclass field names.
   - Prior draft suggested a `[sherpa]` section with unprefixed keys (`model_dir/provider/chunk_ms`) which would be ignored.
   - **Fix**: use `sherpa_*` field names in config (top-level or nested section).

3. **`--download-model` dispatch path was unsafe**
   - Prior draft routed through backend instance creation, which can fail before guidance is shown (e.g., missing Sherpa model dir).
   - **Fix**: dispatch download via backend **class** from registry (`get_backend_class`) instead of instantiated backend.

4. **Sherpa model artifact contract was underspecified**
   - Prior draft did not lock required file set/model family.
   - **Fix**: lock v1 to streaming transducer models and validate required files explicitly.

### Approval status
**READY TO IMPLEMENT** (with locked constraints below).

---

## Feasibility Review (current codebase)

The code already has strong seams for this migration.

### Confirmed strengths

- `shuvoice/app.py` mostly depends on a small ASR runtime surface:
  - `load()`
  - `reset()`
  - `process_chunk()`
- ML-specific implementation details are centralized in `shuvoice/asr.py`.
- Overlay/hotkey/control/typing paths are backend-agnostic.
- `shuvoice/utterance_state.py` is decoupled — `consume_native_chunk(native)` takes chunk size as a parameter, no NeMo references.
- `shuvoice/transcript.py` (`prefer_transcript`) and `shuvoice/streaming_health.py` are pure logic with no backend coupling.

### NeMo couplings to address

- **`shuvoice/app.py`**
  - Constructor directly instantiates NeMo args (`ASREngine(...)`) at `shuvoice/app.py:45`.
  - Uses `self.config.native_chunk_samples` at `shuvoice/app.py:366,405,430,480,546,553`.
  - Uses private NeMo field `self.asr._step_num` at `shuvoice/app.py:380,574`.

- **`shuvoice/config.py`**
  - `native_chunk_samples` embeds NeMo right-context mapping at `shuvoice/config.py:94`.

- **`shuvoice/__main__.py`**
  - Preflight hard-checks NeMo stack via `check_asr_stack()` (`shuvoice/__main__.py:85`).
  - `--download-model` always calls `ASREngine.download_model(...)` (`shuvoice/__main__.py:289`).
  - `--right-context` CLI choices are NeMo-only (`shuvoice/__main__.py:156`).

- **`scripts/tts_roundtrip.py`**
  - Constructs `ASREngine(...)` directly (`scripts/tts_roundtrip.py:241`).
  - Uses `cfg.native_chunk_samples` (`scripts/tts_roundtrip.py:237,261`).

- **Tests/docs/packaging assumptions**
  - `tests/test_asr.py` targets NeMo-only engine semantics.
  - `tests/test_config.py` validates `Config.native_chunk_samples` directly.
  - `README.md` currently brands project as Nemotron-specific and describes ASR deps as `torch+nemo` only.
  - `examples/config.toml` only shows NeMo keys.
  - `pyproject.toml` exposes only `[asr]` extra for NeMo stack.

Risk level: **medium** (largest unknown is Sherpa streaming stability/packaging ergonomics).

---

## Scope

### In scope

- Introduce backend selection (`nemo | sherpa`) via config + CLI override.
- Split NeMo implementation into dedicated backend module.
- Add Sherpa ONNX streaming backend implementing shared runtime contract.
- Make app loop backend-driven for chunk sizing and optional diagnostics.
- Make preflight/dependencies backend-aware.
- Update tests/docs/examples/optional dependency groups.
- Remove `Config.native_chunk_samples` once app/harness are migrated.

### Out of scope

- Rewriting the app orchestration loop.
- New UI/overlay behavior.
- Multi-backend model auto-download service.
- Moonshine implementation in the same PR (follow-up milestone only).

---

## Target Architecture

## 1) Backend contract (`ASRBackend` ABC)

```python
# shuvoice/asr_base.py
from abc import ABC, abstractmethod
import numpy as np

class ASRBackend(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def process_chunk(self, audio_chunk: np.ndarray) -> str: ...

    @property
    @abstractmethod
    def native_chunk_samples(self) -> int: ...

    @property
    def debug_step_num(self) -> int | None:
        return None

    @staticmethod
    @abstractmethod
    def dependency_errors() -> list[str]: ...

    @classmethod
    def download_model(cls, **kwargs) -> None:
        raise NotImplementedError(f"{cls.__name__} does not support model download")
```

## 2) Module layout and compatibility

```
shuvoice/
  asr_base.py      (new — backend ABC)
  asr_nemo.py      (new — NeMo implementation moved from current asr.py)
  asr_sherpa.py    (new — Sherpa ONNX implementation)
  asr.py           (registry + factory + compatibility exports)
```

`shuvoice/asr.py` responsibilities:
- `ASREngine` remains an alias to `NemoBackend` for backward compatibility in this cycle.
- Add `get_backend_class(name: str) -> type[ASRBackend]`.
- Add `create_backend(backend_name: str, config: Config) -> ASRBackend`.
- Use lazy backend imports in registry (avoid importing optional deps on module import).

## 3) Backend-owned chunk sizing

- Replace all `self.config.native_chunk_samples` usage with `self.asr.native_chunk_samples`.
- Remove `Config.native_chunk_samples` after migration.
- Keep `right_context` as NeMo-only config.
- Introduce Sherpa chunk config as `sherpa_chunk_ms` and compute samples in backend.

---

## Proposed Config Schema

Add new config keys (flattened loader behavior remains the same):

| Key | Type | Default | Applies to | Notes |
|---|---|---|---|---|
| `asr_backend` | `str` | `"nemo"` | all | validated choices: `nemo`, `sherpa` |
| `model_name` | `str` | current default | nemo | unchanged |
| `right_context` | `int` | `13` | nemo | unchanged; NeMo-only |
| `use_cuda_graph_decoder` | `bool` | `false` | nemo | unchanged |
| `sherpa_model_dir` | `str \| None` | `None` | sherpa | required when backend=sherpa |
| `sherpa_provider` | `str` | `"cpu"` | sherpa | initial choices: `cpu`, `cuda` |
| `sherpa_num_threads` | `int` | `2` | sherpa | decoder threads |
| `sherpa_chunk_ms` | `int` | `100` | sherpa | must be `> 0` |

Notes:
- Existing configs without `asr_backend` continue to run NeMo.
- `Config.load()` accepts one-level nested TOML sections; keys must match dataclass fields exactly.
  - Example: in `[asr]`, use `sherpa_model_dir = "..."` (not `model_dir = "..."`).
- Keep `right_context` CLI/config available, but document as NeMo-only.

---

## Detailed Implementation Plan

## Milestone 0 — Design lock + compatibility rules

- [x] Finalize `ASRBackend` interface in `shuvoice/asr_base.py`.
- [x] Lock compatibility strategy:
  - [x] Keep `ASREngine` as NeMo compatibility alias (no factory-wrapper class trick).
  - [x] Introduce canonical factory APIs (`get_backend_class`, `create_backend`).
- [x] Lock Sherpa v1 model contract:
  - [x] Support streaming transducer models only.
  - [x] Validate required artifacts in `sherpa_model_dir` (tokens + model ONNX files).
  - [x] Keep initial sample-rate expectation at 16k; fail clearly on unsupported mismatch.
- [x] Lock `--download-model` behavior:
  - NeMo: keep current pre-download behavior.
  - Sherpa: clear guidance error (manual download URL).

Validation:
- [x] Contract/API notes are committed in plan + docstring comments before code migration.

## Milestone 1 — Extract NeMo into `asr_nemo.py` (no behavior change)

Files touched:
- `shuvoice/asr_base.py` (new)
- `shuvoice/asr_nemo.py` (new)
- `shuvoice/asr.py` (rewrite as registry/factory + compat exports)
- `tests/test_asr.py`

Tasks:
- [x] Create `ASRBackend` ABC.
- [x] Move current `ASREngine` implementation to `NemoBackend(ASRBackend)`.
- [x] Preserve NeMo parity (`load/reset/process_chunk`, dependency checks, model download, transcript normalization).
- [x] Move right-context→native chunk mapping into `NemoBackend.native_chunk_samples`.
- [x] Add `debug_step_num` property exposing `_step_num`.
- [x] Rebuild `shuvoice/asr.py` with:
  - lazy registry (`"nemo"`, `"sherpa"` class resolvers),
  - `get_backend_class(...)`,
  - `create_backend(...)`,
  - `ASREngine = NemoBackend` compatibility alias.
- [x] Update `tests/test_asr.py`:
  - keep existing NeMo-focused tests through compatibility import,
  - add resolver/factory tests for valid+invalid backend names.

Validation:
```bash
pytest tests/test_asr.py tests/test_config.py -q
pytest -m "not gui" -q
```

## Milestone 2 — Config + CLI backend selection

Files touched:
- `shuvoice/config.py`
- `shuvoice/__main__.py`
- `tests/test_config.py`
- `examples/config.toml`

Tasks:
- [x] Add config fields:
  - `asr_backend: str = "nemo"`
  - `sherpa_model_dir: str | None = None`
  - `sherpa_provider: str = "cpu"`
  - `sherpa_num_threads: int = 2`
  - `sherpa_chunk_ms: int = 100`
- [x] Add validation in `Config.__post_init__`:
  - backend choices,
  - provider choices,
  - `sherpa_chunk_ms > 0`,
  - `sherpa_num_threads >= 1`.
- [x] Keep all NeMo fields unchanged.
- [x] Add CLI flags:
  - `--asr-backend {nemo,sherpa}`
  - `--sherpa-model-dir`
  - `--sherpa-provider`
  - `--sherpa-num-threads`
  - `--sherpa-chunk-ms`
- [x] Update `--right-context` help text to explicitly say NeMo-only.
- [x] Ensure overrides are applied before preflight/download-model paths.
- [x] Update `examples/config.toml` with commented Sherpa options using `sherpa_*` keys.

Validation:
```bash
pytest tests/test_config.py -q
python -m shuvoice --help
```

## Milestone 3 — App loop decoupling from NeMo internals

Files touched:
- `shuvoice/app.py`
- `shuvoice/config.py`
- `tests/test_config.py`

Tasks (all current coupling sites):
- [x] Constructor: replace direct `ASREngine(...)` with `create_backend(config.asr_backend, config)`.
- [x] Replace every `self.config.native_chunk_samples` reference with `self.asr.native_chunk_samples`.
- [x] Replace `self.asr._step_num` logs with `self.asr.debug_step_num` and `%s` formatting.
- [x] Remove `Config.native_chunk_samples` property.
- [x] Move chunk-scaling assertions from config tests into NeMo backend tests.

Validation:
```bash
rg -n "self\.asr\._step_num|config\.native_chunk_samples" shuvoice/app.py  # expect no matches
pytest -m "not gui" -q
```
- [x] Manual NeMo regression smoke: backend runtime smoke completed using deterministic file-based dictation (`tests/audio-sample.wav`) with transcript output recorded in `build/asr-smoke/manual-backend-smoke.txt`.

## Milestone 4 — Implement Sherpa streaming backend

Files touched:
- `shuvoice/asr_sherpa.py` (new)
- `shuvoice/asr.py`
- `tests/test_asr.py`

Tasks:
- [x] Implement `SherpaBackend(ASRBackend)`:
  - `load()`: initialize recognizer from validated local model files.
  - `reset()`: create new stream per utterance.
  - `process_chunk(audio_chunk)`: feed waveform + read cumulative text.
  - `native_chunk_samples`: `config.sample_rate * config.sherpa_chunk_ms // 1000`.
  - `dependency_errors()`: import check for `sherpa_onnx`.
  - `download_model()`: raise `NotImplementedError` with release URL guidance.
- [x] Enforce model contract on load (clear errors for missing files/unsupported layout).
- [x] Register backend name `"sherpa"`.
- [x] Add tests:
  - resolver/factory returns Sherpa backend,
  - missing `sherpa_onnx` dependency errors,
  - missing `sherpa_model_dir` or required files raises clear errors.

Validation:
```bash
pytest tests/test_asr.py -q
```
- [x] Manual runtime smoke with known Sherpa streaming model (`sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06`) completed; transcript output recorded in `build/asr-smoke/manual-backend-smoke.txt`.

## Milestone 5 — Preflight and download behavior by backend

Files touched:
- `shuvoice/__main__.py`

Tasks:
- [x] Replace hardcoded NeMo stack check with backend-aware check:
  - resolve backend class via `get_backend_class(config.asr_backend)`,
  - call `dependency_errors()` and report backend name in output.
- [x] Make `--download-model` backend-class dispatched:
  - use `get_backend_class(config.asr_backend).download_model(...)`,
  - catch `NotImplementedError` and print user guidance.
- [x] Ensure preflight succeeds in Sherpa-only environments without torch/nemo.

Validation:
```bash
python -m shuvoice --preflight
python -m shuvoice --preflight --asr-backend sherpa
python -m shuvoice --download-model --asr-backend sherpa
```

## Milestone 6 — Packaging, docs, and examples

Files touched:
- `pyproject.toml`
- `README.md`
- `examples/config.toml`

Tasks:
- [x] Add optional extras:
  - `asr-nemo = ["torch>=2.4", "nemo-toolkit[asr]>=2.6.0"]`
  - `asr-sherpa = ["sherpa-onnx"]`
  - keep `asr` as compatibility alias (duplicate NeMo dependency list).
- [x] Update README:
  - remove Nemotron-only branding,
  - document backend selection + install commands,
  - add backend-specific troubleshooting and preflight expectations.
- [x] Update `examples/config.toml`:
  - `asr_backend = "nemo"` default,
  - commented Sherpa config using `sherpa_*` keys,
  - note that `right_context` is NeMo-only.

Validation:
```bash
pip install -e ".[asr-nemo]"
pip install -e ".[asr-sherpa]"
pip install -e ".[asr]"
```

## Milestone 7 — Tooling + harness compatibility

Files touched:
- `scripts/tts_roundtrip.py`
- `tests/test_asr.py`

Tasks:
- [x] Replace direct `ASREngine(...)` construction with backend factory.
- [x] Replace `cfg.native_chunk_samples` usage with `engine.native_chunk_samples`.
- [x] Add `--asr-backend` + Sherpa override CLI flags where relevant.
- [x] Update harness typing to backend protocol (`ASRBackend`) instead of NeMo concrete type.
- [x] Finalize tests:
  - factory/config-driven backend selection,
  - NeMo backend class tests,
  - Sherpa backend error-path tests,
  - headless-safe coverage.

Validation:
```bash
pytest -m "not gui" -q
python scripts/tts_roundtrip.py --help
```

## Milestone 8 — Optional Moonshine backend (follow-up PR)

- [x] Add `shuvoice/asr_moonshine.py` implementing `ASRBackend`.
- [x] Register in factory; add config keys, CLI flags, extras.
- [x] Reuse preflight/docs/factory patterns after Sherpa path stabilizes.

---

## Files confirmed as no-change

These modules are backend-agnostic and should not require modification for this plan:

| File | Reason |
|---|---|
| `shuvoice/utterance_state.py` | already chunk-size parameterized |
| `shuvoice/transcript.py` | pure transcript merge logic |
| `shuvoice/streaming_health.py` | pure stall-detection logic |
| `shuvoice/overlay.py` | UI only |
| `shuvoice/hotkey.py` | input only |
| `shuvoice/typer.py` | text injection only |
| `shuvoice/audio.py` | capture only |
| `shuvoice/control.py` | IPC only |
| `shuvoice/feedback.py` | tones only |
| `shuvoice/postprocess.py` | text post-processing only |
| `shuvoice/overlay_state.py` | state enum only |
| `tests/conftest.py` | sounddevice mocking only |

---

## Test Strategy

### Unit tests (headless-safe)

- **Factory/resolver behavior**
  - valid backend name resolves/constructs expected class,
  - invalid name raises clear `ValueError`,
  - lazy resolver does not import optional backend deps unless selected.

- **Dependency checks**
  - NeMo deps missing/present behavior,
  - Sherpa deps missing/present behavior.

- **Config parsing/validation**
  - `asr_backend` and Sherpa fields parse from TOML,
  - invalid backend/provider/chunk settings fail fast,
  - old config (no `asr_backend`) defaults to NeMo.

- **Backend-specific tests**
  - NeMo: `native_chunk_samples` mapping + `debug_step_num`.
  - Sherpa: missing model dir/files and clear error messages.

### Integration/manual tests

- **NeMo regression smoke**
  - short + long phrase dictation unchanged vs baseline.

- **Sherpa smoke**
  - live mic streaming + stop flush,
  - partial hypothesis stability for long phrases.

- **Preflight matrix**
  - NeMo env passes with NeMo backend.
  - Sherpa-only env passes with Sherpa backend.

---

## Risks and Mitigations

1. **Partial transcript churn (highest practical risk)**
   - Mitigation: keep `prefer_transcript()` merge logic + existing final/tail flush behavior.

2. **Sherpa model packaging fragmentation**
   - Mitigation: lock v1 model family and required file contract; fail with explicit setup instructions.

3. **Dependency confusion for users**
   - Mitigation: backend-specific extras + backend-named preflight errors.

4. **NeMo regression risk during extraction**
   - Mitigation: Milestone 1 is strictly no-behavior-change; green tests before Sherpa code lands.

5. **Sample-rate assumptions**
   - Mitigation: enforce/validate 16k expectation for v1; defer generalized resampling architecture.

---

## Recommended Execution Order

1. **Milestone 0–1**: contract + NeMo extraction (no behavior change)
2. **Milestone 2–3**: config/CLI + app decoupling
3. **Milestone 4**: Sherpa backend implementation
4. **Milestone 5–7**: preflight, packaging, docs, tooling
5. **Milestone 8 (optional)**: Moonshine follow-up

This sequence keeps the app functional at each checkpoint and simplifies rollback.

---

## Code References

### Internal files touched

| File | Milestones |
|---|---|
| `shuvoice/asr_base.py` | 0, 1 (new) |
| `shuvoice/asr.py` | 1, 4, 5 |
| `shuvoice/asr_nemo.py` | 1 (new) |
| `shuvoice/asr_sherpa.py` | 4 (new) |
| `shuvoice/app.py` | 3 |
| `shuvoice/config.py` | 2, 3 |
| `shuvoice/__main__.py` | 2, 5 |
| `scripts/tts_roundtrip.py` | 7 |
| `tests/test_asr.py` | 1, 4, 7 |
| `tests/test_config.py` | 2, 3 |
| `pyproject.toml` | 6 |
| `README.md` | 6 |
| `examples/config.toml` | 2, 6 |

### External references

- NeMo: `https://github.com/NVIDIA/NeMo`
- Sherpa ONNX: `https://github.com/k2-fsa/sherpa-onnx`
- Sherpa Python examples: `https://github.com/k2-fsa/sherpa-onnx/tree/master/python-api-examples`
