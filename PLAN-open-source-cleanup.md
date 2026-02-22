# PLAN: Open-Source Environment Cleanup

## Objective

Remove or generalize environment-specific values (user paths, UID-specific runtime paths, local venv names, personal app IDs, and machine-specific docs) so ShuVoice is ready for public consumption.

---

## Scope

In scope:
- Runtime/source portability cleanup
- Example config cleanup
- Public docs cleanup
- Test/doc command portability
- Add a repeatable "hardcoded env" audit command for release checks

Out of scope:
- Backend behavior changes
- Performance tuning
- New feature work

---

## Current findings (from scan)

### High priority
- `examples/config-sherpa-cpu.toml` and `examples/config-sherpa-cuda.toml`
  - hardcoded `sherpa_model_dir = "/home/shuv/repos/shuvoice/..."`
  - hardcoded `control_socket = "/run/user/1000/shuvoice/control.sock"`
- `examples/config-nemo-cpu.toml`, `examples/config-nemo-cuda.toml`, `examples/config-moonshine-cpu.toml`
  - hardcoded `/run/user/1000/...` socket paths
- `examples/config.toml`
  - commented-out `# control_socket = "/run/user/1000/..."` (line 69) — not active but leaks the UID
- `shuvoice/app.py`
  - `application_id="dev.shuv.shuvoice"` uses personal namespace
- `AGENTS.md`
  - extensive local machine details, user paths, and host-specific stack data (~16 lines with hardcoded paths)

### Medium priority
- `README.md` and `packaging/systemd/user/shuvoice.service`
  - comments reference `.venv312`
- `tests/integration/test_roundtrip_regression.py` and `tests/integration/test_sherpa_low_noise_regression.py`
  - `.venv312` fallback logic in `_roundtrip_python()`
- Planning/internal docs (two files with hardcoded paths):
  - `PLAN-12-13-moonshine-fixes.md` (6 hits: `.venv312` in commands)
  - `PLAN-sherpa-gpu-enable.md` (9 hits: `.venv312` in commands)
  - Remaining plan files (`PLAN-asr-backend-adapters.md`, `PLAN-nemo-gain-accuracy.md`, `PLAN-shuvoice-improvements.md`, `PLAN-shuvoice-production-hardening.md`, `shuvoice-plan.md`) are clean

---

## Decisions (resolve before implementation)

### D1. GTK application ID namespace
**Decision**: Use `io.github.shuv1337.shuvoice`

This aligns with the existing GitHub org/user and follows RDNN conventions.
If the project moves to a GitHub org later, a single rename pass will be needed.

### D2. AGENTS.md handling
**Decision**: Option A — keep in repo, sanitize machine-specific content

Genericize all paths (`.venv312` → `.venv`, `/home/shuv/repos/shuvoice` → repo-relative)
while preserving the GPU build instructions and backend reference tables, which are
genuinely useful for contributors.

### D3. Internal plan docs handling
**Decision**: Move to `docs/internal/` and add `docs/internal/` to `.gitignore`

This removes plan files from the public repo without deleting them locally.
Only two files (`PLAN-12-13-moonshine-fixes.md`, `PLAN-sherpa-gpu-enable.md`) have
hardcoded paths; the move sidesteps per-file cleanup.

---

## Milestone 1 — Runtime + examples (blocker)

### 1.1 Replace hardcoded socket paths in examples

Four example configs have **active** `control_socket` lines with `/run/user/1000/...`.
One (`examples/config.toml`) has it **commented out** — remove the commented line or
replace the UID with a generic placeholder.

For all files: remove explicit `control_socket` lines entirely and let the runtime
default to `$XDG_RUNTIME_DIR/shuvoice/control.sock`. Add a single comment showing
the default pattern.

- [x] `examples/config-nemo-cpu.toml` — remove active `control_socket` line, add comment showing default
- [x] `examples/config-nemo-cuda.toml` — same
- [x] `examples/config-sherpa-cpu.toml` — same
- [x] `examples/config-sherpa-cuda.toml` — same
- [x] `examples/config-moonshine-cpu.toml` — same
- [x] `examples/config.toml` — remove or genericize commented-out line (line 69)

### 1.2 Replace hardcoded local Sherpa model paths in examples

- [x] Replace `/home/shuv/repos/shuvoice/build/asr-models/...` with a placeholder:
      `sherpa_model_dir = "/path/to/sherpa-model-dir"`
- [x] Add a brief inline comment cross-referencing the README for model download/placement instructions
      (e.g. `# See README.md § Sherpa ONNX for model download steps`)

Files:
- `examples/config-sherpa-cpu.toml`
- `examples/config-sherpa-cuda.toml`

### 1.3 Make GTK application ID public-neutral

- [x] Change `application_id` from `"dev.shuv.shuvoice"` to `"io.github.shuv1337.shuvoice"`
- [x] Run `rg -rn 'dev\.shuv\.shuvoice'` across entire repo — confirm only `shuvoice/app.py` matches
      (currently verified: no `.desktop` files, no other references outside PLAN/AGENTS docs)
- [ ] Verify overlay/window behavior unchanged (GTK registers the app ID as a D-Bus well-known name;
      confirm no existing D-Bus client or desktop entry depends on the old name)
- [ ] After rename, test: `python -m shuvoice --preflight` and a live push-to-talk cycle

File:
- `shuvoice/app.py:42`

---

## Milestone 2 — Public docs cleanup (blocker)

### 2.1 README portability pass

- [x] Replace `.venv312` mentions with `.venv` (or generic `<venv>`)
- [x] Ensure all path examples are generic/placeholders
- [x] Keep `/usr/bin/shuvoice` references where package-install specific (valid)
- [x] Keep `shuv1337` in GitHub URLs/badges (these are correct public references)

File:
- `README.md`

### 2.2 systemd template comments

- [x] Replace `%h/repos/shuvoice/.venv312/bin/shuvoice` with generic `%h/.venv/bin/shuvoice`

File:
- `packaging/systemd/user/shuvoice.service:12`

### 2.3 AGENTS.md sanitization (Option A)

Genericize all machine-specific content while preserving the document's value as a
contributor reference. Specific changes:

**Keep as-is** (genuinely useful):
- Backend comparison tables, config key tables, model location table structure
- GPU build instructions (sherpa wheel build, CUDA compat lib steps)
- Known issues table
- System prerequisites list

**Genericize** (replace with portable equivalents):
- [x] `.venv312` → `.venv` throughout (~10 occurrences)
- [x] `/home/shuv/repos/shuvoice` → repo-relative paths or `$REPO_ROOT` (~5 occurrences)
- [x] Service unit example: replace hardcoded `WorkingDirectory` and `ExecStart` with
      generic paths matching `packaging/systemd/user/shuvoice.service` format
- [x] `sherpa_model_dir` examples: use placeholder `/path/to/sherpa-model-dir`
- [x] Environment table: remove user-specific UID, kernel version, driver version — keep
      only the minimum versions required (Python 3.12+, CUDA 12.x, PyTorch 2.x)

**Verify consistency**: after cleanup, the service unit example in AGENTS.md should
match the format of `packaging/systemd/user/shuvoice.service`.

File:
- `AGENTS.md`

---

## Milestone 3 — Test + tooling portability (important)

### 3.1 Integration test interpreter selection

Both test files have identical `_roundtrip_python()` functions that prefer `.venv312`.

- [x] Remove `.venv312` fallback from `_roundtrip_python()` in both files
- [x] Keep `SHUVOICE_ROUNDTRIP_PYTHON` env var override as first choice
- [x] Fall back to `sys.executable` (the interpreter running pytest)
- [x] Keep behavior deterministic across CI/local

Files:
- `tests/integration/test_roundtrip_regression.py:43`
- `tests/integration/test_sherpa_low_noise_regression.py:84`

**Note**: `test_sherpa_low_noise_regression.py:18` also has `DEFAULT_SHERPA_MODEL_DIR`
pointing to `ROOT / "build" / "asr-models" / "sherpa-onnx-..."`. This is a relative
path (not `/home/shuv/`), and the test skips when the directory is absent, so it's
acceptable as-is.

### 3.2 Add a repo audit command/script

- [x] Add `scripts/check-env-hardcoding.sh` that scans tracked files for disallowed patterns
- [x] Implement a file-level exclusion list so `docs/internal/` and allowlisted files
      don't produce false positives
- [x] Exit non-zero if any disallowed hit is found in non-excluded files

**Disallowed patterns**:
- `/home/` (any user home directory)
- `/run/user/1000` (UID-specific runtime paths)
- `\.venv312` (hardcoded venv name)
- `dev\.shuv\.shuvoice` (old app ID)

**Excluded paths** (won't trigger failures):
- `docs/internal/` (internal plan files, moved in Milestone 4)
- `scripts/check-env-hardcoding.sh` (the script itself contains the patterns)
- `PLAN-open-source-cleanup.md` (this plan, if committed)

**Allowlisted patterns** (valid occurrences, never flagged):
- `/usr/bin/shuvoice`
- `/dev/input/event*`
- `/path/to/...` (placeholder patterns)
- `$XDG_RUNTIME_DIR`
- GitHub URLs containing `shuv1337` (these are correct public references)

Example implementation:

```bash
#!/usr/bin/env bash
set -euo pipefail

EXCLUDE_PATHS="docs/internal/|scripts/check-env-hardcoding\.sh|PLAN-open-source-cleanup\.md"
PATTERNS='/home/|/run/user/1000|\.venv312|dev\.shuv\.shuvoice'

hits=$(git ls-files \
  | grep -Ev "$EXCLUDE_PATHS" \
  | xargs rg -n "$PATTERNS" 2>/dev/null || true)

if [ -n "$hits" ]; then
  echo "ERROR: hardcoded environment values found:"
  echo "$hits"
  exit 1
fi

echo "OK: no hardcoded environment values in public files"
```

---

## Milestone 4 — Internal docs handling (recommended)

Move planning/internal docs out of the public tree.

- [x] Create `docs/internal/` directory
- [x] `git mv` the following tracked files into `docs/internal/`:
  - `PLAN-12-13-moonshine-fixes.md`
  - `PLAN-asr-backend-adapters.md`
  - `PLAN-nemo-gain-accuracy.md`
  - `PLAN-sherpa-gpu-enable.md`
  - `PLAN-shuvoice-improvements.md`
  - `PLAN-shuvoice-production-hardening.md`
  - `shuvoice-plan.md`
- [x] Add `docs/internal/` to `.gitignore` so future internal docs stay untracked
- [x] After the move, `git rm --cached` the files so they're untracked going forward
- [x] Verify no public docs (README, AGENTS.md) link to moved files

**Note**: `PLAN-open-source-cleanup.md` is currently untracked. Either commit it to
`docs/internal/` for reference or leave it untracked — it should not be committed to
the repo root.

---

## Validation checklist

### Repo scan

- [x] Run the audit script:

```bash
bash scripts/check-env-hardcoding.sh
```

Expected: exit 0, no hits in public files.

- [x] Manual cross-check (should only hit excluded/internal files):

```bash
git ls-files | xargs rg -n '/home/|/run/user/1000|\.venv312|dev\.shuv\.shuvoice'
```

### D-Bus / app ID verification

- [x] Confirm no `.desktop` files reference the old `dev.shuv.shuvoice` ID
- [ ] Confirm `dbus-monitor` shows the new `io.github.shuv1337.shuvoice` name on launch

### Functional smoke

- [x] `python -m shuvoice --help`
- [x] `python -m shuvoice --preflight`
- [x] `pytest -m "not gui" -v`
- [ ] Live push-to-talk cycle with overlay visible

---

## Suggested execution order

1. Milestone 1 (runtime/examples) — removes user-visible hardcoding
2. Milestone 2 (public docs) — cleans README, systemd template, AGENTS.md
3. Milestone 3 (tests/tooling) — fixes test portability, adds audit guardrail
4. Milestone 4 (internal docs) — moves plan files out of public tree

This ordering removes user-visible hardcoding first, then adds guardrails so
regressions are caught before release.
