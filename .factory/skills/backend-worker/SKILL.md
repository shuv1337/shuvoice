---
name: backend-worker
description: Implements optional ShuVoice model-worker and worker-protocol features with TDD
---

# Backend Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use for features that involve:

- Versioned worker protocol code in `workers/shuvoice_worker_proto/` and/or
  `crates/shuvoice-worker-proto/`
- Optional engine packages under `workers/nemo_asr/`, `workers/moonshine_asr/`,
  `workers/melotts/`
- Worker discovery / spawn composition in
  `crates/shuvoice-cli/src/compose/worker_runtime.rs` and Melo paths in
  `compose/tts_adapter.rs` / `shuvoice-tts` (worker-proto only)
- Golden frame / unittest coverage under `workers/tests/`
- Host-side worker client behavior in `crates/shuvoice-asr` (worker backend)

**Do not** use this skill to revive deleted in-process Python app modules under
`shuvoice/*.py`, legacy Melo helper scripts, or Sherpa wheel/RUNPATH repair.

For pure Rust domain/config/UI/control work without workers, prefer the
relevant crate directly and follow `AGENTS.md`.

## Work Procedure

### 1. Understand the Feature

- Read the feature description, preconditions, expectedBehavior, and verificationSteps
- Read `AGENTS.md` and `docs/adr/0002-versioned-model-workers.md`
- Read `.factory/library/architecture.md` and `.factory/library/environment.md`
- Read `workers/README.md` for wire format, layout, and safety rules
- Identify whether the change is protocol, engine worker, host spawn, or tests

### 2. Study Existing Patterns

Before writing code, read the closest implementation:

- Protocol (Python): `workers/shuvoice_worker_proto/`
- Protocol (Rust): `crates/shuvoice-worker-proto/`
- Engines: `workers/nemo_asr/`, `workers/moonshine_asr/`, `workers/melotts/`
- Host ASR worker client: `crates/shuvoice-asr/src/worker/`
- Host discovery/spawn: `crates/shuvoice-cli/src/compose/worker_runtime.rs`
- Melo host path: `crates/shuvoice-tts` Melo backend + CLI `tts_adapter`
  (worker-proto only; no legacy helper)
- Tests: `workers/tests/` (stdlib unittest + golden bytes), relevant
  `crates/*/tests/*worker*`

### 3. Write Tests First (TDD — Red)

- Prefer `workers/tests/` stdlib unittest for protocol/engine framing
- Prefer `cargo test -p shuvoice-worker-proto` / `-p shuvoice-asr` /
  `-p shuvoice-cli` for host-side changes
- Cover expectedBehavior items; keep tests free of real NeMo/Melo/Moonshine
  packages unless the feature explicitly requires a gated integration
- Confirm new tests fail before implementation

```bash
cd workers && python -m unittest discover -s tests -v
cargo test -p shuvoice-worker-proto
```

### 4. Implement (Green)

- Minimal change to satisfy tests
- Keep workers **out of** the Rust app import/link graph
- Stdio transport only; logs on stderr; no secrets/transcripts in protocol `error` payloads
- Honor frame size caps and cancel/demux rules from `workers/README.md`
- Melo production path must remain worker-proto (`-m melotts`), never a helper script
- Capability flags omitted from manifests default **false** (safe)

### 5. Verify

```bash
cd workers && python -m unittest discover -s tests -v
cargo fmt -- --check
cargo check -p shuvoice-cli
cargo clippy -p shuvoice-cli -- -D warnings
cargo test -p shuvoice-cli
# plus any crate-specific tests you touched
```

Do not start user services, touch live config, or download models unless the
feature explicitly requires it and the user approved.

### 6. Commit

- Stage only files related to the feature
- Message style: `feat(workers): …`, `fix(worker-proto): …`, `test(workers): …`

## Example Handoff

```json
{
  "salientSummary": "Extended worker-proto cancel demux coverage and aligned Melo spawn discovery with SHUVOICE_WORKERS_DIR. workers unittest green; shuvoice-worker-proto + shuvoice-cli tests green.",
  "whatWasImplemented": "Protocol/test updates under workers/ and host discovery assertions in shuvoice-cli compose/worker_runtime.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "python -m unittest discover -s tests -v", "exitCode": 0, "observation": "workers tests passed"},
      {"command": "cargo test -p shuvoice-worker-proto", "exitCode": 0, "observation": "proto crate passed"},
      {"command": "cargo test -p shuvoice-cli", "exitCode": 0, "observation": "cli tests passed"}
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": []
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- A precondition is not met (expected file/module missing)
- The tree still expects deleted `shuvoice/*.py` app modules as the production path
- Existing tests fail before any changes
- Scope requires packaging/CI/service changes outside the assigned files
- Feature description conflicts with ADR 0002 (workers must stay isolated)
