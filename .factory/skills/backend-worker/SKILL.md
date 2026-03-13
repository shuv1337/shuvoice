---
name: backend-worker
description: Implements Python backend features for ShuVoice with TDD
---

# Backend Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use for features that involve:
- Creating or modifying Python modules in `shuvoice/`
- Adding or updating configuration keys in `config.py`
- Modifying the TTS/ASR backend registry
- Updating wizard state or UI logic
- Adding setup automation logic
- Writing tests in `tests/`

## Work Procedure

### 1. Understand the Feature

- Read the feature description, preconditions, expectedBehavior, and verificationSteps carefully
- Read `mission.md` for overall context
- Read `.factory/library/architecture.md` for backend patterns and conventions
- Read `.factory/research/melotts.md` for MeloTTS API details
- Identify which existing files need modification and which new files need creation

### 2. Study Existing Patterns

Before writing any code, read the relevant existing implementations to match style:
- For backend work: read `shuvoice/tts_local.py` (subprocess pattern), `shuvoice/tts_base.py` (interface)
- For config work: read `shuvoice/config.py` (field registration, validation, auto-defaults)
- For registry work: read `shuvoice/tts.py` (lazy resolver pattern)
- For wizard work: read `shuvoice/wizard_state.py` and `shuvoice/wizard/__init__.py`
- For setup work: read `shuvoice/setup_helpers.py` and `shuvoice/setup_command.py`
- For tests: read the corresponding `tests/test_tts_*.py` file for the pattern

### 3. Write Tests First (TDD - Red Phase)

- Create test file(s) following existing naming: `tests/test_tts_melotts.py`, etc.
- Write tests that cover ALL expectedBehavior items from the feature description
- Match existing test patterns exactly (fixtures, monkeypatching, assertions style)
- Tests MUST fail at this point (the implementation doesn't exist yet)
- Run the new tests to confirm they fail: `uv run pytest tests/test_<file>.py -x -q`

### 4. Implement (Green Phase)

- Write the minimum code to make all tests pass
- Follow existing code style exactly (imports, docstrings, type hints, naming)
- For subprocess-based backends: ensure no real MeloTTS dependency in the main venv
- All MeloTTS-specific imports must be inside the helper script only

### 5. Verify

- Run the new tests: `uv run pytest tests/test_<file>.py -x -v`
- Run the full test suite: `uv run pytest tests/ -x -q`
- Run linter: `uv run ruff check shuvoice/ tests/`
- Run formatter check: `uv run ruff format --check shuvoice/ tests/`
- If ruff reports issues, fix them and re-run
- Every verification step must pass before completing

### 6. Commit

- Stage only files related to this feature
- Write a descriptive commit message following repo conventions (e.g., `feat(tts): add MeloTTS backend module`)

## Example Handoff

```json
{
  "salientSummary": "Implemented MeloTTSBackend in shuvoice/tts_melotts.py with subprocess helper protocol. Backend manages helper lifecycle, yields PCM int16 mono at 44100 Hz, lists 5 English voices, reports dependency errors. All 8 new tests pass, full suite (543 passed) clean, ruff clean.",
  "whatWasImplemented": "Created shuvoice/tts_melotts.py with MeloTTSBackend class subclassing TTSBackend. Implements synthesize_stream() via subprocess stdin/stdout protocol, list_voices() returning 5 VoiceInfo entries, sample_rate_hz() returning 44100, dependency_errors() checking venv existence. Created shuvoice/melo_helper.py as the subprocess entry point.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "uv run pytest tests/test_tts_melotts.py -x -v", "exitCode": 0, "observation": "8 tests passed"},
      {"command": "uv run pytest tests/ -x -q", "exitCode": 0, "observation": "543 passed, 2 skipped"},
      {"command": "uv run ruff check shuvoice/ tests/", "exitCode": 0, "observation": "no issues found"},
      {"command": "uv run ruff format --check shuvoice/ tests/", "exitCode": 0, "observation": "all files formatted"}
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {
        "file": "tests/test_tts_melotts.py",
        "cases": [
          {"name": "test_sample_rate_hz_returns_44100", "verifies": "Backend reports correct sample rate"},
          {"name": "test_synthesize_stream_yields_pcm_chunks", "verifies": "Synthesis produces valid PCM audio"},
          {"name": "test_list_voices_returns_five_english_voices", "verifies": "All 5 voices listed with correct info"},
          {"name": "test_capabilities_include_speed_control", "verifies": "Speed control enabled with correct range"},
          {"name": "test_dependency_errors_missing_venv", "verifies": "Reports missing venv"},
          {"name": "test_dependency_errors_missing_helper", "verifies": "Reports missing helper script"},
          {"name": "test_subprocess_crash_raises_error", "verifies": "Subprocess errors handled"},
          {"name": "test_speed_forwarded_to_helper", "verifies": "Speed param reaches helper"}
        ]
      }
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- A precondition is not met (e.g., a file that should exist doesn't)
- The existing code structure differs significantly from what was described
- Existing tests fail before any changes are made (pre-existing issue)
- A required pattern or utility function is missing from the codebase
- The feature scope is ambiguous or contradictory
