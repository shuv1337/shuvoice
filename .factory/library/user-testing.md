# User Testing

Testing surface, resource cost classification, and validation approach.

**What belongs here:** How to test the application, what surfaces exist, concurrency limits.

---

## Validation Surface

ShuVoice is a desktop Wayland/GTK4 application. The primary testing surfaces are:

1. **pytest unit tests** (primary) — All backend behavior is tested via mocked subprocess interactions. Run with `uv run pytest tests/ -x -q`.
2. **CLI commands** — `shuvoice setup`, `shuvoice wizard`, `shuvoice config` can be tested via subprocess invocations in tests.
3. **Direct Python verification** — Import and instantiate backends, verify config loading.

### Surfaces NOT testable in automation
- GTK4 overlay UI (requires Wayland display server + layer shell)
- Audio playback (requires audio hardware + PipeWire)
- Push-to-talk interaction (requires keyboard + Hyprland)

## Validation Concurrency

- Machine: 24 cores, 123GB RAM, ~35GB used baseline
- Max concurrent validators: **5** (pytest is lightweight, ~200MB per instance)
- No browser or heavy GUI testing involved

## Testing Approach for MeloTTS Backend

All MeloTTS backend assertions are verified through pytest unit tests:
- Subprocess interactions are mocked (no real MeloTTS venv needed for tests)
- Config validation tested via direct Config instantiation
- Registry tested via get_tts_backend_class() calls
- Wizard tested via write_config() + state assertions
- Setup tested via mocked subprocess commands

## Flow Validator Guidance: pytest

**Surface**: pytest unit tests
**Tool**: Direct `uv run pytest` execution (no special skill needed)
**Isolation**: Each validator runs `uv run pytest` on separate test files — no shared state conflict.
**Boundaries**:
- Do NOT modify any source or test files
- Do NOT install packages or modify the venv
- Run tests in read-only mode only
- Each validator targets specific test files and uses `-k` filters when needed
- Use verbose mode (`-v`) to get individual test names for evidence
**Concurrency**: Safe to run up to 5 validators in parallel (separate test files, no shared state)
**Evidence**: Capture test output showing pass/fail for each test mapped to an assertion
