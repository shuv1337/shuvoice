# Architecture

Architectural decisions, patterns, and conventions for TTS backend development.

**What belongs here:** Backend interface contract, audio format requirements, registration patterns, config conventions.

---

## TTS Backend Contract

All TTS backends must subclass `TTSBackend` from `shuvoice/tts_base.py` and implement:

| Method | Returns | Purpose |
|---|---|---|
| `sample_rate_hz()` | `int` | PCM sample rate |
| `synthesize_stream(request)` | `Iterator[bytes]` | Yield raw PCM int16 mono chunks |
| `list_voices()` | `list[VoiceInfo]` | Available voices for UI |
| `dependency_errors()` | `list[str]` (static) | Missing deps/runtime errors |

Set class-level `capabilities: TTSCapabilities` with speed control flags.

## Audio Format

- **Format**: Raw PCM int16 mono (little-endian, `<i2`)
- **Sample rate**: Backend-specific (24000 for cloud, 22050 for Piper, 44100 for MeloTTS)
- **Streaming**: Yield `bytes` chunks via `Iterator[bytes]`

## Backend Registration

In `shuvoice/tts.py`, add a lazy resolver to `_TTS_BACKEND_REGISTRY`:
```python
"melotts": _resolve_melotts,
```
Resolver does lazy import to avoid loading heavy modules.

## Config Pattern

1. Add fields to `Config` dataclass in `config.py`
2. Add field names to `CONFIG_SECTION_FIELDS["tts"]` tuple
3. Add validation in `Config.__post_init__`
4. Add auto-default logic for backend-specific defaults

## Wizard Pattern

1. Add backend tuple to `TTS_BACKENDS` in `wizard_state.py`
2. Add case to `default_tts_voice_for_backend()`
3. Add voice label logic to `tts_voice_label()`
4. Handle UI controls in `wizard/__init__.py` `_build_tts_page()`

## Subprocess Isolation Pattern (Piper reference)

Local TTS backends use subprocess to avoid dependency conflicts:
- Backend spawns a subprocess per synthesis request
- Text goes in via stdin/args, PCM audio comes out via stdout
- The `tts_local.py` Piper backend is the reference implementation

### Cross-venv subprocess invocation (MeloTTS pattern)

When a helper script runs in an **isolated venv** (not the main ShuVoice venv), you must invoke it by **absolute file path**, not `-m module`:

```python
# CORRECT — works because the file path is independent of which packages are installed:
helper_path = Path(__file__).with_name("melo_helper.py")
command = [venv_python_bin, str(helper_path), ...]

# WRONG — fails with ModuleNotFoundError because `shuvoice` is NOT installed in the isolated venv:
command = [venv_python_bin, "-m", "shuvoice.melo_helper", ...]
```

The Piper backend doesn't have this issue because it invokes a standalone binary (`piper`) that's in PATH, not a Python module from another package.
