# Polish, Stability & Performance Sprint Plan

> **Project**: ShuVoice v0.1.3  
> **Scope**: ~28K LoC Python, 60+ test files (~11.5K test LoC)  
> **Date**: 2026-04-15  

---

## Executive Summary

ShuVoice is in solid functional shape — ruff passes clean, no TODO/FIXME markers
remain, and test coverage spans unit, integration, and e2e tiers. This sprint
targets the next maturity level: hardening thread safety, closing resource leaks,
improving user-facing feedback, shrinking oversized modules, and eliminating
latency hiccups.

The plan is organized into **5 tracks** with prioritized work items. Each item
includes the affected files, the problem, and the concrete fix.

---

## Track 1: Stability — Thread Safety & Resource Cleanup

These are the highest-priority items. Race conditions and leaked resources cause
the hardest-to-diagnose production bugs.

### 1.1 Add lock protection to module-level caches

**Files**: `shuvoice/waybar/hyprland.py:11-12,76-84`  
**Problem**: `_cached_value` and `_cached_at_monotonic` are read/written from
multiple threads without synchronization. Under concurrent waybar polling the
cache can return stale or torn values.  
**Fix**: Wrap the cache read/write in a module-level `threading.Lock`. The
critical section is tiny (dict lookup + monotonic compare), so contention is
negligible.

### 1.2 Protect ASR NeMo encode cache from concurrent corruption

**Files**: `shuvoice/asr_nemo.py:178-237,265-266`  
**Problem**: `_cache_last_channel`, `_cache_last_time`, `_pre_encode_cache` are
mutated during `process_chunk()` without a lock. The code even has a comment
acknowledging the race at line 266.  
**Fix**: Guard cache reads/writes with a `threading.Lock` scoped to the
`NemoBackend` instance. The cache is only accessed in the hot inference path,
so use a non-reentrant lock to keep overhead minimal.

### 1.3 Fix double-check locking on `_asr_disabled` flag

**Files**: `shuvoice/app.py:158,379-384,412-417`  
**Problem**: The `_asr_disabled` flag is checked before acquiring a lock, then
re-checked after. Without a memory barrier (Python's GIL partially helps, but
the pattern is still fragile), another thread could flip the flag between the
two checks.  
**Fix**: Move the first check inside the lock, or switch to an
`threading.Event` which is inherently thread-safe for boolean signaling.

### 1.4 Ensure subprocess file handles are closed on exception paths

**Files**: `shuvoice/tts_melotts.py:136-192,229-241`  
**Problem**: `proc.stdin.close()` and `proc.stdout.close()` are called in the
happy path but not in exception branches. Repeated synthesis failures can
exhaust file descriptors.  
**Fix**: Use `try/finally` blocks (or a context-manager wrapper) so handles are
always closed. Pattern:
```python
try:
    proc.stdin.write(...)
    proc.stdin.close()
    ...
finally:
    for f in (proc.stdin, proc.stdout, proc.stderr):
        try:
            f.close()
        except Exception:
            pass
```

### 1.5 Clean shutdown: join daemon threads with timeout + force-kill

**Files**: `shuvoice/app.py:210,328,333,338-355`, `shuvoice/tts_player.py:410-420`  
**Problem**: Daemon threads are started without explicit shutdown tracking.
`do_shutdown()` doesn't wait for them, and `TTSPlayer.stop()` joins with a 1s
timeout but has no fallback if the thread doesn't exit.  
**Fix**:
- Track all spawned threads in a list on `ShuVoiceApp`.
- In `do_shutdown()`, set a shared stop event, then join each thread with a
  reasonable timeout (2-3s).
- For `TTSPlayer`, after join timeout expires, log a warning with thread state
  for diagnostics.

### 1.6 Fix potential deadlock in Piper subprocess IPC

**Files**: `shuvoice/tts_local.py:216-245`  
**Problem**: Writing to Piper's stdin while reading from stdout can deadlock if
Piper's stderr buffer fills (OS pipe buffers are typically 64KB).  
**Fix**: Drain stderr in a separate thread, or use `subprocess.communicate()`
with a timeout for the entire exchange. Since this is streaming, a dedicated
stderr-drain thread (started when the process launches) is the cleanest
approach.

---

## Track 2: Performance — Latency & CPU

### 2.1 Eliminate blocking sleep loop in ASR drain

**Files**: `shuvoice/app.py:677-685`  
**Problem**: `time.sleep(0.02)` in a tight loop (up to 5s timeout) called from
the main thread during TTS speak. This blocks the GTK main loop and causes
visible UI jank during selection-to-speak transitions.  
**Fix**: Replace the busy-wait with a `threading.Event.wait(timeout=5.0)` that
the ASR processing thread sets when it finishes draining. This yields the CPU
immediately and wakes up with no latency.

### 2.2 Replace tight queue retry loops with backoff

**Files**: `shuvoice/tts_player.py:216-245`  
**Problem**: `Queue.put()` failures trigger immediate retry in an infinite loop
with only a 0.1s timeout per attempt. Under sustained load this burns CPU.  
**Fix**: Use exponential backoff (0.1s, 0.2s, 0.4s, capped at 1s) with a max
retry count. After max retries, drop the chunk and log a warning rather than
looping forever.

### 2.3 Memoize `resolved_sherpa_decode_mode` property

**Files**: `shuvoice/config.py:631-715`  
**Problem**: This property is recomputed on every access and is called in hot
paths (`asr_sherpa.py:55,81,525`) during every audio chunk.  
**Fix**: Cache the result on first access using `functools.cached_property` or
a simple `_resolved_sherpa_decode_mode` backing field that's invalidated only
when the underlying config changes.

### 2.4 Optimize transcript overlap detection

**Files**: `shuvoice/transcript.py:70-82`  
**Problem**: `prefer_transcript()` uses nested loops for word overlap detection
— O(n*m) complexity. On long transcripts with many overlapping words, this
causes stuttering.  
**Fix**: Use a set-based or suffix-match approach. Since overlaps are at
boundaries, only compare the last N words of the existing transcript against
the first N words of the new one (where N is bounded by the maximum expected
overlap window).

### 2.5 Add configurable timeouts for subprocess calls in typer

**Files**: `shuvoice/typer.py:58-77,100-107`  
**Problem**: All subprocess calls (clipboard write, window detection) use
hardcoded 2-3 second timeouts. On loaded systems these can fire spuriously,
causing typing interruptions.  
**Fix**: Make the timeout configurable via `Config` with a sensible default
(3s). Use a single config key `typing_subprocess_timeout` to avoid
proliferating knobs.

---

## Track 3: Polish — Error Handling & User Feedback

### 3.1 Add circuit-breaker pattern for ASR failures

**Files**: `shuvoice/app.py:420-437`  
**Problem**: ASR exceptions are counted but there's no recovery path — after 10
consecutive failures the app becomes unusable and requires a full restart.  
**Fix**: Implement a circuit breaker:
- After `_ASR_MAX_FAILURES` consecutive errors, enter "open" state and stop
  sending audio to the backend.
- After a configurable cooldown (e.g. 30s), enter "half-open" and attempt one
  inference.
- On success, reset the failure counter and resume normal operation.
- Show a user-visible notification (via overlay or waybar) when the circuit
  opens and closes.

### 3.2 Surface voice-list loading failures to the user

**Files**: `shuvoice/app.py:796-815`  
**Problem**: `_load_tts_voices()` catches exceptions silently. The TTS overlay
voice dropdown appears empty with no explanation.  
**Fix**: On failure, set a flag that the TTS overlay reads to show a brief
inline error message (e.g. "Could not load voices — check network/API key").
Log the full exception for debugging.

### 3.3 Improve download cancellation feedback in wizard

**Files**: `shuvoice/wizard/__init__.py:1287-1307`  
**Problem**: When a model download is cancelled, the UI doesn't immediately
reflect the state change. Users may think the download is still running.  
**Fix**: On cancellation, immediately update the progress label to "Download
cancelled" and re-enable the download button. Ensure partial files are cleaned
up.

### 3.4 Replace silent `except/pass` blocks with diagnostic logging

**Files**: Multiple — `tts_player.py:274-279`, `config_io.py:163-164`,
`app.py:353-354`, `control.py:146-147,204-205,209-210`  
**Problem**: Bare `except: pass` blocks swallow errors silently, making
production debugging extremely difficult.  
**Fix**: Add `log.debug(...)` with `exc_info=True` to each block. In
shutdown/cleanup paths where exceptions are expected, use
`log.debug("Expected cleanup error", exc_info=True)` to distinguish them in
logs. Priority order:
1. `control.py` — control socket errors are user-facing
2. `app.py` — shutdown errors may mask root causes
3. `tts_player.py` — playback errors affect user experience
4. `config_io.py` — config save errors could lose user settings

### 3.5 Log first dropped audio chunk, not just every 50th

**Files**: `shuvoice/audio.py:133-138`  
**Problem**: `if self._dropped_chunks % 50 == 1` means the very first drop
(count=0, 0%50=0) is not logged. The first drop is the most diagnostic.  
**Fix**: Change to `if self._dropped_chunks == 0 or self._dropped_chunks % 50 == 0`.

---

## Track 4: Security & Input Validation

### 4.1 Sanitize config strings before subprocess use

**Files**: `shuvoice/config.py:526-532`, `shuvoice/tts_local.py:118`  
**Problem**: `tts_model_id` and `voice_id` from user config are interpolated
into subprocess commands without escaping. A malicious config value could inject
shell commands.  
**Fix**: Validate `tts_model_id` and `voice_id` against an allowlist regex
(e.g. `^[a-zA-Z0-9_\-\.]+$`) at config load time. Reject values that don't
match. Since all subprocess calls already use list-form (`subprocess.run([...])`)
rather than shell=True, the actual injection risk is low, but validation adds
defense in depth.

### 4.2 Bound clipboard/selection text size before TTS

**Files**: `shuvoice/selection.py:20-28`, `shuvoice/app.py:697-702`  
**Problem**: No size limit on clipboard text before sending to TTS. A user
accidentally selecting a massive document could cause the app to hang or OOM.  
**Fix**: Add a configurable `tts_max_selection_chars` limit (default 10,000).
Truncate and warn if exceeded. The check should happen right after
`capture_selection()` returns.

### 4.3 Validate model download URLs

**Files**: `shuvoice/piper_setup.py:54-95,225-240`  
**Problem**: Download URLs constructed from user-facing data without scheme
validation. While the URLs are currently hardcoded to GitHub releases, future
config-driven URLs could introduce SSRF.  
**Fix**: Validate that download URLs match `https://` scheme and are from
expected domains (github.com, huggingface.co). Reject other schemes.

---

## Track 5: Code Health — Refactoring & Maintainability

### 5.1 Split `wizard/__init__.py` (1,670 lines)

**Problem**: This is the largest file in the project and mixes UI layout, state
management, event handling, and business logic.  
**Fix**: Extract into focused modules:
- `wizard/pages.py` — individual wizard page classes
- `wizard/layout.py` — shared layout helpers and CSS
- `wizard/navigation.py` — page flow and back/next logic
- Keep `wizard/__init__.py` as the thin entry point that wires pages together.

Target: no single file over ~400 lines.

### 5.2 Split `app.py` (1,130 lines)

**Problem**: `ShuVoiceApp` is a god class handling GTK lifecycle, ASR
orchestration, TTS orchestration, recording state, and IPC dispatch.  
**Fix**: Extract cohesive chunks:
- `app_asr.py` — ASR thread management, chunk pipeline, failure handling
- `app_tts.py` — TTS backend loading, voice list, speak/stop
- `app_recording.py` — recording start/stop/toggle state machine
- Keep `app.py` as the GTK Application subclass that delegates to these.

### 5.3 Split `wizard_state.py` (1,001 lines)

**Problem**: Contains a mix of constants, dataclass definitions, and validation
logic all in one file.  
**Fix**: Extract into:
- `wizard/constants.py` — backend lists, default values, presets
- `wizard/models.py` — dataclass state types
- Keep validation co-located with the models.

### 5.4 Tighten ruff lint rules

**Problem**: Current ruff config only enables `E4,E7,E9,F,I` — basic syntax
and import sorting. Many code quality issues slip through.  
**Fix**: Incrementally enable:
- `B` (bugbear) — catches common Python gotchas
- `SIM` (simplify) — suggests cleaner patterns
- `UP` (pyupgrade) — modernize syntax for Python 3.10+
- `RUF` (ruff-specific) — catches ruff-specific issues
- `BLE` globally (currently only `noqa`'d in a few places)

Roll out one rule set at a time, fix violations, then enable the next.

### 5.5 Add type annotations to public APIs

**Problem**: Several core modules lack type annotations on public methods,
making it harder for IDEs and contributors to understand the codebase.  
**Priority files** (public surface area):
- `shuvoice/asr_base.py` — backend interface
- `shuvoice/tts_base.py` — backend interface
- `shuvoice/config.py` — all public properties
- `shuvoice/control.py` — command handler signatures

---

## Prioritized Execution Order

| Priority | Item | Track | Effort | Impact |
|----------|------|-------|--------|--------|
| P0 | 1.1 Lock module-level caches | Stability | S | High |
| P0 | 1.2 Protect NeMo encode cache | Stability | S | High |
| P0 | 1.4 Close subprocess handles on error | Stability | S | High |
| P0 | 3.4 Replace silent except/pass | Polish | M | High |
| P1 | 1.3 Fix double-check locking | Stability | S | Medium |
| P1 | 1.5 Clean daemon thread shutdown | Stability | M | High |
| P1 | 1.6 Fix Piper IPC deadlock | Stability | M | High |
| P1 | 2.1 Eliminate blocking sleep loop | Perf | M | High |
| P1 | 3.1 ASR circuit breaker | Polish | M | High |
| P1 | 3.5 Log first dropped chunk | Polish | XS | Medium |
| P2 | 2.2 Queue retry backoff | Perf | S | Medium |
| P2 | 2.3 Memoize sherpa decode mode | Perf | XS | Medium |
| P2 | 3.2 Surface voice-list errors | Polish | S | Medium |
| P2 | 3.3 Download cancellation feedback | Polish | S | Medium |
| P2 | 4.1 Sanitize config for subprocess | Security | S | Medium |
| P2 | 4.2 Bound selection text size | Security | S | Medium |
| P3 | 2.4 Optimize overlap detection | Perf | M | Low |
| P3 | 2.5 Configurable typer timeouts | Perf | S | Low |
| P3 | 4.3 Validate download URLs | Security | S | Low |
| P3 | 5.1 Split wizard/__init__.py | Health | L | Medium |
| P3 | 5.2 Split app.py | Health | L | Medium |
| P3 | 5.3 Split wizard_state.py | Health | M | Low |
| P3 | 5.4 Tighten ruff rules | Health | M | Medium |
| P3 | 5.5 Add type annotations | Health | M | Low |

**Effort key**: XS = <30min, S = 1-2h, M = 2-4h, L = 4-8h

---

## Definition of Done

Each item is considered done when:

1. The fix is implemented and passes `ruff check`
2. Existing tests still pass (no regressions)
3. New tests are added for any new logic (circuit breaker, validation, backoff)
4. The change is verified manually where applicable (UI feedback items)
5. Commit message references this plan (e.g. "stability: protect NeMo cache [sprint]")

---

## Out of Scope

The following are explicitly **not** in this sprint:

- New features or backends
- UI redesign or theme changes
- Packaging/distribution changes (AUR, systemd)
- Python version support changes
- Dependency upgrades (unless required by a fix)
