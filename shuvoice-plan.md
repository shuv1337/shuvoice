---
title: "ShuVoice: Revised Next Steps"
date: 2026-06-22
status: actionable
supersedes: shuvoice-recommendations-next-steps.md
---

# ShuVoice: Revised Next Steps

This is the working plan. It keeps the strategic direction of the longer
recommendations draft but corrects its sequencing and its model of the current
code. The draft front-loaded a typed event bus + versioned wire protocol as P0
and described several already-shipped subsystems as greenfield. This plan does
neither: it ships truthful, measurable wins first and gates every large bet on
measured pain.

**Positioning** (unchanged): the most reliable, truthful, low-latency dictation
shell for Wayland/Hyprland, with excellent local defaults and clean backend
boundaries. Models are replaceable engines behind that contract. Win on
correctness and honesty, not engine count.

---

## What already exists — do not rebuild

Before adding anything, treat these as **done** and extend rather than replace:

- **Partial stabilization** — `transcript.py::prefer_transcript()` already does
  stable-prefix growth, rejects pathological repetition, and stitches by word
  overlap. Wired into the live loop (`app.py:1220,1284,1295,1329`). We refine it;
  we do not write a second stabilizer.
- **Display-vs-inject partials** — `config.py` defaults `output_mode="final_only"`;
  partial injection only fires under `streaming_partial` on non-offline-instant
  backends (`app.py:1092-1103`). The safe default the draft "recommends" is
  already shipped.
- **Injection adapter chain** — `typer.py` already does wtype / xdotool / ydotool /
  clipboard with XWayland detection via `hyprctl activewindow`
  (`typer.py:92-156,300-419`). `xdotool` is the current primary XWayland path. Any
  adapter work extends this registry; it does not start from `wtype` alone.
- **In-process crash containment** — CUDA-OOM→CPU fallback (`asr_sherpa.py:93`,
  `app.py:459-487`) plus a real circuit breaker with cooldown/half-open recovery
  (`app.py:490-523`). Worker isolation is later *hardening*, not a rescue from an
  unhandled crash class.
- **Safe model download** — Sherpa already validates tar member paths before
  extract and stages atomically via a temp dir (`asr_sherpa.py:306-315,403-479`).
  A future manifest reuses this, not replaces it.
- **Real defaults** — the wizard's stable "instant" profile is **Parakeet
  offline-instant**, not streaming Zipformer (`wizard_state.py:829-969`,
  `test_wizard.py:69-87`). Any profile change is a *migration from this default*,
  not a fresh taxonomy.
- **Versioned config + migrations** — `config_migrations.py`, `config.py:886-947`.
  Every new config section/field must ship a migration + compatibility alias.

---

## Guiding principle

The next feature should be the contract that makes later features cheaper — but
the cheapest correct version of that, driven by measurement. We do not version a
public wire protocol onto semantics that are still moving, and we do not rewrite
the hot path to fix a bug class we have not yet reproduced.

---

## Phase 1 — truthful overlay, focus safety, real metrics (do first)

None of these require an event bus, a model registry, or worker processes.

1. **Instrument the missing latency timings** in `MetricsCollector` (`metrics.py`):
   time-to-first-partial, release-to-final, final-to-injection, cold/warm model
   load. Additive and small. This is the prerequisite for *every* performance
   target — set targets only after these produce numbers.

2. **Focus-change copy-only safety** (standalone correctness fix). Capture target
   window identity at record start via the existing `hyprctl activewindow` path
   (`typer.py:112`); compare before final injection. Default policy on change:
   copy result to clipboard + non-blocking "target changed" status. Never type a
   delayed result into a newly focused field. Highest user-protection value per
   line; no profile engine required.

3. **Expand overlay state vocabulary** — `overlay_state.py` currently has 3 states
   (listening / processing / error). Add: loading, ready, recording, speech,
   finalizing, degraded/fallback, circuit-open. Self-contained
   (`overlay_state.py` + `overlay.py` label/icon maps). Makes the overlay honest
   immediately. Distinguish "daemon running" from "dictation will start now."

4. **Minimally extend `ASRCapabilities`** — add only `partial_semantics` and
   `endpoint_owner` (plus a field only if a current backend/UI decision consumes
   it). Enough to drive capability-aware overlay rendering (streaming vs
   offline-instant vs hybrid) without a typed-event union. Defer timestamps,
   confidence, hotwords, translation, and worker-lifecycle fields until there is
   an implementation that uses them.

**Stabilizer stays display-only.** Feed `prefer_transcript` + a small history
buffer into overlay rendering and a churn metric. Do **not** route stabilized
text into partial injection until target capture, stale-event rejection, and
destructive-backspace regression coverage all exist.

**Internal ordering discipline, not a protocol.** Add `session_id`/`sequence`
*internally only* where delayed `GLib.idle_add` callbacks (`overlay.py:190-203`,
`app.py:983-1003`) against the ASR daemon thread can actually misfire. Define
main-thread dispatch and callback cancellation. No external socket schema yet.

---

## Open decision — resolve before any packaging work

**GPU in the default `uv sync` path (`pyproject.toml:94-107`).** This is a
deliberate, documented choice (`--no-group gpu` opt-out) and is exactly what the
most recent commit added: `5edcd80 "build: keep CUDA Sherpa backend on bare uv
sync (#65)"`. The recommendations draft proposed reverting it. **Decide #65 on
its own merits first.** If we do split: move GPU out of the default group and
keep a documented maintainer `uv sync --group gpu` path — but do this *before* any
worker-environment work, not buried in Phase 5.

---

## Deferred — gated on measured pain, not on the calendar

Keep these as design notes. Promote each only when there is concrete evidence it
is the bottleneck:

- **Internal event bus across every component** / **external `events --follow`
  wire protocol** — gate: a second consumer that actually needs the stream, or a
  *reproduced* stale-event/cross-session bug. Waybar works without it today.
- **Full model registry + manifest override system** — gate: many models and many
  users choosing among them. A 3-profile strategy needs at most a narrow manifest
  for blessed defaults.
- **Per-application profile engine** (regex → backend/overlay/injection/text) —
  gate: focus-safety (Phase 1.2) shipped and real demand for per-app behavior.
- **Generic external stdin/stdout processor** — gate: core STT flow measured and
  stable; needs a privacy/timeout/failure model first.
- **`shuvoice undo`** — larger than it looks; `typer.py` stores no commit-target
  metadata. Gate: structured `InjectionResult` exists.
- **Worker-process runtime isolation** / **model unload/prewarm policy** — gate:
  the in-process circuit breaker is proven insufficient.
- **Competitor benchmark matrix** — keep the instrumentation; the harness/CLI and
  trial matrix wait for Phase-1 data to justify them.

---

## After Phase 1 — pick exactly one larger bet

Using the latency numbers and setup data from Phase 1, choose **one** of:
outcome-oriented profiles · a model manifest for blessed defaults · worker-process
isolation. Do not start all three. The benchmark data, not the targets table,
decides which.

---

## Definition of success

- Every overlay state is backed by a real session state; batch engines never
  impersonate streaming engines.
- Changing focus mid-utterance cannot dump delayed text into the wrong app.
- Latency (first-partial, release-to-final, final-to-injection) is measured, so
  regressions are visible before release.
- Model/runtime/provider choice and fallback are always inspectable; a CUDA→CPU
  fallback says so.
- CPU-only install is straightforward; GPU is a deliberate, documented opt-in.
- Adding a backend means implementing the capability contract — not editing the
  overlay.

The TTS subsystem stays in scope-awareness: any app/overlay/metrics/lifecycle
change must either include TTS intentionally or explicitly declare it out of
contract — it shares too much of `app.py` to ignore silently.
