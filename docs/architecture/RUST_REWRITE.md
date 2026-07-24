# Rust architecture and cutover record

This document records the architecture and compatibility contract of the Rust
ShuVoice application. During the rewrite, the former Python application served
as the behavioral oracle. Its final pre-removal snapshot was **767 passed,
2 skipped**; the production application, CLI, UI, control plane, setup flow,
audio path, and native backends now live in this Cargo workspace.

## Runtime shape

```text
Hyprland binding / CLI / Waybar
              │
              ▼
      secure control socket
              │
              ▼
        session owner actor
       ┌──────┼────────┐
       ▼      ▼        ▼
  audio q   ASR      TTS player
       │    worker       │
       └──────┬──────────┘
              ▼
        typed session events
       ┌──────┼──────────┐
       ▼      ▼          ▼
   GTK view  metrics   injector
```

Ownership rules:

- The session owner is the only writer of recording, processing, circuit, and
  mutual-exclusion state.
- `shuvoice-app` owns that session state; `shuvoice-cli` only composes and
  drives it.
- The ASR worker is the only owner of the active recognizer.
- The audio callback never blocks on application or model locks.
- GTK widgets are touched only from the GLib main context.
- Injection commands are serialized and never log transcript contents on
  failures.
- Configuration is immutable after validation except for explicit runtime
  selections such as TTS voice/speed.
- Native Sherpa is a static CPU-only build: `sherpa_provider = "cuda"` fails
  closed at setup/preflight/load (no session CUDA fallback, no wheel/RUNPATH
  repair). Worker NeMo may still target CUDA via its own device string; that is
  separate from native Sherpa.

## Compatibility boundaries

The rewrite must preserve:

1. `config_version = 1`, v0 migration, section/key names, XDG paths, and atomic
   backup/write behavior.
2. All control commands and the `OK ...` / `ERROR ...` line protocol.
3. Status strings consumed by Waybar and scripts.
4. Exit status 78 and `RestartPreventExitStatus=78`.
5. The default Parakeet CPU offline-instant profile and Kokoro 1.25× wizard
   profile.
6. Preroll, silence gating, app-gain/raw-audio policy, and utterance length
   caps.
7. Local-streaming, offline-utterance, and remote-manual-commit finalization.
8. Transcript stabilization, text replacements, case policy, and newline-safe
   final injection.
9. XWayland, clipboard-watcher, retry, and clipboard-preservation injection
   behavior.
10. STT/TTS mutual exclusion, TTS speed restart, and player cancellation.
11. Overlay namespaces, keyboard modes, click-through behavior, and Waybar JSON.

## Cutover result

The Rust binary is the packaged default. The repository gates cover:

- the complete workspace across default, no-default, and all-feature builds;
- every example configuration and the public CLI/control protocol;
- the cached Parakeet reference-audio smoke;
- injection ordering, exactly-once commit, and sensitive-data handling;
- exit status 78 for dependency/configuration failures;
- locked Rust packaging with no Python application dependency; and
- isolated, versioned worker-protocol subprocesses for optional NeMo,
  Moonshine, and MeloTTS engines.

Python is not part of the application shell. The small Python packages retained
under `workers/` are independently launched model adapters: they communicate
only through the bounded worker protocol, run from isolated virtual
environments, and receive a scrubbed child environment. A real Hyprland
PTT/overlay/Waybar/TTS exercise remains the post-install operational smoke; it
is deliberately not performed by repository tests or by a source-tree cutover.
