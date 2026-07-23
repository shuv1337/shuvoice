# ADR 0001: Rewrite ShuVoice as a Rust modular monolith

- Status: Accepted
- Date: 2026-07-23

## Context

ShuVoice's Python implementation has mature product behavior but its main
process combines GTK lifecycle, audio capture, ASR and TTS ownership, text
injection, control IPC, metrics, recovery, setup, and configuration. The
existing package already contains useful seams—backend contracts, versioned
configuration, pure runtime helpers, and headless state models—but Python does
not enforce them.

The rewrite must preserve the observable contract encoded by the existing test
suite and by real Hyprland integrations:

- existing `config.toml` files and migrations;
- the Unix control socket and its exact command vocabulary;
- safe final text injection and Wayland/XWayland fallback policy;
- the three ASR finalization modes;
- STT/TTS mutual exclusion;
- systemd dependency exit status 78;
- Waybar JSON, overlay namespaces, and wizard defaults.

## Decision

ShuVoice is one deployable application built from a small Cargo workspace:

| Crate | Responsibility |
|---|---|
| `shuvoice-core` | Configuration, domain types, transcript policy, metrics, and pure state machines |
| `shuvoice-worker-proto` | Versioned framed protocol for model runtimes that cannot run natively |
| `shuvoice-control` | Secure Unix socket server/client and stable command protocol |
| `shuvoice-io` | Audio, selection, text injection, Hyprland, and process adapters |
| `shuvoice-asr` | ASR contract, native Sherpa/OpenAI backends, and worker-backed engines |
| `shuvoice-tts` | TTS contract, providers, player state machine, and worker-backed engines |
| `shuvoice-ui` | Headless view models and optional GTK4/layer-shell surfaces |
| `shuvoice-app` | Session owner, runtime orchestration, and STT/TTS lifecycle |
| `shuvoice-cli` | Application composition, CLI, setup/preflight, and Waybar binaries |

The runtime is an actor-shaped modular monolith:

1. one session owner serializes state transitions;
2. the audio callback only writes to a bounded queue;
3. one ASR worker owns the model and all model calls;
4. network, IPC, and timers use Tokio;
5. GTK objects remain on the GLib main thread;
6. text injection is serialized and happens only after final-text safety policy.

The workspace produces the existing public binaries:

- `shuvoice`
- `shuvoice-waybar`

## Consequences

- Headless crates compile and test without GTK, a display, a microphone, or a
  model.
- Backend failures become typed domain events instead of reaching directly
  into UI state.
- The control protocol and configuration schema remain compatibility
  boundaries, not implementation details.
- Native library dependencies remain explicit Cargo features.
- There is no crate-per-file fragmentation and no multi-service coordination
  tax.

## Rejected alternatives

- A single giant crate: too easy to re-create the Python dependency graph.
- A crate per module: excessive public API and compile-time churn.
- Multiple long-running services: creates socket/config races and complicates
  GTK ownership without improving the normal desktop workload.
- A big-bang cutover without parity tests: cannot distinguish a rewrite from a
  regression.
