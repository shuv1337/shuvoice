# ADR 0002: Isolate Python-only model runtimes behind one worker protocol

- Status: Accepted
- Date: 2026-07-23

## Context

Most ShuVoice behavior has a direct Rust implementation:

- Sherpa-ONNX exposes an official Rust API;
- OpenAI Realtime and cloud TTS are network protocols;
- Piper is already an external executable;
- audio, GTK, control IPC, configuration, and injection all have mature Rust
  libraries or stable process boundaries.

NeMo's streaming API and MeloTTS remain Python/PyTorch runtimes. Embedding
CPython in the application would couple the Rust shell to a venv, the GIL, and
model-specific imports. Removing those backends would silently discard
shipping behavior.

## Decision

All non-native model runtimes use one versioned framed worker protocol.

The protocol provides:

- a negotiated protocol version;
- a manifest containing backend identity and capabilities;
- request IDs and bounded frame sizes;
- lifecycle operations such as load, reset, cancel, and close;
- ASR chunk, utterance, and finalization operations;
- TTS synthesis and voice-list operations;
- JSON control messages plus binary PCM/data frames;
- explicit error and cancellation responses.

The Rust application owns worker process lifecycle, health checks, timeouts,
and restart policy. Worker crashes enter the same typed backend failure and
circuit-breaker path as native failures.

Bundled reference workers remain optional model-runtime packages:

- NeMo ASR
- MeloTTS
- Moonshine only until the native model format is validated

User-supplied workers use the same protocol. No Python interpreter is linked
into or imported by the Rust application.

## Consequences

- The desktop shell, control plane, UI, configuration, injection, native
  backends, setup, and packaging are Rust.
- Python is an optional backend runtime, comparable to Piper or a local Kokoro
  server, rather than an application implementation language.
- New model engines can be isolated without adding one-off subprocess
  protocols.
- Protocol compatibility becomes a tested public contract.

## Rejected alternatives

- Drop NeMo/MeloTTS: unacceptable parity loss.
- Embed Python with PyO3: preserves the lifetime and packaging problems the
  rewrite is meant to remove.
- A custom protocol per backend: duplicates framing, cancellation, health, and
  error handling.
