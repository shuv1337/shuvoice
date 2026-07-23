# Configuration

ShuVoice reads `~/.config/shuvoice/config.toml`. The wizard writes this file;
manual examples live under `examples/`. Schema ownership is
`crates/shuvoice-core` (`Config` + section field map).

Local environment variables can be stored in `~/.config/shuvoice/local.dev`
(`KEY=value` or `export KEY=value`). Process environment values take
precedence. The CLI loads `local.dev` on startup.

Top-level marker: `config_version = 1` (legacy unversioned files migrate as v0).

Sections: `[audio]`, `[asr]`, `[overlay]`, `[control]`, `[tts]`, `[typing]`,
`[streaming]`, `[feedback]`, plus nested `[typing.text_replacements]`.

Inspect at runtime:

```bash
shuvoice config path
shuvoice config effective
shuvoice config validate
shuvoice config set typing_final_injection_mode auto
```

## ASR backends

```toml
[asr]
asr_backend = "sherpa"     # sherpa | nemo | moonshine | openai_realtime
```

| Backend | Implementation | Notes |
|---|---|---|
| `sherpa` | Native static Sherpa-ONNX in-process | Default path; **CPU only** |
| `openai_realtime` | Native WebSocket client | Cloud; API key via env |
| `nemo` | Optional Python worker | Isolated venv + workers tree |
| `moonshine` | Optional Python worker | Isolated venv + workers tree |

### Sherpa (native)

```toml
[asr]
asr_backend = "sherpa"
sherpa_provider = "cpu"              # cpu | cuda — cuda fails closed
sherpa_model_name = "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
sherpa_decode_mode = "auto"          # auto | streaming | offline_instant
sherpa_num_threads = 2
sherpa_chunk_ms = 100                # streaming only
sherpa_offline_max_utterance_sec = 60.0
# sherpa_model_dir = "/path/to/model"  # optional; else auto-download
# sherpa_enable_parakeet_streaming = false
```

Parakeet instant mode (wizard stable profile):

```toml
[asr]
asr_backend = "sherpa"
sherpa_model_name = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
sherpa_provider = "cpu"
instant_mode = true
sherpa_decode_mode = "offline_instant"
```

`sherpa_decode_mode = "auto"` resolves to `offline_instant` for Parakeet when
`instant_mode = true`, otherwise `streaming`.

**CUDA:** native static Sherpa does not support `sherpa_provider = "cuda"`.
Setup, preflight, and load fail closed with guidance to use `cpu`. There is no
Python wheel install or RUNPATH repair.

### NeMo (worker)

```toml
[asr]
asr_backend = "nemo"
model_name = "nvidia/nemotron-speech-streaming-en-0.6b"
device = "cuda"          # worker-side device string
right_context = 13       # 0 | 1 | 6 | 13
use_cuda_graph_decoder = false
```

Requires a discoverable workers tree and
`~/.local/share/shuvoice/workers-nemo-venv/` (or setup-created equivalent).

### Moonshine (worker)

```toml
[asr]
asr_backend = "moonshine"
moonshine_model_name = "moonshine/tiny"   # or moonshine/base
moonshine_provider = "cpu"
moonshine_max_window_sec = 5.0
moonshine_max_tokens = 64
moonshine_chunk_ms = 100
moonshine_onnx_threads = 0                # 0 = auto
# moonshine_model_dir = "/path/to/model"
# moonshine_model_precision = "float"
```

### OpenAI Realtime (native)

```toml
[asr]
asr_backend = "openai_realtime"
openai_realtime_model = "gpt-4o-transcribe"
openai_realtime_api_key_env = "OPENAI_API_KEY"
openai_realtime_language = "en"
openai_realtime_turn_detection = "manual"   # v1: keep manual
openai_realtime_vad_eagerness = "auto"
openai_realtime_request_timeout_sec = 10.0
openai_realtime_commit_timeout_sec = 5.0
# openai_realtime_latency_target_sec = 0.8
```

Keys are env-only (`~/.config/shuvoice/local.dev`):

```bash
export OPENAI_API_KEY=sk-...
```

### Instant mode

```toml
[asr]
instant_mode = true
```

Backend-specific tuning at runtime:

- NeMo: forces `right_context = 0`
- Sherpa streaming: caps `sherpa_chunk_ms` to `80`
- Sherpa offline_instant: one-shot utterance decode on key release
- Moonshine: forces `moonshine/tiny`, caps window to `3.0s`, caps tokens to `48`

### Gain / raw audio

| Backend | App auto-gain |
|---|---|
| Sherpa | Enabled (`wants_raw_audio = false`) |
| NeMo, Moonshine, OpenAI Realtime | Bypassed (raw audio) |

Keys: `auto_gain_target_peak`, `auto_gain_max`, `auto_gain_settle_chunks`,
`recording_preroll_ms` under `[audio]`.

## Typing

```toml
[typing]
output_mode = "final_only"                 # final_only | streaming_partial
typing_final_injection_mode = "auto"       # auto | clipboard | direct
typing_text_case = "default"               # default | lowercase
preserve_clipboard = false
typing_clipboard_settle_delay_ms = 40
typing_retry_attempts = 2
typing_retry_delay_ms = 40
typing_subprocess_timeout = 5.0
auto_capitalize = true
# use_clipboard_for_final = true           # legacy; prefer typing_final_injection_mode
```

| Mode | Behavior |
|---|---|
| `auto` | Clipboard paste by default; watcher-aware / XWayland policy may choose direct |
| `clipboard` | `wl-copy` + simulated paste |
| `direct` | `wtype` (and related) without touching clipboard |

Text replacements (case-insensitive, whole words/phrases, longest first).
Empty values delete the match. Built-ins cover common “ShuVoice” / “Hyprland”
ASR variants.

```toml
[typing.text_replacements]
"speech to text" = "speech-to-text"
"um" = ""
```

## Overlay

```toml
[overlay]
font_size = 22
# font_family = "JetBrains Mono"
bg_opacity = 0.75
border_radius = 16
bottom_margin = 60
overlay_debug_mode = false
overlay_debug_max_lines = 12
```

Hyprland layer rules (namespaces `stt-overlay` / `tts-overlay`):

```ini
layerrule = blur, stt-overlay
layerrule = ignorealpha 0.20, stt-overlay
layerrule = xray 1, stt-overlay
layerrule = blur, tts-overlay
layerrule = ignorealpha 0.20, tts-overlay
```

## Control

```toml
[control]
# control_socket = "/path/to/control.sock"
```

Default socket: `$XDG_RUNTIME_DIR/shuvoice/control.sock`.

Allowlisted commands: `start`, `stop`, `toggle`, `status`, `ping`, `metrics`,
`debug_status`, `tts_speak`, `tts_speak_clipboard`, `tts_pause`, `tts_resume`,
`tts_toggle_pause`, `tts_restart`, `tts_stop`, `tts_status`.

## TTS

```toml
[tts]
tts_enabled = true
tts_backend = "elevenlabs"    # elevenlabs | openai | local | melotts | kokoro
tts_default_voice_id = "zNsotODqUhvbJ5wMG7Ei"
tts_model_id = "eleven_flash_v2_5"
tts_api_key_env = "ELEVENLABS_API_KEY"
tts_output_format = "pcm_24000"
tts_max_chars = 5000
tts_request_timeout_sec = 30.0
tts_playback_speed = 1.0      # 0.5–2.0; wizard Kokoro default is 1.25
# tts_playback_device = ...
tts_overlay_auto_hide_sec = 2.0
```

API keys are env-only:

```bash
# ~/.config/shuvoice/local.dev
ELEVENLABS_API_KEY=sk-...
OPENAI_API_KEY=sk-...
```

### OpenAI TTS

When `tts_backend = "openai"` and stock ElevenLabs defaults are still present,
voice/model/env names normalize to OpenAI defaults (`onyx`, `gpt-4o-mini-tts`,
`OPENAI_API_KEY`). Prefer `tts_output_format = "pcm_24000"`.

### Local Piper

```toml
[tts]
tts_backend = "local"
tts_default_voice_id = "default"
# tts_local_voice = "en_US-amy-medium"
# tts_local_model_path = "~/.local/share/shuvoice/models/piper"
# tts_local_device = 3
```

Managed voices: `~/.local/share/shuvoice/models/piper/`. Requires `piper` or
`piper-tts` on `PATH`.

### MeloTTS (worker-proto only)

```toml
[tts]
tts_backend = "melotts"
tts_default_voice_id = "EN-US"
tts_model_id = "melotts"
tts_melotts_device = "auto"    # auto | cpu | cuda
# tts_melotts_venv_path = "~/.local/share/shuvoice/melotts-venv"
```

Requires CLI feature `tts-worker`, a workers tree with `melotts/`, and the
isolated venv. No legacy helper script path.

### Kokoro

```toml
[tts]
tts_backend = "kokoro"
tts_default_voice_id = "af_heart"
tts_model_id = "kokoro"
tts_kokoro_base_url = "http://localhost:8880/v1"
tts_playback_speed = 1.25
```

Local OpenAI-compatible HTTP API; no API key required.

## Audio

```toml
[audio]
sample_rate = 16000
chunk_ms = 100
fallback_sample_rate = 48000
# audio_device = "..."
input_gain = 1.0
audio_queue_max_size = 200
recording_preroll_ms = 200
silence_rms_threshold = 0.008
silence_rms_multiplier = 1.8
min_speech_ms = 80
auto_gain_target_peak = 0.15
auto_gain_max = 10.0
auto_gain_settle_chunks = 2
```

## Streaming and feedback

Defaults from `shuvoice-core` `Config::default()`:

```toml
[streaming]
streaming_stall_guard = true
streaming_stall_chunks = 4
streaming_stall_rms_ratio = 0.7
streaming_stall_flush_chunks = 1

[feedback]
audio_feedback = true
feedback_start_freq = 880
feedback_stop_freq = 660
feedback_duration_ms = 70
feedback_volume = 0.08
```

## Wizard defaults

Persisted by the setup wizard (stable instant profile):

| Key | Value |
|---|---|
| `asr_backend` | `sherpa` |
| `sherpa_model_name` | `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` |
| `sherpa_provider` | `cpu` |
| `instant_mode` | `true` |
| `sherpa_decode_mode` | `offline_instant` |
| `output_mode` | `final_only` |
| `typing_final_injection_mode` | `auto` |
| `typing_text_case` | `default` |
| `tts_backend` | `kokoro` |
| `tts_default_voice_id` | `af_heart` |
| `tts_kokoro_base_url` | `http://localhost:8880/v1` |
| `tts_playback_speed` | `1.25` |

## Example configs

| File | Description |
|---|---|
| [`../examples/config.toml`](../examples/config.toml) | Full reference |
| [`../examples/config-sherpa-cpu.toml`](../examples/config-sherpa-cpu.toml) | Sherpa on CPU (supported) |
| [`../examples/config-sherpa-cuda.toml`](../examples/config-sherpa-cuda.toml) | **Diagnostic only**: `sherpa_provider = "cuda"` — intentional fail-closed (not a working profile) |
| [`../examples/config-sherpa-parakeet-offline.toml`](../examples/config-sherpa-parakeet-offline.toml) | Parakeet instant mode |
| [`../examples/config-sherpa-parakeet-streaming.toml`](../examples/config-sherpa-parakeet-streaming.toml) | Parakeet streaming gate |
| [`../examples/config-nemo-cuda.toml`](../examples/config-nemo-cuda.toml) | NeMo worker (CUDA device string) |
| [`../examples/config-nemo-cpu.toml`](../examples/config-nemo-cpu.toml) | NeMo worker CPU |
| [`../examples/config-moonshine-cpu.toml`](../examples/config-moonshine-cpu.toml) | Moonshine worker CPU |

`examples/config-sherpa-cuda.toml` is an intentional fail-closed diagnostic: the
schema accepts `sherpa_provider = "cuda"`, but the native static Sherpa build is
CPU-only, so setup/preflight/load reject it (often exit 78) with guidance to use
`cpu`. There is no runtime CUDA→CPU fallback for native Sherpa. Worker NeMo may
still use a CUDA device string independently — that is a separate backend.
