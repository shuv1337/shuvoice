# Configuration

ShuVoice reads `~/.config/shuvoice/config.toml`. The wizard writes this file,
and manual examples live under `examples/`.

Local environment variables can be stored in `~/.config/shuvoice/local.dev`.
That file supports `KEY=value` and `export KEY=value` lines. Process
environment values take precedence.

## ASR Backends

```toml
[asr]
asr_backend = "sherpa"     # sherpa | nemo | moonshine | openai_realtime
```

| Backend | Best for | GPU required | Notes |
|---|---|---:|---|
| Sherpa-ONNX | General use | No | Fast CPU default, optional CUDA |
| NeMo | Maximum accuracy | Recommended | NVIDIA/CUDA path |
| Moonshine | Low-resource use | No | Best for shorter utterances |
| OpenAI Realtime Whisper | Cloud transcription | No | Sends audio to OpenAI |

Sherpa profiles:

```toml
[asr]
asr_backend = "sherpa"
sherpa_provider = "cpu"              # cpu | cuda
sherpa_model_name = "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
sherpa_decode_mode = "auto"          # auto | streaming | offline_instant
```

Parakeet instant mode:

```toml
[asr]
asr_backend = "sherpa"
sherpa_model_name = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
instant_mode = true
sherpa_decode_mode = "offline_instant"
```

NeMo:

```toml
[asr]
asr_backend = "nemo"
model_name = "nvidia/nemotron-speech-streaming-en-0.6b"
device = "cuda"
right_context = 13
```

Moonshine:

```toml
[asr]
asr_backend = "moonshine"
moonshine_model_name = "moonshine/tiny"
moonshine_provider = "cpu"
moonshine_max_window_sec = 5.0
```

OpenAI Realtime Whisper:

```toml
[asr]
asr_backend = "openai_realtime"
openai_realtime_model = "gpt-4o-transcribe"
openai_realtime_api_key_env = "OPENAI_API_KEY"
openai_realtime_language = "en"
openai_realtime_turn_detection = "manual"
```

## Typing

```toml
[typing]
typing_final_injection_mode = "auto"   # auto | clipboard | direct
typing_text_case = "default"           # default | lowercase
```

| Mode | Behavior |
|---|---|
| auto | Chooses clipboard or direct typing based on the focused app and clipboard watchers |
| clipboard | Copies text and simulates paste |
| direct | Types via `wtype` and avoids the clipboard |

Text replacements:

```toml
[typing.text_replacements]
"speech to text" = "speech-to-text"
"um" = ""
```

Matches are case-insensitive and applied to whole words or phrases, longest
first. Built-in replacements cover common variants of "ShuVoice" and
"Hyprland".

## Instant Mode

```toml
[asr]
instant_mode = true
```

Effects:

- NeMo forces `right_context = 0`
- Sherpa streaming caps `sherpa_chunk_ms` at 80
- Sherpa offline uses one-shot release-to-final decode
- Moonshine uses `moonshine/tiny`, a 3-second window, and lower token caps

## Overlay

```toml
[overlay]
font_size = 24
font_family = "JetBrains Mono"
bg_opacity = 0.55
```

Hyprland layer rules:

```ini
layerrule = blur, stt-overlay
layerrule = ignorealpha 0.20, stt-overlay
layerrule = xray 1, stt-overlay
layerrule = blur, tts-overlay
layerrule = ignorealpha 0.20, tts-overlay
```

## TTS

```toml
[tts]
tts_enabled = true
tts_backend = "elevenlabs"             # elevenlabs | openai | local | melotts | kokoro
tts_default_voice_id = "zNsotODqUhvbJ5wMG7Ei"
tts_model_id = "eleven_flash_v2_5"
tts_api_key_env = "ELEVENLABS_API_KEY"
tts_playback_speed = 1.0
```

Set keys in `~/.config/shuvoice/local.dev`:

```bash
ELEVENLABS_API_KEY=sk-your-key-here
OPENAI_API_KEY=sk-your-key-here
```

Kokoro example:

```toml
[tts]
tts_enabled = true
tts_backend = "kokoro"
tts_default_voice_id = "af_heart"
tts_model_id = "kokoro"
tts_kokoro_base_url = "http://localhost:8880/v1"
tts_playback_speed = 1.0
```

MeloTTS example:

```toml
[tts]
tts_enabled = true
tts_backend = "melotts"
tts_default_voice_id = "EN-US"
tts_playback_speed = 1.0
```

## Example Configs

| File | Description |
|---|---|
| [`../examples/config.toml`](../examples/config.toml) | Full reference |
| [`../examples/config-sherpa-cpu.toml`](../examples/config-sherpa-cpu.toml) | Sherpa on CPU |
| [`../examples/config-sherpa-cuda.toml`](../examples/config-sherpa-cuda.toml) | Sherpa on GPU |
| [`../examples/config-sherpa-parakeet-offline.toml`](../examples/config-sherpa-parakeet-offline.toml) | Parakeet instant mode |
| [`../examples/config-nemo-cuda.toml`](../examples/config-nemo-cuda.toml) | NeMo on GPU |
| [`../examples/config-nemo-cpu.toml`](../examples/config-nemo-cpu.toml) | NeMo on CPU |
| [`../examples/config-moonshine-cpu.toml`](../examples/config-moonshine-cpu.toml) | Moonshine on CPU |
