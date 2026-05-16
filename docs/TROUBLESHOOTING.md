# Troubleshooting

Run these first:

```bash
shuvoice preflight
systemctl --user status shuvoice.service
journalctl --user -u shuvoice.service -n 80 --no-pager
```

## Missing Python Modules

| Error | Fix |
|---|---|
| `No module named 'torch'` or `No module named 'nemo'` | `uv sync --extra asr-nemo` or install `python-pytorch-cuda` on Arch |
| `No module named 'sherpa_onnx'` | AUR: `yay -S python-sherpa-onnx-bin`; venv: `uv sync --extra asr-sherpa` |
| `No module named 'moonshine_onnx'` | `uv sync --extra asr-moonshine` |
| `No module named 'gi'` | `sudo pacman -S python-gobject gtk4 gtk4-layer-shell` |

## Sherpa and Parakeet

| Error | Fix |
|---|---|
| Missing `encoder`, `decoder`, or `joiner` files | Point `sherpa_model_dir` at a valid model directory or unset it to auto-download |
| `Parakeet requires offline instant mode` | Use `sherpa_decode_mode = "offline_instant"` with `instant_mode = true` |
| `window_size does not exist in the metadata` | Use offline instant mode for that model or switch to Zipformer |
| `CUDAExecutionProvider` not found | Install CUDA-enabled Sherpa or use `sherpa_provider = "cpu"` |
| GPU allocation errors | Switch Sherpa to CPU or close other GPU workloads |

For memory-constrained hosts, CPU is the preferred Sherpa Parakeet default:

```toml
[asr]
asr_backend = "sherpa"
sherpa_provider = "cpu"
sherpa_num_threads = 4
sherpa_model_name = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
instant_mode = true
sherpa_decode_mode = "offline_instant"
sherpa_offline_max_utterance_sec = 60.0
```

## Audio and Recognition

| Problem | Fix |
|---|---|
| Wrong microphone | Run `shuvoice audio list-devices`, then set `audio_device` by name |
| Mic too quiet | Increase `input_gain`, for example `1.3` to `1.8` |
| Phantom text on silent presses | Raise `silence_rms_threshold` or `silence_rms_multiplier` |
| Long phrases cut out | Keep `streaming_stall_guard = true` and tune `streaming_stall_chunks` |
| Clipboard pollution | Use `typing_final_injection_mode = "auto"` |
| Need lowercase chat output | Set `typing_text_case = "lowercase"` |

## System Runtime

| Error | Fix |
|---|---|
| `libgtk4-layer-shell.so not found` | `sudo pacman -S gtk4-layer-shell` |
| `wtype not found in PATH` | `sudo pacman -S wtype` |
| `Control socket not found` | Start ShuVoice before sending control commands |
| `espeak-ng not found` | `sudo pacman -S espeak-ng` |
| `tts_speak` says no selected text | Highlight text first and verify `wl-paste` works |
| ElevenLabs/OpenAI 401 | Export the API key named by `tts_api_key_env`; run `shuvoice preflight` |
| `Failed to build kaldialign` on Python 3.14 | Use `uv sync --extra asr-nemo --override packaging/constraints/py314-overrides.txt` |

## ASR Runtime Errors

ASR errors surface through the overlay as transient failure messages, not only
in logs. CUDA out-of-memory families trigger a one-shot Sherpa fallback to CPU
when possible. Persistent non-OOM failures keep the circuit breaker behavior:
after repeated errors, ASR pauses and retries later.
