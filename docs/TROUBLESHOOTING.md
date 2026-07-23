# Troubleshooting

Run these first:

```bash
shuvoice preflight
systemctl --user status shuvoice.service
journalctl --user -u shuvoice.service -n 80 --no-pager
```

ShuVoice is a Rust application. Optional NeMo / Moonshine / MeloTTS engines are
external workers — missing Python ML packages in a random venv are **not** the
default install path.

## Exit status 78 (dependency / composition)

Exit **78** is intentional. The user unit sets `RestartPreventExitStatus=78` so
systemd does not restart-loop on unrecoverable composition failures.

Common causes:

| Cause | What to do |
|---|---|
| Binary built without required features | Rebuild with `--features desktop` (or the specific feature: `asr-sherpa`, `asr-openai`, `ui`, `tts`, `tts-worker`, `audio`) |
| TTS enabled but no `tts` feature | Rebuild with `tts` / `desktop`, or set `tts_enabled = false` |
| MeloTTS without `tts-worker` | Rebuild with `tts-worker` (worker-proto only; no legacy helper) |
| `libgtk4-layer-shell.so` missing | `sudo pacman -S gtk4-layer-shell` (or distro equivalent) |
| `sherpa_provider = "cuda"` | Set `sherpa_provider = "cpu"` — native static Sherpa is CPU-only |
| Worker root / venv missing (NeMo, Moonshine, Melo) | Install workers tree + `shuvoice setup --install-missing`; see [Worker discovery](#worker-discovery) |

Check status after a forced failure:

```bash
systemctl --user show -p ExecMainStatus -p NRestarts shuvoice.service
# Expect ExecMainStatus=78 and restarts blocked
```

## Native Sherpa (static CPU)

| Error / symptom | Fix |
|---|---|
| Missing `encoder` / `decoder` / `joiner` | Point `sherpa_model_dir` at a complete model dir, or unset it to auto-download into `~/.local/share/shuvoice/models/sherpa/<sherpa_model_name>/` |
| `sherpa_provider=cuda` / “effective=unsupported” | Set `sherpa_provider = "cpu"`. There is no CUDA EP on the static binding and no wheel/RUNPATH repair. |
| Parakeet streaming rejected | Prefer `instant_mode = true` + `sherpa_decode_mode = "offline_instant"`. Streaming requires `sherpa_enable_parakeet_streaming = true` and compatible encoder metadata. |
| Long stuck PTT / huge utterance | Keep `sherpa_offline_max_utterance_sec` (default `60.0`); `0` disables the cap. |

Recommended CPU Parakeet profile:

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

## Worker discovery

Used for **NeMo**, **Moonshine**, and **MeloTTS** (never for native Sherpa/OpenAI).

Priority (first valid wins):

1. `SHUVOICE_WORKERS_DIR` — must be an **absolute** UTF-8 path to a workers tree
2. `/usr/lib/shuvoice/workers`
3. `/usr/libexec/shuvoice/workers`
4. Repo `workers/` in **debug** builds; in **release**, only if
   `SHUVOICE_ALLOW_DEV_WORKERS=1` (or `true` / `yes` / `on`)

Tree must contain `shuvoice_worker_proto/` plus the backend package
(`nemo_asr`, `moonshine_asr`, or `melotts`) with `__init__.py` and `__main__.py`.

Isolated interpreters (under `$XDG_DATA_HOME/shuvoice/`, default
`~/.local/share/shuvoice/`):

| Backend | Venv directory |
|---|---|
| NeMo | `workers-nemo-venv/` |
| Moonshine | `workers-moonshine-venv/` |
| MeloTTS | `melotts-venv/` (or `tts_melotts_venv_path`) |

```bash
export SHUVOICE_WORKERS_DIR=/path/to/shuvoice/workers
shuvoice setup --install-missing
shuvoice preflight
```

| Error | Fix |
|---|---|
| workers root not found | Install packaged workers or set `SHUVOICE_WORKERS_DIR` |
| `SHUVOICE_WORKERS_DIR` invalid | Use absolute UTF-8 path; relative paths fail closed |
| worker Python unusable | Run `shuvoice setup --install-missing` for the isolated venv |
| missing package / module | Reinstall the workers tree |

MeloTTS is **worker-proto only**. Legacy `melo_helper.py` subprocess paths are
not used by the Rust app.

## Audio and recognition

| Problem | Fix |
|---|---|
| Wrong microphone | `shuvoice audio list-devices`, then set `audio_device` by name/index |
| Mic too quiet | Raise `input_gain` (for example `1.3`–`1.8`) |
| Phantom text on silent presses | Raise `silence_rms_threshold` or `silence_rms_multiplier` |
| Long phrases cut out | Keep `streaming_stall_guard = true`; tune `streaming_stall_chunks` |
| Clipboard pollution | Prefer `typing_final_injection_mode = "auto"` |
| Need lowercase chat output | `typing_text_case = "lowercase"` |

App-side auto-gain applies when the ASR backend does **not** request raw audio
(Sherpa). NeMo, Moonshine, and OpenAI Realtime request raw audio and bypass it.

## Layer-shell, control, and injection

| Error | Fix |
|---|---|
| `libgtk4-layer-shell.so not found` | Install `gtk4-layer-shell` (Arch) / `libgtk-4-layer-shell0` (Debian) |
| Overlay missing on non-Hyprland | Compositor must support wlr-layer-shell; GTK4 stack required |
| `wtype not found` | `sudo pacman -S wtype` |
| Control socket not found | Start the service/app before `shuvoice control …` |
| Socket path | Default `$XDG_RUNTIME_DIR/shuvoice/control.sock`; override `[control].control_socket` |
| XWayland paste issues | Install `xdotool`; or force `typing_final_injection_mode = "direct"` |
| `tts_speak` says no selected text | Highlight text; verify `wl-paste --primary` |
| Zellij selection not detected | Copy first, then `shuvoice control tts_speak_clipboard --control-wait-sec 0` |
| ElevenLabs/OpenAI 401 | Set the env named by `tts_api_key_env` / `openai_realtime_api_key_env` in `~/.config/shuvoice/local.dev` |
| Kokoro unreachable | Ensure the local server matches `tts_kokoro_base_url` (default `http://localhost:8880/v1`); `shuvoice preflight` |

## Service and environment

```bash
systemctl --user restart shuvoice.service
systemctl --user status shuvoice.service --no-pager
journalctl --user -u shuvoice.service -n 80 --no-pager
```

Ensure the user service sees the graphical session environment:

```bash
systemctl --user import-environment WAYLAND_DISPLAY DISPLAY XDG_RUNTIME_DIR \
  HYPRLAND_INSTANCE_SIGNATURE DBUS_SESSION_BUS_ADDRESS XDG_CURRENT_DESKTOP XDG_SESSION_TYPE
```

Source builds: confirm `ExecStart` points at your `target/release/shuvoice`
override, not a stale path.

## ASR runtime UX

ASR failures surface on the STT overlay as transient messages (not only logs).
Persistent non-recoverable failures still feed the circuit breaker (pause and
later retry). Native static Sherpa does **not** offer a CUDA session fallback —
unsupported CUDA is rejected up front.

## What not to do

- Do **not** install old `python-sherpa-onnx` wheels or run `patchelf` RUNPATH
  repair for the Rust Sherpa path — it is static CPU.
- Do **not** `pip install` workers into the application environment expecting
  in-process imports — workers are stdio protocol processes only.
- Do **not** treat exit 78 as a transient crash; fix the dependency, then
  `systemctl --user restart shuvoice.service`.
