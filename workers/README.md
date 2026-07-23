# ShuVoice optional model workers

Standalone **reference workers** that speak the versioned framed protocol
implemented by `crates/shuvoice-worker-proto`. They are **optional** runtimes
for heavy Python ML stacks (NeMo, MeloTTS, Moonshine). The Rust shell must
never import or build-depend on this tree.

## Wire format (v1)

Identical to Rust:

```text
u32 BE length | u8 kind | payload (length-1 bytes)
```

| kind | payload |
|------|---------|
| `1`  | UTF-8 JSON control message |
| `2`  | `request_id[16] \|\| f32 LE mono PCM` |
| `3`  | `request_id[16] \|\| i16 LE mono PCM` |
| `4`  | `request_id[16] \|\| opaque bytes` |

`length` is rejected unless `1 ..= 16 MiB` **before** payload allocation.

Handshake: client `hello` → worker `hello_ok` + manifest (or `hello_err`).

## Layout

```text
workers/
  shuvoice_worker_proto/   # shared framing + server/client helpers
  nemo_asr/                # NeMo streaming ASR worker
  melotts/                 # MeloTTS worker (44100 Hz i16 PCM)
  moonshine_asr/           # optional Moonshine ONNX ASR worker
  tests/                   # unittest + golden bytes (no ML deps)
  README.md
```

## Run (from this directory)

```bash
cd workers

# Fake engines — no NeMo/Melo/Moonshine packages required
python -m nemo_asr --fake
python -m melotts --fake
python -m moonshine_asr --fake

# Real engines (install deps in the active environment / Melo venv)
python -m nemo_asr
# Prefer MeloTTS venv interpreter:
~/.local/share/shuvoice/melotts-venv/bin/python -m melotts --device auto
python -m moonshine_asr
```

Stdio is the transport: the host spawns the worker and speaks framed messages
on the child’s stdin/stdout. Logs go to stderr only.

### Environment

| Variable | Effect |
|----------|--------|
| `SHUVOICE_WORKER_FAKE=1` | Force fake engines |
| `SHUVOICE_MELOTTS_VENV` | MeloTTS venv path for dependency checks |
| `SHUVOICE_MELOTTS_DEVICE` | `auto` / `cpu` / `cuda` |

## Install notes

These packages are **not** part of the Rust `Cargo.toml` workspace and are
**not** installed by the core ShuVoice app.

Suggested isolated installs:

```bash
# NeMo worker env
uv venv --python 3.12 ~/.local/share/shuvoice/workers-nemo-venv
~/.local/share/shuvoice/workers-nemo-venv/bin/python -m pip install torch 'nemo-toolkit[asr]'
# run with PYTHONPATH=workers

# MeloTTS already uses ~/.local/share/shuvoice/melotts-venv from setup
~/.local/share/shuvoice/melotts-venv/bin/python -m pip install melotts
# run: PYTHONPATH=/path/to/shuvoice/workers ... -m melotts

# Moonshine
uv pip install useful-moonshine-onnx
```

No `pip install -e` of this tree into the Rust build is required or desired.

## Host integration sketch

```text
shuvoice (Rust)
  └── spawn: python -m nemo_asr
        stdin/stdout: worker-proto v1 frames
```

Capabilities are advertised in the handshake manifest
(`wants_raw_audio`, `native_chunk_samples`, sample rates, etc.).

## Tests

```bash
cd workers
python -m unittest discover -s tests -v
```

Tests use `--fake` engines and golden frame bytes under `tests/golden/`.
They do **not** require NeMo, MeloTTS, or Moonshine packages.

## Safety

- Protocol `error` messages never include transcripts, API keys, or raw text.
- Oversize frames are rejected before allocation (`MAX_FRAME_LEN`); JSON control
  payloads are additionally capped at `MAX_JSON_PAYLOAD_LEN` (1 MiB).
- `close` / clean EOF end the process deterministically (exit 0).
- Cancel between `process_*` meta and the PCM frame is demuxed (same-request
  cancel acks and returns `cancelled` without desync).
- Capability flags omitted from manifests default to **false** (safe).
- Third-party engine calls (NeMo `conformer_stream_step`, Moonshine `generate`,
  MeloTTS `tts_to_file`) are **not preemptible** mid-call. Cooperative cancel is
  checked around those calls and between PCM chunks; hosts bound runaway work by
  killing the worker process (Rust `WorkerProcess` uses `kill_on_drop`).
