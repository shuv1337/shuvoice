# MeloTTS Research

## Overview
MeloTTS is a high-quality multi-lingual TTS by MIT/MyShell.ai. MIT license. Based on VITS2/Bert-VITS2.
- GitHub: https://github.com/myshell-ai/MeloTTS
- PyPI: `melotts` (v0.1.2)

## Python API

```python
from melo.api import TTS

# Init (per-language model)
model = TTS(language='EN_V2', device='auto')  # 'auto', 'cpu', 'cuda', 'mps'

# Get speakers
speaker_ids = model.hps.data.spk2id
# EN_V2: {'EN-US': 0, 'EN-BR': 1, 'EN-INDIA': 2, 'EN-AU': 4}
# EN_NEWEST: {'EN-Newest': 0}

# Sample rate
sr = model.hps.data.sampling_rate  # 44100

# Synthesize (returns np.float32 array when output_path=None)
audio = model.tts_to_file(
    text,
    speaker_id,        # integer from spk2id
    output_path=None,  # None = return numpy array
    speed=1.0,         # 0.5-2.0
    quiet=True,        # suppress debug prints
)

# Convert to PCM int16
pcm = (audio * 32768).clip(-32768, 32767).astype(np.int16)
```

## Available English Voices (confirmed working)

| Language Key | Speaker IDs | Notes |
|---|---|---|
| `EN_V2` | `EN-US` (0), `EN-BR` (1), `EN-INDIA` (2), `EN-AU` (4) | 4 accent variants |
| `EN_NEWEST` | `EN-Newest` (0) | Single improved voice |

EN-Default is NOT present in EN_V2 despite some docs claiming it.

## Key Facts
- Sample rate: 44100 Hz
- Output: float32 numpy mono array
- Speed: `speed` param (internally `length_scale = 1.0 / speed`)
- Batch-only (no streaming API)
- CPU real-time capable
- Model size: ~200MB per language model (HuggingFace cache)
- Venv size: ~9GB (PyTorch included)
- Works on Python 3.12 via uv

## Dependency Conflicts
- Pins `transformers==4.27.4` and `librosa==0.9.1` — conflicts with NeMo
- Solution: subprocess isolation with separate venv

## Installation (in isolated venv)
```bash
python3.12 -m venv /path/to/melotts-venv
/path/to/melotts-venv/bin/pip install melotts
/path/to/melotts-venv/bin/python -m unidic download
```

## HuggingFace Repos
- `myshell-ai/MeloTTS-English-v2` (EN_V2)
- `myshell-ai/MeloTTS-English-v3` (EN_NEWEST)
