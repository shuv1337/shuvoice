"""MeloTTS helper script — runs inside the isolated MeloTTS venv.

This module is designed to be executed as ``__main__`` by the MeloTTS venv
Python interpreter.  It reads JSON synthesis requests from *stdin* (one per
line), synthesises audio via :pymod:`melo.api`, and writes framed raw PCM
int16 mono audio to *stdout*.

Framing protocol
----------------
For each request the helper writes:

1. **4-byte little-endian uint32** — byte length of the PCM payload.
2. **PCM payload** — raw int16 mono audio at 44 100 Hz.

On error the helper writes a JSON object to *stderr* and continues reading
the next request (it never crashes on a single bad input).

Voice → model mapping
---------------------
- ``EN-US``, ``EN-BR``, ``EN-INDIA``, ``EN-AU`` → ``EN_V2`` model
- ``EN-Newest`` → ``EN_NEWEST`` model
"""

from __future__ import annotations

import json
import struct
import sys

# ---------------------------------------------------------------------------
# Voice / model mapping (importable for tests)
# ---------------------------------------------------------------------------

_VOICE_TO_MODEL: dict[str, str] = {
    "EN-US": "EN_V2",
    "EN-BR": "EN_V2",
    "EN-INDIA": "EN_V2",
    "EN-AU": "EN_V2",
    "EN-Newest": "EN_NEWEST",
}

_DEFAULT_MODEL = "EN_V2"


def _model_for_voice(voice_id: str) -> str:
    """Return the MeloTTS language model key for *voice_id*."""
    return _VOICE_TO_MODEL.get(voice_id, _DEFAULT_MODEL)


def _build_request_json(text: str, voice_id: str, speed: float) -> str:
    """Serialise a synthesis request to a JSON string (no trailing newline)."""
    return json.dumps({"text": text, "voice_id": voice_id, "speed": speed})


# ---------------------------------------------------------------------------
# Main loop (only executed when running as __main__ in the MeloTTS venv)
# ---------------------------------------------------------------------------


def _main() -> None:  # pragma: no cover — requires MeloTTS venv
    import numpy as np  # type: ignore[import-untyped]
    from melo.api import TTS  # type: ignore[import-untyped]

    # Lazy model cache: load each language model at most once.
    _models: dict[str, TTS] = {}

    device = "auto"
    if len(sys.argv) > 1:
        device = sys.argv[1]

    stdout_bin = sys.stdout.buffer
    stderr_bin = sys.stderr

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            text = str(req.get("text", ""))
            voice_id = str(req.get("voice_id", "EN-US"))
            speed = float(req.get("speed", 1.0))

            if not text:
                raise ValueError("empty text")

            model_key = _model_for_voice(voice_id)

            if model_key not in _models:
                _models[model_key] = TTS(language=model_key, device=device)

            model = _models[model_key]
            speaker_ids = model.hps.data.spk2id
            speaker_id = speaker_ids.get(voice_id, 0)

            # Synthesise — returns float32 numpy array when output_path=None
            audio = model.tts_to_file(
                text,
                speaker_id,
                output_path=None,
                speed=speed,
                quiet=True,
            )

            # Convert float32 → int16 PCM
            pcm = (np.asarray(audio) * 32768).clip(-32768, 32767).astype(np.int16)
            pcm_bytes = pcm.tobytes()

            # Write framing: 4-byte LE uint32 length + PCM payload
            stdout_bin.write(struct.pack("<I", len(pcm_bytes)))
            stdout_bin.write(pcm_bytes)
            stdout_bin.flush()

        except Exception as exc:  # noqa: BLE001
            error_msg = json.dumps({"error": str(exc)})
            stderr_bin.write(error_msg + "\n")
            stderr_bin.flush()


if __name__ == "__main__":
    _main()
