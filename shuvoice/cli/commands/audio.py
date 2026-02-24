"""Audio utility commands."""

from __future__ import annotations

import sys


def list_audio_devices() -> int:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        print("Audio devices:")
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                print(
                    f"[{idx}] {dev['name']} "
                    f"(in={dev['max_input_channels']}, "
                    f"default_sr={dev['default_samplerate']})"
                )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Could not list audio devices: {exc}", file=sys.stderr)
        return 1
    return 0
