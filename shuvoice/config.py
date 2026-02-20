"""XDG-compliant configuration management."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def _xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))


def _xdg_data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))


@dataclass
class Config:
    # Audio
    sample_rate: int = 16000
    chunk_ms: int = 100
    fallback_sample_rate: int = 48000
    audio_device: str | int | None = None  # sounddevice device name/index
    input_gain: float = 1.0  # multiply PCM before ASR (eg. 1.5)
    silence_rms_threshold: float = 0.008  # absolute floor for speech RMS gating
    silence_rms_multiplier: float = 1.8  # dynamic threshold = noise_floor * multiplier
    min_speech_ms: int = 80  # minimum above-threshold speech before committing text

    # ASR
    model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    # 13 gives the highest streaming accuracy (at the cost of latency).
    # Lower values are snappier but significantly reduce recognition quality.
    right_context: int = 13  # 0=80ms, 1=160ms, 6=560ms, 13=1120ms
    device: str = "cuda"
    use_cuda_graph_decoder: bool = False

    # Overlay
    font_size: int = 22
    bg_opacity: float = 0.75
    border_radius: int = 16
    bottom_margin: int = 60

    # Hotkey / control
    hotkey_backend: str = "evdev"  # evdev | ipc
    hotkey: str = "KEY_RIGHTCTRL"
    hold_threshold_ms: int = 300
    hotkey_device: str | None = None  # /dev/input/eventX; None => auto-detect
    hotkey_listen_all_devices: bool = False  # default false to avoid duplicate key events
    control_socket: str | None = None  # default: $XDG_RUNTIME_DIR/shuvoice/control.sock

    # Text injection
    output_mode: str = "final_only"  # final_only | streaming_partial
    use_clipboard_for_final: bool = True
    preserve_clipboard: bool = False
    typing_retry_attempts: int = 2
    typing_retry_delay_ms: int = 40

    @property
    def chunk_samples(self) -> int:
        return self.sample_rate * self.chunk_ms // 1000

    @property
    def native_chunk_samples(self) -> int:
        """Return expected audio samples per streaming chunk based on right_context.

        0  = 80ms  = 1280
        1  = 160ms = 2560
        6  = 560ms = 8960
        13 = 1120ms = 17920
        """
        match self.right_context:
            case 0:
                return 1280
            case 1:
                return 2560
            case 6:
                return 8960
            case _:
                return 17920

    @classmethod
    def config_dir(cls) -> Path:
        d = _xdg_config_home() / "shuvoice"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def data_dir(cls) -> Path:
        d = _xdg_data_home() / "shuvoice"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def load(cls) -> "Config":
        config_file = cls.config_dir() / "config.toml"
        if not config_file.exists():
            return cls()

        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        # Flatten nested sections into a single dict
        flat: dict = {}
        for key, value in data.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in flat.items() if k in valid_fields}
        return cls(**filtered)
