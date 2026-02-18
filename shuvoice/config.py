"""XDG-compliant configuration management."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


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

    # ASR
    model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    right_context: int = 1  # 0=80ms, 1=160ms, 6=560ms, 13=1120ms
    device: str = "cuda"

    # Overlay
    font_size: int = 22
    bg_opacity: float = 0.75
    border_radius: int = 16
    bottom_margin: int = 60

    # Hotkey
    hotkey: str = "KEY_RIGHTCTRL"
    hold_threshold_ms: int = 300

    # Text injection
    use_clipboard_for_final: bool = True

    @property
    def chunk_samples(self) -> int:
        return self.sample_rate * self.chunk_ms // 1000

    @property
    def native_chunk_samples(self) -> int:
        """1120ms at 16kHz = 17920 samples."""
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
