"""XDG-compliant configuration management."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    audio_queue_max_size: int = 200
    silence_rms_threshold: float = 0.008  # absolute floor for speech RMS gating
    silence_rms_multiplier: float = 1.8  # dynamic threshold = noise_floor * multiplier
    min_speech_ms: int = 80  # minimum above-threshold speech before committing text
    auto_gain_target_peak: float = 0.15  # per-utterance RMS target for app-side gain
    auto_gain_max: float = 10.0  # max app-side gain multiplier
    auto_gain_settle_chunks: int = 2  # speech-level chunks before gain updates

    # ASR
    asr_backend: str = "sherpa"  # sherpa | nemo | moonshine
    model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    # 13 gives the highest streaming accuracy (at the cost of latency).
    # Lower values are snappier but significantly reduce recognition quality.
    right_context: int = 13  # 0=80ms, 1=160ms, 6=560ms, 13=1120ms (NeMo-only)
    device: str = "cuda"
    use_cuda_graph_decoder: bool = False

    # Sherpa (when asr_backend = "sherpa")
    sherpa_model_dir: str | None = None
    sherpa_provider: str = "cpu"  # cpu | cuda
    sherpa_num_threads: int = 2
    sherpa_chunk_ms: int = 100

    # Moonshine ONNX (when asr_backend = "moonshine")
    moonshine_model_name: str = "moonshine/base"
    moonshine_model_dir: str | None = None
    moonshine_model_precision: str = "float"
    moonshine_chunk_ms: int = 100
    moonshine_max_window_sec: float = 5.0
    moonshine_max_tokens: int = 128

    # Overlay
    font_size: int = 22
    bg_opacity: float = 0.75
    border_radius: int = 16
    bottom_margin: int = 60

    # Control socket (Hyprland bind/bindr integration)
    control_socket: str | None = None  # default: $XDG_RUNTIME_DIR/shuvoice/control.sock

    # Text injection
    output_mode: str = "final_only"  # final_only | streaming_partial
    use_clipboard_for_final: bool = True
    preserve_clipboard: bool = False
    typing_retry_attempts: int = 2
    typing_retry_delay_ms: int = 40
    auto_capitalize: bool = True
    text_replacements: dict[str, str] = field(default_factory=dict)

    # Streaming stability
    streaming_stall_guard: bool = True
    streaming_stall_chunks: int = 4
    streaming_stall_rms_ratio: float = 0.7
    streaming_stall_flush_chunks: int = 1

    # Audio feedback
    audio_feedback: bool = True
    feedback_start_freq: int = 880
    feedback_stop_freq: int = 660
    feedback_duration_ms: int = 70
    feedback_volume: float = 0.08

    def __post_init__(self):
        self.asr_backend = str(self.asr_backend).strip().lower()
        if self.asr_backend not in {"nemo", "sherpa", "moonshine"}:
            raise ValueError("asr_backend must be one of: nemo, sherpa, moonshine")

        self.sherpa_provider = str(self.sherpa_provider).strip().lower()
        if self.sherpa_provider not in {"cpu", "cuda"}:
            raise ValueError("sherpa_provider must be one of: cpu, cuda")

        if int(self.sherpa_chunk_ms) <= 0:
            raise ValueError("sherpa_chunk_ms must be > 0")

        if int(self.sherpa_num_threads) < 1:
            raise ValueError("sherpa_num_threads must be >= 1")

        if int(self.moonshine_chunk_ms) <= 0:
            raise ValueError("moonshine_chunk_ms must be > 0")

        if float(self.moonshine_max_window_sec) <= 0:
            raise ValueError("moonshine_max_window_sec must be > 0")

        if int(self.moonshine_max_tokens) < 1:
            raise ValueError("moonshine_max_tokens must be >= 1")

        self.moonshine_model_name = str(self.moonshine_model_name).strip()
        if not self.moonshine_model_name:
            raise ValueError("moonshine_model_name must not be empty")

        self.moonshine_model_precision = str(self.moonshine_model_precision).strip().lower()
        if not self.moonshine_model_precision:
            raise ValueError("moonshine_model_precision must not be empty")

        if int(self.audio_queue_max_size) < 1:
            raise ValueError("audio_queue_max_size must be >= 1")
        if float(self.auto_gain_target_peak) <= 0:
            raise ValueError("auto_gain_target_peak must be > 0")
        if float(self.auto_gain_max) < 1:
            raise ValueError("auto_gain_max must be >= 1")
        if int(self.auto_gain_settle_chunks) < 1:
            raise ValueError("auto_gain_settle_chunks must be >= 1")
        if int(self.streaming_stall_chunks) < 1:
            raise ValueError("streaming_stall_chunks must be >= 1")
        if int(self.streaming_stall_flush_chunks) < 1:
            raise ValueError("streaming_stall_flush_chunks must be >= 1")
        if float(self.streaming_stall_rms_ratio) <= 0:
            raise ValueError("streaming_stall_rms_ratio must be > 0")

        # Normalize text_replacements: strip whitespace and validate string types.
        # Empty values are allowed (they delete the matched word/phrase).
        if not isinstance(self.text_replacements, dict):
            raise ValueError("text_replacements must be a table/map of string keys to values")
        normalized_replacements: dict[str, str] = {}
        for key, value in self.text_replacements.items():
            if not isinstance(key, str):
                raise ValueError("text_replacements keys must be strings")
            if not isinstance(value, str):
                raise ValueError("text_replacements values must be strings")

            key_text = key.strip()
            if not key_text:
                raise ValueError("text_replacements keys must not be empty or whitespace-only")
            normalized_replacements[key_text] = value.strip()
        self.text_replacements = normalized_replacements

    @property
    def chunk_samples(self) -> int:
        return self.sample_rate * self.chunk_ms // 1000

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

        flat: dict = {}
        for key, value in data.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in flat.items() if k in valid_fields}
        return cls(**filtered)
