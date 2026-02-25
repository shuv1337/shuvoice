"""XDG-compliant configuration management."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger(__name__)

CURRENT_CONFIG_VERSION = 1
"""Current config schema version.

Versioning policy
-----------------
- Bump this only when config shape/semantics require migration.
- Missing ``config_version`` is treated as legacy v0 and migrated forward.
- Migration steps must be deterministic and idempotent.
"""


CONFIG_SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "audio": (
        "sample_rate",
        "chunk_ms",
        "fallback_sample_rate",
        "audio_device",
        "input_gain",
        "audio_queue_max_size",
        "silence_rms_threshold",
        "silence_rms_multiplier",
        "min_speech_ms",
        "auto_gain_target_peak",
        "auto_gain_max",
        "auto_gain_settle_chunks",
    ),
    "asr": (
        "asr_backend",
        "instant_mode",
        "model_name",
        "right_context",
        "device",
        "use_cuda_graph_decoder",
        "sherpa_model_name",
        "sherpa_model_dir",
        "sherpa_provider",
        "sherpa_num_threads",
        "sherpa_chunk_ms",
        "moonshine_model_name",
        "moonshine_model_dir",
        "moonshine_model_precision",
        "moonshine_chunk_ms",
        "moonshine_max_window_sec",
        "moonshine_max_tokens",
        "moonshine_provider",
        "moonshine_onnx_threads",
    ),
    "overlay": (
        "font_size",
        "font_family",
        "bg_opacity",
        "border_radius",
        "bottom_margin",
    ),
    "control": ("control_socket",),
    "typing": (
        "output_mode",
        "use_clipboard_for_final",
        "preserve_clipboard",
        "typing_retry_attempts",
        "typing_retry_delay_ms",
        "auto_capitalize",
        "text_replacements",
    ),
    "streaming": (
        "streaming_stall_guard",
        "streaming_stall_chunks",
        "streaming_stall_rms_ratio",
        "streaming_stall_flush_chunks",
    ),
    "feedback": (
        "audio_feedback",
        "feedback_start_freq",
        "feedback_stop_freq",
        "feedback_duration_ms",
        "feedback_volume",
    ),
}


def _xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))


def _xdg_data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))


DEFAULT_TEXT_REPLACEMENTS: dict[str, str] = {
    # ShuVoice name variants (common ASR confusions)
    "shove voice": "ShuVoice",
    "shove-voice": "ShuVoice",
    "shovevoice": "ShuVoice",
    "shu voice": "ShuVoice",
    "shu-voice": "ShuVoice",
    "shuvoice": "ShuVoice",
    "shoo voice": "ShuVoice",
    "shoo-voice": "ShuVoice",
    "shoovoice": "ShuVoice",
    "shoe voice": "ShuVoice",
    "shoe-voice": "ShuVoice",
    "shoevoice": "ShuVoice",
    "show voice": "ShuVoice",
    "show-voice": "ShuVoice",
    "showvoice": "ShuVoice",
    # Hyprland name variants (common ASR confusions)
    "hyper land": "Hyprland",
    "hyper-land": "Hyprland",
    "hyperland": "Hyprland",
    "hypr land": "Hyprland",
    "hypr-land": "Hyprland",
    "hyprland": "Hyprland",
    "hype land": "Hyprland",
    "hype-land": "Hyprland",
    "high per land": "Hyprland",
    "high-per-land": "Hyprland",
    "highper land": "Hyprland",
    "highper-land": "Hyprland",
    "highperland": "Hyprland",
    "hyper lend": "Hyprland",
    "hyper-lend": "Hyprland",
    "hyperlend": "Hyprland",
}


def _default_text_replacements() -> dict[str, str]:
    return dict(DEFAULT_TEXT_REPLACEMENTS)


_FONT_FAMILY_RE = re.compile(r"^[A-Za-z0-9 ._-]+$")


@dataclass
class Config:
    # Schema metadata
    config_version: int = CURRENT_CONFIG_VERSION

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
    instant_mode: bool = False  # low-latency profile with backend-specific tuning
    model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    # 13 gives the highest streaming accuracy (at the cost of latency).
    # Lower values are snappier but significantly reduce recognition quality.
    right_context: int = 13  # 0=80ms, 1=160ms, 6=560ms, 13=1120ms (NeMo-only)
    device: str = "cuda"
    use_cuda_graph_decoder: bool = False

    # Sherpa (when asr_backend = "sherpa")
    sherpa_model_name: str = "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
    sherpa_model_dir: str | None = None
    sherpa_provider: str = "cpu"  # cpu | cuda
    sherpa_num_threads: int = 2
    sherpa_chunk_ms: int = 100

    # Moonshine ONNX (when asr_backend = "moonshine")
    moonshine_model_name: str = "moonshine/tiny"
    moonshine_model_dir: str | None = None
    moonshine_model_precision: str = "float"
    moonshine_chunk_ms: int = 100
    moonshine_max_window_sec: float = 5.0
    moonshine_max_tokens: int = 64
    moonshine_provider: str = "cpu"  # cpu | cuda
    moonshine_onnx_threads: int = 0  # 0 = auto (os.cpu_count())

    # Overlay
    font_size: int = 22
    font_family: str | None = None
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
    text_replacements: dict[str, str] = field(default_factory=_default_text_replacements)

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

    # Runtime cache for hot-path post-processing.
    _compiled_text_replacements: tuple[tuple[re.Pattern[str], str], ...] = field(
        init=False,
        repr=False,
        compare=False,
        default_factory=tuple,
    )

    def __post_init__(self):
        try:
            self.config_version = int(self.config_version)
        except (TypeError, ValueError) as exc:
            raise ValueError("config_version must be an integer") from exc
        if self.config_version < 1:
            raise ValueError("config_version must be >= 1")
        if self.config_version > CURRENT_CONFIG_VERSION:
            raise ValueError(
                "config_version is newer than this ShuVoice build supports "
                f"(got {self.config_version}, max {CURRENT_CONFIG_VERSION})"
            )

        self.asr_backend = str(self.asr_backend).strip().lower()
        if self.asr_backend not in {"nemo", "sherpa", "moonshine"}:
            raise ValueError("asr_backend must be one of: nemo, sherpa, moonshine")

        if not isinstance(self.instant_mode, bool):
            raise ValueError("instant_mode must be true or false")

        self.sherpa_model_name = str(self.sherpa_model_name).strip()
        if not self.sherpa_model_name:
            raise ValueError("sherpa_model_name must not be empty")

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

        self.moonshine_provider = str(self.moonshine_provider).strip().lower()
        if self.moonshine_provider not in {"cpu", "cuda"}:
            raise ValueError("moonshine_provider must be one of: cpu, cuda")

        if int(self.moonshine_onnx_threads) < 0:
            raise ValueError("moonshine_onnx_threads must be >= 0 (0 = auto)")

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

        # Validate core audio settings
        if int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be > 0")
        if int(self.chunk_ms) <= 0:
            raise ValueError("chunk_ms must be > 0")
        if int(self.fallback_sample_rate) <= 0:
            raise ValueError("fallback_sample_rate must be > 0")
        if float(self.input_gain) <= 0:
            raise ValueError("input_gain must be > 0")

        # Validate overlay styling (security: prevent CSS injection)
        if int(self.font_size) <= 0:
            raise ValueError("font_size must be > 0")
        self.font_size = int(self.font_size)

        if self.font_family is None:
            pass
        elif not isinstance(self.font_family, str):
            raise ValueError("font_family must be a string")
        else:
            normalized_font_family = self.font_family.strip()
            if not normalized_font_family:
                self.font_family = None
            elif not _FONT_FAMILY_RE.fullmatch(normalized_font_family):
                raise ValueError(
                    "font_family contains unsupported characters "
                    "(allowed: letters, numbers, spaces, '.', '_' and '-')"
                )
            else:
                self.font_family = normalized_font_family

        if not (0.0 <= float(self.bg_opacity) <= 1.0):
            raise ValueError("bg_opacity must be between 0.0 and 1.0")
        self.bg_opacity = float(self.bg_opacity)

        if int(self.border_radius) < 0:
            raise ValueError("border_radius must be >= 0")
        self.border_radius = int(self.border_radius)

        if int(self.bottom_margin) < 0:
            raise ValueError("bottom_margin must be >= 0")
        self.bottom_margin = int(self.bottom_margin)

        # Validate typing retry
        if int(self.typing_retry_attempts) < 0:
            raise ValueError("typing_retry_attempts must be >= 0")
        if int(self.typing_retry_delay_ms) < 0:
            raise ValueError("typing_retry_delay_ms must be >= 0")

        # Normalize text_replacements: strip whitespace and validate string types.
        # Built-in brand corrections are always included; user config can add
        # or override entries. Empty values are allowed (word/phrase deletion).
        if not isinstance(self.text_replacements, dict):
            raise ValueError("text_replacements must be a table/map of string keys to values")
        normalized_replacements: dict[str, str] = _default_text_replacements()
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

        self._apply_instant_mode_profile()

        from .postprocess import compile_text_replacements

        self._compiled_text_replacements = compile_text_replacements(self.text_replacements)

    def _apply_instant_mode_profile(self) -> None:
        """Apply low-latency backend tuning when ``instant_mode`` is enabled."""
        if not self.instant_mode:
            return

        if self.asr_backend == "nemo":
            if int(self.right_context) != 0:
                log.info(
                    "instant_mode enabled: forcing NeMo right_context=0 (was %s)",
                    self.right_context,
                )
            self.right_context = 0
            return

        if self.asr_backend == "sherpa":
            tuned_chunk_ms = min(int(self.sherpa_chunk_ms), 80)
            if tuned_chunk_ms != int(self.sherpa_chunk_ms):
                log.info(
                    "instant_mode enabled: lowering sherpa_chunk_ms to %dms (was %s)",
                    tuned_chunk_ms,
                    self.sherpa_chunk_ms,
                )
            self.sherpa_chunk_ms = tuned_chunk_ms
            return

        if self.asr_backend == "moonshine":
            if self.moonshine_model_name != "moonshine/tiny":
                log.info(
                    "instant_mode enabled: forcing moonshine_model_name='moonshine/tiny' (was %s)",
                    self.moonshine_model_name,
                )
                self.moonshine_model_name = "moonshine/tiny"

            tuned_window = min(float(self.moonshine_max_window_sec), 3.0)
            if tuned_window != float(self.moonshine_max_window_sec):
                log.info(
                    "instant_mode enabled: capping moonshine_max_window_sec to %.1fs (was %.1fs)",
                    tuned_window,
                    float(self.moonshine_max_window_sec),
                )
                self.moonshine_max_window_sec = tuned_window

            tuned_tokens = min(int(self.moonshine_max_tokens), 48)
            if tuned_tokens != int(self.moonshine_max_tokens):
                log.info(
                    "instant_mode enabled: capping moonshine_max_tokens to %d (was %s)",
                    tuned_tokens,
                    self.moonshine_max_tokens,
                )
                self.moonshine_max_tokens = tuned_tokens

    @property
    def compiled_text_replacements(self) -> tuple[tuple[re.Pattern[str], str], ...]:
        return self._compiled_text_replacements

    @property
    def chunk_samples(self) -> int:
        return self.sample_rate * self.chunk_ms // 1000

    @classmethod
    def config_dir(cls) -> Path:
        d = _xdg_config_home() / "shuvoice"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def config_path(cls) -> Path:
        return cls.config_dir() / "config.toml"

    @classmethod
    def data_dir(cls) -> Path:
        d = _xdg_data_home() / "shuvoice"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def _flatten_raw(cls, raw: Mapping[str, Any]) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for key, value in raw.items():
            if key == "config_version":
                flat[key] = value
                continue
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value
        return flat

    @classmethod
    def load(cls) -> "Config":
        from .config_io import load_raw, write_atomic
        from .config_migrations import migrate_to_latest

        config_file = cls.config_path()

        raw = load_raw(config_file)
        migrated, report = migrate_to_latest(raw)

        flat = cls._flatten_raw(migrated)

        valid_fields = {
            f.name for f in cls.__dataclass_fields__.values() if not f.name.startswith("_")
        }
        ignored = sorted(k for k in flat if k not in valid_fields)
        if ignored:
            log.debug("Ignoring unknown config keys: %s", ", ".join(ignored))

        filtered = {k: v for k, v in flat.items() if k in valid_fields}
        cfg = cls(**filtered)

        if config_file.exists() and report.to_version != report.from_version:
            try:
                write_atomic(config_file, migrated)
                log.info(
                    "Migrated config schema v%d -> v%d",
                    report.from_version,
                    report.to_version,
                )
            except Exception:  # noqa: BLE001
                log.warning("Failed to persist migrated config file", exc_info=True)

        return cfg

    @classmethod
    def config_field_names(cls) -> set[str]:
        return {f.name for f in cls.__dataclass_fields__.values() if not f.name.startswith("_")}

    def to_nested_dict(self, *, include_none: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {"config_version": int(self.config_version)}

        for section, fields in CONFIG_SECTION_FIELDS.items():
            section_data: dict[str, Any] = {}
            for key in fields:
                value = getattr(self, key)
                if value is None and not include_none:
                    continue
                if key == "text_replacements":
                    section_data[key] = dict(value)
                else:
                    section_data[key] = value
            if section_data:
                data[section] = section_data

        return data
