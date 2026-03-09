"""Formatting helpers for Waybar JSON + tooltip content."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..config import DEFAULT_ELEVENLABS_TTS_VOICE_ID, Config
from ..tts_speed import format_tts_playback_speed

_OPENAI_VOICE_LABELS = {
    "alloy": "Alloy",
    "ash": "Ash",
    "coral": "Coral",
    "echo": "Echo",
    "fable": "Fable",
    "nova": "Nova",
    "onyx": "Onyx",
    "sage": "Sage",
    "shimmer": "Shimmer",
}


def _tts_backend_label(backend: str) -> str:
    return {
        "elevenlabs": "ElevenLabs",
        "openai": "OpenAI",
        "local": "Local Piper",
    }.get(str(backend).strip().lower(), str(backend).strip() or "unknown")


def _tts_voice_label(backend: str, voice_id: str) -> str:
    backend_id = str(backend).strip().lower()
    voice = str(voice_id).strip()
    if backend_id == "openai":
        return _OPENAI_VOICE_LABELS.get(voice.lower(), voice or "default")
    if backend_id == "elevenlabs" and voice == DEFAULT_ELEVENLABS_TTS_VOICE_ID:
        return "Default"
    if backend_id == "local" and not voice:
        return "configured local default"
    return voice or "default"


def config_info_lines(config: Config) -> list[str]:
    """Build tooltip info lines from the active configuration."""
    backend = config.asr_backend
    backend_labels = {
        "nemo": "NeMo (NVIDIA)",
        "sherpa": "Sherpa-ONNX",
        "moonshine": "Moonshine-ONNX",
    }

    lines = [f"Backend:  {backend_labels.get(backend, backend)}"]
    if config.instant_mode:
        lines.append("Profile:  Instant")

    # Model name (shortened for readability)
    if backend == "nemo":
        model = config.model_name
        if "/" in model:
            model = model.rsplit("/", 1)[-1]
        lines.append(f"Model:    {model}")
    elif backend == "sherpa":
        if config.sherpa_model_dir:
            model = Path(config.sherpa_model_dir).name
        else:
            model = f"{config.sherpa_model_name} (auto-download)"
        lines.append(f"Model:    {model}")
    elif backend == "moonshine":
        lines.append(f"Model:    {config.moonshine_model_name}")

    # Device (GPU vs CPU)
    if backend == "nemo":
        device = "GPU (CUDA)" if config.device == "cuda" else "CPU"
    elif backend == "sherpa":
        device = "GPU (CUDA)" if config.sherpa_provider == "cuda" else "CPU"
    elif backend == "moonshine":
        device = "GPU (CUDA)" if config.moonshine_provider == "cuda" else "CPU"
    else:
        device = "unknown"
    lines.append(f"Device:   {device}")
    if config.tts_enabled:
        lines.append(f"TTS:      {_tts_backend_label(config.tts_backend)}")
        lines.append(f"Voice:    {_tts_voice_label(config.tts_backend, config.tts_default_voice_id)}")
        lines.append(f"Speed:    {format_tts_playback_speed(config.tts_playback_speed)}")
    else:
        lines.append("TTS:      Disabled")

    return lines


def sanitize_class(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")
    return cleaned or "unknown"


def build_waybar_payload(
    state: str,
    *,
    config_lines: list[str] | None = None,
    service_state: str | None = None,
    control_error: str | None = None,
    action_error: str | None = None,
) -> dict[str, Any]:
    base_state = state
    reason = ""
    if ":" in state:
        base_state, reason = state.split(":", 1)

    icons = {
        "recording": "",
        "processing": "",
        "idle": "",
        "starting": "",
        "stopped": "",
        "error": "",
    }

    labels = {
        "recording": "Recording",
        "processing": "Processing",
        "idle": "Ready",
        "starting": "Starting",
        "stopped": "Stopped",
        "error": "Error",
    }

    if base_state not in labels:
        base_state = "error"
        reason = reason or "unknown_state"

    lines = [f"ShuVoice: {labels[base_state]}"]

    if reason:
        lines.append(f"Reason: {reason}")
    if service_state and service_state != "unknown":
        lines.append(f"Service: {service_state}")
    if control_error and base_state in {"starting", "error"}:
        lines.append(f"Control: {control_error}")
    if action_error:
        lines.append(f"Action: {action_error}")

    if config_lines:
        lines.append("")
        lines.extend(config_lines)

    lines.extend(
        [
            "",
            "Left click: toggle recording",
            "Middle click: toggle service",
            "Right click: open action menu",
        ]
    )

    class_name = sanitize_class(base_state)

    return {
        "text": icons[base_state],
        "alt": base_state,
        "class": class_name,
        "tooltip": "\n".join(lines),
    }
