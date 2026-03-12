"""Wizard flow helpers."""

from __future__ import annotations

from ..wizard_state import format_summary


def summary_text(
    asr_backend: str,
    keybind_id: str,
    *,
    auto_add_keybind: bool,
    sherpa_model_name: str | None = None,
    sherpa_enable_parakeet_streaming: bool = False,
    sherpa_provider: str | None = None,
    typing_final_injection_mode: str = "auto",
    typing_text_case: str = "default",
    tts_backend: str = "elevenlabs",
    tts_default_voice_id: str | None = None,
    tts_local_model_path: str | None = None,
) -> str:
    return format_summary(
        asr_backend,
        keybind_id,
        auto_add_keybind=auto_add_keybind,
        sherpa_model_name=sherpa_model_name,
        sherpa_enable_parakeet_streaming=sherpa_enable_parakeet_streaming,
        sherpa_provider=sherpa_provider,
        typing_final_injection_mode=typing_final_injection_mode,
        typing_text_case=typing_text_case,
        tts_backend=tts_backend,
        tts_default_voice_id=tts_default_voice_id,
        tts_local_model_path=tts_local_model_path,
    )
