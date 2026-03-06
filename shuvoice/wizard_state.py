"""Pure wizard state helpers (headless-test friendly).

Contains first-run detection, config writing, and marker logic that
do not depend on GTK / gi.
"""

from __future__ import annotations

import logging
import re
import shlex
import shutil
import sys
from pathlib import Path

from .asr import get_backend_class
from .config import (
    CURRENT_CONFIG_VERSION,
    DEFAULT_ELEVENLABS_TTS_VOICE_ID,
    DEFAULT_OPENAI_TTS_VOICE_ID,
    Config,
)
from .config_io import load_raw, write_atomic
from .config_migrations import migrate_to_latest

log = logging.getLogger(__name__)

_MARKER_FILE = ".wizard-done"

# ASR backend descriptions shown in the wizard.
ASR_BACKENDS = [
    (
        "sherpa",
        "Sherpa-ONNX",
        "Fast ONNX ASR with profiles for Streaming (Zipformer) or Instant (Parakeet).",
    ),
    (
        "nemo",
        "NeMo (NVIDIA)",
        "Highest accuracy streaming ASR.  Requires an NVIDIA GPU with CUDA.",
    ),
    (
        "moonshine",
        "Moonshine-ONNX",
        "Lightweight ONNX ASR with low resource usage.  CPU-friendly.",
    ),
]

DEFAULT_SHERPA_MODEL_NAME = "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
PARAKEET_TDT_V3_INT8_MODEL_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
DEFAULT_KEYBIND_ID = "right_ctrl"

# Final text-injection presets for wizard configuration.
# (id, display_label, description)
FINAL_INJECTION_MODES = [
    (
        "auto",
        "Auto (recommended)",
        "Uses clipboard paste by default, and falls back to direct typing when clipboard watchers are detected.",
    ),
    (
        "clipboard",
        "Clipboard paste (Ctrl+V)",
        "Copies final text to the clipboard and pastes with Ctrl+V. Best for apps that reject synthetic typing.",
    ),
    (
        "direct",
        "Direct typing (keystroke simulation)",
        "Types final text directly with wtype and avoids clipboard changes.",
    ),
]
DEFAULT_FINAL_INJECTION_MODE = "auto"
_FINAL_INJECTION_MODE_SET = {mode for mode, _label, _desc in FINAL_INJECTION_MODES}


def _is_parakeet_sherpa_model_name(model_name: str) -> bool:
    return "parakeet" in str(model_name).strip().lower()


# Keybind presets for push-to-talk setup.
# (id, display_label, hyprland_bind_key_spec, description)
# hyprland_bind_key_spec is the "MODS, KEY" portion for bind/bindr lines.
KEYBIND_PRESETS = [
    (
        "right_ctrl",
        "Right Control",
        ", Control_R",
        "Recommended default for hold-to-talk on most keyboards.",
    ),
    ("insert", "Insert", ", Insert", "Usually unused and easy to dedicate."),
    ("f9", "F9", ", F9", "Simple single-key push-to-talk."),
    (
        "super_v",
        "Super + V",
        "SUPER, V",
        "Modifier combo — mnemonic for Voice.",
    ),
    ("custom", "Custom", None, "Set your own key in Hyprland config later."),
]

TTS_KEYBIND_PRESETS = [
    (
        "super_ctrl_s",
        "Super + Ctrl + S",
        "SUPER CTRL, S",
        "Speak selected text via ShuVoice TTS (primary selection + clipboard fallback).",
    ),
]
DEFAULT_TTS_KEYBIND_ID = "super_ctrl_s"
DEFAULT_TTS_BACKEND = "elevenlabs"
_OPENAI_TTS_VOICE_LABELS: dict[str, str] = {
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
TTS_BACKENDS = [
    (
        "elevenlabs",
        "ElevenLabs",
        "Cloud TTS with custom voice IDs. Keep the default voice or paste your own voice ID.",
    ),
    (
        "openai",
        "OpenAI",
        "Cloud TTS with built-in voice names like onyx, nova, and shimmer.",
    ),
]


def default_tts_voice_for_backend(backend: str) -> str:
    backend_id = str(backend).strip().lower()
    if backend_id == "openai":
        return DEFAULT_OPENAI_TTS_VOICE_ID
    return DEFAULT_ELEVENLABS_TTS_VOICE_ID


def tts_backend_label(backend: str) -> str:
    backend_id = str(backend).strip().lower()
    return next((label for value, label, _desc in TTS_BACKENDS if value == backend_id), backend_id)


def tts_voice_label(backend: str, voice_id: str) -> str:
    backend_id = str(backend).strip().lower()
    value = str(voice_id).strip()
    if not value:
        value = default_tts_voice_for_backend(backend_id)

    if backend_id == "openai":
        return _OPENAI_TTS_VOICE_LABELS.get(value.lower(), value)
    if backend_id == "elevenlabs" and value == DEFAULT_ELEVENLABS_TTS_VOICE_ID:
        return f"Default ({value})"
    return value


def format_hyprland_bind(hypr_key_spec: str, *, shuvoice_command: str = "shuvoice") -> str:
    """Format Hyprland bind/bindr lines for a push-to-talk keybind."""
    return (
        f"bind = {hypr_key_spec}, exec, {_control_exec('start', shuvoice_command=shuvoice_command)}\n"
        f"bindr = {hypr_key_spec}, exec, {_control_exec('stop', shuvoice_command=shuvoice_command)}"
    )


def _tts_bind_lines(*, shuvoice_command: str) -> list[str]:
    _, _label, hypr_key_spec, _description = TTS_KEYBIND_PRESETS[0]
    return [
        "bind = "
        + f"{hypr_key_spec}, exec, "
        + _control_exec("tts_speak", shuvoice_command=shuvoice_command)
    ]


def _bind_lines_for_preset(
    keybind_id: str,
    hypr_key_spec: str,
    *,
    shuvoice_command: str,
) -> list[str]:
    """Return bind lines for the selected preset.

    Right Control gets an extra release mapping with ``CTRL`` modifier to make
    key-up handling robust on setups where release events include modmask.
    """
    base = format_hyprland_bind(hypr_key_spec, shuvoice_command=shuvoice_command).splitlines()
    if keybind_id == "right_ctrl":
        base.append(
            "bindr = CTRL, Control_R, exec, "
            + _control_exec("stop", shuvoice_command=shuvoice_command)
        )
    base.extend(_tts_bind_lines(shuvoice_command=shuvoice_command))
    return base


def format_hyprland_bind_for_keybind(
    keybind_id: str,
    hypr_key_spec: str,
    *,
    shuvoice_command: str = "shuvoice",
) -> str:
    """Format Hyprland bind/bindr lines for a specific keybind preset."""
    return "\n".join(
        _bind_lines_for_preset(
            keybind_id,
            hypr_key_spec,
            shuvoice_command=shuvoice_command,
        )
    )


def hyprland_config_path() -> Path:
    """Return the default Hyprland config path (~/.config/hypr/hyprland.conf)."""
    return Config.config_dir().parent / "hypr" / "hyprland.conf"


def _bindings_config_path() -> Path:
    return Config.config_dir().parent / "hypr" / "bindings.conf"


def _hypr_config_candidates() -> list[Path]:
    """Return user-writable Hypr config files in priority order.

    Prefer ``bindings.conf`` when present (common in Omarchy-style setups),
    then fall back to ``hyprland.conf``.
    """
    bindings = _bindings_config_path()
    hyprland = hyprland_config_path()

    candidates: list[Path] = []
    if bindings.exists():
        candidates.append(bindings)
    if hyprland.exists() and hyprland != bindings:
        candidates.append(hyprland)

    if candidates:
        return candidates

    # If neither exists, default to the canonical main config path.
    return [hyprland]


def _hypr_key_spec_for_preset(keybind_id: str) -> str | None:
    return next((hk for kid, _label, hk, _desc in KEYBIND_PRESETS if kid == keybind_id), None)


def _resolve_shuvoice_command() -> str:
    """Resolve an executable command for Hyprland exec lines.

    Prefers a sibling ``shuvoice`` next to the running Python interpreter
    (venv-safe), then falls back to ``shutil.which('shuvoice')``.
    """
    python_bin = Path(sys.executable)
    sibling = python_bin.with_name("shuvoice")
    if sibling.exists() and sibling.is_file():
        return str(sibling)

    discovered = shutil.which("shuvoice")
    if discovered:
        return discovered

    return "shuvoice"


def _control_exec(command: str, *, shuvoice_command: str | None = None) -> str:
    binary = shuvoice_command or _resolve_shuvoice_command()
    quoted = shlex.quote(binary)
    return f"{quoted} control {command} --control-wait-sec 0"


def _normalize_bind_spec(mods: str, key: str) -> str | None:
    mods_norm = " ".join(mods.strip().upper().split())
    key_norm = " ".join(key.strip().upper().split())
    if not key_norm:
        return None
    return f"{mods_norm},{key_norm}" if mods_norm else f",{key_norm}"


def _normalize_hypr_key_spec(hypr_key_spec: str) -> str | None:
    if "," not in hypr_key_spec:
        return None
    mods, key = hypr_key_spec.split(",", 1)
    return _normalize_bind_spec(mods, key)


def _strip_inline_comment(line: str) -> str:
    # Good enough for Hyprland bind syntax in practice.
    return line.split("#", 1)[0].strip()


def _parse_hypr_bind_line(line: str) -> tuple[str, str] | None:
    """Parse a Hyprland bind line into (normalized_spec, command_text)."""
    text = _strip_inline_comment(line)
    if not text:
        return None

    match = re.match(r"^\s*bind[a-z]*\s*=\s*(.+)$", text, flags=re.IGNORECASE)
    if not match:
        return None

    rhs = match.group(1)
    parts = [part.strip() for part in rhs.split(",", 3)]
    if len(parts) < 2:
        return None

    spec = _normalize_bind_spec(parts[0], parts[1])
    if spec is None:
        return None

    command = ",".join(parts[2:]).strip()
    return spec, command


def _is_shuvoice_control_command(command_lc: str) -> bool:
    if "shuvoice" not in command_lc:
        return False
    # Support both legacy and canonical control invocation styles:
    #   shuvoice --control start
    #   shuvoice control start [--control-wait-sec ...]
    return "--control" in command_lc or " control " in f" {command_lc} "


def _is_shuvoice_start_command(command_lc: str) -> bool:
    if not _is_shuvoice_control_command(command_lc):
        return False
    return "--control start" in command_lc or " control start" in command_lc


def _is_shuvoice_stop_command(command_lc: str) -> bool:
    if not _is_shuvoice_control_command(command_lc):
        return False
    return "--control stop" in command_lc or " control stop" in command_lc


def _is_shuvoice_tts_speak_command(command_lc: str) -> bool:
    if not _is_shuvoice_control_command(command_lc):
        return False
    return "--control tts_speak" in command_lc or " control tts_speak" in command_lc


def auto_add_hyprland_keybind(keybind_id: str) -> tuple[str, str]:
    """Try to add/update ShuVoice bind/bindr lines in Hyprland config.

    Returns ``(status, message)`` where ``status`` is one of:
    - ``added``: ShuVoice keybind lines were added/updated.
    - ``already_configured``: desired start/stop lines already exist.
    - ``conflict``: selected key is already used by another command.
    - ``missing_config``: no Hyprland config file was found.
    - ``skipped_custom``: keybind preset has no concrete key spec.
    - ``error``: unexpected error while reading/writing config.
    """
    hypr_key_spec = _hypr_key_spec_for_preset(keybind_id)
    if hypr_key_spec is None:
        return "skipped_custom", "Selected keybind is custom; no automatic Hyprland edit attempted."

    target_spec = _normalize_hypr_key_spec(hypr_key_spec)
    if target_spec is None:
        return "error", f"Invalid Hyprland key spec for preset '{keybind_id}': {hypr_key_spec!r}"

    candidates = _hypr_config_candidates()
    existing_files = [path for path in candidates if path.exists()]
    if not existing_files:
        return "missing_config", f"Hyprland config not found: {hyprland_config_path()}"

    content_by_file: dict[Path, str] = {}
    for config_file in existing_files:
        try:
            content_by_file[config_file] = config_file.read_text()
        except Exception as exc:  # noqa: BLE001
            return "error", f"Failed to read {config_file}: {exc}"

    shuvoice_command = _resolve_shuvoice_command()
    desired_lines = _bind_lines_for_preset(
        keybind_id,
        hypr_key_spec,
        shuvoice_command=shuvoice_command,
    )

    desired_start_specs = {target_spec}
    desired_stop_specs = {target_spec}
    conflict_specs = {target_spec}

    tts_key_spec = TTS_KEYBIND_PRESETS[0][2]
    desired_tts_specs: set[str] = set()
    normalized_tts_spec = _normalize_hypr_key_spec(tts_key_spec)
    if normalized_tts_spec is not None:
        desired_tts_specs.add(normalized_tts_spec)
        conflict_specs.add(normalized_tts_spec)

    extra_stop_spec: str | None = None
    if keybind_id == "right_ctrl":
        extra_stop_spec = _normalize_bind_spec("CTRL", "Control_R")
        if extra_stop_spec:
            desired_stop_specs.add(extra_stop_spec)
            conflict_specs.add(extra_stop_spec)

    has_target_start = False
    has_target_stop = False
    has_extra_stop = False
    has_tts_speak = False
    has_other_shuvoice_bind = False
    shuvoice_binding_count = 0

    conflict_files: dict[Path, list[int]] = {}
    shuvoice_lines: dict[Path, list[int]] = {}

    for config_file, content in content_by_file.items():
        for line_no, raw_line in enumerate(content.splitlines(), start=1):
            parsed = _parse_hypr_bind_line(raw_line)
            if parsed is None:
                continue

            spec, command = parsed
            command_lc = command.lower()
            is_shuvoice = _is_shuvoice_control_command(command_lc)
            is_start = _is_shuvoice_start_command(command_lc)
            is_stop = _is_shuvoice_stop_command(command_lc)
            is_tts_speak = _is_shuvoice_tts_speak_command(command_lc)

            if spec in conflict_specs and not is_shuvoice:
                conflict_files.setdefault(config_file, []).append(line_no)

            if not is_shuvoice:
                continue

            if is_start or is_stop or is_tts_speak:
                shuvoice_binding_count += 1
                shuvoice_lines.setdefault(config_file, []).append(line_no)

            if is_start and spec in desired_start_specs:
                has_target_start = True
                continue
            if is_stop and spec in desired_stop_specs:
                if spec == target_spec:
                    has_target_stop = True
                if extra_stop_spec and spec == extra_stop_spec:
                    has_extra_stop = True
                continue
            if is_tts_speak and spec in desired_tts_specs:
                has_tts_speak = True
                continue
            if is_start or is_stop or is_tts_speak:
                has_other_shuvoice_bind = True

    if conflict_files:
        parts: list[str] = []
        for path, lines in conflict_files.items():
            line_refs = ", ".join(str(n) for n in lines[:3])
            if len(lines) > 3:
                line_refs += ", ..."
            parts.append(f"{path} line(s): {line_refs}")
        return (
            "conflict",
            "Key is already bound; not adding ShuVoice binds (" + "; ".join(parts) + ").",
        )

    is_fully_configured = has_target_start and has_target_stop
    if extra_stop_spec is not None:
        is_fully_configured = is_fully_configured and has_extra_stop
    if desired_tts_specs:
        is_fully_configured = is_fully_configured and has_tts_speak

    if (
        is_fully_configured
        and not has_other_shuvoice_bind
        and shuvoice_binding_count == len(desired_lines)
    ):
        configured_in = next(
            (path for path, lines in shuvoice_lines.items() if lines),
            existing_files[0],
        )
        return "already_configured", f"ShuVoice keybind already configured in {configured_in}."

    destination = next(
        (path for path, lines in shuvoice_lines.items() if lines),
        existing_files[0],
    )

    try:
        for config_file, content in content_by_file.items():
            original_lines = content.splitlines(keepends=True)
            filtered_lines: list[str] = []
            for raw_line in original_lines:
                parsed = _parse_hypr_bind_line(raw_line)
                if parsed is None:
                    if raw_line.strip() == "# Added by ShuVoice setup wizard":
                        continue
                    filtered_lines.append(raw_line)
                    continue
                _spec, command = parsed
                command_lc = command.lower()
                is_shuvoice_control = _is_shuvoice_control_command(command_lc)
                is_control_bind = (
                    _is_shuvoice_start_command(command_lc)
                    or _is_shuvoice_stop_command(command_lc)
                    or _is_shuvoice_tts_speak_command(command_lc)
                )
                if is_shuvoice_control and is_control_bind:
                    continue
                filtered_lines.append(raw_line)

            if config_file == destination:
                if filtered_lines and not filtered_lines[-1].endswith("\n"):
                    filtered_lines[-1] += "\n"
                if filtered_lines and filtered_lines[-1].strip():
                    filtered_lines.append("\n")
                filtered_lines.append("# Added by ShuVoice setup wizard\n")
                filtered_lines.extend(f"{line}\n" for line in desired_lines)

            new_content = "".join(filtered_lines)
            if new_content != content:
                config_file.write_text(new_content)
    except Exception as exc:  # noqa: BLE001
        return "error", f"Failed to update Hyprland config: {exc}"

    return "added", f"Added ShuVoice keybind to {destination}."


def needs_wizard() -> bool:
    """Return True when the welcome wizard should run (first launch).

    Returns False if the wizard-done marker exists **or** if a config.toml
    already exists (upgrade path — existing installations should not be
    forced through the wizard just because the marker is absent).
    """
    if (Config.data_dir() / _MARKER_FILE).exists():
        return False
    if (Config.config_dir() / "config.toml").exists():
        return False
    return True


def write_marker():
    """Create the .wizard-done marker so the wizard won't run again."""
    marker = Config.data_dir() / _MARKER_FILE
    marker.write_text("done\n")


def _upsert_asr_key(config_file: Path, key: str, value: str):
    """Update or insert a key under ``[asr]`` in config.toml.

    Preserves unrelated lines/comments and only patches the relevant field.
    """
    lines = config_file.read_text().splitlines(keepends=True)

    section_re = re.compile(r"^\s*\[([^\]]+)\]\s*$")
    key_re = re.compile(rf"^\s*{re.escape(key)}\s*=")

    asr_start: int | None = None
    asr_end = len(lines)

    for idx, line in enumerate(lines):
        match = section_re.match(line.strip())
        if not match:
            continue
        section_name = match.group(1).strip().lower()
        if asr_start is None:
            if section_name == "asr":
                asr_start = idx
            continue
        asr_end = idx
        break

    new_line = f'{key} = "{value}"\n'

    if asr_start is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        if lines and lines[-1].strip():
            lines.append("\n")
        lines.extend(["[asr]\n", new_line])
        config_file.write_text("".join(lines))
        return

    for idx in range(asr_start + 1, asr_end):
        stripped = lines[idx].lstrip()
        if stripped.startswith("#"):
            continue
        if key_re.match(stripped):
            indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
            lines[idx] = f"{indent}{new_line}"
            config_file.write_text("".join(lines))
            return

    insert_at = asr_start + 1
    while insert_at < asr_end and lines[insert_at].strip().startswith("#"):
        insert_at += 1
    lines.insert(insert_at, new_line)
    config_file.write_text("".join(lines))


def _detect_cuda() -> bool:
    """Return True if CUDA is likely usable for inference."""
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        pass
    # Fallback: check for nvidia-smi
    import shutil  # noqa: PLC0415

    return shutil.which("nvidia-smi") is not None


def _detect_sherpa_cuda_provider() -> bool:
    """Return True when installed sherpa-onnx runtime exposes CUDA provider."""
    try:
        sherpa_cls = get_backend_class("sherpa")
        checker = getattr(sherpa_cls, "_cuda_provider_available", None)
        if callable(checker):
            ok, _detail = checker()
            return bool(ok)
    except Exception:  # noqa: BLE001
        pass
    return False


# Maps backend id -> config key for the provider/device setting.
_BACKEND_PROVIDER_KEY: dict[str, str] = {
    "sherpa": "sherpa_provider",
    "nemo": "device",
    "moonshine": "moonshine_provider",
}


def write_config(
    asr_backend: str,
    *,
    overwrite_existing: bool = False,
    sherpa_model_name: str | None = None,
    sherpa_enable_parakeet_streaming: bool = False,
    sherpa_provider: str | None = None,
    typing_final_injection_mode: str = DEFAULT_FINAL_INJECTION_MODE,
    tts_backend: str = DEFAULT_TTS_BACKEND,
    tts_default_voice_id: str | None = None,
):
    """Write wizard selections to config.toml.

    By default, preserves existing config files. When ``overwrite_existing`` is
    true, ``[asr].asr_backend`` and the provider/device key are updated.

    For Sherpa, optionally persists ``sherpa_model_name`` so the runtime can
    auto-download the selected model variant on first launch. For Parakeet,
    wizard can write either:
    - stable/default instant mode (``instant_mode = true`` +
      ``sherpa_decode_mode = "offline_instant"``), or
    - explicit streaming override (``sherpa_decode_mode = "streaming"`` +
      ``sherpa_enable_parakeet_streaming = true``).

    Wizard maps Sherpa profiles to overlay-only typing by default:
    - Streaming profiles -> ``[typing].output_mode = "final_only"``
    - Instant profile -> ``[typing].output_mode = "final_only"``

    Users can still opt into live partial typing manually with
    ``[typing].output_mode = "streaming_partial"``.

    Final text injection mode is persisted to ``[typing].typing_final_injection_mode``
    so users can pick between clipboard paste and direct keystroke typing from
    the wizard UI.

    TTS provider + default voice are persisted to ``[tts]`` so the wizard can
    configure which backend handles ``tts_speak`` and which voice it should use.

    Writes use the config I/O durability path (atomic write + backup).

    Provider selection:
    - If ``sherpa_provider`` is explicitly set, that value is written as-is
      (``cpu`` or ``cuda``).
    - Otherwise, Sherpa defaults to CUDA only when the installed sherpa-onnx
      runtime already exposes CUDAExecutionProvider; falls back to CPU.
    """
    injection_mode = str(typing_final_injection_mode).strip().lower()
    if injection_mode not in _FINAL_INJECTION_MODE_SET:
        allowed = ", ".join(sorted(_FINAL_INJECTION_MODE_SET))
        raise ValueError(f"typing_final_injection_mode must be one of: {allowed}")

    tts_backend_value = str(tts_backend).strip().lower()
    if tts_backend_value not in {"elevenlabs", "openai", "local"}:
        raise ValueError("tts_backend must be one of: elevenlabs, openai, local")

    tts_voice_value = str(
        tts_default_voice_id or default_tts_voice_for_backend(tts_backend_value)
    ).strip()
    if not tts_voice_value:
        raise ValueError("tts_default_voice_id must not be empty")

    if asr_backend == "sherpa":
        explicit_provider = None
        if sherpa_provider is not None:
            explicit_provider = str(sherpa_provider).strip().lower()
            if explicit_provider not in {"cpu", "cuda"}:
                raise ValueError("sherpa_provider must be one of: cpu, cuda")

        if explicit_provider is not None:
            provider = explicit_provider
            if provider == "cuda" and not _detect_sherpa_cuda_provider():
                log.info(
                    "Sherpa CUDA explicitly selected in wizard, but current runtime does not yet "
                    "expose CUDAExecutionProvider; setup will attempt runtime install."
                )
        else:
            sherpa_cuda_available = _detect_sherpa_cuda_provider()
            provider = "cuda" if sherpa_cuda_available else "cpu"
            if not sherpa_cuda_available and _detect_cuda():
                log.info(
                    "Sherpa runtime CUDAExecutionProvider not detected; defaulting to "
                    "sherpa_provider='cpu'"
                )
    else:
        provider = "cuda" if _detect_cuda() else "cpu"

    provider_key = _BACKEND_PROVIDER_KEY.get(asr_backend)

    config_file = Config.config_dir() / "config.toml"
    if config_file.exists() and not overwrite_existing:
        log.info("config.toml already exists; wizard will not overwrite it")
        return

    if config_file.exists():
        raw = load_raw(config_file)
        migrated, _report = migrate_to_latest(raw)
    else:
        migrated = {"config_version": CURRENT_CONFIG_VERSION}

    asr_table = migrated.get("asr")
    if not isinstance(asr_table, dict):
        asr_table = {}
        migrated["asr"] = asr_table

    typing_table = migrated.get("typing")
    if not isinstance(typing_table, dict):
        typing_table = {}
        migrated["typing"] = typing_table

    tts_table = migrated.get("tts")
    if not isinstance(tts_table, dict):
        tts_table = {}
        migrated["tts"] = tts_table

    asr_table["asr_backend"] = asr_backend
    if provider_key:
        asr_table[provider_key] = provider

    typing_table["typing_final_injection_mode"] = injection_mode
    # Keep legacy compatibility flag in sync for older tooling/parsers.
    typing_table["use_clipboard_for_final"] = injection_mode != "direct"

    if asr_backend == "sherpa":
        chosen_model = (sherpa_model_name or DEFAULT_SHERPA_MODEL_NAME).strip()
        is_parakeet = _is_parakeet_sherpa_model_name(chosen_model)
        enable_parakeet_streaming = bool(sherpa_enable_parakeet_streaming and is_parakeet)

        asr_table["sherpa_model_name"] = chosen_model
        asr_table["sherpa_enable_parakeet_streaming"] = enable_parakeet_streaming

        if is_parakeet:
            if enable_parakeet_streaming:
                # Explicit override for Parakeet streaming path.
                asr_table["instant_mode"] = False
                asr_table["sherpa_decode_mode"] = "streaming"
                typing_table["output_mode"] = "final_only"
            else:
                # Stable/default Parakeet path.
                asr_table["instant_mode"] = True
                asr_table["sherpa_decode_mode"] = "offline_instant"
                typing_table["output_mode"] = "final_only"
        else:
            # Keep Sherpa defaults for Zipformer/non-Parakeet streaming models.
            asr_table["sherpa_decode_mode"] = "auto"
            asr_table["instant_mode"] = False
            typing_table["output_mode"] = "final_only"

    tts_table["tts_backend"] = tts_backend_value
    tts_table["tts_default_voice_id"] = tts_voice_value

    migrated["config_version"] = CURRENT_CONFIG_VERSION

    backup = write_atomic(config_file, migrated)
    if backup is not None:
        log.info("Updated %s (provider=%s, backup=%s)", config_file, provider, backup)
    else:
        log.info("Wrote %s (provider=%s)", config_file, provider)


def format_summary(
    asr_backend: str,
    keybind_id: str = DEFAULT_KEYBIND_ID,
    *,
    auto_add_keybind: bool = True,
    sherpa_model_name: str | None = None,
    sherpa_enable_parakeet_streaming: bool = False,
    sherpa_provider: str | None = None,
    typing_final_injection_mode: str = DEFAULT_FINAL_INJECTION_MODE,
    tts_backend: str = DEFAULT_TTS_BACKEND,
    tts_default_voice_id: str | None = None,
) -> str:
    """Build a human-readable summary of wizard selections."""
    asr_name = next(
        (label for bid, label, _ in ASR_BACKENDS if bid == asr_backend),
        asr_backend,
    )
    keybind_label, hypr_key = next(
        ((label, hk) for kid, label, hk, _ in KEYBIND_PRESETS if kid == keybind_id),
        ("Custom", None),
    )

    injection_mode = str(typing_final_injection_mode).strip().lower()
    injection_label = {
        "auto": "Auto (recommended)",
        "clipboard": "Clipboard paste (Ctrl+V)",
        "direct": "Direct typing (keystroke simulation)",
    }.get(injection_mode, injection_mode or DEFAULT_FINAL_INJECTION_MODE)

    tts_backend_value = str(tts_backend).strip().lower() or DEFAULT_TTS_BACKEND
    tts_voice_value = str(
        tts_default_voice_id or default_tts_voice_for_backend(tts_backend_value)
    ).strip()

    lines = [
        f"ASR backend:      {asr_name}",
        f"Final injection:  {injection_label}",
        f"TTS provider:     {tts_backend_label(tts_backend_value)}",
        f"TTS voice:        {tts_voice_label(tts_backend_value, tts_voice_value)}",
        f"Push-to-talk:     {keybind_label}",
    ]

    if asr_backend == "sherpa":
        chosen_model = (sherpa_model_name or DEFAULT_SHERPA_MODEL_NAME).strip()
        is_parakeet = _is_parakeet_sherpa_model_name(chosen_model)
        parakeet_streaming = bool(sherpa_enable_parakeet_streaming and is_parakeet)

        if chosen_model == PARAKEET_TDT_V3_INT8_MODEL_NAME:
            model_label = "Parakeet TDT v3 (int8)"
        elif chosen_model == DEFAULT_SHERPA_MODEL_NAME:
            model_label = "Zipformer Kroko (default)"
        else:
            model_label = chosen_model

        if parakeet_streaming:
            profile_label = "Streaming (Parakeet)"
            decode_label = "Streaming (explicit override)"
            output_mode_label = "final_only"
        elif is_parakeet:
            profile_label = "Instant (Parakeet)"
            decode_label = "Offline instant (auto-enabled)"
            output_mode_label = "final_only"
        else:
            profile_label = "Streaming"
            decode_label = "Streaming (auto)"
            output_mode_label = "final_only"

        provider_value = str(sherpa_provider or "cpu").strip().lower()
        provider_label = "GPU (CUDA)" if provider_value == "cuda" else "CPU"

        lines.insert(1, f"Sherpa profile: {profile_label}")
        lines.insert(2, f"Sherpa device:  {provider_label}")
        lines.insert(3, f"Sherpa model:   {model_label}")
        lines.insert(4, f"Sherpa decode:  {decode_label}")
        lines.insert(5, f"Output mode:    {output_mode_label}")

    if hypr_key:
        bind_lines = format_hyprland_bind_for_keybind(
            keybind_id,
            hypr_key,
            shuvoice_command="shuvoice",
        )
        indented = "\n".join(f"  {line}" for line in bind_lines.splitlines())
        if auto_add_keybind:
            lines.extend(
                [
                    "",
                    "Wizard will try to add this to ~/.config/hypr/hyprland.conf",
                    "(only if no conflicting bind already uses that key):",
                    "",
                    indented,
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Add to ~/.config/hypr/hyprland.conf:",
                    "",
                    indented,
                ]
            )
    else:
        lines.extend(
            [
                "",
                "Configure your keybind in ~/.config/hypr/hyprland.conf",
                "See README.md for bind/bindr examples.",
            ]
        )

    return "\n".join(lines)
