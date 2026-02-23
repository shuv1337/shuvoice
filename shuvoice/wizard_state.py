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

from .config import Config

log = logging.getLogger(__name__)

_MARKER_FILE = ".wizard-done"

# ASR backend descriptions shown in the wizard.
ASR_BACKENDS = [
    (
        "sherpa",
        "Sherpa-ONNX",
        "Fast ONNX-based streaming ASR.  Works on CPU — no GPU needed.",
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

# Keybind presets for push-to-talk setup.
# (id, display_label, hyprland_bind_key_spec, description)
# hyprland_bind_key_spec is the "MODS, KEY" portion for bind/bindr lines.
KEYBIND_PRESETS = [
    ("insert", "Insert", ", Insert", "Usually unused and easy to dedicate."),
    (
        "right_ctrl",
        "Right Control",
        ", Control_R",
        "Comfortable hold-to-talk key on many keyboards.",
    ),
    ("f9", "F9", ", F9", "Simple single-key push-to-talk."),
    (
        "super_v",
        "Super + V",
        "SUPER, V",
        "Modifier combo — mnemonic for Voice.",
    ),
    ("custom", "Custom", None, "Set your own key in Hyprland config later."),
]


def format_hyprland_bind(hypr_key_spec: str, *, shuvoice_command: str = "shuvoice") -> str:
    """Format Hyprland bind/bindr lines for a push-to-talk keybind."""
    return (
        f"bind = {hypr_key_spec}, exec, {_control_exec('start', shuvoice_command=shuvoice_command)}\n"
        f"bindr = {hypr_key_spec}, exec, {_control_exec('stop', shuvoice_command=shuvoice_command)}"
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
    return f"{shlex.quote(binary)} --control {command}"


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
    bind_text = format_hyprland_bind(hypr_key_spec, shuvoice_command=shuvoice_command)
    start_line, stop_line = bind_text.splitlines()

    has_target_start = False
    has_target_stop = False
    has_other_shuvoice_bind = False

    conflict_files: dict[Path, list[int]] = {}
    shuvoice_lines: dict[Path, list[int]] = {}

    for config_file, content in content_by_file.items():
        for line_no, raw_line in enumerate(content.splitlines(), start=1):
            parsed = _parse_hypr_bind_line(raw_line)
            if parsed is None:
                continue

            spec, command = parsed
            command_lc = command.lower()
            is_shuvoice = "shuvoice" in command_lc and "--control" in command_lc
            is_start = is_shuvoice and "--control start" in command_lc
            is_stop = is_shuvoice and "--control stop" in command_lc

            if spec == target_spec and not is_shuvoice:
                conflict_files.setdefault(config_file, []).append(line_no)

            if not is_shuvoice:
                continue

            if is_start or is_stop:
                shuvoice_lines.setdefault(config_file, []).append(line_no)

            if spec == target_spec and is_start:
                has_target_start = True
            elif spec == target_spec and is_stop:
                has_target_stop = True
            elif is_start or is_stop:
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

    if has_target_start and has_target_stop and not has_other_shuvoice_bind:
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
                    filtered_lines.append(raw_line)
                    continue
                _spec, command = parsed
                command_lc = command.lower()
                is_shuvoice_control = "shuvoice" in command_lc and "--control" in command_lc
                is_start_or_stop = "--control start" in command_lc or "--control stop" in command_lc
                if is_shuvoice_control and is_start_or_stop:
                    continue
                filtered_lines.append(raw_line)

            if config_file == destination:
                if filtered_lines and not filtered_lines[-1].endswith("\n"):
                    filtered_lines[-1] += "\n"
                if filtered_lines and filtered_lines[-1].strip():
                    filtered_lines.append("\n")
                filtered_lines.append("# Added by ShuVoice setup wizard\n")
                filtered_lines.append(f"{start_line}\n")
                filtered_lines.append(f"{stop_line}\n")

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


# Maps backend id -> config key for the provider/device setting.
_BACKEND_PROVIDER_KEY: dict[str, str] = {
    "sherpa": "sherpa_provider",
    "nemo": "device",
    "moonshine": "moonshine_provider",
}


def write_config(asr_backend: str, *, overwrite_existing: bool = False):
    """Write wizard selections to config.toml.

    By default, preserves existing config files. When ``overwrite_existing`` is
    true, ``[asr].asr_backend`` and the provider/device key are updated
    in-place.

    Automatically sets CUDA as the provider when a GPU is detected.
    """
    provider = "cuda" if _detect_cuda() else "cpu"
    provider_key = _BACKEND_PROVIDER_KEY.get(asr_backend)

    config_file = Config.config_dir() / "config.toml"
    if config_file.exists():
        if not overwrite_existing:
            log.info("config.toml already exists; wizard will not overwrite it")
            return
        _upsert_asr_key(config_file, "asr_backend", asr_backend)
        if provider_key:
            _upsert_asr_key(config_file, provider_key, provider)
        log.info("Updated asr config in %s (provider=%s)", config_file, provider)
        return

    lines = [
        "# Generated by ShuVoice welcome wizard\n",
        "\n",
        "[asr]\n",
        f'asr_backend = "{asr_backend}"\n',
    ]
    if provider_key:
        lines.append(f'{provider_key} = "{provider}"\n')
    config_file.write_text("".join(lines))
    log.info("Wrote %s (provider=%s)", config_file, provider)


def format_summary(
    asr_backend: str,
    keybind_id: str = "insert",
    *,
    auto_add_keybind: bool = True,
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

    lines = [
        f"ASR backend:    {asr_name}",
        f"Push-to-talk:   {keybind_label}",
    ]

    if hypr_key:
        bind_lines = format_hyprland_bind(hypr_key)
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
