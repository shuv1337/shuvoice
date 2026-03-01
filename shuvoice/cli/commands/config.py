"""Config CLI subcommands."""

from __future__ import annotations

import sys
from typing import Any

from ...config import Config
from ...config_io import load_raw, toml_dumps, write_atomic
from ...config_migrations import migrate_to_latest

_ALLOWED_FINAL_INJECTION_MODES = {"auto", "clipboard", "direct"}


def _flatten_candidate(raw: dict[str, Any]) -> dict[str, Any]:
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


def _validate_candidate(raw: dict[str, Any]) -> None:
    flat = _flatten_candidate(raw)

    # Preserve legacy compatibility behavior used by Config.load().
    has_explicit_mode = "typing_final_injection_mode" in flat
    has_legacy_flag = "use_clipboard_for_final" in flat
    if not has_explicit_mode and has_legacy_flag:
        legacy_flag = flat.get("use_clipboard_for_final")
        if isinstance(legacy_flag, bool):
            flat["typing_final_injection_mode"] = "clipboard" if legacy_flag else "direct"

    valid_fields = Config.config_field_names()
    filtered = {k: v for k, v in flat.items() if k in valid_fields}
    Config(**filtered)


def config_set(key: str, value: str) -> int:
    key_norm = str(key).strip()
    value_norm = str(value).strip().lower()

    if key_norm != "typing_final_injection_mode":
        print(
            f"ERROR: unsupported config key '{key_norm}'. Supported keys: typing_final_injection_mode",
            file=sys.stderr,
        )
        return 1

    if value_norm not in _ALLOWED_FINAL_INJECTION_MODES:
        allowed = ", ".join(sorted(_ALLOWED_FINAL_INJECTION_MODES))
        print(
            f"ERROR: typing_final_injection_mode must be one of: {allowed}",
            file=sys.stderr,
        )
        return 1

    config_file = Config.config_path()

    try:
        raw = load_raw(config_file)
        migrated, _report = migrate_to_latest(raw)

        typing_table = migrated.get("typing")
        if not isinstance(typing_table, dict):
            typing_table = {}
            migrated["typing"] = typing_table

        typing_table["typing_final_injection_mode"] = value_norm
        # Keep legacy flag synchronized for old config consumers.
        typing_table["use_clipboard_for_final"] = value_norm != "direct"

        _validate_candidate(migrated)
        backup = write_atomic(config_file, migrated)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    legacy_value = "true" if value_norm != "direct" else "false"
    if backup is not None:
        print(
            "OK "
            f"set {key_norm}={value_norm} "
            f"(use_clipboard_for_final={legacy_value}, path={config_file}, backup={backup})"
        )
    else:
        print(
            "OK "
            f"set {key_norm}={value_norm} "
            f"(use_clipboard_for_final={legacy_value}, path={config_file})"
        )

    return 0


def config_path() -> int:
    print(Config.config_path())
    return 0


def config_validate() -> int:
    try:
        raw = load_raw(Config.config_path())
        _migrated, report = migrate_to_latest(raw)
        cfg = Config.load()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "OK "
        f"(schema={cfg.config_version}, migrated_from={report.from_version}, "
        f"path={Config.config_path()})"
    )
    return 0


def config_effective() -> int:
    try:
        cfg = Config.load()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(toml_dumps(cfg.to_nested_dict()).rstrip())
    return 0
