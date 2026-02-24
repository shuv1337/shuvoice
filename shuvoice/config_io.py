"""Low-level TOML config I/O helpers (atomic writes + backups)."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .config import CURRENT_CONFIG_VERSION


def load_raw(path: str | Path) -> dict[str, Any]:
    """Load raw TOML config data from ``path``.

    Missing files return an empty vCurrent skeleton.
    Existing unversioned files are tagged as ``config_version = 0`` for migration.
    """
    config_path = Path(path).expanduser()
    if not config_path.exists():
        return {"config_version": CURRENT_CONFIG_VERSION}

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level TOML table: {config_path}")

    if "config_version" not in data:
        data["config_version"] = 0

    return data


def backup_config(path: str | Path) -> Path | None:
    """Create timestamped backup beside ``path``.

    Returns backup path when source exists, otherwise ``None``.
    """
    config_path = Path(path).expanduser()
    if not config_path.exists():
        return None

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = config_path.with_name(f"{config_path.name}.bak-{stamp}")
    shutil.copy2(config_path, backup_path)
    return backup_path


_BARE_TOML_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _format_toml_key(key: str) -> str:
    if _BARE_TOML_KEY_RE.fullmatch(key):
        return key
    return json.dumps(key, ensure_ascii=False)


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Keep deterministic representation for snapshot-like tests.
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def _serialize_table(data: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> list[str]:
    lines: list[str] = []

    if prefix:
        dotted = ".".join(_format_toml_key(part) for part in prefix)
        lines.append(f"[{dotted}]")

    nested: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in data.items():
        if not isinstance(key, str):
            raise TypeError("TOML table keys must be strings")

        if value is None:
            continue

        if isinstance(value, Mapping):
            nested.append((key, value))
            continue

        lines.append(f"{_format_toml_key(key)} = {_format_toml_value(value)}")

    for idx, (key, child) in enumerate(nested):
        if lines:
            lines.append("")
        lines.extend(_serialize_table(child, prefix + (key,)))
        if idx < len(nested) - 1:
            lines.append("")

    return lines


def toml_dumps(data: Mapping[str, Any]) -> str:
    """Serialize nested config data to TOML text."""
    lines = _serialize_table(data)
    text = "\n".join(lines).strip()
    return text + "\n"


def _fsync_directory(path: Path) -> None:
    try:
        dir_fd = os.open(path, os.O_DIRECTORY | os.O_RDONLY)
    except OSError:
        return

    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def write_atomic(path: str | Path, data: Mapping[str, Any]) -> Path | None:
    """Atomically write TOML data to ``path``.

    Uses a same-directory temporary file + fsync + rename.
    Returns backup path when an existing file was backed up.
    """
    config_path = Path(path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    backup = backup_config(config_path)
    payload = toml_dumps(data)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{config_path.name}.",
        suffix=".tmp",
        dir=config_path.parent,
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_path, config_path)
        _fsync_directory(config_path.parent)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

        # Recovery path for catastrophic replacement failure where destination
        # disappears and a backup exists.
        if backup is not None and not config_path.exists():
            shutil.copy2(backup, config_path)
        raise

    return backup
