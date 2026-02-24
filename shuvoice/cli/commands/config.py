"""Config CLI subcommands."""

from __future__ import annotations

import sys

from ...config import Config
from ...config_io import load_raw, toml_dumps
from ...config_migrations import migrate_to_latest


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
