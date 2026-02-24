"""Config schema migrations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from .config import CURRENT_CONFIG_VERSION

MigrationStep = Callable[[dict[str, Any]], tuple[dict[str, Any], set[str]]]


@dataclass(frozen=True)
class MigrationReport:
    from_version: int
    to_version: int
    changed_keys: tuple[str, ...]


def _detect_version(raw_config: dict[str, Any]) -> int:
    value = raw_config.get("config_version", 0)
    try:
        version = int(value)
    except (TypeError, ValueError):
        version = 0
    return max(version, 0)


def _migrate_v0_to_v1(raw_config: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    migrated = deepcopy(raw_config)
    changed: set[str] = set()
    if migrated.get("config_version") != 1:
        migrated["config_version"] = 1
        changed.add("config_version")
    return migrated, changed


MIGRATIONS: dict[int, MigrationStep] = {
    0: _migrate_v0_to_v1,
}


def migrate_to_latest(raw_config: dict[str, Any]) -> tuple[dict[str, Any], MigrationReport]:
    """Migrate an arbitrary raw config map to ``CURRENT_CONFIG_VERSION``."""
    current = deepcopy(raw_config) if isinstance(raw_config, dict) else {}

    from_version = _detect_version(current)
    if from_version > CURRENT_CONFIG_VERSION:
        raise ValueError(
            "Config schema version is newer than this ShuVoice build supports "
            f"(got {from_version}, max {CURRENT_CONFIG_VERSION})"
        )

    changed_keys: set[str] = set()
    version = from_version

    while version < CURRENT_CONFIG_VERSION:
        step = MIGRATIONS.get(version)
        if step is None:
            raise RuntimeError(
                f"Missing migration step for config schema {version} -> {version + 1}"
            )

        current, changed = step(current)
        changed_keys.update(changed)
        version += 1

    if int(current.get("config_version", version)) != CURRENT_CONFIG_VERSION:
        current["config_version"] = CURRENT_CONFIG_VERSION
        changed_keys.add("config_version")

    report = MigrationReport(
        from_version=from_version,
        to_version=CURRENT_CONFIG_VERSION,
        changed_keys=tuple(sorted(changed_keys)),
    )
    return current, report
