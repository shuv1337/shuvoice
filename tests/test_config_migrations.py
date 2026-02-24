from __future__ import annotations

import pytest

from shuvoice.config import CURRENT_CONFIG_VERSION
from shuvoice.config_migrations import migrate_to_latest


def test_migrate_unversioned_config_to_latest():
    raw = {
        "asr": {
            "asr_backend": "sherpa",
        }
    }

    migrated, report = migrate_to_latest(raw)

    assert migrated["config_version"] == CURRENT_CONFIG_VERSION
    assert report.from_version == 0
    assert report.to_version == CURRENT_CONFIG_VERSION
    assert "config_version" in report.changed_keys


def test_migrate_current_version_is_noop():
    raw = {"config_version": CURRENT_CONFIG_VERSION, "audio": {"sample_rate": 16000}}

    migrated, report = migrate_to_latest(raw)

    assert migrated == raw
    assert report.from_version == CURRENT_CONFIG_VERSION
    assert report.to_version == CURRENT_CONFIG_VERSION
    assert report.changed_keys == ()


def test_migrate_rejects_future_schema_version():
    raw = {"config_version": CURRENT_CONFIG_VERSION + 1}

    with pytest.raises(ValueError, match="newer"):
        migrate_to_latest(raw)
