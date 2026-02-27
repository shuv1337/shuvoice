from __future__ import annotations

from pathlib import Path


def test_service_unit_prevents_missing_dependency_restart_loop() -> None:
    unit_path = (
        Path(__file__).resolve().parents[1] / "packaging" / "systemd" / "user" / "shuvoice.service"
    )
    content = unit_path.read_text(encoding="utf-8")

    assert "RestartPreventExitStatus=78" in content
