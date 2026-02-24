from __future__ import annotations

import logging

from shuvoice.cli import main as cli_main
from shuvoice.config import Config


def test_cli_main_dispatches_default_run(monkeypatch):
    monkeypatch.setattr("shuvoice.cli.run_app", lambda args: 7)

    assert cli_main([]) == 7


def test_cli_main_dispatches_wizard(monkeypatch):
    monkeypatch.setattr("shuvoice.cli.run_wizard_command", lambda: 0)

    assert cli_main(["wizard"]) == 0


def test_cli_main_dispatches_audio_list_devices(monkeypatch):
    monkeypatch.setattr("shuvoice.cli.list_audio_devices", lambda: 0)

    assert cli_main(["audio", "list-devices"]) == 0


def test_cli_main_dispatches_config_path_without_loading_config(monkeypatch):
    called = {"path": False, "load": False}

    def fake_path() -> int:
        called["path"] = True
        return 0

    def fake_load(_args):
        called["load"] = True
        return Config()

    monkeypatch.setattr("shuvoice.cli.config_path", fake_path)
    monkeypatch.setattr("shuvoice.cli._load_config_or_exit", fake_load)

    assert cli_main(["config", "path"]) == 0
    assert called["path"] is True
    assert called["load"] is False


def test_cli_main_dispatches_preflight(monkeypatch):
    monkeypatch.setattr("shuvoice.cli._load_config_or_exit", lambda _args: Config())
    monkeypatch.setattr("shuvoice.cli.run_preflight", lambda _cfg: True)

    assert cli_main(["preflight"]) == 0


def test_cli_main_legacy_control_logs_warning_and_dispatches(monkeypatch, caplog):
    monkeypatch.setattr("shuvoice.cli._load_config_or_exit", lambda _args: Config())

    called: dict[str, object] = {}

    def fake_control(command: str, _config: Config, *, wait_sec: float) -> int:
        called["command"] = command
        called["wait_sec"] = wait_sec
        return 0

    monkeypatch.setattr("shuvoice.cli.run_control", fake_control)

    with caplog.at_level(logging.WARNING):
        result = cli_main(["--control", "status"])

    assert result == 0
    assert called["command"] == "status"
    assert called["wait_sec"] == 2.0
    assert any("deprecated" in record.message.lower() for record in caplog.records)
