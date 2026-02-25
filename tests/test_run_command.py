from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

from shuvoice.cli.commands import run as run_cmd
from shuvoice.cli.parser import create_parser
from shuvoice.config import Config
from shuvoice.setup_helpers import DEPENDENCY_EXIT_CODE, BackendSetupReport


def test_check_backend_dependencies_prints_actionable_message(capsys, monkeypatch):
    report = BackendSetupReport(
        backend="sherpa",
        missing_dependencies=("Missing sherpa-onnx dependency",),
        install_hints=("Arch (AUR): yay -S --needed python-sherpa-onnx",),
        model_status="missing (/tmp/model); will auto-download on first successful startup",
    )

    monkeypatch.setattr(run_cmd, "build_backend_setup_report", lambda _cfg: report)

    assert run_cmd._check_backend_dependencies(Config()) is False

    stderr = capsys.readouterr().err
    assert "Missing sherpa-onnx dependency" in stderr
    assert "Model status:" in stderr
    assert "shuvoice setup" in stderr


def test_run_app_returns_dependency_exit_code_when_backend_check_fails(monkeypatch):
    parser = create_parser()
    args: Namespace = parser.parse_args([])

    monkeypatch.setattr(run_cmd.Config, "load", classmethod(lambda cls: Config()))
    monkeypatch.setattr(run_cmd, "CDLL", lambda _lib: None)
    monkeypatch.setattr("shuvoice.wizard_state.needs_wizard", lambda: False)
    monkeypatch.setattr(run_cmd, "_check_backend_dependencies", lambda _cfg: False)

    assert run_cmd.run_app(args) == DEPENDENCY_EXIT_CODE


def test_check_backend_startup_guards_prints_warnings_and_errors(capsys, monkeypatch):
    dummy_backend = SimpleNamespace(
        startup_warnings=lambda _cfg, apply_fixes=False: ["warn-1"],
        startup_errors=lambda _cfg: ["err-1"],
    )
    monkeypatch.setattr(run_cmd, "get_backend_class", lambda _name: dummy_backend)

    ok = run_cmd._check_backend_startup_guards(Config())

    assert ok is False
    stderr = capsys.readouterr().err
    assert "WARNING: warn-1" in stderr
    assert "backend startup guard failed" in stderr
    assert "err-1" in stderr


def test_run_app_returns_dependency_exit_code_when_startup_guard_fails(monkeypatch):
    parser = create_parser()
    args: Namespace = parser.parse_args([])

    monkeypatch.setattr(run_cmd.Config, "load", classmethod(lambda cls: Config()))
    monkeypatch.setattr(run_cmd, "CDLL", lambda _lib: None)
    monkeypatch.setattr("shuvoice.wizard_state.needs_wizard", lambda: False)
    monkeypatch.setattr(run_cmd, "_check_backend_dependencies", lambda _cfg: True)
    monkeypatch.setattr(run_cmd, "_check_backend_startup_guards", lambda _cfg: False)

    assert run_cmd.run_app(args) == DEPENDENCY_EXIT_CODE


def test_run_app_returns_dependency_exit_code_when_config_validation_fails(monkeypatch):
    parser = create_parser()
    args: Namespace = parser.parse_args([])

    class BadConfig:
        asr_backend = "sherpa"

        def __post_init__(self):
            raise ValueError("bad config")

    monkeypatch.setattr(run_cmd.Config, "load", classmethod(lambda cls: BadConfig()))

    assert run_cmd.run_app(args) == DEPENDENCY_EXIT_CODE
