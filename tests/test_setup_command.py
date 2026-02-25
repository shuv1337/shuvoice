from __future__ import annotations

from shuvoice.cli.commands import setup as setup_cmd
from shuvoice.config import Config
from shuvoice.setup_helpers import DEPENDENCY_EXIT_CODE, BackendSetupReport


def test_run_setup_returns_dependency_exit_code_when_missing(capsys, monkeypatch):
    report = BackendSetupReport(
        backend="sherpa",
        missing_dependencies=("missing sherpa_onnx",),
        install_hints=("uv sync --extra asr-sherpa",),
        model_status="missing (/tmp/model)",
    )
    monkeypatch.setattr(setup_cmd, "build_backend_setup_report", lambda _cfg: report)

    code = setup_cmd.run_setup(
        Config(),
        install_missing=False,
        skip_model_download=True,
        skip_preflight=True,
    )

    assert code == DEPENDENCY_EXIT_CODE
    stdout = capsys.readouterr().out
    assert "Model status: missing" in stdout
    assert "Setup incomplete" in stdout


def test_run_setup_success_path_with_skips(monkeypatch):
    report = BackendSetupReport(
        backend="sherpa",
        missing_dependencies=(),
        install_hints=(),
        model_status="present (/tmp/model)",
    )
    monkeypatch.setattr(setup_cmd, "build_backend_setup_report", lambda _cfg: report)

    code = setup_cmd.run_setup(
        Config(),
        install_missing=False,
        skip_model_download=True,
        skip_preflight=True,
    )

    assert code == 0
