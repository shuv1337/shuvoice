from __future__ import annotations

from shuvoice.cli.commands import setup as setup_cmd
from shuvoice.config import Config
from shuvoice.setup_helpers import install_hints_for_backend, model_status_for_backend


def test_install_hints_for_sherpa_prefer_bin_provider(monkeypatch):
    monkeypatch.setattr(
        "shuvoice.setup_helpers.shutil.which",
        lambda exe: "/usr/bin/yay" if exe == "yay" else None,
    )

    hints = install_hints_for_backend("sherpa")

    assert any("python-sherpa-onnx-bin" in hint for hint in hints)
    assert any("python-sherpa-onnx" in hint for hint in hints)


def test_auto_install_commands_prefers_bin_provider(monkeypatch):
    monkeypatch.setattr(setup_cmd, "_running_in_venv", lambda: False)

    commands = setup_cmd._auto_install_commands("sherpa")

    assert commands[0][-1] == "python-sherpa-onnx-bin"
    assert commands[1][-1] == "python-sherpa-onnx"


def test_model_status_mentions_configured_sherpa_model_name_when_missing(monkeypatch):
    cfg = Config(
        asr_backend="sherpa", sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    )

    monkeypatch.setattr(
        "shuvoice.setup_helpers._is_complete_sherpa_model_dir",
        lambda _path: False,
    )

    status = model_status_for_backend(cfg)

    assert "parakeet-tdt-0.6b-v3-int8" in status
