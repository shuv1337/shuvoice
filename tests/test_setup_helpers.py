from __future__ import annotations

from shuvoice.cli.commands import setup as setup_cmd
from shuvoice.setup_helpers import install_hints_for_backend


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
