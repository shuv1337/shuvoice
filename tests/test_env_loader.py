from __future__ import annotations

import os
from pathlib import Path

from shuvoice.env_loader import load_local_dev_env, local_dev_env_path


def test_local_dev_env_path_uses_xdg_config_home(monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg-config-home")
    path = local_dev_env_path()
    assert path == Path("/tmp/xdg-config-home/shuvoice/local.dev")


def test_load_local_dev_env_loads_keys_and_export_lines(tmp_path, monkeypatch):
    env_file = tmp_path / "local.dev"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "ELEVENLABS_API_KEY=abc123",
                'export OPENAI_API_KEY="xyz"',
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_local_dev_env(env_file)

    assert loaded == 2
    assert os.environ["ELEVENLABS_API_KEY"] == "abc123"
    assert os.environ["OPENAI_API_KEY"] == "xyz"


def test_load_local_dev_env_does_not_override_existing_by_default(tmp_path, monkeypatch):
    env_file = tmp_path / "local.dev"
    env_file.write_text("ELEVENLABS_API_KEY=from_file\n", encoding="utf-8")

    monkeypatch.setenv("ELEVENLABS_API_KEY", "already_set")
    loaded = load_local_dev_env(env_file)

    assert loaded == 0
    assert os.environ["ELEVENLABS_API_KEY"] == "already_set"


def test_load_local_dev_env_overrides_when_requested(tmp_path, monkeypatch):
    env_file = tmp_path / "local.dev"
    env_file.write_text("ELEVENLABS_API_KEY=from_file\n", encoding="utf-8")

    monkeypatch.setenv("ELEVENLABS_API_KEY", "already_set")
    loaded = load_local_dev_env(env_file, override=True)

    assert loaded == 1
    assert os.environ["ELEVENLABS_API_KEY"] == "from_file"
