from __future__ import annotations

import json
import sys
import types
from argparse import Namespace

import pytest

from shuvoice.cli.commands import audio as audio_cmd
from shuvoice.cli.commands import common as common_cmd
from shuvoice.cli.commands import config as config_cmd
from shuvoice.cli.commands import control as control_cmd
from shuvoice.cli.commands import diagnostics as diagnostics_cmd
from shuvoice.cli.commands import model as model_cmd
from shuvoice.cli.commands import wizard as wizard_cmd
from shuvoice.config import Config


def test_list_audio_devices_success_filters_input_devices(monkeypatch, capsys):
    fake_sd = types.SimpleNamespace(
        query_devices=lambda: [
            {"name": "Input Mic", "max_input_channels": 1, "default_samplerate": 48000},
            {"name": "Output Only", "max_input_channels": 0, "default_samplerate": 48000},
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    assert audio_cmd.list_audio_devices() == 0

    out = capsys.readouterr().out
    assert "Audio devices:" in out
    assert "Input Mic" in out
    assert "Output Only" not in out


def test_list_audio_devices_error(monkeypatch, capsys):
    fake_sd = types.SimpleNamespace(query_devices=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    assert audio_cmd.list_audio_devices() == 1

    err = capsys.readouterr().err
    assert "Could not list audio devices" in err


def test_load_effective_config_applies_overrides_and_revalidates(monkeypatch):
    cfg = Config()

    monkeypatch.setattr(common_cmd.Config, "load", classmethod(lambda cls: cfg))

    def fake_apply(args: Namespace, config: Config) -> None:
        assert args.command == "setup"
        config.asr_backend = "moonshine"

    monkeypatch.setattr(common_cmd, "apply_cli_overrides", fake_apply)

    args = Namespace(command="setup")
    loaded = common_cmd.load_effective_config(args)

    assert loaded is cfg
    assert loaded.asr_backend == "moonshine"


def test_load_effective_config_raises_when_overrides_make_config_invalid(monkeypatch):
    cfg = Config()
    monkeypatch.setattr(common_cmd.Config, "load", classmethod(lambda cls: cfg))

    def fake_apply(_args: Namespace, config: Config) -> None:
        config.asr_backend = "invalid"

    monkeypatch.setattr(common_cmd, "apply_cli_overrides", fake_apply)

    with pytest.raises(ValueError):
        common_cmd.load_effective_config(Namespace())


def test_config_path_prints_path(capsys):
    assert config_cmd.config_path() == 0
    out = capsys.readouterr().out.strip()
    assert out.endswith("config.toml")


def test_config_validate_success(monkeypatch, capsys):
    cfg = Config()

    monkeypatch.setattr(config_cmd, "load_raw", lambda _path: {"config_version": 1})
    monkeypatch.setattr(
        config_cmd,
        "migrate_to_latest",
        lambda _raw: ({"config_version": 1}, types.SimpleNamespace(from_version=1)),
    )
    monkeypatch.setattr(config_cmd.Config, "load", classmethod(lambda cls: cfg))

    assert config_cmd.config_validate() == 0
    out = capsys.readouterr().out
    assert "OK" in out
    assert "schema=1" in out


def test_config_validate_error(monkeypatch, capsys):
    monkeypatch.setattr(config_cmd, "load_raw", lambda _path: (_ for _ in ()).throw(RuntimeError("bad")))

    assert config_cmd.config_validate() == 1
    err = capsys.readouterr().err
    assert "ERROR: bad" in err


def test_config_effective_success(monkeypatch, capsys):
    cfg = Config()
    monkeypatch.setattr(config_cmd.Config, "load", classmethod(lambda cls: cfg))

    assert config_cmd.config_effective() == 0
    out = capsys.readouterr().out
    assert "[asr]" in out


def test_config_effective_error(monkeypatch, capsys):
    monkeypatch.setattr(
        config_cmd.Config,
        "load",
        classmethod(lambda cls: (_ for _ in ()).throw(RuntimeError("broken"))),
    )

    assert config_cmd.config_effective() == 1
    err = capsys.readouterr().err
    assert "ERROR: broken" in err


def test_config_set_updates_typing_final_injection_mode(monkeypatch, tmp_path, capsys):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[typing]\ntyping_final_injection_mode = \"auto\"\n", encoding="utf-8")

    monkeypatch.setattr(config_cmd.Config, "config_path", classmethod(lambda cls: config_file))

    assert config_cmd.config_set("typing_final_injection_mode", "direct") == 0
    out = capsys.readouterr().out
    assert "OK set typing_final_injection_mode=direct" in out

    content = config_file.read_text(encoding="utf-8")
    assert 'typing_final_injection_mode = "direct"' in content
    assert "use_clipboard_for_final = false" in content


def test_config_set_rejects_invalid_value(capsys):
    assert config_cmd.config_set("typing_final_injection_mode", "invalid") == 1
    err = capsys.readouterr().err
    assert "typing_final_injection_mode must be one of" in err


def test_config_set_rejects_unsupported_key(capsys):
    assert config_cmd.config_set("unknown_key", "value") == 1
    err = capsys.readouterr().err
    assert "unsupported config key" in err


def test_run_control_stop_waits_for_processing_to_finish(monkeypatch, capsys):
    calls: list[str] = []

    def fake_send(command: str, _socket: str | None, timeout: float | None = None) -> str:
        calls.append(command)
        if command == "stop":
            return "OK stopped"
        if command == "status":
            # first poll returns processing, second poll returns idle
            status_calls = [c for c in calls if c == "status"]
            return "OK processing" if len(status_calls) == 1 else "OK idle"
        raise AssertionError(f"unexpected command {command}")

    monkeypatch.setattr(control_cmd, "send_control_command", fake_send)
    monkeypatch.setattr(control_cmd.time, "sleep", lambda _sec: None)

    cfg = Config()
    assert control_cmd.run_control("stop", cfg, wait_sec=1.0) == 0

    out = capsys.readouterr().out
    assert "OK stopped" in out
    assert calls.count("status") == 2


def test_run_control_toggle_waits_only_when_pre_state_is_recording(monkeypatch):
    calls: list[str] = []

    def fake_send(command: str, _socket: str | None, timeout: float | None = None) -> str:
        calls.append(command)
        if command == "status":
            status_calls = [c for c in calls if c == "status"]
            if len(status_calls) == 1:
                return "OK recording"
            return "OK idle"
        if command == "toggle":
            return "OK toggled"
        raise AssertionError(f"unexpected command {command}")

    monkeypatch.setattr(control_cmd, "send_control_command", fake_send)
    monkeypatch.setattr(control_cmd.time, "sleep", lambda _sec: None)

    cfg = Config()
    assert control_cmd.run_control("toggle", cfg, wait_sec=1.0) == 0
    assert calls.count("status") >= 2


def test_run_control_tts_command_skips_processing_wait(monkeypatch, capsys):
    calls: list[str] = []

    def fake_send(command: str, _socket: str | None, timeout: float | None = None) -> str:
        calls.append(command)
        if command == "tts_stop":
            return "OK tts stopped"
        if command == "status":
            return "OK idle"
        raise AssertionError(f"unexpected command {command}")

    monkeypatch.setattr(control_cmd, "send_control_command", fake_send)

    cfg = Config()
    assert control_cmd.run_control("tts_stop", cfg, wait_sec=1.0) == 0

    out = capsys.readouterr().out
    assert "OK tts stopped" in out
    assert calls == ["tts_stop"]


def test_run_control_error(monkeypatch, capsys):
    monkeypatch.setattr(
        control_cmd,
        "send_control_command",
        lambda _command, _socket, timeout=None: (_ for _ in ()).throw(RuntimeError("no socket")),
    )

    cfg = Config()
    assert control_cmd.run_control("status", cfg, wait_sec=0.0) == 1
    assert "ERROR: no socket" in capsys.readouterr().err


def test_diagnostics_text_and_error_status(monkeypatch, capsys):
    responses = {
        "status": RuntimeError("down"),
        "metrics": "OK {}",
    }

    def fake_send(command: str, _socket: str | None, timeout: float | None = None) -> str:
        value = responses[command]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(diagnostics_cmd, "send_control_command", fake_send)

    cfg = Config()
    assert diagnostics_cmd.diagnostics(cfg, json_output=False) == 1

    out = capsys.readouterr().out
    assert "status: ERROR: down" in out
    assert "metrics: OK {}" in out


def test_diagnostics_json_success(monkeypatch, capsys):
    monkeypatch.setattr(
        diagnostics_cmd,
        "send_control_command",
        lambda command, _socket, timeout=None: "OK recording" if command == "status" else "OK {}",
    )

    cfg = Config()
    assert diagnostics_cmd.diagnostics(cfg, json_output=True) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"].startswith("OK")
    assert payload["metrics"].startswith("OK")


def test_download_model_passes_nemo_model_name(monkeypatch, capsys):
    seen: dict[str, object] = {}

    class FakeBackend:
        @staticmethod
        def download_model(**kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(model_cmd, "get_backend_class", lambda _name: FakeBackend)

    cfg = Config(asr_backend="nemo")
    assert model_cmd.download_model(cfg) == 0
    assert seen["model_name"] == cfg.model_name
    assert "downloaded successfully" in capsys.readouterr().out.lower()


def test_download_model_passes_sherpa_args(monkeypatch):
    seen: dict[str, object] = {}

    class FakeBackend:
        @staticmethod
        def download_model(**kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(model_cmd, "get_backend_class", lambda _name: FakeBackend)

    cfg = Config(asr_backend="sherpa", sherpa_model_dir="/tmp/sherpa")
    assert model_cmd.download_model(cfg) == 0
    assert seen["model_name"] == cfg.sherpa_model_name
    assert seen["model_dir"] == "/tmp/sherpa"


def test_download_model_error(monkeypatch, capsys):
    class FakeBackend:
        @staticmethod
        def download_model(**_kwargs):
            raise RuntimeError("download failed")

    monkeypatch.setattr(model_cmd, "get_backend_class", lambda _name: FakeBackend)

    cfg = Config()
    assert model_cmd.download_model(cfg) == 1
    assert "ERROR: download failed" in capsys.readouterr().err


def test_run_welcome_wizard_returns_false_when_layer_shell_missing(monkeypatch, capsys):
    monkeypatch.setattr(
        wizard_cmd,
        "CDLL",
        lambda _lib: (_ for _ in ()).throw(OSError("missing")),
    )

    assert wizard_cmd.run_welcome_wizard() is False
    assert "libgtk4-layer-shell.so not found" in capsys.readouterr().err


def test_run_welcome_wizard_success(monkeypatch):
    class FakeWelcomeWizard:
        def __init__(self, *, force_reconfigure: bool):
            self.force_reconfigure = force_reconfigure
            self.completed = True

        def run(self, _arg):
            return None

    fake_module = types.SimpleNamespace(WelcomeWizard=FakeWelcomeWizard)

    monkeypatch.setattr(wizard_cmd, "CDLL", lambda _lib: None)
    monkeypatch.setitem(sys.modules, "shuvoice.wizard", fake_module)

    assert wizard_cmd.run_welcome_wizard(force_reconfigure=True) is True
