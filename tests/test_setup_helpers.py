from __future__ import annotations

from pathlib import Path

from shuvoice.cli.commands import setup as setup_cmd
from shuvoice.config import Config
from shuvoice.setup_helpers import (
    build_local_tts_setup_report,
    install_hints_for_backend,
    model_status_for_backend,
)


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
    monkeypatch.setattr(setup_cmd, "_detect_cuda_gpu", lambda: False)

    commands = setup_cmd._auto_install_commands("sherpa")

    assert commands[0][-1] == "python-sherpa-onnx-bin"
    assert commands[1][-1] == "python-sherpa-onnx"


def test_auto_install_commands_prefers_cuda_capable_provider_when_gpu_detected(monkeypatch):
    monkeypatch.setattr(setup_cmd, "_running_in_venv", lambda: False)
    monkeypatch.setattr(setup_cmd, "_detect_cuda_gpu", lambda: True)

    commands = setup_cmd._auto_install_commands("sherpa")

    assert commands[0][-1] == "python-sherpa-onnx"
    assert commands[1][-1] == "python-sherpa-onnx"
    assert commands[2][-1] == "python-sherpa-onnx-bin"


def test_auto_install_commands_venv_cuda_includes_gpu_build_and_compat_packages(monkeypatch):
    monkeypatch.setattr(setup_cmd, "_running_in_venv", lambda: True)
    monkeypatch.setattr(setup_cmd, "_detect_cuda_gpu", lambda: True)
    monkeypatch.setattr(setup_cmd, "_detect_cuda_architectures", lambda: "89")
    monkeypatch.setattr(
        setup_cmd.shutil,
        "which",
        lambda exe: "/usr/bin/uv" if exe == "uv" else None,
    )

    commands = setup_cmd._auto_install_commands("sherpa")

    assert any("SHERPA_ONNX_CMAKE_ARGS=-DSHERPA_ONNX_ENABLE_GPU=ON" in part for cmd in commands for part in cmd)
    assert any("nvidia-cublas-cu12" in cmd for cmd in commands)
    assert any("nvidia-cudnn-cu12" in cmd for cmd in commands)


def test_auto_install_commands_venv_prefers_uv_pip(monkeypatch):
    monkeypatch.setattr(setup_cmd, "_running_in_venv", lambda: True)
    monkeypatch.setattr(setup_cmd, "_detect_cuda_gpu", lambda: False)
    monkeypatch.setattr(
        setup_cmd.shutil,
        "which",
        lambda exe: "/usr/bin/uv" if exe == "uv" else None,
    )

    commands = setup_cmd._auto_install_commands("sherpa")
    assert any(cmd[:3] == ["uv", "pip", "install"] for cmd in commands)


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


def test_model_status_mentions_parakeet_streaming_enabled(monkeypatch):
    cfg = Config(
        asr_backend="sherpa",
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        sherpa_decode_mode="streaming",
        sherpa_enable_parakeet_streaming=True,
    )

    monkeypatch.setattr(
        "shuvoice.setup_helpers._is_complete_sherpa_model_dir",
        lambda _path: False,
    )

    status = model_status_for_backend(cfg)

    assert "Parakeet streaming mode enabled" in status


def test_build_local_tts_setup_report_marks_missing_binary_and_path(monkeypatch):
    monkeypatch.setattr("shuvoice.setup_helpers.find_piper_binary", lambda: None)

    report = build_local_tts_setup_report(Config(tts_backend="local"))

    assert report.binary_present is False
    assert report.model_dir is None
    assert report.missing_artifacts == ("tts_local_model_path is not configured",)



def test_build_local_tts_setup_report_lists_installed_voices(monkeypatch, tmp_path: Path):
    (tmp_path / "en_US-amy-medium.onnx").write_bytes(b"model")
    (tmp_path / "en_US-amy-medium.onnx.json").write_text('{"audio": {"sample_rate": 22050}}')
    monkeypatch.setattr("shuvoice.setup_helpers.find_piper_binary", lambda: "piper-tts")

    report = build_local_tts_setup_report(
        Config(
            tts_backend="local",
            tts_local_model_path=str(tmp_path),
            tts_local_voice="en_US-amy-medium",
        )
    )

    assert report.binary_present is True
    assert report.binary_name == "piper-tts"
    assert report.installed_voices == ("en_US-amy-medium",)
    assert report.missing_artifacts == tuple()
