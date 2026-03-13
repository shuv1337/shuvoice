from __future__ import annotations

from pathlib import Path

from shuvoice.cli.commands import setup as setup_cmd
from shuvoice.config import Config
from shuvoice.setup_helpers import (
    build_local_tts_setup_report,
    build_melotts_setup_report,
    format_melotts_report,
    install_hints_for_backend,
    melotts_install_commands,
    melotts_venv_valid,
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

    assert any(
        "SHERPA_ONNX_CMAKE_ARGS=-DSHERPA_ONNX_ENABLE_GPU=ON" in part
        for cmd in commands
        for part in cmd
    )
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


# ------------------------------------------------------------------ #
# MeloTTS helpers                                                     #
# ------------------------------------------------------------------ #


def test_build_melotts_setup_report_missing_venv(tmp_path: Path, monkeypatch):
    """When the MeloTTS venv does not exist the report flags it as missing."""
    venv_dir = tmp_path / "melotts-venv"
    cfg = Config(tts_backend="melotts", tts_melotts_venv_path=str(venv_dir))

    report = build_melotts_setup_report(cfg)

    assert report.venv_present is False
    assert report.python_executable is False
    assert "not installed" in report.model_status
    assert len(report.missing_dependencies) > 0


def test_build_melotts_setup_report_ready_venv(tmp_path: Path):
    """When the MeloTTS venv exists with a working python the report is positive."""
    venv_dir = tmp_path / "melotts-venv"
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    python_bin = bin_dir / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)

    # Also create the helper script where the backend expects it
    import shuvoice.tts_melotts as melotts_mod

    helper_path = Path(melotts_mod.__file__).with_name("melo_helper.py")
    helper_existed = helper_path.exists()

    if not helper_existed:
        helper_path.write_text("# stub")

    try:
        cfg = Config(tts_backend="melotts", tts_melotts_venv_path=str(venv_dir))
        report = build_melotts_setup_report(cfg)

        assert report.venv_present is True
        assert report.python_executable is True
        assert report.missing_dependencies == ()
        assert "ready" in report.model_status
    finally:
        if not helper_existed:
            helper_path.unlink(missing_ok=True)


def test_melotts_venv_valid_true(tmp_path: Path):
    """An executable python inside the venv makes it valid."""
    venv_dir = tmp_path / "venv"
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    python_bin = bin_dir / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)

    assert melotts_venv_valid(venv_dir) is True


def test_melotts_venv_valid_false_missing(tmp_path: Path):
    """No venv directory → not valid."""
    assert melotts_venv_valid(tmp_path / "nonexistent") is False


def test_melotts_venv_valid_false_not_executable(tmp_path: Path):
    """Python exists but is not executable → not valid."""
    venv_dir = tmp_path / "venv"
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    python_bin = bin_dir / "python"
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o644)

    assert melotts_venv_valid(venv_dir) is False


def test_melotts_install_commands_default_structure(tmp_path: Path):
    """Install commands follow expected create-venv → pip install → unidic pattern."""
    commands = melotts_install_commands(tmp_path / "melotts-venv")

    assert len(commands) == 4

    # Step 1: uv python install 3.12
    assert commands[0] == ["uv", "python", "install", "3.12"]

    # Step 2: uv venv --python 3.12 <path>
    assert commands[1][:3] == ["uv", "venv", "--python"]
    assert "3.12" in commands[1]
    assert str(tmp_path / "melotts-venv") in commands[1]

    # Step 3: pip install melotts
    python_bin = str(tmp_path / "melotts-venv" / "bin" / "python")
    assert commands[2] == [python_bin, "-m", "pip", "install", "melotts"]

    # Step 4: unidic download
    assert commands[3] == [python_bin, "-m", "unidic", "download"]


def test_melotts_install_commands_idempotent():
    """Calling install_commands multiple times returns the same sequence."""
    a = melotts_install_commands()
    b = melotts_install_commands()

    assert a == b
    # The default path uses the expanded home dir
    assert "melotts-venv" in str(a[1])


def test_format_melotts_report_missing_venv(tmp_path: Path):
    """Format report for missing venv includes 'missing'."""
    from shuvoice.setup_helpers import MeloTTSSetupReport

    report = MeloTTSSetupReport(
        venv_present=False,
        venv_dir=tmp_path / "melotts-venv",
        python_executable=False,
        missing_dependencies=("MeloTTS venv directory does not exist",),
        model_status="not installed",
    )

    text = format_melotts_report(report)

    assert "missing" in text.lower()
    assert "MeloTTS" in text
    assert "not installed" in text


def test_format_melotts_report_ready(tmp_path: Path):
    """Format report for ready venv includes 'present' and 'ready'."""
    from shuvoice.setup_helpers import MeloTTSSetupReport

    report = MeloTTSSetupReport(
        venv_present=True,
        venv_dir=tmp_path / "melotts-venv",
        python_executable=True,
        missing_dependencies=(),
        model_status="ready",
    )

    text = format_melotts_report(report)

    assert "present" in text.lower()
    assert "executable" in text.lower()
    assert "ready" in text
