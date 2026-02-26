from __future__ import annotations

from types import SimpleNamespace

from shuvoice.asr import get_backend_class
from shuvoice.cli.commands import preflight as preflight_cmd
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


def test_run_setup_reports_sherpa_decode_mode_and_provider(capsys, monkeypatch):
    report = BackendSetupReport(
        backend="sherpa",
        missing_dependencies=(),
        install_hints=(),
        model_status="present (/tmp/model)",
    )
    monkeypatch.setattr(setup_cmd, "build_backend_setup_report", lambda _cfg: report)

    class _DummySherpaBackend:
        capabilities = SimpleNamespace(supports_model_download=True)

        @staticmethod
        def startup_warnings(cfg, *, apply_fixes: bool = False):
            if apply_fixes and cfg.sherpa_provider == "cuda":
                cfg.sherpa_provider = "cpu"
                return ["CUDA provider unavailable; falling back to CPU"]
            return []

        @staticmethod
        def startup_errors(_cfg):
            return []

        @staticmethod
        def _looks_like_parakeet_model(cfg):
            return "parakeet" in cfg.sherpa_model_name.lower()

    monkeypatch.setattr(setup_cmd, "get_backend_class", lambda _name: _DummySherpaBackend)

    cfg = Config(
        asr_backend="sherpa",
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        sherpa_provider="cuda",
        sherpa_decode_mode="auto",
        instant_mode=True,
    )

    code = setup_cmd.run_setup(
        cfg,
        install_missing=False,
        skip_model_download=True,
        skip_preflight=True,
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "[INFO] Sherpa decode mode: offline_instant" in out
    assert "[INFO] Sherpa provider: requested=cuda effective=cpu" in out
    assert "[INFO] Sherpa Parakeet model: yes" in out
    assert "[INFO] Sherpa Parakeet runnable: yes" in out


def test_run_setup_reports_clear_failure_for_parakeet_streaming(capsys, monkeypatch):
    report = BackendSetupReport(
        backend="sherpa",
        missing_dependencies=(),
        install_hints=(),
        model_status="present (/tmp/model)",
    )
    monkeypatch.setattr(setup_cmd, "build_backend_setup_report", lambda _cfg: report)

    sherpa_cls = get_backend_class("sherpa")
    monkeypatch.setattr(setup_cmd, "get_backend_class", lambda _name: sherpa_cls)

    cfg = Config(
        asr_backend="sherpa",
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        sherpa_decode_mode="streaming",
    )

    code = setup_cmd.run_setup(
        cfg,
        install_missing=False,
        skip_model_download=True,
        skip_preflight=True,
    )

    assert code == DEPENDENCY_EXIT_CODE
    out = capsys.readouterr().out
    assert "[INFO] Sherpa decode mode: streaming" in out
    assert "[FAIL] Backend runtime compatibility" in out
    assert "offline_instant" in out


def test_run_setup_allows_parakeet_streaming_when_enabled(capsys, monkeypatch):
    report = BackendSetupReport(
        backend="sherpa",
        missing_dependencies=(),
        install_hints=(),
        model_status="present (/tmp/model)",
    )
    monkeypatch.setattr(setup_cmd, "build_backend_setup_report", lambda _cfg: report)

    sherpa_cls = get_backend_class("sherpa")
    monkeypatch.setattr(setup_cmd, "get_backend_class", lambda _name: sherpa_cls)
    monkeypatch.setattr(
        sherpa_cls,
        "_parakeet_streaming_model_compatible",
        classmethod(lambda cls, _cfg: (True, "compatible")),
    )

    cfg = Config(
        asr_backend="sherpa",
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        sherpa_decode_mode="streaming",
        sherpa_enable_parakeet_streaming=True,
    )

    code = setup_cmd.run_setup(
        cfg,
        install_missing=False,
        skip_model_download=True,
        skip_preflight=True,
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "[INFO] Sherpa decode mode: streaming" in out
    assert "[PASS] Backend runtime compatibility" in out


def test_run_setup_blocks_parakeet_streaming_when_runtime_incompatible(capsys, monkeypatch):
    report = BackendSetupReport(
        backend="sherpa",
        missing_dependencies=(),
        install_hints=(),
        model_status="present (/tmp/model)",
    )
    monkeypatch.setattr(setup_cmd, "build_backend_setup_report", lambda _cfg: report)

    sherpa_cls = get_backend_class("sherpa")
    monkeypatch.setattr(setup_cmd, "get_backend_class", lambda _name: sherpa_cls)
    monkeypatch.setattr(
        sherpa_cls,
        "_parakeet_streaming_model_compatible",
        classmethod(lambda cls, _cfg: (False, "missing window_size")),
    )

    cfg = Config(
        asr_backend="sherpa",
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        sherpa_decode_mode="streaming",
        sherpa_enable_parakeet_streaming=True,
    )

    code = setup_cmd.run_setup(
        cfg,
        install_missing=False,
        skip_model_download=True,
        skip_preflight=True,
    )

    assert code == DEPENDENCY_EXIT_CODE
    out = capsys.readouterr().out
    assert "missing window_size" in out


def test_preflight_reports_sherpa_decode_mode_status(capsys, monkeypatch):
    class _DummySherpaBackend:
        capabilities = SimpleNamespace(
            supports_gpu=True,
            expected_chunking="streaming",
            wants_raw_audio=False,
        )

        @staticmethod
        def dependency_errors():
            return []

        @staticmethod
        def startup_warnings(cfg, *, apply_fixes: bool = False):
            if apply_fixes and cfg.sherpa_provider == "cuda":
                cfg.sherpa_provider = "cpu"
                return ["CUDA provider unavailable; falling back to CPU"]
            return []

        @staticmethod
        def startup_errors(_cfg):
            return []

        @staticmethod
        def _looks_like_parakeet_model(cfg):
            return "parakeet" in cfg.sherpa_model_name.lower()

    monkeypatch.setattr(preflight_cmd, "get_backend_class", lambda _name: _DummySherpaBackend)
    monkeypatch.setattr(preflight_cmd.importlib, "import_module", lambda _name: object())
    monkeypatch.setattr(preflight_cmd.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr(preflight_cmd, "CDLL", lambda _lib: None)

    cfg = Config(
        asr_backend="sherpa",
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        sherpa_decode_mode="auto",
        instant_mode=True,
        sherpa_provider="cuda",
    )

    ready = preflight_cmd.run_preflight(cfg)

    assert ready is True
    out = capsys.readouterr().out
    assert "sherpa_decode_mode=offline_instant" in out
    assert "sherpa_provider=cuda->cpu" in out
    assert "parakeet_runnable=yes" in out
