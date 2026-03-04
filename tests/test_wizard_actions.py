from __future__ import annotations

from types import SimpleNamespace

from shuvoice.config import Config
from shuvoice.wizard.actions import maybe_download_model


class _DownloadCapableBackend:
    capabilities = SimpleNamespace(supports_model_download=True)
    _last_kwargs: dict[str, object] | None = None

    @staticmethod
    def dependency_errors() -> list[str]:
        return []

    @classmethod
    def download_model(cls, **kwargs):
        cls._last_kwargs = kwargs


class _MissingDepsBackend:
    capabilities = SimpleNamespace(supports_model_download=True)

    @staticmethod
    def dependency_errors() -> list[str]:
        return ["missing backend dependency"]

    @staticmethod
    def download_model(**_kwargs):
        raise AssertionError("download_model should not run when dependencies are missing")


class _LazyBackend:
    capabilities = SimpleNamespace(supports_model_download=False)

    @staticmethod
    def dependency_errors() -> list[str]:
        return []


class _ProgressBackend:
    capabilities = SimpleNamespace(supports_model_download=True)

    @staticmethod
    def dependency_errors() -> list[str]:
        return []

    @staticmethod
    def download_model(**kwargs):
        callback = kwargs.get("progress_callback")
        if callable(callback):
            callback(0.5, "halfway")


class _CancelableBackend:
    capabilities = SimpleNamespace(supports_model_download=True)

    @staticmethod
    def dependency_errors() -> list[str]:
        return []

    @staticmethod
    def download_model(**kwargs):
        cancel_check = kwargs.get("cancel_check")
        if callable(cancel_check) and cancel_check():
            raise RuntimeError("Model download cancelled")


class _IncompatibleParakeetStreamingBackend(_DownloadCapableBackend):
    @staticmethod
    def _parakeet_streaming_model_compatible(_cfg: Config) -> tuple[bool, str]:
        return False, "encoder metadata missing window_size"


class _PostDownloadIncompatibleParakeetBackend(_DownloadCapableBackend):
    _compat_calls = 0

    @classmethod
    def _parakeet_streaming_model_compatible(cls, _cfg: Config) -> tuple[bool, str]:
        cls._compat_calls += 1
        if cls._compat_calls == 1:
            return True, "model directory not present yet"
        return False, "encoder metadata missing window_size"


class _CudaWarningBackend(_DownloadCapableBackend):
    @staticmethod
    def startup_warnings(cfg: Config, *, apply_fixes: bool = False) -> list[str]:
        if cfg.sherpa_provider != "cuda":
            return []

        warning = "CUDAExecutionProvider missing"
        if apply_fixes:
            cfg.sherpa_provider = "cpu"
            return [f"{warning}; falling back to CPU"]
        return [warning]


def test_maybe_download_model_sherpa_uses_selected_model_name(monkeypatch):
    monkeypatch.setattr(
        "shuvoice.wizard.actions.Config.load",
        classmethod(
            lambda cls: Config(
                asr_backend="sherpa",
                instant_mode=True,
                sherpa_decode_mode="offline_instant",
            )
        ),
    )
    monkeypatch.setattr(
        "shuvoice.wizard.actions.get_backend_class", lambda _name: _DownloadCapableBackend
    )

    status, _message = maybe_download_model(
        "sherpa",
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    )

    assert status == "downloaded"
    assert _DownloadCapableBackend._last_kwargs is not None
    assert (
        _DownloadCapableBackend._last_kwargs["model_name"]
        == "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    )
    assert "progress_callback" in _DownloadCapableBackend._last_kwargs
    assert _DownloadCapableBackend._last_kwargs["progress_callback"] is None
    assert "cancel_check" in _DownloadCapableBackend._last_kwargs
    assert _DownloadCapableBackend._last_kwargs["cancel_check"] is None


def test_maybe_download_model_rejects_parakeet_without_offline_mode(monkeypatch):
    monkeypatch.setattr("shuvoice.wizard.actions.Config.load", classmethod(lambda cls: Config()))
    monkeypatch.setattr(
        "shuvoice.wizard.actions.get_backend_class", lambda _name: _DownloadCapableBackend
    )

    status, message = maybe_download_model(
        "sherpa",
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    )

    assert status == "error"
    assert "offline_instant" in message


def test_maybe_download_model_allows_parakeet_streaming_when_enabled(monkeypatch):
    monkeypatch.setattr(
        "shuvoice.wizard.actions.Config.load",
        classmethod(
            lambda cls: Config(
                asr_backend="sherpa",
                sherpa_decode_mode="streaming",
                sherpa_enable_parakeet_streaming=True,
            )
        ),
    )
    monkeypatch.setattr(
        "shuvoice.wizard.actions.get_backend_class", lambda _name: _DownloadCapableBackend
    )

    status, _message = maybe_download_model(
        "sherpa",
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    )

    assert status == "downloaded"


def test_maybe_download_model_marks_incompatible_parakeet_streaming(monkeypatch):
    monkeypatch.setattr(
        "shuvoice.wizard.actions.Config.load",
        classmethod(
            lambda cls: Config(
                asr_backend="sherpa",
                sherpa_decode_mode="streaming",
                sherpa_enable_parakeet_streaming=True,
            )
        ),
    )
    monkeypatch.setattr(
        "shuvoice.wizard.actions.get_backend_class",
        lambda _name: _IncompatibleParakeetStreamingBackend,
    )

    status, message = maybe_download_model(
        "sherpa",
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    )

    assert status == "incompatible_streaming"
    assert "Zipformer streaming profile" in message


def test_maybe_download_model_marks_post_download_incompatibility(monkeypatch):
    _PostDownloadIncompatibleParakeetBackend._compat_calls = 0
    _PostDownloadIncompatibleParakeetBackend._last_kwargs = None

    monkeypatch.setattr(
        "shuvoice.wizard.actions.Config.load",
        classmethod(
            lambda cls: Config(
                asr_backend="sherpa",
                sherpa_decode_mode="streaming",
                sherpa_enable_parakeet_streaming=True,
            )
        ),
    )
    monkeypatch.setattr(
        "shuvoice.wizard.actions.get_backend_class",
        lambda _name: _PostDownloadIncompatibleParakeetBackend,
    )

    status, _message = maybe_download_model(
        "sherpa",
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    )

    assert status == "incompatible_streaming"
    assert _PostDownloadIncompatibleParakeetBackend._last_kwargs is not None


def test_maybe_download_model_skips_when_deps_missing(monkeypatch):
    monkeypatch.setattr("shuvoice.wizard.actions.Config.load", classmethod(lambda cls: Config()))
    monkeypatch.setattr(
        "shuvoice.wizard.actions.get_backend_class", lambda _name: _MissingDepsBackend
    )

    status, message = maybe_download_model("sherpa")

    assert status == "skipped_missing_deps"
    assert "missing" in message


def test_maybe_download_model_skips_for_lazy_backends(monkeypatch):
    monkeypatch.setattr("shuvoice.wizard.actions.Config.load", classmethod(lambda cls: Config()))
    monkeypatch.setattr("shuvoice.wizard.actions.get_backend_class", lambda _name: _LazyBackend)

    status, message = maybe_download_model("moonshine")

    assert status == "skipped"
    assert "lazily" in message


def test_maybe_download_model_reports_progress(monkeypatch):
    monkeypatch.setattr(
        "shuvoice.wizard.actions.Config.load",
        classmethod(
            lambda cls: Config(
                asr_backend="sherpa",
                instant_mode=True,
                sherpa_decode_mode="offline_instant",
            )
        ),
    )
    monkeypatch.setattr("shuvoice.wizard.actions.get_backend_class", lambda _name: _ProgressBackend)

    events: list[tuple[float | None, str]] = []

    status, _message = maybe_download_model(
        "sherpa",
        sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        progress_callback=lambda fraction, text: events.append((fraction, text)),
    )

    assert status == "downloaded"
    assert any("Preparing" in text for _, text in events)
    assert any(text == "halfway" for _, text in events)
    assert any("completed" in text.lower() for _, text in events)


def test_maybe_download_model_auto_install_attempts_missing_dependencies(monkeypatch):
    monkeypatch.setattr("shuvoice.wizard.actions.Config.load", classmethod(lambda cls: Config()))
    monkeypatch.setattr(
        "shuvoice.wizard.actions.get_backend_class", lambda _name: _MissingDepsBackend
    )

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        "shuvoice.wizard.actions._attempt_auto_install_backend",
        lambda backend, *, prefer_cuda: calls.append((backend, prefer_cuda)) or False,
    )

    status, message = maybe_download_model("sherpa", auto_install_missing=True)

    assert status == "skipped_missing_deps"
    assert calls == [("sherpa", False)]
    assert "attempted" in message.lower()


def test_maybe_download_model_auto_install_prefers_cuda_runtime_path(monkeypatch):
    monkeypatch.setattr(
        "shuvoice.wizard.actions.Config.load",
        classmethod(lambda cls: Config(asr_backend="sherpa", sherpa_provider="cuda")),
    )
    monkeypatch.setattr(
        "shuvoice.wizard.actions.get_backend_class", lambda _name: _CudaWarningBackend
    )

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        "shuvoice.wizard.actions._attempt_auto_install_backend",
        lambda backend, *, prefer_cuda: calls.append((backend, prefer_cuda)) or False,
    )

    status, _message = maybe_download_model("sherpa", auto_install_missing=True)

    assert status == "downloaded"
    assert calls == [("sherpa", True)]


def test_maybe_download_model_cancelled_before_start(monkeypatch):
    monkeypatch.setattr("shuvoice.wizard.actions.Config.load", classmethod(lambda cls: Config()))
    monkeypatch.setattr(
        "shuvoice.wizard.actions.get_backend_class", lambda _name: _DownloadCapableBackend
    )

    status, message = maybe_download_model(
        "sherpa",
        cancel_requested=lambda: True,
    )

    assert status == "cancelled"
    assert "cancelled" in message.lower()


def test_maybe_download_model_cancelled_during_download(monkeypatch):
    monkeypatch.setattr("shuvoice.wizard.actions.Config.load", classmethod(lambda cls: Config()))
    monkeypatch.setattr(
        "shuvoice.wizard.actions.get_backend_class", lambda _name: _CancelableBackend
    )

    checks = iter([False, False, False, True])

    status, message = maybe_download_model(
        "sherpa",
        cancel_requested=lambda: next(checks, True),
    )

    assert status == "cancelled"
    assert "cancelled" in message.lower()
