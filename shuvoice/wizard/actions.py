"""Wizard finish actions (config + marker + optional keybind/model setup)."""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..asr import get_backend_class
from ..config import Config
from ..wizard_state import needs_wizard, write_config, write_marker
from .hyprland import setup_keybind

log = logging.getLogger(__name__)


def maybe_download_model(
    asr_backend: str,
    *,
    sherpa_model_name: str | None = None,
    progress_callback: Callable[[float | None, str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[str, str]:
    """Best-effort model download for wizard-selected backend/model.

    Returns ``(status, message)`` where status is one of:
    - ``downloaded``
    - ``skipped``
    - ``skipped_missing_deps``
    - ``cancelled``
    - ``error``

    ``progress_callback`` receives ``(fraction, message)`` where fraction is
    ``0.0..1.0`` when known, or ``None`` for indeterminate progress.

    ``cancel_requested`` can be provided to abort long downloads.
    """

    def _emit(fraction: float | None, message: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(fraction, message)
        except Exception:  # noqa: BLE001
            log.debug("Wizard progress callback failed", exc_info=True)

    def _is_cancelled() -> bool:
        if cancel_requested is None:
            return False
        try:
            return bool(cancel_requested())
        except Exception:  # noqa: BLE001
            log.debug("Wizard cancel callback failed", exc_info=True)
            return False

    if _is_cancelled():
        _emit(1.0, "Model download cancelled")
        return "cancelled", "Model download cancelled by user."

    _emit(0.0, "Preparing model download…")

    try:
        cfg = Config.load()
        cfg.asr_backend = asr_backend
        if sherpa_model_name is not None:
            cfg.sherpa_model_name = sherpa_model_name

        # Re-validate after overrides.
        cfg.__post_init__()

        backend_cls = get_backend_class(cfg.asr_backend)
    except Exception as exc:  # noqa: BLE001
        _emit(1.0, "Model download setup failed")
        return "error", f"Could not prepare model download: {exc}"

    if _is_cancelled():
        _emit(1.0, "Model download cancelled")
        return "cancelled", "Model download cancelled by user."

    if not backend_cls.capabilities.supports_model_download:
        _emit(1.0, "Model download skipped (lazy backend)")
        return "skipped", "Selected backend downloads models lazily at runtime."

    missing = backend_cls.dependency_errors()
    if missing:
        _emit(1.0, "Model download skipped (missing dependencies)")
        return (
            "skipped_missing_deps",
            "Dependencies for selected backend are missing. Run `shuvoice setup` to install them.",
        )

    kwargs: dict[str, object] = {}
    if cfg.asr_backend == "nemo":
        kwargs["model_name"] = cfg.model_name
    elif cfg.asr_backend == "sherpa":
        kwargs["model_name"] = cfg.sherpa_model_name
        kwargs["model_dir"] = cfg.sherpa_model_dir
        kwargs["progress_callback"] = progress_callback
        kwargs["cancel_check"] = cancel_requested

    if cfg.asr_backend == "nemo":
        _emit(None, f"Downloading NeMo model: {cfg.model_name}")
    elif cfg.asr_backend == "sherpa":
        _emit(0.0, f"Downloading Sherpa model: {cfg.sherpa_model_name}")

    if _is_cancelled():
        _emit(1.0, "Model download cancelled")
        return "cancelled", "Model download cancelled by user."

    try:
        backend_cls.download_model(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if _is_cancelled() or "cancelled" in str(exc).lower():
            _emit(1.0, "Model download cancelled")
            return "cancelled", "Model download cancelled by user."

        log.warning("Wizard model download failed: %s", exc)
        _emit(1.0, "Model download failed")
        return "error", f"Model download failed: {exc}"

    _emit(1.0, "Model download completed")
    return "downloaded", "Model download completed."


def finish_setup(
    asr_backend: str,
    keybind_id: str,
    *,
    auto_add_keybind: bool,
    overwrite_existing: bool,
    sherpa_model_name: str | None = None,
    auto_download_model: bool = True,
) -> tuple[str, str, str, str]:
    """Persist wizard selections and optionally configure keybind/model download."""
    write_config(
        asr_backend,
        overwrite_existing=overwrite_existing,
        sherpa_model_name=sherpa_model_name,
    )

    keybind_status = "not_attempted"
    keybind_message = "automatic keybind setup disabled"
    if auto_add_keybind:
        keybind_status, keybind_message = setup_keybind(keybind_id)

    model_status = "not_attempted"
    model_message = "automatic model download disabled"
    if auto_download_model:
        model_status, model_message = maybe_download_model(
            asr_backend,
            sherpa_model_name=sherpa_model_name,
        )

    write_marker()
    return keybind_status, keybind_message, model_status, model_message


__all__ = [
    "finish_setup",
    "maybe_download_model",
    "needs_wizard",
    "write_config",
    "write_marker",
]
