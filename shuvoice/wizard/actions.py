"""Wizard finish actions (config + marker + optional keybind/model setup)."""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..asr import get_backend_class
from ..config import Config
from ..wizard_state import DEFAULT_SHERPA_MODEL_NAME, needs_wizard, write_config, write_marker
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
    - ``incompatible_streaming``
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

    provider_note = ""
    parakeet_streaming_requested = False

    def _with_provider_note(message: str) -> str:
        return f"{message}{provider_note}" if provider_note else message

    def _check_parakeet_streaming_compatibility(
        backend_cls: type,
        cfg: Config,
        *,
        phase: str,
    ) -> tuple[str, str] | None:
        if cfg.asr_backend != "sherpa":
            return None

        model_name = str(getattr(cfg, "sherpa_model_name", "")).strip().lower()
        if "parakeet" not in model_name:
            return None

        allow_parakeet_streaming = bool(getattr(cfg, "sherpa_enable_parakeet_streaming", False))
        if cfg.resolved_sherpa_decode_mode != "streaming" or not allow_parakeet_streaming:
            return None

        checker = getattr(backend_cls, "_parakeet_streaming_model_compatible", None)
        if not callable(checker):
            return None

        try:
            compatible, detail = checker(cfg)
        except Exception as exc:  # noqa: BLE001
            compatible, detail = False, f"compatibility probe failed ({exc})"

        if compatible:
            return None

        _emit(1.0, "Parakeet streaming is incompatible on this runtime")
        return (
            "incompatible_streaming",
            _with_provider_note(
                "Parakeet streaming profile is incompatible with this Sherpa runtime "
                f"({detail}; phase={phase}). "
                "Wizard will switch to Zipformer streaming profile to keep streaming available."
            ),
        )

    try:
        cfg = Config.load()
        cfg.asr_backend = asr_backend
        if sherpa_model_name is not None:
            cfg.sherpa_model_name = sherpa_model_name

        # Re-validate after overrides.
        cfg.__post_init__()

        backend_cls = get_backend_class(cfg.asr_backend)

        if cfg.asr_backend == "sherpa":
            requested_provider = cfg.sherpa_provider
            startup_warnings = getattr(backend_cls, "startup_warnings", None)
            warnings = startup_warnings(cfg, apply_fixes=True) if callable(startup_warnings) else []
            if warnings:
                for warning in warnings:
                    log.warning("Wizard Sherpa runtime warning: %s", warning)
                _emit(None, "Sherpa runtime warning detected; see logs for details")

            if requested_provider != cfg.sherpa_provider:
                provider_note = (
                    f" Requested sherpa_provider='{requested_provider}' but runtime will use "
                    f"'{cfg.sherpa_provider}'."
                )

            if "parakeet" in cfg.sherpa_model_name.lower():
                allow_parakeet_streaming = bool(
                    getattr(cfg, "sherpa_enable_parakeet_streaming", False)
                )
                if (
                    cfg.resolved_sherpa_decode_mode != "offline_instant"
                    and not allow_parakeet_streaming
                ):
                    _emit(1.0, "Parakeet streaming is disabled by default")
                    return (
                        "error",
                        _with_provider_note(
                            "Parakeet model selection requires sherpa_decode_mode='offline_instant' "
                            "(or instant_mode=true with sherpa_decode_mode='auto'). "
                            "To use Parakeet in streaming mode, set "
                            "sherpa_enable_parakeet_streaming=true and "
                            "sherpa_decode_mode='streaming'."
                        ),
                    )

            parakeet_streaming_requested = bool(
                "parakeet" in cfg.sherpa_model_name.lower()
                and cfg.resolved_sherpa_decode_mode == "streaming"
                and getattr(cfg, "sherpa_enable_parakeet_streaming", False)
            )
            if parakeet_streaming_requested:
                incompatible = _check_parakeet_streaming_compatibility(
                    backend_cls,
                    cfg,
                    phase="pre-download",
                )
                if incompatible is not None:
                    return incompatible
    except Exception as exc:  # noqa: BLE001
        _emit(1.0, "Model download setup failed")
        return "error", f"Could not prepare model download: {exc}"

    if _is_cancelled():
        _emit(1.0, "Model download cancelled")
        return "cancelled", _with_provider_note("Model download cancelled by user.")

    if not backend_cls.capabilities.supports_model_download:
        _emit(1.0, "Model download skipped (lazy backend)")
        return "skipped", _with_provider_note("Selected backend downloads models lazily at runtime.")

    missing = backend_cls.dependency_errors()
    if missing:
        _emit(1.0, "Model download skipped (missing dependencies)")
        return (
            "skipped_missing_deps",
            _with_provider_note(
                "Dependencies for selected backend are missing. Run `shuvoice setup` to install them."
            ),
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
        return "cancelled", _with_provider_note("Model download cancelled by user.")

    try:
        backend_cls.download_model(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if _is_cancelled() or "cancelled" in str(exc).lower():
            _emit(1.0, "Model download cancelled")
            return "cancelled", _with_provider_note("Model download cancelled by user.")

        log.warning("Wizard model download failed: %s", exc)
        _emit(1.0, "Model download failed")
        return "error", _with_provider_note(f"Model download failed: {exc}")

    if cfg.asr_backend == "sherpa" and parakeet_streaming_requested:
        incompatible = _check_parakeet_streaming_compatibility(
            backend_cls,
            cfg,
            phase="post-download",
        )
        if incompatible is not None:
            return incompatible

    _emit(1.0, "Model download completed")
    return "downloaded", _with_provider_note("Model download completed.")


def finish_setup(
    asr_backend: str,
    keybind_id: str,
    *,
    auto_add_keybind: bool,
    overwrite_existing: bool,
    sherpa_model_name: str | None = None,
    sherpa_enable_parakeet_streaming: bool = False,
    typing_final_injection_mode: str = "auto",
    auto_download_model: bool = True,
) -> tuple[str, str, str, str]:
    """Persist wizard selections and optionally configure keybind/model download."""
    write_config(
        asr_backend,
        overwrite_existing=overwrite_existing,
        sherpa_model_name=sherpa_model_name,
        sherpa_enable_parakeet_streaming=sherpa_enable_parakeet_streaming,
        typing_final_injection_mode=typing_final_injection_mode,
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

    if model_status == "incompatible_streaming" and asr_backend == "sherpa":
        write_config(
            "sherpa",
            overwrite_existing=True,
            sherpa_model_name=DEFAULT_SHERPA_MODEL_NAME,
            sherpa_enable_parakeet_streaming=False,
            typing_final_injection_mode=typing_final_injection_mode,
        )
        model_message = (
            f"{model_message} "
            "Applied fallback profile: Streaming (Zipformer default model)."
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
