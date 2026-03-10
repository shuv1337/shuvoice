"""First-run welcome wizard for ShuVoice.

Presents a guided setup dialog on the first launch, allowing the user
to select an ASR backend and review the default IPC push-to-talk flow.
Writes ``config.toml`` and a ``.wizard-done`` marker so it only runs once.

Uses the Hyprland layer-shell system for visual consistency with the
rest of ShuVoice.

IMPORTANT: ctypes.CDLL('libgtk4-layer-shell.so') must be called
before this module is imported. See __main__.py.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gtk4LayerShell", "1.0")
from gi.repository import GLib, Gtk
from gi.repository import Gtk4LayerShell as LayerShell

from ..wizard_state import (
    ASR_BACKENDS,
    KEYBIND_PRESETS,
    DEFAULT_FINAL_INJECTION_MODE,
    DEFAULT_KEYBIND_ID,
    DEFAULT_SHERPA_MODEL_NAME,
    DEFAULT_TTS_BACKEND,
    FINAL_INJECTION_MODES,
    PARAKEET_TDT_V3_INT8_MODEL_NAME,
    TTS_BACKENDS,
    default_tts_voice_for_backend,
)
from .actions import maybe_download_model, needs_wizard, write_config, write_marker
from .flow import summary_text
from .hyprland import (
    KeybindSetupStatus,
    format_hyprland_bind,
    format_hyprland_bind_for_keybind,
    setup_keybind as auto_add_hyprland_keybind,
)
from .ui import find_logo, setup_css

log = logging.getLogger(__name__)

# Wizard UX default: stable Parakeet instant profile.
DEFAULT_WIZARD_SHERPA_MODEL_NAME = PARAKEET_TDT_V3_INT8_MODEL_NAME

# Re-export for __main__.py and backward compatibility.
__all__ = [
    "WelcomeWizard",
    "needs_wizard",
    "write_config",
    "write_marker",
    "format_hyprland_bind",
    "format_hyprland_bind_for_keybind",
    "auto_add_hyprland_keybind",
    "LayerShell",
]


class WelcomeWizard(Gtk.Application):
    """First-run guided setup wizard."""

    def __init__(self, *, force_reconfigure: bool = False):
        super().__init__(application_id="io.github.shuv1337.shuvoice.wizard")
        self.completed = False
        self._force_reconfigure = force_reconfigure
        self._asr_backend = "sherpa"
        self._sherpa_model_name = DEFAULT_WIZARD_SHERPA_MODEL_NAME
        self._sherpa_enable_parakeet_streaming = False
        self._sherpa_provider = "cpu"
        self._typing_final_injection_mode = DEFAULT_FINAL_INJECTION_MODE
        self._tts_backend = DEFAULT_TTS_BACKEND
        self._tts_voice_by_backend = {
            "elevenlabs": default_tts_voice_for_backend("elevenlabs"),
            "openai": default_tts_voice_for_backend("openai"),
            "local": default_tts_voice_for_backend("local"),
        }
        self._tts_local_model_path = ""
        self._tts_voice_id = self._tts_voice_by_backend[self._tts_backend]
        self._keybind = DEFAULT_KEYBIND_ID
        self._finish_in_progress = False
        self._download_pulse_source_id: int | None = None
        self._download_cancel_event = threading.Event()
        self._win: Gtk.Window | None = None
        self._stack: Gtk.Stack | None = None

    # -- GTK lifecycle ---------------------------------------------------------

    def do_activate(self):
        win = Gtk.Window(application=self)
        self._win = win

        if LayerShell.is_supported():
            LayerShell.init_for_window(win)
            LayerShell.set_layer(win, LayerShell.Layer.TOP)
            # Avoid global keyboard grabs during setup; keeps input recoverable
            # even if window teardown is delayed by compositor timing.
            LayerShell.set_keyboard_mode(win, LayerShell.KeyboardMode.ON_DEMAND)
            LayerShell.set_exclusive_zone(win, -1)
            LayerShell.set_namespace(win, "shuvoice-wizard")

        self._setup_css()

        self._stack = Gtk.Stack()
        self._stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        self._stack.set_transition_duration(200)

        self._stack.add_named(self._build_welcome_page(), "welcome")
        self._stack.add_named(self._build_asr_page(), "asr")
        self._stack.add_named(self._build_keybind_page(), "keybind")
        self._stack.add_named(self._build_done_page(), "done")

        win.set_child(self._stack)
        win.add_css_class("wizard-window")
        win.present()

    def _setup_css(self):
        setup_css()

    # -- Page builders ---------------------------------------------------------

    def _build_welcome_page(self) -> Gtk.Widget:
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        page.add_css_class("wizard-page")
        page.set_halign(Gtk.Align.CENTER)
        page.set_valign(Gtk.Align.CENTER)
        page.set_spacing(4)

        logo_path = find_logo()
        if logo_path:
            try:
                logo = Gtk.Picture.new_for_filename(str(logo_path))
                logo.set_can_shrink(True)
                logo.set_alternative_text("ShuVoice logo")
                logo.set_size_request(320, -1)
                logo.set_halign(Gtk.Align.CENTER)
                logo.set_margin_bottom(12)
                page.append(logo)
            except Exception:
                self._add_text_title(page, "ShuVoice")
        else:
            self._add_text_title(page, "ShuVoice")

        sub = Gtk.Label(label="Streaming speech-to-text for Hyprland")
        sub.add_css_class("wizard-subtitle")
        sub.set_halign(Gtk.Align.CENTER)
        page.append(sub)

        desc = Gtk.Label(
            label="Let\u2019s set up a few things before you start.\nThis will only take a moment."
        )
        desc.add_css_class("wizard-desc")
        desc.set_halign(Gtk.Align.CENTER)
        desc.set_justify(Gtk.Justification.CENTER)
        desc.set_margin_bottom(24)
        page.append(desc)

        btn = self._make_button("Get Started", primary=True)
        btn.connect("clicked", lambda _: self._stack.set_visible_child_name("asr"))
        btn.set_halign(Gtk.Align.CENTER)
        page.append(btn)

        return page

    def _build_asr_page(self) -> Gtk.Widget:
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        page.add_css_class("wizard-page")
        page.set_halign(Gtk.Align.CENTER)
        page.set_valign(Gtk.Align.CENTER)
        page.set_spacing(4)

        self._add_text_title(page, "Speech Recognition Engine")

        sub = Gtk.Label(label="Choose the ASR backend that fits your hardware.")
        sub.add_css_class("wizard-subtitle")
        page.append(sub)

        group: Gtk.CheckButton | None = None
        self._asr_radios: dict[str, Gtk.CheckButton] = {}

        for backend_id, label, description in ASR_BACKENDS:
            radio = Gtk.CheckButton(label=label)
            radio.add_css_class("wizard-radio")
            if group is None:
                group = radio
            else:
                radio.set_group(group)

            if backend_id == "sherpa":
                radio.set_active(True)

            radio.connect("toggled", self._on_asr_toggled, backend_id)
            page.append(radio)
            self._asr_radios[backend_id] = radio

            desc_label = Gtk.Label(label=description)
            desc_label.add_css_class("wizard-radio-desc")
            desc_label.set_halign(Gtk.Align.START)
            page.append(desc_label)
            self._set_accessible_description(radio, description)

        self._sherpa_profile_title = Gtk.Label(label="Sherpa profile")
        self._sherpa_profile_title.add_css_class("wizard-subtitle")
        self._sherpa_profile_title.set_halign(Gtk.Align.START)
        self._sherpa_profile_title.set_margin_top(8)
        page.append(self._sherpa_profile_title)

        self._sherpa_profile_help = Gtk.Label(
            label=(
                "Streaming = live overlay updates while holding push-to-talk.\n"
                "Text is committed only on key release (no live partial typing)."
            )
        )
        self._sherpa_profile_help.add_css_class("wizard-desc")
        self._sherpa_profile_help.set_halign(Gtk.Align.START)
        self._sherpa_profile_help.set_justify(Gtk.Justification.LEFT)
        self._sherpa_profile_help.set_margin_bottom(4)
        page.append(self._sherpa_profile_help)

        self._sherpa_streaming_radio = Gtk.CheckButton(label="Streaming (Zipformer Kroko model)")
        self._sherpa_streaming_radio.add_css_class("wizard-radio")
        self._sherpa_streaming_radio.connect(
            "toggled",
            self._on_sherpa_profile_toggled,
            (DEFAULT_SHERPA_MODEL_NAME, False),
        )
        page.append(self._sherpa_streaming_radio)

        sherpa_streaming_desc = (
            "Shows incremental transcript updates in the overlay while you hold the key. "
            "Final text is committed on key release."
        )
        self._sherpa_streaming_desc = Gtk.Label(label=sherpa_streaming_desc)
        self._sherpa_streaming_desc.add_css_class("wizard-radio-desc")
        self._sherpa_streaming_desc.set_halign(Gtk.Align.START)
        page.append(self._sherpa_streaming_desc)
        self._set_accessible_description(self._sherpa_streaming_radio, sherpa_streaming_desc)

        self._sherpa_parakeet_radio = Gtk.CheckButton(
            label="Instant (Parakeet TDT v3 int8 model, recommended)"
        )
        self._sherpa_parakeet_radio.add_css_class("wizard-radio")
        self._sherpa_parakeet_radio.set_group(self._sherpa_streaming_radio)
        self._sherpa_parakeet_radio.connect(
            "toggled",
            self._on_sherpa_profile_toggled,
            (PARAKEET_TDT_V3_INT8_MODEL_NAME, False),
        )
        page.append(self._sherpa_parakeet_radio)

        sherpa_parakeet_desc = (
            "Stable default profile. Emits one final transcript on key release and "
            "auto-enables instant_mode + sherpa_decode_mode=offline_instant."
        )
        self._sherpa_parakeet_desc = Gtk.Label(label=sherpa_parakeet_desc)
        self._sherpa_parakeet_desc.add_css_class("wizard-radio-desc")
        self._sherpa_parakeet_desc.set_halign(Gtk.Align.START)
        page.append(self._sherpa_parakeet_desc)
        self._set_accessible_description(self._sherpa_parakeet_radio, sherpa_parakeet_desc)

        self._sherpa_provider_title = Gtk.Label(label="Sherpa compute device")
        self._sherpa_provider_title.add_css_class("wizard-subtitle")
        self._sherpa_provider_title.set_halign(Gtk.Align.START)
        self._sherpa_provider_title.set_margin_top(8)
        page.append(self._sherpa_provider_title)

        self._sherpa_provider_help = Gtk.Label(
            label=(
                "Pick CPU or GPU (CUDA). If GPU is selected, wizard will try to install "
                "a CUDA-capable Sherpa runtime automatically."
            )
        )
        self._sherpa_provider_help.add_css_class("wizard-desc")
        self._sherpa_provider_help.set_halign(Gtk.Align.START)
        self._sherpa_provider_help.set_justify(Gtk.Justification.LEFT)
        self._sherpa_provider_help.set_margin_bottom(4)
        page.append(self._sherpa_provider_help)

        self._sherpa_provider_cpu_radio = Gtk.CheckButton(label="CPU")
        self._sherpa_provider_cpu_radio.add_css_class("wizard-radio")
        self._sherpa_provider_cpu_radio.connect(
            "toggled",
            self._on_sherpa_provider_toggled,
            "cpu",
        )
        page.append(self._sherpa_provider_cpu_radio)

        sherpa_provider_cpu_desc = "Best compatibility and lowest setup friction."
        self._sherpa_provider_cpu_desc = Gtk.Label(label=sherpa_provider_cpu_desc)
        self._sherpa_provider_cpu_desc.add_css_class("wizard-radio-desc")
        self._sherpa_provider_cpu_desc.set_halign(Gtk.Align.START)
        page.append(self._sherpa_provider_cpu_desc)
        self._set_accessible_description(self._sherpa_provider_cpu_radio, sherpa_provider_cpu_desc)

        self._sherpa_provider_cuda_radio = Gtk.CheckButton(label="GPU (CUDA)")
        self._sherpa_provider_cuda_radio.add_css_class("wizard-radio")
        self._sherpa_provider_cuda_radio.set_group(self._sherpa_provider_cpu_radio)
        self._sherpa_provider_cuda_radio.connect(
            "toggled",
            self._on_sherpa_provider_toggled,
            "cuda",
        )
        page.append(self._sherpa_provider_cuda_radio)

        sherpa_provider_cuda_desc = (
            "Higher throughput when CUDA runtime is available. "
            "Wizard will attempt CUDA-capable Sherpa runtime install at finish."
        )
        self._sherpa_provider_cuda_desc = Gtk.Label(label=sherpa_provider_cuda_desc)
        self._sherpa_provider_cuda_desc.add_css_class("wizard-radio-desc")
        self._sherpa_provider_cuda_desc.set_halign(Gtk.Align.START)
        page.append(self._sherpa_provider_cuda_desc)
        self._set_accessible_description(
            self._sherpa_provider_cuda_radio,
            sherpa_provider_cuda_desc,
        )

        self._sync_sherpa_model_controls()

        nav = self._make_nav_row(
            back_page="welcome",
            next_page="keybind",
        )
        nav.set_margin_top(20)
        page.append(nav)

        return page

    def _build_keybind_page(self) -> Gtk.Widget:
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        page.add_css_class("wizard-page")
        page.set_halign(Gtk.Align.CENTER)
        page.set_valign(Gtk.Align.CENTER)
        page.set_spacing(4)

        self._add_text_title(page, "Push-to-Talk Keybind")

        sub = Gtk.Label(
            label="ShuVoice uses Hyprland bind/bindr for push-to-talk.\n"
            "Hold the key to record, release to stop. Default: Right Control."
        )
        sub.add_css_class("wizard-subtitle")
        sub.set_justify(Gtk.Justification.CENTER)
        page.append(sub)

        group: Gtk.CheckButton | None = None
        self._keybind_radios: dict[str, Gtk.CheckButton] = {}

        for kb_id, label, _hypr_key, description in KEYBIND_PRESETS:
            radio = Gtk.CheckButton(label=label)
            radio.add_css_class("wizard-radio")
            if group is None:
                group = radio
            else:
                radio.set_group(group)

            if kb_id == DEFAULT_KEYBIND_ID:
                radio.set_active(True)

            radio.connect("toggled", self._on_keybind_toggled, kb_id)
            page.append(radio)
            self._keybind_radios[kb_id] = radio

            desc_label = Gtk.Label(label=description)
            desc_label.add_css_class("wizard-radio-desc")
            desc_label.set_halign(Gtk.Align.START)
            page.append(desc_label)
            self._set_accessible_description(radio, description)

        self._auto_add_last_non_custom = True
        self._auto_add_keybind = Gtk.CheckButton(
            label="Try to add this keybind to ~/.config/hypr/hyprland.conf automatically"
        )
        self._auto_add_keybind.add_css_class("wizard-radio")
        self._auto_add_keybind.set_active(True)
        self._auto_add_keybind.connect("toggled", self._on_auto_add_keybind_toggled)
        page.append(self._auto_add_keybind)

        auto_add_desc = Gtk.Label(
            label=(
                "Only applies when the selected key is not already used by another bind. "
                "If there is a conflict, wizard will leave your Hyprland config unchanged."
            )
        )
        auto_add_desc.add_css_class("wizard-radio-desc")
        auto_add_desc.set_halign(Gtk.Align.START)
        page.append(auto_add_desc)

        # Live preview of the Hyprland config lines
        self._keybind_preview = Gtk.Label()
        self._keybind_preview.add_css_class("wizard-summary")
        self._keybind_preview.set_halign(Gtk.Align.START)
        self._keybind_preview.set_selectable(True)
        self._sync_auto_add_keybind_state()
        self._update_keybind_preview()
        page.append(self._keybind_preview)

        injection_title = Gtk.Label(label="Final text injection")
        injection_title.add_css_class("wizard-subtitle")
        injection_title.set_halign(Gtk.Align.START)
        injection_title.set_margin_top(12)
        page.append(injection_title)

        injection_help = Gtk.Label(
            label=(
                "Choose how the final transcript is committed when you release push-to-talk.\n"
                "If one mode misbehaves in an app, rerun wizard and switch modes."
            )
        )
        injection_help.add_css_class("wizard-desc")
        injection_help.set_halign(Gtk.Align.START)
        injection_help.set_justify(Gtk.Justification.LEFT)
        page.append(injection_help)

        mode_group: Gtk.CheckButton | None = None
        self._final_injection_radios: dict[str, Gtk.CheckButton] = {}
        active_mode = getattr(self, "_typing_final_injection_mode", DEFAULT_FINAL_INJECTION_MODE)

        for mode_id, mode_label, mode_desc in FINAL_INJECTION_MODES:
            mode_radio = Gtk.CheckButton(label=mode_label)
            mode_radio.add_css_class("wizard-radio")
            if mode_group is None:
                mode_group = mode_radio
            else:
                mode_radio.set_group(mode_group)

            if mode_id == active_mode:
                mode_radio.set_active(True)

            mode_radio.connect("toggled", self._on_final_injection_toggled, mode_id)
            page.append(mode_radio)
            self._final_injection_radios[mode_id] = mode_radio

            mode_desc_label = Gtk.Label(label=mode_desc)
            mode_desc_label.add_css_class("wizard-radio-desc")
            mode_desc_label.set_halign(Gtk.Align.START)
            page.append(mode_desc_label)
            self._set_accessible_description(mode_radio, mode_desc)

        tts_title = Gtk.Label(label="Text-to-Speech provider")
        tts_title.add_css_class("wizard-subtitle")
        tts_title.set_halign(Gtk.Align.START)
        tts_title.set_margin_top(12)
        page.append(tts_title)

        tts_help = Gtk.Label(
            label=(
                "Choose which backend handles the TTS keybind and set the default voice.\n"
                "For ElevenLabs, paste a voice ID. For OpenAI, use built-in names like onyx or nova.\n"
                "For Local Piper, point ShuVoice at a .onnx model file or a directory of .onnx voices."
            )
        )
        tts_help.add_css_class("wizard-desc")
        tts_help.set_halign(Gtk.Align.START)
        tts_help.set_justify(Gtk.Justification.LEFT)
        page.append(tts_help)

        tts_group: Gtk.CheckButton | None = None
        self._tts_backend_radios: dict[str, Gtk.CheckButton] = {}
        active_tts_backend = str(getattr(self, "_tts_backend", DEFAULT_TTS_BACKEND)).strip().lower()

        for backend_id, backend_label, backend_desc in TTS_BACKENDS:
            backend_radio = Gtk.CheckButton(label=backend_label)
            backend_radio.add_css_class("wizard-radio")
            if tts_group is None:
                tts_group = backend_radio
            else:
                backend_radio.set_group(tts_group)

            if backend_id == active_tts_backend:
                backend_radio.set_active(True)

            backend_radio.connect("toggled", self._on_tts_backend_toggled, backend_id)
            page.append(backend_radio)
            self._tts_backend_radios[backend_id] = backend_radio

            backend_desc_label = Gtk.Label(label=backend_desc)
            backend_desc_label.add_css_class("wizard-radio-desc")
            backend_desc_label.set_halign(Gtk.Align.START)
            page.append(backend_desc_label)
            self._set_accessible_description(backend_radio, backend_desc)

        self._tts_local_model_path_entry = Gtk.Entry()
        self._tts_local_model_path_entry.add_css_class("wizard-entry")
        self._tts_local_model_path_entry.set_halign(Gtk.Align.FILL)
        self._tts_local_model_path_entry.set_hexpand(True)
        self._tts_local_model_path_entry.connect("changed", self._on_tts_local_model_path_changed)
        page.append(self._tts_local_model_path_entry)

        self._tts_local_model_path_help = Gtk.Label(label="")
        self._tts_local_model_path_help.add_css_class("wizard-radio-desc")
        self._tts_local_model_path_help.set_halign(Gtk.Align.START)
        page.append(self._tts_local_model_path_help)

        self._tts_voice_entry = Gtk.Entry()
        self._tts_voice_entry.add_css_class("wizard-entry")
        self._tts_voice_entry.set_halign(Gtk.Align.FILL)
        self._tts_voice_entry.set_hexpand(True)
        self._tts_voice_entry.connect("changed", self._on_tts_voice_changed)
        page.append(self._tts_voice_entry)

        self._tts_voice_help = Gtk.Label(label="")
        self._tts_voice_help.add_css_class("wizard-radio-desc")
        self._tts_voice_help.set_halign(Gtk.Align.START)
        page.append(self._tts_voice_help)

        self._tts_config_error_label = Gtk.Label(label="")
        self._tts_config_error_label.add_css_class("wizard-radio-desc")
        self._tts_config_error_label.set_halign(Gtk.Align.START)
        self._tts_config_error_label.set_visible(False)
        page.append(self._tts_config_error_label)

        self._sync_tts_voice_controls()

        nav = self._make_nav_row(
            back_page="asr",
            next_page="done",
            next_label="Finish",
        )
        nav.set_margin_top(20)
        page.append(nav)

        return page

    def _build_done_page(self) -> Gtk.Widget:
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        page.add_css_class("wizard-page")
        page.set_halign(Gtk.Align.CENTER)
        page.set_valign(Gtk.Align.CENTER)
        page.set_spacing(4)

        self._add_text_title(page, "You\u2019re All Set!")

        sub = Gtk.Label(label="Here\u2019s a summary of your settings:")
        sub.add_css_class("wizard-subtitle")
        page.append(sub)

        self._summary_label = Gtk.Label()
        self._summary_label.add_css_class("wizard-summary")
        self._summary_label.set_halign(Gtk.Align.CENTER)
        page.append(self._summary_label)

        tip = Gtk.Label(
            label="Edit ~/.config/shuvoice/config.toml to adjust ASR, TTS, and overlay settings.\n"
            "If enabled, wizard will try to add your push-to-talk keybind in ~/.config/hypr/hyprland.conf.\n"
            "Model setup starts automatically on this screen."
        )
        tip.add_css_class("wizard-desc")
        tip.set_halign(Gtk.Align.CENTER)
        tip.set_justify(Gtk.Justification.CENTER)
        tip.set_margin_top(16)
        page.append(tip)

        self._finish_status_label = Gtk.Label(label="")
        self._finish_status_label.add_css_class("wizard-desc")
        self._finish_status_label.set_halign(Gtk.Align.CENTER)
        self._finish_status_label.set_justify(Gtk.Justification.CENTER)
        self._finish_status_label.set_visible(False)
        self._finish_status_label.set_margin_top(8)
        page.append(self._finish_status_label)

        self._download_progress = Gtk.ProgressBar()
        self._download_progress.set_show_text(True)
        self._download_progress.set_text("Preparing model download…")
        self._download_progress.set_fraction(0.0)
        self._download_progress.set_hexpand(True)
        self._download_progress.set_visible(False)
        self._download_progress.set_margin_top(6)
        self._download_progress.set_margin_start(16)
        self._download_progress.set_margin_end(16)
        page.append(self._download_progress)

        self._download_note_label = Gtk.Label(
            label=(
                "Note: extraction can pause for 10–60s on slower disks. "
                "Please keep this window open."
            )
        )
        self._download_note_label.add_css_class("wizard-desc")
        self._download_note_label.set_halign(Gtk.Align.CENTER)
        self._download_note_label.set_justify(Gtk.Justification.CENTER)
        self._download_note_label.set_margin_top(6)
        self._download_note_label.set_visible(False)
        page.append(self._download_note_label)

        self._cancel_download_button = self._make_button("Cancel download")
        self._cancel_download_button.set_visible(False)
        self._cancel_download_button.set_halign(Gtk.Align.CENTER)
        self._cancel_download_button.set_margin_top(6)
        self._cancel_download_button.connect("clicked", self._on_cancel_download_clicked)
        page.append(self._cancel_download_button)

        self._launch_button = self._make_button("Launch ShuVoice", primary=True)
        self._launch_button.connect("clicked", self._on_launch_clicked)
        self._launch_button.set_halign(Gtk.Align.CENTER)
        self._launch_button.set_margin_top(20)
        self._launch_button.set_visible(False)
        page.append(self._launch_button)

        return page

    # -- Callbacks -------------------------------------------------------------

    def _on_asr_toggled(self, button: Gtk.CheckButton, backend_id: str):
        if button.get_active():
            self._asr_backend = backend_id
            self._sync_sherpa_model_controls()

    def _on_sherpa_profile_toggled(
        self,
        button: Gtk.CheckButton,
        profile: tuple[str, bool],
    ):
        if not button.get_active():
            return

        model_name, enable_parakeet_streaming = profile
        self._sherpa_model_name = model_name
        self._sherpa_enable_parakeet_streaming = bool(enable_parakeet_streaming)

    def _on_sherpa_provider_toggled(self, button: Gtk.CheckButton, provider: str):
        if not button.get_active():
            return
        self._sherpa_provider = provider

    def _sync_sherpa_model_controls(self):
        title = getattr(self, "_sherpa_profile_title", None)
        help_text = getattr(self, "_sherpa_profile_help", None)
        streaming_radio = getattr(self, "_sherpa_streaming_radio", None)
        streaming_desc = getattr(self, "_sherpa_streaming_desc", None)
        parakeet_radio = getattr(self, "_sherpa_parakeet_radio", None)
        parakeet_desc = getattr(self, "_sherpa_parakeet_desc", None)

        provider_title = getattr(self, "_sherpa_provider_title", None)
        provider_help = getattr(self, "_sherpa_provider_help", None)
        provider_cpu_radio = getattr(self, "_sherpa_provider_cpu_radio", None)
        provider_cpu_desc = getattr(self, "_sherpa_provider_cpu_desc", None)
        provider_cuda_radio = getattr(self, "_sherpa_provider_cuda_radio", None)
        provider_cuda_desc = getattr(self, "_sherpa_provider_cuda_desc", None)

        if (
            title is None
            or help_text is None
            or streaming_radio is None
            or streaming_desc is None
            or parakeet_radio is None
            or parakeet_desc is None
            or provider_title is None
            or provider_help is None
            or provider_cpu_radio is None
            or provider_cpu_desc is None
            or provider_cuda_radio is None
            or provider_cuda_desc is None
        ):
            return

        is_sherpa = self._asr_backend == "sherpa"
        for widget in (
            title,
            help_text,
            streaming_radio,
            streaming_desc,
            parakeet_radio,
            parakeet_desc,
            provider_title,
            provider_help,
            provider_cpu_radio,
            provider_cpu_desc,
            provider_cuda_radio,
            provider_cuda_desc,
        ):
            widget.set_visible(is_sherpa)

        streaming_radio.set_sensitive(is_sherpa)
        parakeet_radio.set_sensitive(is_sherpa)
        provider_cpu_radio.set_sensitive(is_sherpa)
        provider_cuda_radio.set_sensitive(is_sherpa)

        if not is_sherpa:
            self._sherpa_model_name = DEFAULT_WIZARD_SHERPA_MODEL_NAME
            self._sherpa_enable_parakeet_streaming = False
            return

        if self._sherpa_model_name == PARAKEET_TDT_V3_INT8_MODEL_NAME:
            parakeet_radio.set_active(True)
        else:
            streaming_radio.set_active(True)

        if parakeet_radio.get_active():
            self._sherpa_model_name = PARAKEET_TDT_V3_INT8_MODEL_NAME
            self._sherpa_enable_parakeet_streaming = False
        else:
            self._sherpa_model_name = DEFAULT_SHERPA_MODEL_NAME
            self._sherpa_enable_parakeet_streaming = False

        if str(getattr(self, "_sherpa_provider", "cpu")).strip().lower() == "cuda":
            provider_cuda_radio.set_active(True)
            self._sherpa_provider = "cuda"
        else:
            provider_cpu_radio.set_active(True)
            self._sherpa_provider = "cpu"

    def _on_keybind_toggled(self, button: Gtk.CheckButton, kb_id: str):
        if button.get_active():
            self._keybind = kb_id
            self._sync_auto_add_keybind_state()
            self._update_keybind_preview()

    def _on_auto_add_keybind_toggled(self, _button: Gtk.CheckButton):
        if not hasattr(self, "_auto_add_keybind"):
            return

        if self._auto_add_keybind.get_sensitive():
            self._auto_add_last_non_custom = self._auto_add_keybind.get_active()
        self._update_keybind_preview()

    def _on_final_injection_toggled(self, button: Gtk.CheckButton, mode_id: str):
        if not button.get_active():
            return
        self._typing_final_injection_mode = mode_id

    def _on_tts_backend_toggled(self, button: Gtk.CheckButton, backend_id: str):
        if not button.get_active():
            return
        current_backend = str(getattr(self, "_tts_backend", DEFAULT_TTS_BACKEND)).strip().lower()
        current_voice = str(getattr(self, "_tts_voice_id", "")).strip()
        voice_by_backend = getattr(self, "_tts_voice_by_backend", None)
        if not isinstance(voice_by_backend, dict):
            voice_by_backend = {}
            self._tts_voice_by_backend = voice_by_backend
        if current_backend:
            voice_by_backend[current_backend] = current_voice or default_tts_voice_for_backend(
                current_backend
            )

        self._tts_backend = backend_id
        self._tts_voice_id = voice_by_backend.get(
            backend_id, default_tts_voice_for_backend(backend_id)
        )
        self._set_tts_config_error(None)
        self._sync_tts_voice_controls()

    def _on_tts_local_model_path_changed(self, entry: Gtk.Entry):
        self._tts_local_model_path = entry.get_text().strip()
        self._set_tts_config_error(None)

    def _on_tts_voice_changed(self, entry: Gtk.Entry):
        value = entry.get_text().strip()
        backend_id = str(getattr(self, "_tts_backend", DEFAULT_TTS_BACKEND)).strip().lower()
        if not value:
            value = default_tts_voice_for_backend(backend_id)
        self._tts_voice_id = value
        voice_by_backend = getattr(self, "_tts_voice_by_backend", None)
        if isinstance(voice_by_backend, dict):
            voice_by_backend[backend_id] = value
        self._set_tts_config_error(None)

    def _set_tts_config_error(self, message: str | None) -> None:
        label = getattr(self, "_tts_config_error_label", None)
        if label is None:
            return
        label.set_text(message or "")
        label.set_visible(bool(message))

    def _local_tts_resolved_voice(self) -> str:
        requested = str(getattr(self, "_tts_voice_id", "")).strip().lower()
        path_text = str(getattr(self, "_tts_local_model_path", "")).strip()
        if not path_text:
            return default_tts_voice_for_backend("local")

        path = Path(path_text).expanduser()
        if path.is_file():
            return path.stem

        if requested and requested != default_tts_voice_for_backend("local"):
            return str(getattr(self, "_tts_voice_id", "")).strip()

        first = next(iter(sorted(path.glob("*.onnx"))), None) if path.is_dir() else None
        if first is not None:
            return first.stem
        return default_tts_voice_for_backend("local")

    def _validate_tts_selection_for_finish(self) -> bool:
        self._set_tts_config_error(None)
        backend_id = str(getattr(self, "_tts_backend", DEFAULT_TTS_BACKEND)).strip().lower()
        if backend_id != "local":
            return True

        path_text = str(getattr(self, "_tts_local_model_path", "")).strip()
        if not path_text:
            self._set_tts_config_error(
                "Local Piper requires a .onnx model path or a directory of .onnx voices."
            )
            return False

        path = Path(path_text).expanduser()
        if not path.exists():
            self._set_tts_config_error(f"Local Piper path does not exist: {path}")
            return False

        if path.is_file():
            if path.suffix.lower() != ".onnx":
                self._set_tts_config_error(f"Local Piper model path must end in .onnx: {path}")
                return False
            return True

        if not path.is_dir():
            self._set_tts_config_error(f"Local Piper path must be a file or directory: {path}")
            return False

        model_files = sorted(path.glob("*.onnx"))
        if not model_files:
            self._set_tts_config_error(f"No .onnx model files found under: {path}")
            return False

        requested_voice = str(getattr(self, "_tts_voice_id", "")).strip()
        if requested_voice and requested_voice.lower() != default_tts_voice_for_backend("local"):
            requested_file = path / f"{requested_voice}.onnx"
            if not requested_file.is_file():
                self._set_tts_config_error(
                    f"Local Piper voice '{requested_voice}' was not found in: {path}"
                )
                return False

        return True

    def _sync_tts_voice_controls(self):
        entry = getattr(self, "_tts_voice_entry", None)
        help_label = getattr(self, "_tts_voice_help", None)
        path_entry = getattr(self, "_tts_local_model_path_entry", None)
        path_help = getattr(self, "_tts_local_model_path_help", None)
        if entry is None or help_label is None or path_entry is None or path_help is None:
            return

        backend_id = str(getattr(self, "_tts_backend", DEFAULT_TTS_BACKEND)).strip().lower()
        voice_by_backend = getattr(self, "_tts_voice_by_backend", None)
        if not isinstance(voice_by_backend, dict):
            voice_by_backend = {}
            self._tts_voice_by_backend = voice_by_backend

        default_voice = default_tts_voice_for_backend(backend_id)
        current_voice = str(voice_by_backend.get(backend_id, default_voice)).strip() or default_voice
        self._tts_voice_id = current_voice
        voice_by_backend[backend_id] = current_voice

        display_voice = "" if backend_id == "local" and current_voice == default_voice else current_voice
        if entry.get_text() != display_voice:
            entry.set_text(display_voice)

        is_local = backend_id == "local"
        path_entry.set_visible(is_local)
        path_help.set_visible(is_local)
        if is_local:
            current_path = str(getattr(self, "_tts_local_model_path", "")).strip()
            if path_entry.get_text() != current_path:
                path_entry.set_text(current_path)
            path_entry.set_placeholder_text("/path/to/piper-model.onnx")
            path_help.set_text(
                "Point to a Piper .onnx model file or a directory containing .onnx voices."
            )
            entry.set_placeholder_text("Optional voice/model stem, e.g. en_US-amy-medium")
            help_label.set_text(
                "Leave the voice blank to use the first discovered .onnx model automatically."
            )
            return

        if backend_id == "openai":
            entry.set_placeholder_text("onyx")
            help_label.set_text("Examples: onyx, nova, shimmer, alloy, sage")
        else:
            entry.set_placeholder_text(default_tts_voice_for_backend("elevenlabs"))
            help_label.set_text(
                "Paste an ElevenLabs voice ID here, or keep the default voice ID."
            )

    def _sync_auto_add_keybind_state(self):
        if not hasattr(self, "_auto_add_keybind"):
            return

        hypr_key = next(
            (hk for kid, _, hk, _ in KEYBIND_PRESETS if kid == self._keybind),
            None,
        )
        if hypr_key is None:
            self._auto_add_last_non_custom = self._auto_add_keybind.get_active()
            self._auto_add_keybind.set_sensitive(False)
            self._auto_add_keybind.set_active(False)
            return

        self._auto_add_keybind.set_sensitive(True)
        self._auto_add_keybind.set_active(getattr(self, "_auto_add_last_non_custom", True))

    def _auto_add_enabled(self) -> bool:
        return bool(
            hasattr(self, "_auto_add_keybind")
            and self._auto_add_keybind is not None
            and self._auto_add_keybind.get_active()
        )

    def _update_keybind_preview(self):
        """Refresh the keybind preview box based on current selection."""
        if not hasattr(self, "_keybind_preview") or self._keybind_preview is None:
            return
        hypr_key = next(
            (hk for kid, _, hk, _ in KEYBIND_PRESETS if kid == self._keybind),
            None,
        )
        if hypr_key:
            bind_text = format_hyprland_bind_for_keybind(self._keybind, hypr_key)
            indented = "\n".join(f"  {line}" for line in bind_text.splitlines())
            if self._auto_add_enabled():
                text = (
                    "Wizard will try to add this to ~/.config/hypr/hyprland.conf\n"
                    "(only if no conflicting bind already uses that key):\n\n"
                    f"{indented}"
                )
            else:
                text = f"Add to ~/.config/hypr/hyprland.conf:\n\n{indented}"
        else:
            text = (
                "Configure your keybind in ~/.config/hypr/hyprland.conf\n"
                "See README.md for bind/bindr examples."
            )
        self._keybind_preview.set_text(text)

    def _release_input_and_destroy_window(self):
        win = self._win
        if win is None:
            return

        if LayerShell.is_supported():
            try:
                LayerShell.set_keyboard_mode(win, LayerShell.KeyboardMode.NONE)
            except Exception:
                log.debug("Failed to release wizard keyboard mode", exc_info=True)

        try:
            win.set_visible(False)
        except Exception:
            log.debug("Failed to hide wizard window", exc_info=True)

        try:
            win.destroy()
        except Exception:
            log.debug("Failed to destroy wizard window", exc_info=True)

        self._win = None

    def do_shutdown(self):
        self._release_input_and_destroy_window()
        Gtk.Application.do_shutdown(self)

    def _on_finish(self, button):
        if not hasattr(self, "_download_pulse_source_id"):
            self._download_pulse_source_id = None

        if getattr(self, "_finish_in_progress", False):
            return
        self._finish_in_progress = True

        self._set_launch_button_visible(False)

        if button is not None:
            try:
                button.set_sensitive(False)
            except Exception:
                log.debug("Failed to disable finish button", exc_info=True)

        sherpa_model_name = getattr(self, "_sherpa_model_name", DEFAULT_SHERPA_MODEL_NAME)
        sherpa_enable_parakeet_streaming = bool(
            getattr(self, "_sherpa_enable_parakeet_streaming", False)
        )
        tts_backend = str(getattr(self, "_tts_backend", DEFAULT_TTS_BACKEND)).strip().lower()
        if not self._validate_tts_selection_for_finish():
            self._finish_in_progress = False
            return

        tts_voice_id = str(
            getattr(self, "_tts_voice_id", default_tts_voice_for_backend(tts_backend))
        ).strip() or default_tts_voice_for_backend(tts_backend)
        tts_local_model_path = str(getattr(self, "_tts_local_model_path", "")).strip() or None
        tts_local_voice = None
        if tts_backend == "local":
            tts_voice_id = self._local_tts_resolved_voice()
            tts_local_voice = tts_voice_id

        write_kwargs: dict[str, object] = {
            "overwrite_existing": getattr(self, "_force_reconfigure", False),
            "sherpa_model_name": sherpa_model_name,
            "typing_final_injection_mode": getattr(
                self,
                "_typing_final_injection_mode",
                DEFAULT_FINAL_INJECTION_MODE,
            ),
            "tts_backend": tts_backend,
            "tts_default_voice_id": tts_voice_id,
            "tts_local_model_path": tts_local_model_path,
            "tts_local_voice": tts_local_voice,
        }
        if self._asr_backend == "sherpa":
            write_kwargs["sherpa_enable_parakeet_streaming"] = sherpa_enable_parakeet_streaming
            write_kwargs["sherpa_provider"] = getattr(self, "_sherpa_provider", "cpu")

        write_config(
            self._asr_backend,
            **write_kwargs,
        )

        keybind_status = KeybindSetupStatus.NOT_ATTEMPTED.value
        keybind_message = "automatic keybind setup disabled"
        if self._auto_add_enabled():
            keybind_status, keybind_message = auto_add_hyprland_keybind(self._keybind)

        if keybind_status in {
            KeybindSetupStatus.ADDED.value,
            KeybindSetupStatus.ALREADY_CONFIGURED.value,
        }:
            log.info("Wizard keybind setup: %s", keybind_message)
        elif keybind_status != KeybindSetupStatus.NOT_ATTEMPTED.value:
            log.warning("Wizard keybind setup: %s", keybind_message)

        if hasattr(self, "_download_cancel_event"):
            self._download_cancel_event.clear()

        # If UI widgets are present, run model download in a worker thread so
        # progress can be rendered live in the done screen.
        if hasattr(self, "_download_progress") and self._download_progress is not None:
            self._show_finish_status(self._starting_model_setup_status_text(keybind_status))
            self._set_download_progress_visible(True)
            self._set_download_note_visible(self._asr_backend == "sherpa")
            self._set_cancel_download_visible(True)
            initial_progress_text = "Preparing model download…"
            if (
                self._asr_backend == "sherpa"
                and str(getattr(self, "_sherpa_provider", "cpu")).strip().lower() == "cuda"
            ):
                initial_progress_text = "Preparing Sherpa CUDA runtime setup…"
            self._apply_download_progress(0.0, initial_progress_text)

            threading.Thread(
                target=self._download_model_async,
                args=(keybind_status, sherpa_model_name),
                name="wizard-model-download",
                daemon=True,
            ).start()
            return

        # Fallback path used by tests/headless invocation without full UI.
        model_status, model_message = maybe_download_model(
            self._asr_backend,
            sherpa_model_name=sherpa_model_name,
            progress_callback=None,
            auto_install_missing=True,
        )
        self._complete_finish(
            keybind_status,
            sherpa_model_name,
            model_status,
            model_message,
        )

    def _download_model_async(self, keybind_status: str, sherpa_model_name: str) -> None:
        def _progress(fraction: float | None, message: str) -> None:
            GLib.idle_add(self._apply_download_progress, fraction, message)

        model_status, model_message = maybe_download_model(
            self._asr_backend,
            sherpa_model_name=sherpa_model_name,
            progress_callback=_progress,
            cancel_requested=self._is_download_cancelled,
            auto_install_missing=True,
        )

        GLib.idle_add(
            self._complete_finish,
            keybind_status,
            sherpa_model_name,
            model_status,
            model_message,
        )

    def _is_download_cancelled(self) -> bool:
        event = getattr(self, "_download_cancel_event", None)
        return bool(event is not None and event.is_set())

    def _on_cancel_download_clicked(self, _button):
        if not hasattr(self, "_download_cancel_event"):
            self._download_cancel_event = threading.Event()

        self._download_cancel_event.set()

        btn = getattr(self, "_cancel_download_button", None)
        if btn is not None:
            btn.set_sensitive(False)
            btn.set_label("Canceling…")

        self._apply_download_progress(None, "Cancelling model download…")

    def _on_launch_clicked(self, _button):
        self.completed = True
        self._finalize_and_quit()

    def _set_download_progress_visible(self, visible: bool) -> None:
        progress = getattr(self, "_download_progress", None)
        if progress is None:
            return
        progress.set_visible(visible)

    def _set_cancel_download_visible(self, visible: bool) -> None:
        btn = getattr(self, "_cancel_download_button", None)
        if btn is None:
            return
        btn.set_visible(visible)
        btn.set_sensitive(visible)
        if visible:
            btn.set_label("Cancel download")

    def _set_download_note_visible(self, visible: bool) -> None:
        note = getattr(self, "_download_note_label", None)
        if note is None:
            return
        note.set_visible(visible)

    def _set_launch_button_visible(self, visible: bool) -> None:
        launch = getattr(self, "_launch_button", None)
        if launch is None:
            return
        launch.set_visible(visible)
        launch.set_sensitive(visible)

    def _apply_download_progress(self, fraction: float | None, message: str):
        progress = getattr(self, "_download_progress", None)
        if progress is None:
            return False

        self._set_download_progress_visible(True)

        if message:
            progress.set_text(message)

        if fraction is None:
            if self._download_pulse_source_id is None:
                self._download_pulse_source_id = GLib.timeout_add(
                    120, self._pulse_download_progress
                )
        else:
            if self._download_pulse_source_id is not None:
                GLib.source_remove(self._download_pulse_source_id)
                self._download_pulse_source_id = None
            bounded = max(0.0, min(1.0, float(fraction)))
            progress.set_fraction(bounded)

        return False

    def _pulse_download_progress(self):
        progress = getattr(self, "_download_progress", None)
        if (
            progress is None
            or not progress.get_visible()
            or not getattr(self, "_finish_in_progress", False)
        ):
            self._download_pulse_source_id = None
            return False

        progress.pulse()
        return True

    def _complete_finish(
        self,
        keybind_status: str,
        sherpa_model_name: str,
        model_status: str,
        model_message: str,
    ):
        if model_status == "incompatible_streaming" and self._asr_backend == "sherpa":
            try:
                resolved_tts_backend = getattr(self, "_tts_backend", DEFAULT_TTS_BACKEND)
                resolved_tts_voice = getattr(
                    self,
                    "_tts_voice_id",
                    default_tts_voice_for_backend(resolved_tts_backend),
                )
                resolved_local_voice = None
                if str(resolved_tts_backend).strip().lower() == "local":
                    resolved_tts_voice = self._local_tts_resolved_voice()
                    resolved_local_voice = resolved_tts_voice

                write_config(
                    "sherpa",
                    overwrite_existing=True,
                    sherpa_model_name=DEFAULT_SHERPA_MODEL_NAME,
                    sherpa_enable_parakeet_streaming=False,
                    sherpa_provider=getattr(self, "_sherpa_provider", "cpu"),
                    typing_final_injection_mode=getattr(
                        self,
                        "_typing_final_injection_mode",
                        DEFAULT_FINAL_INJECTION_MODE,
                    ),
                    tts_backend=resolved_tts_backend,
                    tts_default_voice_id=resolved_tts_voice,
                    tts_local_model_path=getattr(self, "_tts_local_model_path", None),
                    tts_local_voice=resolved_local_voice,
                )
            except Exception:  # noqa: BLE001
                log.exception("Wizard fallback to Zipformer streaming profile failed")
            else:
                self._sherpa_model_name = DEFAULT_SHERPA_MODEL_NAME
                self._sherpa_enable_parakeet_streaming = False
                sherpa_model_name = DEFAULT_SHERPA_MODEL_NAME
                log.warning(
                    "Wizard fallback applied: switched to Sherpa Zipformer streaming profile"
                )

        if model_status == "downloaded":
            log.info("Wizard model setup: %s", model_message)
        elif model_status == "cancelled":
            log.info("Wizard model setup cancelled by user")
        elif model_status not in {"skipped", "skipped_missing_deps"}:
            log.warning("Wizard model setup: %s", model_message)

        if self._download_pulse_source_id is not None:
            GLib.source_remove(self._download_pulse_source_id)
            self._download_pulse_source_id = None

        self._set_cancel_download_visible(False)
        self._set_download_note_visible(False)
        if model_status == "cancelled":
            self._apply_download_progress(0.0, "Model download cancelled")
        else:
            self._apply_download_progress(1.0, "Model setup finished")

        write_marker()
        resolved_tts_backend = getattr(self, "_tts_backend", DEFAULT_TTS_BACKEND)
        resolved_tts_voice = getattr(
            self,
            "_tts_voice_id",
            default_tts_voice_for_backend(resolved_tts_backend),
        )
        if str(resolved_tts_backend).strip().lower() == "local":
            resolved_tts_voice = self._local_tts_resolved_voice()

        log.info(
            "Wizard setup ready: asr_backend=%s sherpa_model=%s sherpa_provider=%s tts_backend=%s tts_voice=%s keybind=%s final_injection=%s keybind_setup=%s model_setup=%s",
            self._asr_backend,
            sherpa_model_name,
            getattr(self, "_sherpa_provider", "cpu"),
            resolved_tts_backend,
            resolved_tts_voice,
            self._keybind,
            getattr(self, "_typing_final_injection_mode", DEFAULT_FINAL_INJECTION_MODE),
            keybind_status,
            model_status,
        )

        status_text = self._finish_status_text(keybind_status)
        model_status_text = self._model_download_status_text(model_status)
        if model_status_text:
            status_text = f"{status_text}\n{model_status_text}"

        self._show_finish_status(status_text)
        self._finish_in_progress = False

        if hasattr(self, "_launch_button") and self._launch_button is not None:
            self._set_launch_button_visible(True)
            return False

        # Headless/tests fallback path.
        self.completed = True
        self._finalize_and_quit()
        return False

    def _show_finish_status(self, text: str):
        label = getattr(self, "_finish_status_label", None)
        if label is None:
            return
        label.set_text(text)
        label.set_visible(True)

    @staticmethod
    def _finish_status_text(keybind_status: str) -> str:
        messages = {
            "added": "✓ Added push-to-talk keybind to Hyprland config.",
            "already_configured": "✓ Push-to-talk keybind already configured.",
            "conflict": "⚠ Selected key is already bound; Hyprland config unchanged.",
            "missing_config": "⚠ Hyprland config not found; add keybind manually.",
            "skipped_custom": "ℹ Custom keybind selected; configure it manually in Hyprland.",
            "not_attempted": "ℹ Automatic keybind setup disabled.",
            "error": "⚠ Could not update Hyprland config; check logs.",
        }
        return messages.get(keybind_status, "⚠ Keybind setup status unknown; check logs.")

    def _starting_model_setup_status_text(self, keybind_status: str) -> str:
        status = self._finish_status_text(keybind_status)

        if (
            self._asr_backend == "sherpa"
            and str(getattr(self, "_sherpa_provider", "cpu")).strip().lower() == "cuda"
        ):
            status = (
                f"{status}\n"
                "ℹ Sherpa GPU (CUDA) selected. Wizard will attempt CUDA runtime setup "
                "before model download."
            )

        return status

    @staticmethod
    def _model_download_status_text(model_status: str) -> str:
        messages = {
            "downloaded": "✓ Model downloaded and ready.",
            "skipped": "ℹ Model download skipped (backend downloads lazily).",
            "skipped_missing_deps": "⚠ Model not downloaded (missing dependencies). Run `shuvoice setup`.",
            "cancelled": "ℹ Model download cancelled. You can run `shuvoice model download` later.",
            "incompatible_streaming": (
                "⚠ Parakeet streaming is incompatible with this Sherpa runtime. "
                "Switched to Zipformer streaming profile."
            ),
            "error": "⚠ Model download failed. You can run `shuvoice model download` later.",
        }
        return messages.get(model_status, "")

    def _finalize_and_quit(self):
        if self._download_pulse_source_id is not None:
            GLib.source_remove(self._download_pulse_source_id)
            self._download_pulse_source_id = None

        if hasattr(self, "_download_cancel_event"):
            self._download_cancel_event.clear()
        self._set_cancel_download_visible(False)
        self._set_download_note_visible(False)
        self._set_launch_button_visible(False)
        self._finish_in_progress = False
        self._release_input_and_destroy_window()
        self.quit()
        return False

    # -- Helpers ---------------------------------------------------------------

    def _update_summary(self):
        """Refresh the summary label text based on current selections."""
        if not hasattr(self, "_summary_label") or self._summary_label is None:
            return
        tts_backend = getattr(self, "_tts_backend", DEFAULT_TTS_BACKEND)
        tts_voice_id = getattr(
            self,
            "_tts_voice_id",
            default_tts_voice_for_backend(tts_backend),
        )
        if str(tts_backend).strip().lower() == "local":
            tts_voice_id = self._local_tts_resolved_voice()

        self._summary_label.set_text(
            summary_text(
                self._asr_backend,
                self._keybind,
                auto_add_keybind=self._auto_add_enabled(),
                sherpa_model_name=getattr(self, "_sherpa_model_name", DEFAULT_SHERPA_MODEL_NAME),
                sherpa_enable_parakeet_streaming=bool(
                    getattr(self, "_sherpa_enable_parakeet_streaming", False)
                ),
                sherpa_provider=getattr(self, "_sherpa_provider", "cpu"),
                typing_final_injection_mode=getattr(
                    self,
                    "_typing_final_injection_mode",
                    DEFAULT_FINAL_INJECTION_MODE,
                ),
                tts_backend=tts_backend,
                tts_default_voice_id=tts_voice_id,
                tts_local_model_path=getattr(self, "_tts_local_model_path", ""),
            )
        )

    @staticmethod
    def _set_accessible_description(widget: Gtk.Widget, description: str) -> None:
        setter = getattr(widget, "set_accessible_description", None)
        if callable(setter):
            setter(description)
            return
        widget.update_property([Gtk.AccessibleProperty.DESCRIPTION], [description])

    @staticmethod
    def _add_text_title(box: Gtk.Box, text: str):
        title = Gtk.Label(label=text)
        title.add_css_class("wizard-title")
        title.set_halign(Gtk.Align.CENTER)
        box.append(title)

    @staticmethod
    def _make_button(label: str, *, primary: bool = False) -> Gtk.Button:
        btn = Gtk.Button(label=label)
        btn.add_css_class("wizard-btn")
        if primary:
            btn.add_css_class("wizard-btn-primary")
        return btn

    def _make_nav_row(
        self,
        back_page: str | None = None,
        next_page: str | None = None,
        next_label: str = "Next",
    ) -> Gtk.Box:
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        row.set_halign(Gtk.Align.CENTER)
        row.set_spacing(16)

        if back_page:
            back = self._make_button("Back")
            back.connect("clicked", lambda _: self._stack.set_visible_child_name(back_page))
            row.append(back)

        if next_page:
            nxt = self._make_button(next_label, primary=True)

            def _go_next(_btn, page=next_page):
                if page == "done":
                    if not self._validate_tts_selection_for_finish():
                        return
                    self._update_summary()
                    self._set_launch_button_visible(False)
                    self._set_cancel_download_visible(False)
                    self._set_download_note_visible(False)
                    self._set_download_progress_visible(False)
                    self._show_finish_status("Applying settings…")
                    self._stack.set_visible_child_name(page)
                    GLib.idle_add(self._on_finish, None)
                    return
                self._stack.set_visible_child_name(page)

            nxt.connect("clicked", _go_next)
            row.append(nxt)

        return row
