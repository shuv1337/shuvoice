"""Interactive GTK4 layer-shell overlay for TTS controls."""

from __future__ import annotations

import logging
from collections.abc import Callable

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gtk4LayerShell", "1.0")
from gi.repository import Gdk, GLib, Gtk
from gi.repository import Gtk4LayerShell as LayerShell

from .tts_base import VoiceInfo
from .tts_overlay_state import (
    TTS_OVERLAY_ERROR,
    TTS_OVERLAY_IDLE,
    TTS_OVERLAY_PAUSED,
    summarize_preview,
    status_label_for_state,
)

log = logging.getLogger(__name__)


class TTSOverlay:
    """Interactive TTS overlay with transport controls and voice selection."""

    def __init__(
        self,
        app: Gtk.Application,
        config,
        *,
        on_pause: Callable[[], None] | None = None,
        on_resume: Callable[[], None] | None = None,
        on_restart: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        on_voice_selected: Callable[[str], None] | None = None,
    ):
        self._config = config
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._on_restart = on_restart
        self._on_stop = on_stop
        self._on_voice_selected = on_voice_selected

        self._window = Gtk.Window(application=app)
        self._visible = False
        self._state = TTS_OVERLAY_IDLE
        self._preview_text = ""
        self._auto_hide_source_id: int | None = None

        self._voices: list[VoiceInfo] = []
        self._voice_selected_id = str(config.tts_default_voice_id)

        self._status_label: Gtk.Label | None = None
        self._preview_label: Gtk.Label | None = None
        self._pause_btn: Gtk.Button | None = None
        self._restart_btn: Gtk.Button | None = None
        self._stop_btn: Gtk.Button | None = None
        self._voice_store: Gtk.StringList | None = None
        self._voice_dropdown: Gtk.DropDown | None = None

        self._setup_layer_shell()
        self._setup_css()
        self._setup_widgets()
        self._window.present()
        self._window.set_visible(False)

    def _setup_layer_shell(self) -> None:
        window = self._window
        if not LayerShell.is_supported():
            log.error("Layer shell not supported for TTS overlay")
            return

        LayerShell.init_for_window(window)
        LayerShell.set_layer(window, LayerShell.Layer.OVERLAY)
        LayerShell.set_keyboard_mode(window, LayerShell.KeyboardMode.ON_DEMAND)
        LayerShell.set_exclusive_zone(window, -1)
        LayerShell.set_namespace(window, "tts-overlay")
        LayerShell.set_anchor(window, LayerShell.Edge.BOTTOM, True)
        LayerShell.set_margin(window, LayerShell.Edge.BOTTOM, int(self._config.bottom_margin + 96))

    def _setup_css(self) -> None:
        css = Gtk.CssProvider()
        css.load_from_string(
            "window { background-color: transparent; }\n"
            ".tts-overlay-box {\n"
            "  background-color: rgba(0, 0, 0, 0.82);\n"
            "  border-radius: 14px;\n"
            "  padding: 14px 18px;\n"
            "}\n"
            ".tts-status-label {\n"
            "  color: white;\n"
            "  font-size: 16px;\n"
            "  font-weight: 700;\n"
            "}\n"
            ".tts-preview-label {\n"
            "  color: rgba(255, 255, 255, 0.88);\n"
            "  font-size: 13px;\n"
            "}\n"
            ".tts-control-btn {\n"
            "  font-weight: 600;\n"
            "}\n"
        )
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _setup_widgets(self) -> None:
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        root.add_css_class("tts-overlay-box")
        root.set_spacing(10)

        status_label = Gtk.Label(label="🔈 Idle")
        status_label.add_css_class("tts-status-label")
        status_label.set_halign(Gtk.Align.START)
        self._status_label = status_label
        root.append(status_label)

        preview_label = Gtk.Label(label="")
        preview_label.add_css_class("tts-preview-label")
        preview_label.set_halign(Gtk.Align.START)
        preview_label.set_wrap(True)
        preview_label.set_max_width_chars(56)
        self._preview_label = preview_label
        root.append(preview_label)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        controls.set_spacing(8)

        pause_btn = Gtk.Button(label="⏸ Pause")
        pause_btn.add_css_class("tts-control-btn")
        pause_btn.connect("clicked", self._on_pause_clicked)
        self._pause_btn = pause_btn
        controls.append(pause_btn)

        restart_btn = Gtk.Button(label="⟲ Restart")
        restart_btn.add_css_class("tts-control-btn")
        restart_btn.connect("clicked", self._on_restart_clicked)
        self._restart_btn = restart_btn
        controls.append(restart_btn)

        stop_btn = Gtk.Button(label="■ Stop")
        stop_btn.add_css_class("tts-control-btn")
        stop_btn.connect("clicked", self._on_stop_clicked)
        self._stop_btn = stop_btn
        controls.append(stop_btn)

        voice_store = Gtk.StringList.new(["Default"])
        voice_dropdown = Gtk.DropDown.new(voice_store, None)
        voice_dropdown.connect("notify::selected", self._on_voice_changed)
        voice_dropdown.set_hexpand(True)
        self._voice_store = voice_store
        self._voice_dropdown = voice_dropdown
        controls.append(voice_dropdown)

        root.append(controls)
        self._window.set_child(root)

    def _clear_auto_hide_timer(self) -> None:
        if self._auto_hide_source_id is None:
            return
        GLib.source_remove(self._auto_hide_source_id)
        self._auto_hide_source_id = None

    def _schedule_auto_hide(self) -> None:
        self._clear_auto_hide_timer()
        delay_ms = int(max(0.0, float(self._config.tts_overlay_auto_hide_sec)) * 1000)
        if delay_ms <= 0:
            self._do_hide()
            return
        self._auto_hide_source_id = GLib.timeout_add(delay_ms, self._do_hide)

    def _render(self) -> None:
        if self._status_label is not None:
            error_message = None
            if self._state == TTS_OVERLAY_ERROR:
                error_message = self._preview_text
            self._status_label.set_text(
                status_label_for_state(self._state, error_message=error_message)
            )

        if self._preview_label is not None:
            if self._state == TTS_OVERLAY_ERROR:
                self._preview_label.set_text("")
            else:
                self._preview_label.set_text(self._preview_text)

        if self._pause_btn is not None:
            self._pause_btn.set_label(
                "▶ Resume" if self._state == TTS_OVERLAY_PAUSED else "⏸ Pause"
            )

    def _on_pause_clicked(self, _button: Gtk.Button) -> None:
        if self._state == TTS_OVERLAY_PAUSED:
            if self._on_resume is not None:
                self._on_resume()
        else:
            if self._on_pause is not None:
                self._on_pause()

    def _on_restart_clicked(self, _button: Gtk.Button) -> None:
        if self._on_restart is not None:
            self._on_restart()

    def _on_stop_clicked(self, _button: Gtk.Button) -> None:
        if self._on_stop is not None:
            self._on_stop()
        self.hide()

    def _on_voice_changed(self, dropdown: Gtk.DropDown, _param_spec) -> None:
        idx = int(dropdown.get_selected())
        if idx < 0 or idx >= len(self._voices):
            return

        selected = self._voices[idx]
        self._voice_selected_id = selected.id
        if self._on_voice_selected is not None:
            self._on_voice_selected(selected.id)

    # -- Thread-safe public API ---------------------------------------------

    def show(self) -> None:
        GLib.idle_add(self._do_show)

    def hide(self) -> None:
        GLib.idle_add(self._do_hide)

    def set_state(
        self, state: str, *, preview_text: str = "", error_message: str | None = None
    ) -> None:
        GLib.idle_add(self._do_set_state, state, preview_text, error_message)

    def set_voices(self, voices: list[VoiceInfo], selected_voice_id: str | None = None) -> None:
        GLib.idle_add(self._do_set_voices, list(voices), selected_voice_id)

    # -- GLib callbacks ------------------------------------------------------

    def _do_show(self):
        self._clear_auto_hide_timer()
        self._window.set_visible(True)
        self._visible = True
        return GLib.SOURCE_REMOVE

    def _do_hide(self):
        self._clear_auto_hide_timer()
        self._window.set_visible(False)
        self._visible = False
        if self._state != TTS_OVERLAY_ERROR:
            self._state = TTS_OVERLAY_IDLE
            self._preview_text = ""
            self._render()
        return GLib.SOURCE_REMOVE

    def _do_set_state(
        self,
        state: str,
        preview_text: str,
        error_message: str | None,
    ):
        self._state = state
        if state == TTS_OVERLAY_ERROR:
            self._preview_text = (error_message or "TTS failed").strip()
        else:
            self._preview_text = summarize_preview(preview_text)
        self._render()

        if state == TTS_OVERLAY_IDLE:
            self._schedule_auto_hide()
        else:
            self._do_show()
        return GLib.SOURCE_REMOVE

    def _do_set_voices(self, voices: list[VoiceInfo], selected_voice_id: str | None):
        self._voices = list(voices)
        target_id = str(selected_voice_id or self._voice_selected_id).strip()

        store = self._voice_store
        dropdown = self._voice_dropdown
        if store is None or dropdown is None:
            return GLib.SOURCE_REMOVE

        while store.get_n_items() > 0:
            store.remove(0)

        if not self._voices:
            store.append("Voice: default")
            dropdown.set_selected(0)
            dropdown.set_sensitive(False)
            return GLib.SOURCE_REMOVE

        dropdown.set_sensitive(True)

        selected_index = 0
        for idx, voice in enumerate(self._voices):
            store.append(f"Voice: {voice.name}")
            if voice.id == target_id:
                selected_index = idx

        dropdown.set_selected(selected_index)
        self._voice_selected_id = self._voices[selected_index].id
        return GLib.SOURCE_REMOVE
