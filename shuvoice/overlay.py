"""GTK4 layer-shell transparent overlay for speech-to-text captions.

IMPORTANT: ctypes.CDLL('libgtk4-layer-shell.so') must be called
before this module is imported. See __main__.py.
"""

from __future__ import annotations

import logging

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gtk4LayerShell", "1.0")
from gi.repository import Gdk, GLib, Gtk
from gi.repository import Gtk4LayerShell as LayerShell

from .overlay_state import (
    OVERLAY_STATE_CLASSES,
    OVERLAY_STATE_ERROR,
    OVERLAY_STATE_LISTENING,
    OVERLAY_STATE_PROCESSING,
    overlay_state_class,
)

log = logging.getLogger(__name__)


class CaptionOverlay:
    """Manages a transparent caption overlay window on a layer shell surface."""

    def __init__(self, app: Gtk.Application, config):
        self._config = config
        self._window = Gtk.Window(application=app)
        self._label: Gtk.Label | None = None
        self._box: Gtk.Box | None = None
        self._visible = False
        self._state = OVERLAY_STATE_LISTENING
        self._setup_layer_shell()
        self._setup_css()
        self._setup_widgets()
        self._window.connect("realize", self._make_click_through)
        self._window.present()

    def _setup_layer_shell(self):
        w = self._window
        if not LayerShell.is_supported():
            log.error("Layer shell not supported — not on a wlroots compositor?")
            return

        LayerShell.init_for_window(w)
        LayerShell.set_layer(w, LayerShell.Layer.OVERLAY)
        LayerShell.set_keyboard_mode(w, LayerShell.KeyboardMode.NONE)
        LayerShell.set_exclusive_zone(w, -1)
        LayerShell.set_namespace(w, "stt-overlay")
        LayerShell.set_anchor(w, LayerShell.Edge.BOTTOM, True)
        LayerShell.set_margin(w, LayerShell.Edge.BOTTOM, self._config.bottom_margin)

    @staticmethod
    def _make_click_through(window):
        """Set an empty input region so pointer events pass through."""
        try:
            import cairo

            surface = window.get_surface()
            if surface:
                empty = cairo.Region()
                surface.set_input_region(empty)
                log.debug("Set empty input region for pointer passthrough")
        except Exception as e:
            log.debug(
                "Could not set input region (%s) — pointer passthrough may depend on compositor",
                e,
            )

    def _setup_css(self):
        cfg = self._config
        css = Gtk.CssProvider()
        css.load_from_string(
            f"window {{ background-color: transparent; }}\n"
            ".caption-box {\n"
            "  background-color: rgba(0, 0, 0, 0.75);\n"
            f"  border-radius: {cfg.border_radius}px;\n"
            "  padding: 16px 28px;\n"
            "}\n"
            f".caption-box.state-listening {{ background-color: rgba(0, 0, 0, {cfg.bg_opacity}); }}\n"
            f".caption-box.state-processing {{ background-color: rgba(20, 45, 90, {cfg.bg_opacity}); }}\n"
            f".caption-box.state-error {{ background-color: rgba(120, 20, 20, {cfg.bg_opacity}); }}\n"
            ".caption-label {\n"
            "  color: white;\n"
            f"  font-size: {cfg.font_size}px;\n"
            "  font-weight: bold;\n"
            "}\n"
            ".recording-icon {\n"
            "  color: white;\n"
            "}\n"
        )
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _setup_widgets(self):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        box.add_css_class("caption-box")
        box.set_spacing(12)

        self._icon = Gtk.Image.new_from_icon_name("microphone-sensitivity-high-symbolic")
        icon_size = int(self._config.font_size * 1.2)
        self._icon.set_pixel_size(icon_size)
        self._icon.set_valign(Gtk.Align.CENTER)
        self._icon.add_css_class("recording-icon")
        self._icon.set_tooltip_text("Microphone active")
        box.append(self._icon)

        self._label = Gtk.Label(label="")
        self._label.add_css_class("caption-label")
        self._label.set_wrap(True)
        self._label.set_max_width_chars(60)
        box.append(self._label)

        self._box = box
        self._window.set_child(box)
        self._apply_state(self._state)

        self._window.set_visible(False)
        self._visible = False

    def _apply_state(self, state: str):
        if not self._box:
            return

        for css_class in OVERLAY_STATE_CLASSES.values():
            self._box.remove_css_class(css_class)

        self._box.add_css_class(overlay_state_class(state))
        self._state = state

        if self._icon:
            status_text = {
                OVERLAY_STATE_LISTENING: "Listening…",
                OVERLAY_STATE_PROCESSING: "Processing…",
                OVERLAY_STATE_ERROR: "Error",
            }.get(state, state)
            self._icon.set_tooltip_text(status_text)
            self._icon.update_property([Gtk.AccessibleProperty.LABEL], [status_text])

    # -- Thread-safe public API (called from ASR / hotkey threads) ----------

    def set_text(self, text: str):
        GLib.idle_add(self._do_set_text, text)

    def set_state(self, state: str):
        GLib.idle_add(self._do_set_state, state)

    def show(self):
        GLib.idle_add(self._do_show)

    def hide(self):
        GLib.idle_add(self._do_hide)

    # -- GLib callbacks (run on GTK main thread) ----------------------------

    def _do_set_text(self, text: str):
        if self._label:
            self._label.set_text(text)
            if text and not self._visible:
                self._do_show()
        return GLib.SOURCE_REMOVE

    def _do_set_state(self, state: str):
        try:
            self._apply_state(state)
        except ValueError:
            log.debug("Ignoring unknown overlay state: %s", state)
        return GLib.SOURCE_REMOVE

    def _do_show(self):
        if self._window:
            self._window.set_visible(True)
            self._visible = True
        return GLib.SOURCE_REMOVE

    def _do_hide(self):
        if self._window:
            self._window.set_visible(False)
            self._visible = False
            self._apply_state(OVERLAY_STATE_LISTENING)
            if self._label:
                self._label.set_text("")
        return GLib.SOURCE_REMOVE
