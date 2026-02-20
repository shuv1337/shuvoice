"""GTK4 layer-shell transparent overlay for speech-to-text captions.

IMPORTANT: ctypes.CDLL('libgtk4-layer-shell.so') must be called
before this module is imported. See __main__.py.
"""

import logging

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gtk4LayerShell", "1.0")
from gi.repository import Gdk, GLib, Gtk
from gi.repository import Gtk4LayerShell as LayerShell

log = logging.getLogger(__name__)


class CaptionOverlay:
    """Manages a transparent caption overlay window on a layer shell surface."""

    def __init__(self, app: Gtk.Application, config):
        self._config = config
        self._window = Gtk.Window(application=app)
        self._label: Gtk.Label | None = None
        self._visible = False
        self._setup_layer_shell()
        self._setup_css()
        self._setup_widgets()
        # Make pointer-click-through after the window is mapped (fix #1)
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
        LayerShell.set_exclusive_zone(w, -1)  # Don't reserve screen space
        LayerShell.set_namespace(w, "stt-overlay")

        # Anchor bottom only — auto-centers horizontally on wlroots compositors.
        # Fix #2: if centering doesn't work, anchor LEFT+RIGHT with margins.
        LayerShell.set_anchor(w, LayerShell.Edge.BOTTOM, True)
        LayerShell.set_margin(w, LayerShell.Edge.BOTTOM, self._config.bottom_margin)

    @staticmethod
    def _make_click_through(window):
        """Set an empty input region so pointer events pass through (fix #1)."""
        try:
            import cairo

            surface = window.get_surface()
            if surface:
                empty = cairo.Region()
                surface.set_input_region(empty)
                log.debug("Set empty input region for pointer passthrough")
        except Exception as e:
            log.debug(
                "Could not set input region (%s) — "
                "pointer passthrough may depend on compositor behavior",
                e,
            )

    def _setup_css(self):
        cfg = self._config
        css = Gtk.CssProvider()
        css.load_from_string(
            f"window {{ background-color: transparent; }}\n"
            f".caption-box {{\n"
            f"  background-color: rgba(0, 0, 0, {cfg.bg_opacity});\n"
            f"  border-radius: {cfg.border_radius}px;\n"
            f"  padding: 16px 28px;\n"
            f"}}\n"
            f".caption-label {{\n"
            f"  color: white;\n"
            f"  font-size: {cfg.font_size}px;\n"
            f"  font-weight: bold;\n"
            f"}}\n"
            f".recording-icon {{\n"
            f"  color: white;\n"
            f"}}\n"
        )
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _setup_widgets(self):
        # Use a horizontal box to place icon next to text
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        box.add_css_class("caption-box")
        box.set_spacing(12)

        # Active microphone icon
        icon = Gtk.Image.new_from_icon_name("microphone-sensitivity-high-symbolic")
        # Scale icon with font size (roughly 1.2x)
        icon_size = int(self._config.font_size * 1.2)
        icon.set_pixel_size(icon_size)
        icon.set_valign(Gtk.Align.CENTER)
        icon.add_css_class("recording-icon")
        # Avoid PyGObject API mismatch across GTK builds (get_accessible may not exist)
        # Keep a human hint without crashing the overlay init.
        icon.set_tooltip_text("Microphone active")

        box.append(icon)

        self._label = Gtk.Label(label="")
        self._label.add_css_class("caption-label")
        self._label.set_wrap(True)
        self._label.set_max_width_chars(60)

        box.append(self._label)
        self._window.set_child(box)

        # Start hidden
        self._window.set_visible(False)
        self._visible = False

    # -- Thread-safe public API (called from ASR / hotkey threads) ----------

    def set_text(self, text: str):
        GLib.idle_add(self._do_set_text, text)

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

    def _do_show(self):
        if self._window:
            self._window.set_visible(True)
            self._visible = True
        return GLib.SOURCE_REMOVE

    def _do_hide(self):
        if self._window:
            self._window.set_visible(False)
            self._visible = False
            if self._label:
                self._label.set_text("")
        return GLib.SOURCE_REMOVE
