"""Branded splash overlay shown during model loading.

Uses the Hyprland layer-shell overlay system to display a centered,
click-through splash with the ShuVoice branding while the ASR model
loads in the background.

IMPORTANT: ctypes.CDLL('libgtk4-layer-shell.so') must be called
before this module is imported. See __main__.py.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gtk4LayerShell", "1.0")
from gi.repository import Gdk, GLib, Gtk
from gi.repository import Gtk4LayerShell as LayerShell

log = logging.getLogger(__name__)

# Try to locate branding logo relative to the project root.
_LOGO_CANDIDATES = [
    Path(__file__).resolve().parent.parent
    / "docs"
    / "assets"
    / "branding"
    / "shuvoice-variant-dark-lockup.png",
    Path(__file__).resolve().parent.parent
    / "docs"
    / "assets"
    / "branding"
    / "shuvoice-variant-dark-badge.png",
]


def _find_logo() -> Path | None:
    for candidate in _LOGO_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


class SplashOverlay:
    """Layer-shell overlay that shows a branded splash screen."""

    def __init__(self, app: Gtk.Application):
        self._window = Gtk.Window(application=app)
        self._status: Gtk.Label | None = None
        self._shown_monotonic: float | None = None
        self._setup_layer_shell()
        self._setup_css()
        self._setup_widgets()
        self._window.connect("realize", self._on_realize)
        self._window.connect("realize", self._make_click_through)
        self._window.present()

    def _setup_layer_shell(self):
        w = self._window
        if not LayerShell.is_supported():
            log.warning("Layer shell not supported for splash")
            return

        LayerShell.init_for_window(w)
        LayerShell.set_layer(w, LayerShell.Layer.OVERLAY)
        LayerShell.set_keyboard_mode(w, LayerShell.KeyboardMode.NONE)
        LayerShell.set_exclusive_zone(w, -1)
        LayerShell.set_namespace(w, "shuvoice-splash")

    def _setup_css(self):
        css = Gtk.CssProvider()
        css.load_from_string(
            "window.splash-window { background-color: transparent; }\n"
            ".splash-box {\n"
            "  background-color: rgba(15, 15, 20, 0.92);\n"
            "  border-radius: 24px;\n"
            "  padding: 48px 64px;\n"
            "}\n"
            ".splash-title {\n"
            "  color: white;\n"
            "  font-size: 32px;\n"
            "  font-weight: bold;\n"
            "}\n"
            ".splash-status {\n"
            "  color: rgba(255, 255, 255, 0.6);\n"
            "  font-size: 16px;\n"
            "  margin-top: 8px;\n"
            "}\n"
        )
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _setup_widgets(self):
        self._window.add_css_class("splash-window")

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box.add_css_class("splash-box")
        box.set_halign(Gtk.Align.CENTER)
        box.set_valign(Gtk.Align.CENTER)
        box.set_spacing(8)

        logo_path = _find_logo()
        if logo_path:
            try:
                logo = Gtk.Picture.new_for_filename(str(logo_path))
                logo.set_can_shrink(True)
                logo.set_size_request(300, -1)
                logo.set_halign(Gtk.Align.CENTER)
                box.append(logo)
            except Exception as e:
                log.debug("Could not load splash logo: %s", e)
                self._add_text_branding(box)
        else:
            self._add_text_branding(box)

        self._status = Gtk.Label(label="Loading model\u2026")
        self._status.add_css_class("splash-status")
        self._status.set_halign(Gtk.Align.CENTER)
        box.append(self._status)

        self._window.set_child(box)

    @property
    def shown_monotonic(self) -> float | None:
        return self._shown_monotonic

    def _on_realize(self, _window):
        if self._shown_monotonic is None:
            self._shown_monotonic = time.monotonic()

    @staticmethod
    def _add_text_branding(box: Gtk.Box):
        title = Gtk.Label(label="ShuVoice")
        title.add_css_class("splash-title")
        title.set_halign(Gtk.Align.CENTER)
        box.append(title)

    @staticmethod
    def _make_click_through(window):
        """Set an empty input region so pointer events pass through."""
        try:
            import cairo

            surface = window.get_surface()
            if surface:
                empty = cairo.Region()
                surface.set_input_region(empty)
        except Exception:
            pass

    # -- Thread-safe public API ------------------------------------------------

    def set_status(self, text: str):
        GLib.idle_add(self._do_set_status, text)

    def dismiss(self):
        GLib.idle_add(self._do_dismiss)

    # -- GLib callbacks (run on GTK main thread) -------------------------------

    def _do_set_status(self, text: str):
        if self._status:
            self._status.set_text(text)
        return GLib.SOURCE_REMOVE

    def _do_dismiss(self):
        if self._window:
            self._window.set_visible(False)
            self._window.destroy()
            self._window = None
            self._status = None
        return GLib.SOURCE_REMOVE
