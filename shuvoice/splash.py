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

from .branding import logo_candidates

log = logging.getLogger(__name__)

# Keep this as a module variable so tests can patch deterministic candidates.
_LOGO_CANDIDATES = logo_candidates()


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
        self._progress: Gtk.ProgressBar | None = None
        self._pulse_source_id: int | None = None
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
            ".splash-progress {\n"
            "  min-width: 340px;\n"
            "  margin-top: 10px;\n"
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
                logo.set_alternative_text("ShuVoice logo")
                logo.set_size_request(300, -1)
                logo.set_halign(Gtk.Align.CENTER)
                box.append(logo)
            except Exception as e:
                log.debug("Could not load splash logo: %s", e)
                self._add_text_branding(box)
        else:
            self._add_text_branding(box)

        self._status = Gtk.Label(label="Loading model…")
        self._status.add_css_class("splash-status")
        self._status.set_halign(Gtk.Align.CENTER)
        self._status.set_accessible_role(Gtk.AccessibleRole.STATUS)
        box.append(self._status)

        self._progress = Gtk.ProgressBar()
        self._progress.add_css_class("splash-progress")
        self._progress.set_show_text(True)
        self._progress.set_text("Starting…")
        self._progress.set_fraction(0.0)
        self._progress.set_halign(Gtk.Align.FILL)
        self._progress.set_hexpand(True)
        box.append(self._progress)

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

    def set_progress(self, fraction: float | None, text: str | None = None):
        GLib.idle_add(self._do_set_progress, fraction, text)

    def dismiss(self):
        GLib.idle_add(self._do_dismiss)

    # -- GLib callbacks (run on GTK main thread) -------------------------------

    def _do_set_status(self, text: str):
        if self._status:
            self._status.set_text(text)
            self._status.update_property([Gtk.AccessibleProperty.LABEL], [text])
        return GLib.SOURCE_REMOVE

    def _pulse_progress(self):
        progress = getattr(self, "_progress", None)
        window = getattr(self, "_window", None)
        if progress is None or window is None:
            self._pulse_source_id = None
            return GLib.SOURCE_REMOVE

        progress.pulse()
        return GLib.SOURCE_CONTINUE

    def _do_set_progress(self, fraction: float | None, text: str | None):
        if text:
            self._do_set_status(text)

        progress = getattr(self, "_progress", None)
        if progress is None:
            return GLib.SOURCE_REMOVE

        pulse_source_id = getattr(self, "_pulse_source_id", None)

        if fraction is None:
            if pulse_source_id is None:
                self._pulse_source_id = GLib.timeout_add(120, self._pulse_progress)
            progress.set_show_text(True)
            if text:
                progress.set_text(text)
            return GLib.SOURCE_REMOVE

        if pulse_source_id is not None:
            GLib.source_remove(pulse_source_id)
            self._pulse_source_id = None

        bounded = max(0.0, min(1.0, float(fraction)))
        progress.set_fraction(bounded)
        progress.set_show_text(True)
        if text:
            progress.set_text(text)
        else:
            progress.set_text(f"Loading model… {int(round(bounded * 100))}%")
        return GLib.SOURCE_REMOVE

    def _do_dismiss(self):
        pulse_source_id = getattr(self, "_pulse_source_id", None)
        if pulse_source_id is not None:
            GLib.source_remove(pulse_source_id)
            self._pulse_source_id = None

        if self._window:
            self._window.set_visible(False)
            self._window.destroy()
            self._window = None
            self._status = None
            self._progress = None
        return GLib.SOURCE_REMOVE
